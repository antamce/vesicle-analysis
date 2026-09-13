from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import (
    QObject,
    QRectF,
    QRunnable,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QGuiApplication,
    QIcon,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from vesicle_analysis.analysis import (
    PeakDetectionResult,
    RectangleBounds,
    clip_rectangle,
    detect_recording_peak,
    overlaps_any,
)
from vesicle_analysis.background import (
    BackgroundParameters,
    BackgroundResult,
    background_labels_to_rgba,
    find_recording_backgrounds,
)
from vesicle_analysis.corrections import (
    LabelOverlapError,
    MaskEdit,
    analysis_region_mask,
    apply_shape,
    delete_pools,
    ellipse_mask,
    guided_pool_mask,
    make_edit,
    merge_pools,
    polygon_mask,
    split_pool_with_line,
)
from vesicle_analysis.nd2_io import (
    RecordingMetadata,
    discover_nd2_files,
    read_frame,
    read_metadata,
)
from vesicle_analysis.segmentation import (
    SegmentationParameters,
    SegmentationResult,
    labels_to_rgba,
    segment_recording,
)
from vesicle_analysis.trace_export import (
    AnalysisExportRequest,
    AnalysisExportResult,
    export_analysis_bundle,
)

pg.setConfigOptions(imageAxisOrder="row-major", useOpenGL=False)


def application_icon_path() -> Path | None:
    """Return the best platform icon available in the source checkout."""
    project_root = Path(__file__).resolve().parents[1]
    if sys.platform == "win32":
        platform_icon = project_root / "launcher" / "assets" / "icon.ico"
    elif sys.platform == "darwin":
        platform_icon = project_root / "launcher" / "assets" / "icon.icns"
    else:
        platform_icon = project_root / "icon.png"
    candidates = [platform_icon, project_root / "icon.png"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


@dataclass(slots=True)
class RecordingAnalysisState:
    rectangles: list[RectangleBounds] = field(default_factory=list)
    peak_result: PeakDetectionResult | None = None
    peak_error: str | None = None
    manual_peak_index: int | None = None
    segmentation_parameters: SegmentationParameters = field(
        default_factory=SegmentationParameters
    )
    segmentation_result: SegmentationResult | None = None
    corrected_labels: np.ndarray[Any, Any] | None = None
    correction_undo: list[MaskEdit] = field(default_factory=list)
    correction_redo: list[MaskEdit] = field(default_factory=list)
    selected_pool_ids: set[int] = field(default_factory=set)
    background_parameters: BackgroundParameters = field(
        default_factory=BackgroundParameters
    )
    background_result: BackgroundResult | None = None
    corrected_background_labels: np.ndarray[Any, Any] | None = None
    background_assignments: dict[int, int | None] = field(default_factory=dict)
    selected_background_id: int | None = None
    background_undo: list[_BackgroundEdit] = field(default_factory=list)
    background_redo: list[_BackgroundEdit] = field(default_factory=list)

    @property
    def effective_peak_index(self) -> int | None:
        if self.manual_peak_index is not None:
            return self.manual_peak_index
        if self.peak_result is not None:
            return self.peak_result.automatic_peak_index
        return None


@dataclass(slots=True)
class _PeakBatchResult:
    results: dict[Path, PeakDetectionResult]
    failures: dict[Path, str]


@dataclass(slots=True)
class _BackgroundEdit:
    mask_edit: MaskEdit
    before_assignments: dict[int, int | None]
    after_assignments: dict[int, int | None]
    description: str


def _detect_peak_batch(
    requests: tuple[tuple[Path, tuple[RectangleBounds, ...]], ...],
) -> _PeakBatchResult:
    """Calculate one independent peak per recording in a single batch action."""
    results: dict[Path, PeakDetectionResult] = {}
    failures: dict[Path, str] = {}
    for path, rectangles in requests:
        try:
            results[path] = detect_recording_peak(path, rectangles)
        except Exception as error:  # noqa: BLE001 - preserve other batch results
            failures[path] = f"{type(error).__name__}: {error}"
    return _PeakBatchResult(results=results, failures=failures)


class DrawingViewBox(pg.ViewBox):
    rectangle_drawn = Signal(object)
    correction_drawn = Signal(object)
    image_clicked = Signal(object)

    def __init__(self) -> None:
        super().__init__(lockAspect=True)
        self._rectangle_drawing_enabled = False
        self._correction_mode: str | None = None
        self._drawing_start: Any | None = None
        self._temporary_roi: Any | None = None
        self._temporary_curve: pg.PlotCurveItem | None = None
        self._stroke_points: list[tuple[float, float]] = []

    def set_rectangle_drawing_enabled(self, enabled: bool) -> None:
        self._rectangle_drawing_enabled = enabled
        if enabled:
            self.set_correction_drawing_mode(None)
        if not enabled:
            self._discard_temporary_roi()

    def set_correction_drawing_mode(self, mode: str | None) -> None:
        if mode not in {None, "freehand", "oval", "separate", "hint", "background"}:
            raise ValueError(f"Unknown correction drawing mode: {mode}")
        self._correction_mode = mode
        if mode is not None:
            self._rectangle_drawing_enabled = False
        self._discard_temporary_roi()
        self._discard_temporary_curve()

    def mouseClickEvent(self, event: Any) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseClickEvent(event)
            return
        position = self.mapSceneToView(event.scenePos())
        if self._correction_mode == "hint":
            event.accept()
            self.correction_drawn.emit(("hint", ((position.x(), position.y()),)))
            return
        if self._correction_mode is None and not self._rectangle_drawing_enabled:
            event.accept()
            toggle = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            self.image_clicked.emit((position.x(), position.y(), toggle))
            return
        super().mouseClickEvent(event)

    def mouseDragEvent(self, event: Any, axis: Any | None = None) -> None:
        if (
            self._correction_mode is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._correction_drag_event(event)
            return
        if (
            not self._rectangle_drawing_enabled
            or event.button() != Qt.MouseButton.LeftButton
        ):
            super().mouseDragEvent(event, axis=axis)
            return

        event.accept()
        current = self.mapSceneToView(event.scenePos())
        if event.isStart():
            self._drawing_start = self.mapSceneToView(event.buttonDownScenePos())
            self._temporary_roi = pg.RectROI(
                (self._drawing_start.x(), self._drawing_start.y()),
                (1, 1),
                movable=False,
                resizable=False,
                pen=pg.mkPen("#ffd166", width=2, style=Qt.PenStyle.DashLine),
            )
            self.addItem(self._temporary_roi)

        if self._drawing_start is None or self._temporary_roi is None:
            return
        left = min(self._drawing_start.x(), current.x())
        top = min(self._drawing_start.y(), current.y())
        right = max(self._drawing_start.x(), current.x())
        bottom = max(self._drawing_start.y(), current.y())
        self._temporary_roi.setPos((left, top))
        self._temporary_roi.setSize((right - left, bottom - top))

        if event.isFinish():
            self._discard_temporary_roi()
            self.rectangle_drawn.emit((left, top, right, bottom))

    def _correction_drag_event(self, event: Any) -> None:
        event.accept()
        current = self.mapSceneToView(event.scenePos())
        mode = self._correction_mode
        if event.isStart():
            self._drawing_start = self.mapSceneToView(event.buttonDownScenePos())
            if mode in {"oval", "background"}:
                roi_type = pg.EllipseROI if mode == "oval" else pg.RectROI
                self._temporary_roi = roi_type(
                    (self._drawing_start.x(), self._drawing_start.y()),
                    (1, 1),
                    movable=False,
                    resizable=False,
                    pen=pg.mkPen("#ff7ad9", width=2),
                )
                self.addItem(self._temporary_roi)
            else:
                self._stroke_points = [
                    (self._drawing_start.x(), self._drawing_start.y())
                ]
                self._temporary_curve = pg.PlotCurveItem(
                    pen=pg.mkPen("#ff7ad9", width=3)
                )
                self.addItem(self._temporary_curve)

        if mode in {"oval", "background"} and self._drawing_start is not None:
            left = min(self._drawing_start.x(), current.x())
            top = min(self._drawing_start.y(), current.y())
            right = max(self._drawing_start.x(), current.x())
            bottom = max(self._drawing_start.y(), current.y())
            if self._temporary_roi is not None:
                self._temporary_roi.setPos((left, top))
                self._temporary_roi.setSize((right - left, bottom - top))
            if event.isFinish():
                self._discard_temporary_roi()
                self.correction_drawn.emit((mode, (left, top, right, bottom)))
            return

        point = (current.x(), current.y())
        if not self._stroke_points or point != self._stroke_points[-1]:
            self._stroke_points.append(point)
        if self._temporary_curve is not None:
            self._temporary_curve.setData(
                [value[0] for value in self._stroke_points],
                [value[1] for value in self._stroke_points],
            )
        if event.isFinish():
            points = tuple(self._stroke_points)
            self._discard_temporary_curve()
            if mode is not None:
                self.correction_drawn.emit((mode, points))

    def _discard_temporary_roi(self) -> None:
        if self._temporary_roi is not None:
            self.removeItem(self._temporary_roi)
        self._temporary_roi = None
        self._drawing_start = None

    def _discard_temporary_curve(self) -> None:
        if self._temporary_curve is not None:
            self.removeItem(self._temporary_curve)
        self._temporary_curve = None
        self._stroke_points.clear()


class _WorkerSignals(QObject):
    succeeded = Signal(int, object)
    failed = Signal(int, str)


class _FunctionTask(QRunnable):
    def __init__(self, token: int, function: Callable[[], Any]) -> None:
        super().__init__()
        self.token = token
        self.function = function
        self.signals = _WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.function()
        # This is the thread-to-GUI error boundary. Any worker failure must be
        # converted to a Qt signal rather than escaping the worker thread.
        except Exception as error:  # noqa: BLE001
            self.signals.failed.emit(self.token, f"{type(error).__name__}: {error}")
        else:
            self.signals.succeeded.emit(self.token, result)


class ImageCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame: np.ndarray[Any, Any] | None = None
        self._levels: tuple[float, float] | None = None

        self.graphics = pg.GraphicsLayoutWidget()
        self.view_box = DrawingViewBox()
        self.graphics.addItem(self.view_box, row=0, col=0)
        self.view_box.invertY(True)
        self.image_item = pg.ImageItem()
        self.view_box.addItem(self.image_item)
        self.background_item = pg.ImageItem()
        self.background_item.setZValue(14)
        self.view_box.addItem(self.background_item)
        self.segmentation_item = pg.ImageItem()
        self.segmentation_item.setZValue(15)
        self.view_box.addItem(self.segmentation_item)
        self.histogram = pg.HistogramLUTItem(image=self.image_item)
        self.graphics.addItem(self.histogram, row=0, col=1)

        self.message = QLabel("Open an ND2 file or a folder to begin.")
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.message.setStyleSheet("color: #666; padding: 8px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.message)
        layout.addWidget(self.graphics, 1)

    @property
    def frame(self) -> np.ndarray[Any, Any] | None:
        return self._frame

    @property
    def display_levels(self) -> tuple[float, float] | None:
        return self._levels

    def display_lut(self) -> np.ndarray[Any, Any]:
        lut = np.asarray(
            self.histogram.gradient.getLookupTable(256, alpha=False),
            dtype=np.uint8,
        )
        if lut.shape != (256, 3):
            raise ValueError("The current display gradient is not a 256-color RGB LUT.")
        return lut.copy()

    def clear_for_new_recording(self) -> None:
        self._frame = None
        self._levels = None
        self.image_item.clear()
        self.background_item.clear()
        self.segmentation_item.clear()
        self.message.setText("Loading recording…")
        self.message.show()

    def set_frame(self, frame: np.ndarray[Any, Any]) -> None:
        first_frame = self._frame is None
        self._frame = frame
        if self._levels is None:
            self._levels = _display_levels(frame)
        self.image_item.setImage(frame, autoLevels=False, levels=self._levels)
        self.histogram.setLevels(*self._levels)
        self.message.hide()
        if first_frame:
            self.view_box.autoRange()

    def auto_contrast(self) -> None:
        if self._frame is None:
            return
        self._levels = _display_levels(self._frame)
        self.image_item.setLevels(self._levels)
        self.histogram.setLevels(*self._levels)

    def reset_zoom(self) -> None:
        if self._frame is not None:
            self.view_box.autoRange()

    def set_segmentation(
        self,
        labels: np.ndarray[Any, Any],
        selected_ids: tuple[int, ...] = (),
        warning_ids: tuple[int, ...] = (),
    ) -> None:
        self.segmentation_item.setImage(
            labels_to_rgba(labels, selected_ids, warning_ids), autoLevels=False
        )

    def clear_segmentation(self) -> None:
        self.segmentation_item.clear()

    def set_backgrounds(
        self, labels: np.ndarray[Any, Any], selected_ids: tuple[int, ...] = ()
    ) -> None:
        self.background_item.setImage(
            background_labels_to_rgba(labels, selected_ids), autoLevels=False
        )

    def clear_backgrounds(self) -> None:
        self.background_item.clear()


def _display_levels(frame: np.ndarray[Any, Any]) -> tuple[float, float]:
    low, high = np.percentile(frame, (0.5, 99.8))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vesicle Analysis")
        icon_path = application_icon_path()
        if icon_path is not None:
            self.setWindowIcon(QIcon(str(icon_path)))

        self._thread_pool = QThreadPool(self)
        self._thread_pool.setMaxThreadCount(2)
        self._analysis_thread_pool = QThreadPool(self)
        self._analysis_thread_pool.setMaxThreadCount(1)
        self._metadata_token = 0
        self._frame_token = 0
        self._peak_token = 0
        self._peak_batch_running = False
        self._segmentation_token = 0
        self._background_token = 0
        self._export_token = 0
        self._export_running = False
        self._last_export_directory: Path | None = None
        self._current_path: Path | None = None
        self._metadata: RecordingMetadata | None = None
        self._analysis_states: dict[Path, RecordingAnalysisState] = {}
        self._region_items: list[pg.RectROI] = []
        self._region_labels: list[pg.TextItem] = []
        self._selected_region_index = -1
        self._updating_region = False
        self._background_move_roi: pg.RectROI | None = None
        self._background_move_origin: tuple[int, int] | None = None

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_playback)

        self._build_actions()
        self._build_ui()
        self._build_shortcuts()
        self._apply_screen_safe_initial_size()
        self.statusBar().showMessage("Ready")

    def _apply_screen_safe_initial_size(self) -> None:
        """Choose a normal-state size that stays inside the available desktop."""
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1024, 700)
            return
        available = screen.availableGeometry()
        width = min(1280, max(1, int(available.width() * 0.9)))
        height = min(820, max(1, int(available.height() * 0.9)))
        self.resize(width, height)

    @property
    def current_metadata(self) -> RecordingMetadata | None:
        return self._metadata

    @property
    def current_analysis_state(self) -> RecordingAnalysisState | None:
        if self._current_path is None:
            return None
        return self._analysis_states.get(self._current_path)

    def _build_actions(self) -> None:
        self.open_file_action = QAction("Open ND2 File…", self)
        self.open_file_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_file_action.triggered.connect(self._choose_file)

        self.open_folder_action = QAction("Open Folder…", self)
        self.open_folder_action.setShortcut("Ctrl+Shift+O")
        self.open_folder_action.triggered.connect(self._choose_folder)

        self.draw_region_action = QAction("Edit Regions", self)
        self.draw_region_action.setCheckable(True)
        self.draw_region_action.setEnabled(False)
        self.draw_region_action.setToolTip(
            "Unlock analysis rectangles for selection, addition, moving, and resizing"
        )
        self.draw_region_action.toggled.connect(self._set_region_drawing)

        self.delete_region_action = QAction("Delete Region", self)
        self.delete_region_action.setEnabled(False)
        self.delete_region_action.triggered.connect(self._delete_selected_region)

        self.clear_regions_action = QAction("Clear Regions", self)
        self.clear_regions_action.setEnabled(False)
        self.clear_regions_action.triggered.connect(
            self._clear_regions_with_confirmation
        )

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_file_action)
        file_menu.addAction(self.open_folder_action)

        analysis_menu = self.menuBar().addMenu("&Analysis")
        analysis_menu.addAction(self.draw_region_action)
        analysis_menu.addAction(self.delete_region_action)
        analysis_menu.addAction(self.clear_regions_action)

        toolbar = self.addToolBar("Files")
        toolbar.setMovable(False)
        toolbar.addAction(self.open_file_action)
        toolbar.addAction(self.open_folder_action)
        toolbar.addSeparator()
        toolbar.addAction(self.draw_region_action)

    def _build_ui(self) -> None:
        self.recording_list = QListWidget()
        self.recording_list.setMinimumWidth(100)
        self.recording_list.currentRowChanged.connect(self._recording_selected)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Recordings"))
        left_layout.addWidget(self.recording_list, 1)

        self.canvas = ImageCanvas()
        self.canvas.view_box.rectangle_drawn.connect(self._rectangle_drawn)
        self.canvas.view_box.correction_drawn.connect(self._correction_drawn)
        self.canvas.view_box.image_clicked.connect(self._pool_clicked)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(4, 4, 4, 4)
        center_layout.addWidget(self.canvas, 1)
        center_layout.addWidget(self._create_frame_controls())

        self.metadata_labels: dict[str, QLabel] = {}
        metadata_form = QFormLayout()
        metadata_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        for key, title in (
            ("filename", "File"),
            ("path", "Location"),
            ("dimensions", "Dimensions"),
            ("dtype", "Pixel type"),
            ("pixel_size", "Pixel size"),
            ("file_size", "File size"),
            ("duration", "Duration"),
            ("interval", "Median interval"),
            ("exposure", "Exposure"),
        ):
            label = QLabel("—")
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setWordWrap(key == "path")
            if key == "path":
                label.setSizePolicy(
                    QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
                )
            self.metadata_labels[key] = label
            metadata_form.addRow(f"{title}:", label)

        right = QWidget()
        right.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Recording metadata"))
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        right_layout.addWidget(divider)
        right_layout.addLayout(metadata_form)
        self.analysis_controls = self._create_analysis_controls()
        right_layout.addWidget(self.analysis_controls)
        right_layout.addStretch(1)

        self.analysis_scroll_area = QScrollArea()
        self.analysis_scroll_area.setObjectName("analysisScrollArea")
        self.analysis_scroll_area.setWidgetResizable(True)
        self.analysis_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.analysis_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.analysis_scroll_area.setMinimumWidth(260)
        self.analysis_scroll_area.setWidget(right)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(self.analysis_scroll_area)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([210, 770, 300])
        self.setCentralWidget(splitter)

    def _create_analysis_controls(self) -> QWidget:
        group = QGroupBox("Analysis regions and peak")
        group.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(group)

        help_label = QLabel(
            "Regions are locked during analysis. Enable region editing to add, "
            "select, move, or resize analysis rectangles."
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        self.edit_regions_button = QPushButton("Edit analysis regions")
        self.edit_regions_button.setCheckable(True)
        self.edit_regions_button.setEnabled(False)
        self.edit_regions_button.toggled.connect(self.draw_region_action.setChecked)
        self.draw_region_action.changed.connect(self._sync_region_edit_button)
        layout.addWidget(self.edit_regions_button)

        self.region_list = QListWidget()
        self.region_list.setMinimumHeight(100)
        self.region_list.currentRowChanged.connect(self._region_list_selected)
        layout.addWidget(self.region_list, 1)

        region_buttons = QHBoxLayout()
        delete_button = QToolButton()
        delete_button.setDefaultAction(self.delete_region_action)
        clear_button = QToolButton()
        clear_button.setDefaultAction(self.clear_regions_action)
        region_buttons.addWidget(delete_button)
        region_buttons.addWidget(clear_button)
        region_buttons.addStretch(1)
        layout.addLayout(region_buttons)

        self.find_peak_button = QPushButton("Calculate all recording peaks")
        self.find_peak_button.setEnabled(False)
        self.find_peak_button.setToolTip(
            "One click calculates a separate peak for every loaded recording "
            "that has analysis rectangles"
        )
        self.find_peak_button.clicked.connect(self._start_peak_detection)
        layout.addWidget(self.find_peak_button)

        self.peak_summary_label = QLabel("Add at least one rectangle.")
        self.peak_summary_label.setWordWrap(True)
        self.peak_summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.peak_summary_label)

        self.override_peak_checkbox = QCheckBox("Override automatic peak")
        self.override_peak_checkbox.setEnabled(False)
        self.override_peak_checkbox.toggled.connect(self._override_peak_toggled)
        layout.addWidget(self.override_peak_checkbox)

        self.manual_peak_spin = QSpinBox()
        self.manual_peak_spin.setPrefix("Use frame ")
        self.manual_peak_spin.setEnabled(False)
        self.manual_peak_spin.valueChanged.connect(self._manual_peak_changed)
        layout.addWidget(self.manual_peak_spin)

        self.go_to_peak_button = QPushButton("Go to effective peak")
        self.go_to_peak_button.setEnabled(False)
        self.go_to_peak_button.clicked.connect(self._go_to_effective_peak)
        layout.addWidget(self.go_to_peak_button)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(separator)
        layout.addWidget(QLabel("Automatic pool segmentation"))

        segmentation_form = QFormLayout()
        segmentation_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.segmentation_threshold_spin = QDoubleSpinBox()
        self.segmentation_threshold_spin.setRange(1.5, 8.0)
        self.segmentation_threshold_spin.setSingleStep(0.25)
        self.segmentation_threshold_spin.setValue(6.0)
        self.segmentation_threshold_spin.setSuffix(" sigma")
        self.segmentation_threshold_spin.setToolTip(
            "Lower values detect more and dimmer candidate pools"
        )
        self.segmentation_threshold_spin.valueChanged.connect(
            self._segmentation_parameters_changed
        )
        segmentation_form.addRow(
            "Detection threshold:", self.segmentation_threshold_spin
        )

        self.segmentation_min_area_spin = QSpinBox()
        self.segmentation_min_area_spin.setRange(1, 1000)
        self.segmentation_min_area_spin.setValue(8)
        self.segmentation_min_area_spin.setSuffix(" px²")
        self.segmentation_min_area_spin.valueChanged.connect(
            self._segmentation_parameters_changed
        )
        segmentation_form.addRow("Minimum area:", self.segmentation_min_area_spin)

        self.segmentation_max_area_spin = QSpinBox()
        self.segmentation_max_area_spin.setRange(1, 10000)
        self.segmentation_max_area_spin.setValue(150)
        self.segmentation_max_area_spin.setSuffix(" px²")
        self.segmentation_max_area_spin.valueChanged.connect(
            self._segmentation_parameters_changed
        )
        segmentation_form.addRow("Maximum area:", self.segmentation_max_area_spin)

        self.segmentation_distance_spin = QSpinBox()
        self.segmentation_distance_spin.setRange(1, 30)
        self.segmentation_distance_spin.setValue(4)
        self.segmentation_distance_spin.setSuffix(" px")
        self.segmentation_distance_spin.valueChanged.connect(
            self._segmentation_parameters_changed
        )
        segmentation_form.addRow("Seed separation:", self.segmentation_distance_spin)
        layout.addLayout(segmentation_form)

        self.run_segmentation_button = QPushButton("Run automatic segmentation")
        self.run_segmentation_button.setEnabled(False)
        self.run_segmentation_button.clicked.connect(self._start_segmentation)
        layout.addWidget(self.run_segmentation_button)

        self.show_segmentation_checkbox = QCheckBox("Show segmentation overlay")
        self.show_segmentation_checkbox.setChecked(True)
        self.show_segmentation_checkbox.setEnabled(False)
        self.show_segmentation_checkbox.toggled.connect(
            self._segmentation_visibility_changed
        )
        layout.addWidget(self.show_segmentation_checkbox)

        self.segmentation_summary_label = QLabel(
            "Calculate a recording peak before segmentation."
        )
        self.segmentation_summary_label.setWordWrap(True)
        layout.addWidget(self.segmentation_summary_label)

        correction_separator = QFrame()
        correction_separator.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(correction_separator)
        layout.addWidget(QLabel("Manual pool correction"))
        correction_help = QLabel(
            "Select a pool in the list or click its overlay. Ctrl-click or use "
            "the list to select several pools. Drawing tools apply once."
        )
        correction_help.setWordWrap(True)
        layout.addWidget(correction_help)

        self.pool_list = QListWidget()
        self.pool_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.pool_list.setMinimumHeight(90)
        self.pool_list.setMaximumHeight(150)
        self.pool_list.itemSelectionChanged.connect(self._pool_selection_changed)
        layout.addWidget(self.pool_list)

        history_buttons = QHBoxLayout()
        self.undo_correction_button = QPushButton("Undo")
        self.undo_correction_button.clicked.connect(self._undo_correction)
        self.redo_correction_button = QPushButton("Redo")
        self.redo_correction_button.clicked.connect(self._redo_correction)
        self.reset_corrections_button = QPushButton("Reset")
        self.reset_corrections_button.setToolTip(
            "Reset all manual pool corrections to the automatic segmentation"
        )
        self.reset_corrections_button.clicked.connect(self._reset_corrections)
        history_buttons.addWidget(self.undo_correction_button)
        history_buttons.addWidget(self.redo_correction_button)
        history_buttons.addWidget(self.reset_corrections_button)
        layout.addLayout(history_buttons)

        direct_buttons = QHBoxLayout()
        self.delete_pool_button = QPushButton("Delete selected")
        self.delete_pool_button.clicked.connect(self._delete_selected_pools)
        self.merge_pool_button = QPushButton("Merge selected")
        self.merge_pool_button.clicked.connect(self._merge_selected_pools)
        direct_buttons.addWidget(self.delete_pool_button)
        direct_buttons.addWidget(self.merge_pool_button)
        layout.addLayout(direct_buttons)

        self.freehand_pool_button = QPushButton("Draw freehand boundary")
        self.freehand_pool_button.setToolTip(
            "Replace one selected pool, or add a new pool when none is selected"
        )
        self.freehand_pool_button.clicked.connect(
            lambda: self._begin_correction_drawing("freehand")
        )
        layout.addWidget(self.freehand_pool_button)

        self.oval_pool_button = QPushButton("Draw oval/circle boundary")
        self.oval_pool_button.setToolTip(
            "Replace one selected pool, or add a new pool when none is selected"
        )
        self.oval_pool_button.clicked.connect(
            lambda: self._begin_correction_drawing("oval")
        )
        layout.addWidget(self.oval_pool_button)

        self.separate_pool_button = QPushButton("Separate selected with line")
        self.separate_pool_button.clicked.connect(
            lambda: self._begin_correction_drawing("separate")
        )
        layout.addWidget(self.separate_pool_button)

        self.hint_pool_button = QPushButton("Draw pool hint and resegment")
        self.hint_pool_button.setToolTip(
            "Draw a short stroke or click inside a pool to force local resegmentation"
        )
        self.hint_pool_button.clicked.connect(
            lambda: self._begin_correction_drawing("hint")
        )
        layout.addWidget(self.hint_pool_button)

        self.correction_summary_label = QLabel(
            "Run automatic segmentation before making corrections."
        )
        self.correction_summary_label.setWordWrap(True)
        layout.addWidget(self.correction_summary_label)

        background_separator = QFrame()
        background_separator.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(background_separator)
        layout.addWidget(QLabel("Background regions"))

        self.background_mode_checkbox = QCheckBox("One background per pool")
        self.background_mode_checkbox.setChecked(True)
        self.background_mode_checkbox.setToolTip(
            "Uncheck to use one common background for all accepted pools"
        )
        self.background_mode_checkbox.toggled.connect(
            self._background_parameters_changed
        )
        layout.addWidget(self.background_mode_checkbox)

        self.background_advanced_button = QPushButton("Advanced settings")
        self.background_advanced_button.setCheckable(True)
        layout.addWidget(self.background_advanced_button)

        self.background_advanced_widget = QWidget()
        background_form = QFormLayout(self.background_advanced_widget)
        background_form.setContentsMargins(0, 0, 0, 0)
        background_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.background_min_distance_spin = QSpinBox()
        self.background_min_distance_spin.setRange(0, 100)
        self.background_min_distance_spin.setValue(5)
        self.background_min_distance_spin.setSuffix(" px")
        self.background_min_distance_spin.valueChanged.connect(
            self._background_parameters_changed
        )
        background_form.addRow(
            "Pool minimum distance:", self.background_min_distance_spin
        )
        self.background_max_distance_spin = QSpinBox()
        self.background_max_distance_spin.setRange(0, 150)
        self.background_max_distance_spin.setValue(10)
        self.background_max_distance_spin.setSuffix(" px")
        self.background_max_distance_spin.valueChanged.connect(
            self._background_parameters_changed
        )
        background_form.addRow(
            "Pool maximum distance:", self.background_max_distance_spin
        )
        self.background_common_size_spin = QSpinBox()
        self.background_common_size_spin.setRange(5, 500)
        self.background_common_size_spin.setValue(40)
        self.background_common_size_spin.setSuffix(" px")
        self.background_common_size_spin.valueChanged.connect(
            self._background_parameters_changed
        )
        background_form.addRow("Common square size:", self.background_common_size_spin)
        self.background_clearance_spin = QSpinBox()
        self.background_clearance_spin.setRange(0, 100)
        self.background_clearance_spin.setValue(5)
        self.background_clearance_spin.setSuffix(" px")
        self.background_clearance_spin.valueChanged.connect(
            self._background_parameters_changed
        )
        background_form.addRow("Common pool clearance:", self.background_clearance_spin)
        self.background_advanced_widget.setVisible(False)
        self.background_advanced_button.toggled.connect(
            self.background_advanced_widget.setVisible
        )
        layout.addWidget(self.background_advanced_widget)

        self.run_background_button = QPushButton("Find backgrounds")
        self.run_background_button.clicked.connect(self._start_background_detection)
        layout.addWidget(self.run_background_button)
        self.show_background_checkbox = QCheckBox("Show background overlay")
        self.show_background_checkbox.setChecked(True)
        self.show_background_checkbox.toggled.connect(
            self._background_visibility_changed
        )
        layout.addWidget(self.show_background_checkbox)

        self.missing_background_label = QLabel("Pool needing manual background")
        self.missing_background_combo = QComboBox()
        self.missing_background_combo.setToolTip(
            "Choose a pool whose background must be placed manually"
        )
        self.missing_background_combo.currentIndexChanged.connect(
            self._missing_background_pool_changed
        )
        self.missing_background_label.setVisible(False)
        self.missing_background_combo.setVisible(False)
        layout.addWidget(self.missing_background_label)
        layout.addWidget(self.missing_background_combo)

        self.background_list = QListWidget()
        self.background_list.setMinimumHeight(70)
        self.background_list.setMaximumHeight(130)
        self.background_list.currentRowChanged.connect(
            self._background_selection_changed
        )
        layout.addWidget(self.background_list)
        background_selection_help = QLabel(
            "Select a green background on the image or in this list. "
            "The selected background turns magenta."
        )
        background_selection_help.setWordWrap(True)
        layout.addWidget(background_selection_help)

        background_history = QHBoxLayout()
        self.undo_background_button = QPushButton("Undo")
        self.undo_background_button.clicked.connect(self._undo_background)
        self.redo_background_button = QPushButton("Redo")
        self.redo_background_button.clicked.connect(self._redo_background)
        self.reset_background_button = QPushButton("Reset")
        self.reset_background_button.setToolTip(
            "Reset manual background edits to the automatic result"
        )
        self.reset_background_button.clicked.connect(self._reset_backgrounds)
        background_history.addWidget(self.undo_background_button)
        background_history.addWidget(self.redo_background_button)
        background_history.addWidget(self.reset_background_button)
        layout.addLayout(background_history)

        background_manual = QHBoxLayout()
        self.draw_background_button = QPushButton("Draw/add")
        self.draw_background_button.setToolTip(
            "Draw a common rectangle, or place a pool-shaped background"
        )
        self.draw_background_button.clicked.connect(self._begin_background_drawing)
        self.move_background_button = QPushButton("Move selected")
        self.move_background_button.clicked.connect(self._begin_background_move)
        self.delete_background_button = QPushButton("Delete selected")
        self.delete_background_button.clicked.connect(self._delete_selected_background)
        background_manual.addWidget(self.draw_background_button)
        background_manual.addWidget(self.move_background_button)
        background_manual.addWidget(self.delete_background_button)
        layout.addLayout(background_manual)

        self.background_summary_label = QLabel(
            "Finish pool segmentation before finding backgrounds."
        )
        self.background_summary_label.setWordWrap(True)
        layout.addWidget(self.background_summary_label)

        export_separator = QFrame()
        export_separator.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(export_separator)
        layout.addWidget(QLabel("Trace and ImageJ ROI export"))
        export_help = QLabel(
            "Exports raw pool and background mean-intensity traces to XLSX, "
            "final regions to an ImageJ-compatible ROI ZIP, and the current "
            "display gradient to an ImageJ LUT."
        )
        export_help.setWordWrap(True)
        layout.addWidget(export_help)
        self.export_analysis_button = QPushButton("Export current recording…")
        self.export_analysis_button.clicked.connect(self._start_analysis_export)
        layout.addWidget(self.export_analysis_button)
        self.export_summary_label = QLabel(
            "Complete pool segmentation and background selection before exporting."
        )
        self.export_summary_label.setWordWrap(True)
        layout.addWidget(self.export_summary_label)
        self._sync_background_mode_controls()
        self._show_background_state()
        return group

    def _create_frame_controls(self) -> QWidget:
        previous_button = QToolButton()
        previous_button.setText("◀")
        previous_button.setFixedWidth(32)
        previous_button.setToolTip("Previous frame (Left Arrow)")
        previous_button.clicked.connect(lambda: self._step_frame(-1))

        self.play_button = QToolButton()
        self.play_button.setText("▶")
        self.play_button.setFixedWidth(32)
        self.play_button.setCheckable(True)
        self.play_button.setToolTip("Play/Pause (Space)")
        self.play_button.toggled.connect(self._set_playing)

        next_button = QToolButton()
        next_button.setText("▶|")
        next_button.setFixedWidth(32)
        next_button.setToolTip("Next frame (Right Arrow)")
        next_button.clicked.connect(lambda: self._step_frame(1))

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._frame_slider_changed)

        self.frame_spin = QSpinBox()
        self.frame_spin.setPrefix("Frame ")
        self.frame_spin.setMaximumWidth(110)
        self.frame_spin.setEnabled(False)
        self.frame_spin.valueChanged.connect(self._frame_spin_changed)

        self.frame_label = QLabel("No recording")
        self.frame_label.setMinimumWidth(90)
        self.frame_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 30.0)
        self.fps_spin.setValue(10.0)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.setDecimals(1)
        self.fps_spin.setMaximumWidth(110)
        self.fps_spin.valueChanged.connect(self._update_playback_interval)

        contrast_button = QPushButton("Auto contrast")
        contrast_button.setMaximumWidth(120)
        contrast_button.clicked.connect(self.canvas.auto_contrast)
        zoom_button = QPushButton("Fit image")
        zoom_button.setMaximumWidth(90)
        zoom_button.clicked.connect(self.canvas.reset_zoom)

        controls = QWidget()
        layout = QVBoxLayout(controls)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setSpacing(2)

        navigation_row = QHBoxLayout()
        navigation_row.addWidget(previous_button)
        navigation_row.addWidget(self.play_button)
        navigation_row.addWidget(next_button)
        navigation_row.addWidget(self.frame_slider, 1)
        layout.addLayout(navigation_row)

        frame_row = QHBoxLayout()
        frame_row.addWidget(self.frame_spin)
        frame_row.addWidget(self.frame_label)
        frame_row.addStretch(1)
        layout.addLayout(frame_row)

        display_row = QHBoxLayout()
        display_row.addStretch(1)
        display_row.addWidget(self.fps_spin)
        display_row.addWidget(contrast_button)
        display_row.addWidget(zoom_button)
        layout.addLayout(display_row)
        return controls

    def _build_shortcuts(self) -> None:
        QShortcut(Qt.Key.Key_Left, self, activated=lambda: self._step_frame(-1))
        QShortcut(Qt.Key.Key_Right, self, activated=lambda: self._step_frame(1))
        QShortcut(Qt.Key.Key_Space, self, activated=self.play_button.toggle)
        QShortcut(Qt.Key.Key_Escape, self, activated=self._stop_all_drawing)
        QShortcut(Qt.Key.Key_Delete, self, activated=self._delete_context_selection)
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self._undo_correction)
        QShortcut(QKeySequence.StandardKey.Redo, self, activated=self._redo_correction)

    def _delete_context_selection(self) -> None:
        state = self.current_analysis_state
        if (
            state is not None
            and state.segmentation_result is not None
            and state.selected_pool_ids
        ):
            self._delete_selected_pools()
        else:
            self._delete_selected_region()

    @Slot()
    def _choose_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open ND2 recording", "", "Nikon ND2 recordings (*.nd2)"
        )
        if filename:
            self.open_path(Path(filename))

    @Slot()
    def _choose_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Open recording folder")
        if directory:
            self.open_path(Path(directory))

    def open_path(self, path: Path) -> None:
        path = Path(path).expanduser().resolve()
        try:
            if path.is_dir():
                recordings = discover_nd2_files(path)
                if not recordings:
                    raise FileNotFoundError(f"No ND2 files found directly in {path}")
            elif path.is_file() and path.suffix.casefold() == ".nd2":
                recordings = [path]
            else:
                raise FileNotFoundError(f"ND2 file or directory not found: {path}")
        except (OSError, ValueError) as error:
            self._show_error("Could not open selection", str(error))
            return
        self.load_recordings(recordings)

    def load_recordings(self, recordings: list[Path]) -> None:
        self._peak_token += 1
        self._background_token += 1
        self._export_token += 1
        self._peak_batch_running = False
        self.recording_list.setEnabled(True)
        self._stop_playback()
        self._stop_all_drawing()
        self._clear_region_items()
        self.recording_list.clear()
        resolved_recordings = [Path(path).resolve() for path in recordings]
        self._analysis_states = {
            path: RecordingAnalysisState() for path in resolved_recordings
        }
        for path in resolved_recordings:
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            self.recording_list.addItem(item)
        self.statusBar().showMessage(f"Loaded {len(resolved_recordings)} recording(s)")
        if resolved_recordings:
            self.recording_list.setCurrentRow(0)

    @Slot(int)
    def _recording_selected(self, row: int) -> None:
        if row < 0:
            return
        self._frame_token += 1
        self._peak_token += 1
        self._segmentation_token += 1
        self._background_token += 1
        self._export_token += 1
        self._thread_pool.clear()
        self._analysis_thread_pool.clear()
        self._remove_background_move_roi()
        self._stop_all_drawing()
        self._clear_region_items()
        item = self.recording_list.item(row)
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        self._current_path = path
        self._metadata = None
        self._stop_playback()
        self.canvas.clear_for_new_recording()
        self._clear_metadata()
        self.frame_slider.setEnabled(False)
        self.frame_spin.setEnabled(False)
        self.draw_region_action.setEnabled(False)
        self._clear_analysis_controls()

        self._metadata_token += 1
        token = self._metadata_token
        task = _FunctionTask(token, lambda: read_metadata(path))
        task.signals.succeeded.connect(self._metadata_loaded)
        task.signals.failed.connect(self._metadata_failed)
        self.statusBar().showMessage(f"Reading metadata: {path.name}")
        self._thread_pool.start(task)

    @Slot(int, object)
    def _metadata_loaded(self, token: int, result: object) -> None:
        if token != self._metadata_token or not isinstance(result, RecordingMetadata):
            return
        if result.path != self._current_path:
            return
        self._metadata = result
        self._show_metadata(result)
        self.frame_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.frame_slider.setRange(0, result.frame_count - 1)
        self.frame_spin.setRange(1, result.frame_count)
        self.frame_slider.setValue(0)
        self.frame_spin.setValue(1)
        self.frame_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)
        self.frame_slider.setEnabled(True)
        self.frame_spin.setEnabled(True)
        self.draw_region_action.setEnabled(True)
        self._update_frame_label(0)
        self._restore_analysis_state()
        self._request_frame(0)

    @Slot(int, str)
    def _metadata_failed(self, token: int, message: str) -> None:
        if token != self._metadata_token:
            return
        self.canvas.message.setText("Unable to open recording.")
        self.canvas.message.show()
        self.draw_region_action.setEnabled(False)
        self.statusBar().showMessage("Recording could not be opened")
        self._show_error("Unsupported or unreadable recording", message)

    @Slot(int)
    def _frame_slider_changed(self, index: int) -> None:
        if self._metadata is None:
            return
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(index + 1)
        self.frame_spin.blockSignals(False)
        self._update_frame_label(index)
        self._request_frame(index)

    @Slot(int)
    def _frame_spin_changed(self, one_based_index: int) -> None:
        self.frame_slider.setValue(one_based_index - 1)

    def _request_frame(self, index: int) -> None:
        if self._current_path is None or self._metadata is None:
            return
        path = self._current_path
        self._frame_token += 1
        token = self._frame_token
        self._thread_pool.clear()
        task = _FunctionTask(token, lambda: read_frame(path, index))
        task.signals.succeeded.connect(self._frame_loaded)
        task.signals.failed.connect(self._frame_failed)
        self.statusBar().showMessage(
            f"Loading {path.name}, frame {index + 1}/{self._metadata.frame_count}"
        )
        self._thread_pool.start(task)

    @Slot(int, object)
    def _frame_loaded(self, token: int, result: object) -> None:
        if token != self._frame_token or not isinstance(result, np.ndarray):
            return
        self.canvas.set_frame(result)
        if self._current_path is not None:
            self.statusBar().showMessage(
                f"{self._current_path.name} — frame {self.frame_slider.value() + 1}"
            )

    @Slot(int, str)
    def _frame_failed(self, token: int, message: str) -> None:
        if token != self._frame_token:
            return
        self._stop_playback()
        self.statusBar().showMessage("Frame could not be loaded")
        self._show_error("Could not read frame", message)

    def _step_frame(self, amount: int) -> None:
        if not self.frame_slider.isEnabled():
            return
        target = max(
            self.frame_slider.minimum(),
            min(self.frame_slider.maximum(), self.frame_slider.value() + amount),
        )
        self.frame_slider.setValue(target)

    @Slot(bool)
    def _set_playing(self, playing: bool) -> None:
        if playing and self.frame_slider.isEnabled():
            self.play_button.setText("❚❚")
            self._update_playback_interval()
            self._play_timer.start()
        else:
            self.play_button.setText("▶")
            self._play_timer.stop()
            if playing:
                self.play_button.setChecked(False)

    def _stop_playback(self) -> None:
        self.play_button.setChecked(False)
        self._play_timer.stop()

    @Slot()
    def _advance_playback(self) -> None:
        if self.frame_slider.value() >= self.frame_slider.maximum():
            self._stop_playback()
        else:
            self._step_frame(1)

    @Slot()
    def _update_playback_interval(self) -> None:
        self._play_timer.setInterval(round(1000 / self.fps_spin.value()))

    def _update_frame_label(self, index: int) -> None:
        if self._metadata is None:
            self.frame_label.setText("No recording")
            return
        timestamp = self._metadata.timestamp_for_frame(index)
        time_text = "time unavailable" if timestamp is None else f"{timestamp:.3f} s"
        self.frame_label.setText(
            f"{index + 1}/{self._metadata.frame_count} — {time_text}"
        )

    def _show_metadata(self, metadata: RecordingMetadata) -> None:
        duration = (
            "Unavailable"
            if metadata.duration_s is None
            else f"{metadata.duration_s:.3f} s"
        )
        interval = (
            "Unavailable"
            if metadata.median_interval_s is None
            else f"{metadata.median_interval_s:.6f} s"
        )
        exposure = (
            "Unavailable"
            if metadata.exposure_ms is None
            else f"{metadata.exposure_ms:g} ms"
        )
        values = {
            "filename": metadata.path.name,
            "path": str(metadata.path.parent),
            "dimensions": (
                f"{metadata.frame_count} frames × {metadata.height} × {metadata.width}"
            ),
            "dtype": metadata.dtype,
            "pixel_size": (
                f"{metadata.pixel_size_x_um:g} × {metadata.pixel_size_y_um:g} µm"
            ),
            "file_size": _human_file_size(metadata.file_size_bytes),
            "duration": duration,
            "interval": interval,
            "exposure": exposure,
        }
        for key, value in values.items():
            self.metadata_labels[key].setText(value)
        self._update_frame_label(self.frame_slider.value())

    def _clear_metadata(self) -> None:
        for label in self.metadata_labels.values():
            label.setText("—")
        self.frame_label.setText("Loading metadata…")

    @Slot(bool)
    def _set_region_drawing(self, enabled: bool) -> None:
        if enabled:
            self._stop_correction_drawing()
        self.canvas.view_box.set_rectangle_drawing_enabled(enabled)
        self._apply_region_editability()
        self._sync_region_edit_button()
        if enabled:
            self._stop_playback()
            self.statusBar().showMessage(
                "Region editing enabled: add, select, move, or resize rectangles; "
                "press Escape when finished"
            )
        else:
            self.statusBar().showMessage("Analysis regions locked")

    @Slot()
    def _sync_region_edit_button(self) -> None:
        if not hasattr(self, "edit_regions_button"):
            return
        enabled = self.draw_region_action.isEnabled()
        checked = self.draw_region_action.isChecked()
        self.edit_regions_button.blockSignals(True)
        self.edit_regions_button.setEnabled(enabled)
        self.edit_regions_button.setChecked(checked)
        self.edit_regions_button.setText(
            "Finish region editing" if checked else "Edit analysis regions"
        )
        self.edit_regions_button.blockSignals(False)

    def _apply_region_editability(self) -> None:
        editable = (
            self.draw_region_action.isEnabled() and self.draw_region_action.isChecked()
        )
        accepted_button = (
            Qt.MouseButton.LeftButton if editable else Qt.MouseButton.NoButton
        )
        for roi in self._region_items:
            roi.translatable = editable
            roi.setAcceptedMouseButtons(accepted_button)
            for handle in roi.getHandles():
                handle.setAcceptedMouseButtons(accepted_button)
                handle.setVisible(editable)

    def _stop_region_drawing(self) -> None:
        self.draw_region_action.setChecked(False)
        self.canvas.view_box.set_rectangle_drawing_enabled(False)
        self._apply_region_editability()
        self._sync_region_edit_button()

    def _stop_correction_drawing(self) -> None:
        self.canvas.view_box.set_correction_drawing_mode(None)
        self._apply_region_editability()

    def _stop_all_drawing(self) -> None:
        self._stop_region_drawing()
        self._stop_correction_drawing()
        self._remove_background_move_roi()

    @Slot(object)
    def _rectangle_drawn(self, coordinates: object) -> None:
        if self._metadata is None or not isinstance(coordinates, tuple):
            return
        try:
            left, top, right, bottom = coordinates
            rectangle = clip_rectangle(
                float(left),
                float(top),
                float(right),
                float(bottom),
                image_width=self._metadata.width,
                image_height=self._metadata.height,
            )
        except (TypeError, ValueError):
            self.statusBar().showMessage("Rectangle was too small and was not added")
            return
        self.add_analysis_region(rectangle)

    def add_analysis_region(
        self, rectangle: RectangleBounds, *, allow_overlap: bool | None = None
    ) -> bool:
        """Add a rectangle, optionally bypassing the interactive overlap prompt."""
        state = self.current_analysis_state
        if self._metadata is None or state is None:
            return False
        if rectangle.x1 > self._metadata.width or rectangle.y1 > self._metadata.height:
            raise ValueError("Analysis rectangle exceeds the image bounds.")

        if overlaps_any(rectangle, state.rectangles):
            if allow_overlap is False:
                return False
            if allow_overlap is None and not self._confirm_overlap():
                self.statusBar().showMessage("Overlapping rectangle was not added")
                return False

        state.rectangles.append(rectangle)
        self._invalidate_peak()
        self._rebuild_region_items(selected_index=len(state.rectangles) - 1)
        self.statusBar().showMessage(
            f"Added analysis region {len(state.rectangles)}: "
            f"x={rectangle.x0}:{rectangle.x1}, y={rectangle.y0}:{rectangle.y1}"
        )
        return True

    def _confirm_overlap(self) -> bool:
        answer = QMessageBox.warning(
            self,
            "Overlapping analysis regions",
            "This rectangle overlaps another analysis region. Overlapping pixels "
            "will be counted only once when finding the peak. Allow this overlap?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _restore_analysis_state(self) -> None:
        state = self.current_analysis_state
        if state is None or self._metadata is None:
            return
        self.manual_peak_spin.setRange(1, self._metadata.frame_count)
        self._rebuild_region_items()
        self._show_peak_state()
        self._restore_segmentation_state()
        self._restore_background_state()

    def _rebuild_region_items(self, selected_index: int | None = None) -> None:
        self._clear_region_items()
        state = self.current_analysis_state
        if state is None or self._metadata is None:
            return
        image_bounds = QRectF(0, 0, self._metadata.width, self._metadata.height)
        for index, rectangle in enumerate(state.rectangles):
            roi = pg.RectROI(
                (rectangle.x0, rectangle.y0),
                (rectangle.width, rectangle.height),
                maxBounds=image_bounds,
                pen=pg.mkPen("#35d0e5", width=2),
            )
            roi.setZValue(20)
            roi.sigRegionChangeFinished.connect(
                lambda *args, item=roi: self._region_change_finished(item)
            )
            if hasattr(roi, "sigClicked"):
                roi.sigClicked.connect(
                    lambda *args, item=roi: self._region_item_clicked(item, *args)
                )
            self.canvas.view_box.addItem(roi)
            self._region_items.append(roi)

            label = pg.TextItem(str(index + 1), color="#35d0e5", anchor=(0, 1))
            label.setZValue(21)
            label.setPos(rectangle.x0, rectangle.y0)
            roi.sigRegionChanged.connect(
                lambda *args, item=roi, text=label: text.setPos(
                    item.pos().x(), item.pos().y()
                )
            )
            self.canvas.view_box.addItem(label)
            self._region_labels.append(label)

        self._apply_region_editability()
        self._sync_region_list()
        if selected_index is not None and state.rectangles:
            self.region_list.setCurrentRow(selected_index)
        elif state.rectangles:
            self.region_list.setCurrentRow(
                min(max(self._selected_region_index, 0), len(state.rectangles) - 1)
            )
        self._update_region_actions()

    def _clear_region_items(self) -> None:
        for item in self._region_items:
            self.canvas.view_box.removeItem(item)
        for label in self._region_labels:
            self.canvas.view_box.removeItem(label)
        self._region_items.clear()
        self._region_labels.clear()
        if hasattr(self, "region_list"):
            self.region_list.clear()
        self._selected_region_index = -1

    def _sync_region_list(self) -> None:
        state = self.current_analysis_state
        self.region_list.blockSignals(True)
        self.region_list.clear()
        if state is not None:
            for index, rectangle in enumerate(state.rectangles):
                self.region_list.addItem(
                    f"{index + 1}: x {rectangle.x0}-{rectangle.x1}, "
                    f"y {rectangle.y0}-{rectangle.y1} "
                    f"({rectangle.width} x {rectangle.height})"
                )
        self.region_list.blockSignals(False)

    @Slot(int)
    def _region_list_selected(self, index: int) -> None:
        self._selected_region_index = index
        self._update_region_pens()
        self._update_region_actions()

    def _region_item_clicked(self, item: pg.RectROI, *signal_arguments: object) -> None:
        if self._current_labels() is not None:
            event = next(
                (
                    value
                    for value in reversed(signal_arguments)
                    if hasattr(value, "scenePos")
                ),
                None,
            )
            if event is not None:
                position = self.canvas.view_box.mapSceneToView(event.scenePos())
                toggle = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                self._pool_clicked((position.x(), position.y(), toggle))
                return
        try:
            index = self._region_items.index(item)
        except ValueError:
            return
        self.region_list.setCurrentRow(index)

    def _update_region_pens(self) -> None:
        for index, item in enumerate(self._region_items):
            if index == self._selected_region_index:
                item.setPen(pg.mkPen("#ffd166", width=3))
                self._region_labels[index].setColor("#ffd166")
            else:
                item.setPen(pg.mkPen("#35d0e5", width=2))
                self._region_labels[index].setColor("#35d0e5")

    def _region_change_finished(self, item: pg.RectROI) -> None:
        if self._updating_region or self._metadata is None:
            return
        state = self.current_analysis_state
        if state is None:
            return
        try:
            index = self._region_items.index(item)
        except ValueError:
            return
        previous = state.rectangles[index]
        position, size = item.pos(), item.size()
        try:
            candidate = clip_rectangle(
                position.x(),
                position.y(),
                position.x() + size.x(),
                position.y() + size.y(),
                image_width=self._metadata.width,
                image_height=self._metadata.height,
            )
        except ValueError:
            self._set_region_item_geometry(item, previous)
            return

        if (
            overlaps_any(candidate, state.rectangles, excluded_index=index)
            and not self._confirm_overlap()
        ):
            self._set_region_item_geometry(item, previous)
            self.statusBar().showMessage("Region edit reverted to prevent overlap")
            return

        state.rectangles[index] = candidate
        self._set_region_item_geometry(item, candidate)
        self._region_labels[index].setPos(candidate.x0, candidate.y0)
        self._invalidate_peak()
        self._sync_region_list()
        self.region_list.setCurrentRow(index)
        self.statusBar().showMessage(f"Updated analysis region {index + 1}")

    def _set_region_item_geometry(
        self, item: pg.RectROI, rectangle: RectangleBounds
    ) -> None:
        self._updating_region = True
        item.blockSignals(True)
        item.setPos((rectangle.x0, rectangle.y0))
        item.setSize((rectangle.width, rectangle.height))
        item.blockSignals(False)
        self._updating_region = False
        try:
            index = self._region_items.index(item)
        except ValueError:
            return
        self._region_labels[index].setPos(rectangle.x0, rectangle.y0)

    @Slot()
    def _delete_selected_region(self) -> None:
        state = self.current_analysis_state
        index = self._selected_region_index
        if state is None or not 0 <= index < len(state.rectangles):
            return
        del state.rectangles[index]
        self._invalidate_peak()
        next_index = min(index, len(state.rectangles) - 1)
        self._rebuild_region_items(
            selected_index=next_index if next_index >= 0 else None
        )
        self.statusBar().showMessage("Analysis region deleted")

    @Slot()
    def _clear_regions_with_confirmation(self) -> None:
        state = self.current_analysis_state
        if state is None or not state.rectangles:
            return
        answer = QMessageBox.question(
            self,
            "Clear analysis regions",
            "Delete every analysis rectangle for this recording?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        state.rectangles.clear()
        self._invalidate_peak()
        self._rebuild_region_items()
        self.statusBar().showMessage("All analysis regions cleared")

    def _update_region_actions(self) -> None:
        state = self.current_analysis_state
        has_regions = state is not None and bool(state.rectangles)
        has_any_regions = any(
            analysis_state.rectangles
            for analysis_state in self._analysis_states.values()
        )
        has_selection = (
            has_regions
            and self._selected_region_index >= 0
            and self._selected_region_index < len(state.rectangles)
        )
        can_edit = not self._peak_batch_running
        self.delete_region_action.setEnabled(has_selection and can_edit)
        self.clear_regions_action.setEnabled(has_regions and can_edit)
        self.find_peak_button.setEnabled(has_any_regions and can_edit)

    def _update_peak_button_text(self) -> None:
        has_calculated_peak = any(
            state.peak_result is not None for state in self._analysis_states.values()
        )
        self.find_peak_button.setText(
            "Recalculate all recording peaks"
            if has_calculated_peak
            else "Calculate all recording peaks"
        )

    def _invalidate_peak(self) -> None:
        self._peak_token += 1
        state = self.current_analysis_state
        if state is not None:
            state.peak_result = None
            state.peak_error = None
            state.manual_peak_index = None
        self._invalidate_segmentation()
        self._update_peak_button_text()
        self._show_peak_state()

    @Slot()
    def _start_peak_detection(self) -> None:
        requests = tuple(
            (path, tuple(state.rectangles))
            for path, state in self._analysis_states.items()
            if state.rectangles
        )
        if not requests or self._peak_batch_running:
            return
        self._stop_playback()
        self._stop_all_drawing()
        self._peak_token += 1
        token = self._peak_token
        self._segmentation_token += 1
        for path, _rectangles in requests:
            state = self._analysis_states[path]
            state.peak_result = None
            state.peak_error = None
            state.manual_peak_index = None
            state.segmentation_result = None
            state.corrected_labels = None
            state.correction_undo.clear()
            state.correction_redo.clear()
            state.selected_pool_ids.clear()
            state.background_result = None
            state.corrected_background_labels = None
            state.background_assignments.clear()
            state.background_undo.clear()
            state.background_redo.clear()
            state.selected_background_id = None

        task = _FunctionTask(token, lambda: _detect_peak_batch(requests))
        task.signals.succeeded.connect(self._peak_detection_finished)
        task.signals.failed.connect(self._peak_detection_failed)
        self._peak_batch_running = True
        self.recording_list.setEnabled(False)
        self.draw_region_action.setEnabled(False)
        self._update_region_actions()
        self.canvas.clear_segmentation()
        self.canvas.clear_backgrounds()
        self.find_peak_button.setEnabled(False)
        self.find_peak_button.setText("Calculating all recording peaks…")
        region_count = sum(len(rectangles) for _path, rectangles in requests)
        self.peak_summary_label.setText(
            f"One batch is scanning {len(requests)} recording(s) containing "
            f"{region_count} analysis region(s)…"
        )
        self.statusBar().showMessage(
            f"Calculating strict raw peaks for {len(requests)} recording(s)"
        )
        self._analysis_thread_pool.start(task)

    @Slot(int, object)
    def _peak_detection_finished(self, token: int, result: object) -> None:
        if token != self._peak_token or not isinstance(result, _PeakBatchResult):
            return
        self._peak_batch_running = False
        self.recording_list.setEnabled(True)
        self.draw_region_action.setEnabled(self._metadata is not None)
        for path, peak_result in result.results.items():
            state = self._analysis_states.get(path)
            if state is not None:
                state.peak_result = peak_result
                state.peak_error = None
                state.manual_peak_index = None
        for path, message in result.failures.items():
            state = self._analysis_states.get(path)
            if state is not None:
                state.peak_error = message

        self._update_peak_button_text()
        self._show_peak_state()
        self._show_segmentation_state()
        current_result = result.results.get(self._current_path)
        if current_result is not None:
            self.frame_slider.setValue(current_result.automatic_peak_index)
        completed = len(result.results)
        failed = len(result.failures)
        self.statusBar().showMessage(
            f"Calculated {completed} recording peak(s) in one batch"
            + (f"; {failed} failed" if failed else "")
        )

    @Slot(int, str)
    def _peak_detection_failed(self, token: int, message: str) -> None:
        if token != self._peak_token:
            return
        self._peak_batch_running = False
        self.recording_list.setEnabled(True)
        self.draw_region_action.setEnabled(self._metadata is not None)
        self._update_peak_button_text()
        self._update_region_actions()
        self.peak_summary_label.setText("Peak calculation failed.")
        self.statusBar().showMessage("Peak calculation failed")
        self._show_error("Could not find recording peak", message)

    def _show_peak_state(self) -> None:
        state = self.current_analysis_state
        if state is None or state.peak_result is None:
            self.peak_summary_label.setText(
                "Add at least one rectangle."
                if state is None or not state.rectangles
                else (
                    f"Peak calculation failed: {state.peak_error}"
                    if state.peak_error is not None
                    else "Peak not calculated. Use the batch button once after "
                    "drawing regions for all recordings."
                )
            )
            self.override_peak_checkbox.blockSignals(True)
            self.override_peak_checkbox.setChecked(False)
            self.override_peak_checkbox.blockSignals(False)
            self.override_peak_checkbox.setEnabled(False)
            self.manual_peak_spin.setEnabled(False)
            self.go_to_peak_button.setEnabled(False)
            self._update_peak_button_text()
            self._update_region_actions()
            return

        automatic = state.peak_result.automatic_peak_index
        effective = state.effective_peak_index
        self._update_peak_button_text()
        automatic_text = self._describe_frame(automatic)
        effective_text = (
            self._describe_frame(effective) if effective is not None else "—"
        )
        source = (
            "manual override" if state.manual_peak_index is not None else "automatic"
        )
        self.peak_summary_label.setText(
            f"Automatic peak: {automatic_text}\n"
            f"Effective peak: {effective_text} ({source})\n"
            f"Regions included: {len(state.rectangles)}\n"
            f"Union size: {state.peak_result.union_pixel_count:,} pixels"
        )
        self.override_peak_checkbox.setEnabled(True)
        self.override_peak_checkbox.blockSignals(True)
        self.override_peak_checkbox.setChecked(state.manual_peak_index is not None)
        self.override_peak_checkbox.blockSignals(False)
        self.manual_peak_spin.setEnabled(state.manual_peak_index is not None)
        if effective is not None:
            self.manual_peak_spin.blockSignals(True)
            self.manual_peak_spin.setValue(effective + 1)
            self.manual_peak_spin.blockSignals(False)
        self.go_to_peak_button.setEnabled(True)
        self._update_region_actions()

    def _describe_frame(self, index: int) -> str:
        timestamp = (
            None
            if self._metadata is None
            else self._metadata.timestamp_for_frame(index)
        )
        return (
            f"frame {index + 1}"
            if timestamp is None
            else f"frame {index + 1} at {timestamp:.3f} s"
        )

    @Slot(bool)
    def _override_peak_toggled(self, enabled: bool) -> None:
        state = self.current_analysis_state
        if state is None or state.peak_result is None:
            return
        if enabled:
            state.manual_peak_index = state.peak_result.automatic_peak_index
        else:
            state.manual_peak_index = None
        self._invalidate_segmentation()
        self._show_peak_state()
        self._go_to_effective_peak()

    @Slot(int)
    def _manual_peak_changed(self, one_based_index: int) -> None:
        state = self.current_analysis_state
        if (
            state is None
            or state.peak_result is None
            or not self.override_peak_checkbox.isChecked()
        ):
            return
        previous = state.manual_peak_index
        state.manual_peak_index = one_based_index - 1
        if state.manual_peak_index != previous:
            self._invalidate_segmentation()
        self._show_peak_state()
        self.frame_slider.setValue(state.manual_peak_index)

    @Slot()
    def _go_to_effective_peak(self) -> None:
        state = self.current_analysis_state
        if state is not None and state.effective_peak_index is not None:
            self.frame_slider.setValue(state.effective_peak_index)

    def _restore_segmentation_state(self) -> None:
        state = self.current_analysis_state
        if state is None:
            return
        parameters = state.segmentation_parameters
        controls = (
            (self.segmentation_threshold_spin, parameters.seed_threshold_sigma),
            (self.segmentation_min_area_spin, parameters.minimum_area_px),
            (self.segmentation_max_area_spin, parameters.maximum_area_px),
            (self.segmentation_distance_spin, parameters.minimum_seed_distance),
        )
        for control, value in controls:
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        self._show_segmentation_state()

    def _restore_background_state(self) -> None:
        state = self.current_analysis_state
        if state is None:
            return
        parameters = state.background_parameters
        controls = (
            (self.background_min_distance_spin, parameters.pool_min_distance_px),
            (self.background_max_distance_spin, parameters.pool_max_distance_px),
            (self.background_common_size_spin, parameters.common_size_px),
            (self.background_clearance_spin, parameters.common_clearance_px),
        )
        self.background_mode_checkbox.blockSignals(True)
        self.background_mode_checkbox.setChecked(parameters.mode == "pool-wise")
        self.background_mode_checkbox.blockSignals(False)
        for control, value in controls:
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        self._sync_background_mode_controls()
        self._show_background_state()

    @Slot()
    def _background_parameters_changed(self) -> None:
        state = self.current_analysis_state
        if state is None:
            return
        minimum = self.background_min_distance_spin.value()
        if self.background_max_distance_spin.value() < minimum:
            self.background_max_distance_spin.blockSignals(True)
            self.background_max_distance_spin.setValue(minimum)
            self.background_max_distance_spin.blockSignals(False)
        state.background_parameters = BackgroundParameters(
            mode=(
                "pool-wise" if self.background_mode_checkbox.isChecked() else "common"
            ),
            pool_min_distance_px=minimum,
            pool_max_distance_px=self.background_max_distance_spin.value(),
            common_size_px=self.background_common_size_spin.value(),
            common_clearance_px=self.background_clearance_spin.value(),
        )
        self._sync_background_mode_controls()
        self._invalidate_backgrounds()

    def _sync_background_mode_controls(self) -> None:
        pool_wise = self.background_mode_checkbox.isChecked()
        self.background_mode_checkbox.setText(
            "One background per pool"
            if pool_wise
            else "One common background for all pools"
        )
        self.background_min_distance_spin.setEnabled(pool_wise)
        self.background_max_distance_spin.setEnabled(pool_wise)
        self.background_common_size_spin.setEnabled(not pool_wise)
        self.background_clearance_spin.setEnabled(not pool_wise)

    @Slot()
    def _segmentation_parameters_changed(self) -> None:
        state = self.current_analysis_state
        if state is None:
            return
        minimum_area = self.segmentation_min_area_spin.value()
        if self.segmentation_max_area_spin.value() < minimum_area:
            self.segmentation_max_area_spin.blockSignals(True)
            self.segmentation_max_area_spin.setValue(minimum_area)
            self.segmentation_max_area_spin.blockSignals(False)
        state.segmentation_parameters = SegmentationParameters(
            seed_threshold_sigma=self.segmentation_threshold_spin.value(),
            minimum_seed_distance=self.segmentation_distance_spin.value(),
            minimum_area_px=minimum_area,
            maximum_area_px=self.segmentation_max_area_spin.value(),
        )
        self._invalidate_segmentation()

    def _invalidate_segmentation(self) -> None:
        self._segmentation_token += 1
        self._background_token += 1
        state = self.current_analysis_state
        if state is not None:
            state.segmentation_result = None
            state.corrected_labels = None
            state.correction_undo.clear()
            state.correction_redo.clear()
            state.selected_pool_ids.clear()
            state.background_result = None
            state.corrected_background_labels = None
            state.background_assignments.clear()
            state.background_undo.clear()
            state.background_redo.clear()
            state.selected_background_id = None
        if hasattr(self, "canvas"):
            self.canvas.clear_segmentation()
            self.canvas.clear_backgrounds()
        if hasattr(self, "segmentation_summary_label"):
            self._show_segmentation_state()

    @Slot()
    def _start_segmentation(self) -> None:
        state = self.current_analysis_state
        if (
            self._current_path is None
            or state is None
            or state.effective_peak_index is None
            or not state.rectangles
        ):
            return
        self._stop_playback()
        self._stop_all_drawing()
        self._invalidate_backgrounds()
        self._segmentation_token += 1
        token = self._segmentation_token
        path = self._current_path
        peak_index = state.effective_peak_index
        rectangles = tuple(state.rectangles)
        parameters = state.segmentation_parameters
        task = _FunctionTask(
            token,
            lambda: segment_recording(path, peak_index, rectangles, parameters),
        )
        task.signals.succeeded.connect(self._segmentation_finished)
        task.signals.failed.connect(self._segmentation_failed)
        self.run_segmentation_button.setEnabled(False)
        self.run_segmentation_button.setText("Segmenting…")
        self.segmentation_summary_label.setText(
            f"Segmenting frame {peak_index + 1} inside {len(rectangles)} region(s)…"
        )
        self.statusBar().showMessage(f"Segmenting vesicle pools in {path.name}")
        self._analysis_thread_pool.start(task)

    @Slot(int, object)
    def _segmentation_finished(self, token: int, result: object) -> None:
        if token != self._segmentation_token or not isinstance(
            result, SegmentationResult
        ):
            return
        state = self.current_analysis_state
        if state is None:
            return
        state.segmentation_result = result
        state.corrected_labels = result.labels.copy()
        state.correction_undo.clear()
        state.correction_redo.clear()
        state.selected_pool_ids.clear()
        state.background_result = None
        state.corrected_background_labels = None
        state.background_assignments.clear()
        state.background_undo.clear()
        state.background_redo.clear()
        state.selected_background_id = None
        self.run_segmentation_button.setText("Rerun automatic segmentation")
        self._show_segmentation_state()
        self._go_to_effective_peak()
        self.statusBar().showMessage(
            f"Automatic segmentation found {len(result.pools)} pool(s)"
        )

    @Slot(int, str)
    def _segmentation_failed(self, token: int, message: str) -> None:
        if token != self._segmentation_token:
            return
        self.run_segmentation_button.setText("Run automatic segmentation")
        self._show_segmentation_state()
        self.segmentation_summary_label.setText("Automatic segmentation failed.")
        self.statusBar().showMessage("Automatic segmentation failed")
        self._show_error("Could not segment vesicle pools", message)

    def _show_segmentation_state(self) -> None:
        state = self.current_analysis_state
        can_segment = (
            state is not None
            and state.effective_peak_index is not None
            and bool(state.rectangles)
        )
        self.run_segmentation_button.setEnabled(can_segment)
        if state is None or state.segmentation_result is None:
            self.run_segmentation_button.setText("Run automatic segmentation")
            self.show_segmentation_checkbox.setEnabled(False)
            self.canvas.clear_segmentation()
            self.segmentation_summary_label.setText(
                "Calculate a recording peak before segmentation."
                if not can_segment
                else "Ready. Lower the detection threshold to find dimmer pools."
            )
            self._show_correction_state()
            self._show_background_state()
            return

        result = state.segmentation_result
        labels = self._current_labels()
        if labels is None:
            return
        self.run_segmentation_button.setText("Rerun automatic segmentation")
        self.show_segmentation_checkbox.setEnabled(True)
        self.canvas.set_segmentation(
            labels,
            tuple(sorted(state.selected_pool_ids)),
            self._missing_background_pool_ids(state, labels),
        )
        self.canvas.segmentation_item.setVisible(
            self.show_segmentation_checkbox.isChecked()
        )
        current_count = len(np.unique(labels)) - 1
        self.segmentation_summary_label.setText(
            f"Found {len(result.pools)} automatic pool(s); {current_count} currently "
            f"accepted on frame "
            f"{result.peak_frame_index + 1}.\n"
            f"Threshold: {result.parameters.seed_threshold_sigma:g} sigma; "
            f"area: {result.parameters.minimum_area_px}-"
            f"{result.parameters.maximum_area_px} px²."
        )
        self._show_correction_state()
        self._show_background_state()

    @Slot(bool)
    def _segmentation_visibility_changed(self, visible: bool) -> None:
        self.canvas.segmentation_item.setVisible(visible)

    def _current_labels(self) -> np.ndarray[Any, Any] | None:
        state = self.current_analysis_state
        if state is None or state.segmentation_result is None:
            return None
        if state.corrected_labels is None:
            state.corrected_labels = state.segmentation_result.labels.copy()
        return state.corrected_labels

    def _show_correction_state(self) -> None:
        if not hasattr(self, "pool_list"):
            return
        state = self.current_analysis_state
        labels = self._current_labels()
        has_labels = state is not None and labels is not None

        self.pool_list.blockSignals(True)
        self.pool_list.clear()
        existing_ids: set[int] = set()
        if has_labels and labels is not None and state is not None:
            counts = np.bincount(labels.ravel())
            for pool_id in np.flatnonzero(counts[1:]) + 1:
                value = int(pool_id)
                existing_ids.add(value)
                item = QListWidgetItem(f"Pool {value} — {int(counts[value])} px²")
                item.setData(Qt.ItemDataRole.UserRole, value)
                self.pool_list.addItem(item)
                item.setSelected(value in state.selected_pool_ids)
            state.selected_pool_ids.intersection_update(existing_ids)
        self.pool_list.blockSignals(False)

        selected_count = len(state.selected_pool_ids) if state is not None else 0
        self.pool_list.setEnabled(has_labels)
        self.undo_correction_button.setEnabled(
            has_labels and bool(state and state.correction_undo)
        )
        self.redo_correction_button.setEnabled(
            has_labels and bool(state and state.correction_redo)
        )
        self.reset_corrections_button.setEnabled(
            has_labels and bool(state and state.correction_undo)
        )
        self.delete_pool_button.setEnabled(has_labels and selected_count > 0)
        self.merge_pool_button.setEnabled(has_labels and selected_count >= 2)
        self.freehand_pool_button.setEnabled(has_labels)
        self.oval_pool_button.setEnabled(has_labels)
        self.separate_pool_button.setEnabled(has_labels and selected_count == 1)
        self.hint_pool_button.setEnabled(has_labels)

        if not has_labels or state is None or labels is None:
            self.correction_summary_label.setText(
                "Run automatic segmentation before making corrections."
            )
            return
        correction_count = len(state.correction_undo)
        selection_text = (
            "none"
            if not state.selected_pool_ids
            else ", ".join(str(value) for value in sorted(state.selected_pool_ids))
        )
        self.correction_summary_label.setText(
            f"{len(existing_ids)} accepted pool(s); selected: {selection_text}. "
            f"{correction_count} correction(s) applied."
        )
        self.canvas.set_segmentation(
            labels,
            tuple(sorted(state.selected_pool_ids)),
            self._missing_background_pool_ids(state, labels),
        )
        self.canvas.segmentation_item.setVisible(
            self.show_segmentation_checkbox.isChecked()
        )

    @Slot()
    def _pool_selection_changed(self) -> None:
        state = self.current_analysis_state
        if state is None:
            return
        state.selected_pool_ids = {
            int(item.data(Qt.ItemDataRole.UserRole))
            for item in self.pool_list.selectedItems()
        }
        self._show_correction_state()

    @Slot(object)
    def _pool_clicked(self, payload: object) -> None:
        state = self.current_analysis_state
        labels = self._current_labels()
        if state is None or labels is None or not isinstance(payload, tuple):
            return
        x, y, toggle = payload
        column, row = int(np.floor(float(x))), int(np.floor(float(y)))
        if not (0 <= row < labels.shape[0] and 0 <= column < labels.shape[1]):
            return

        background_labels = self._current_background_labels()
        if (
            background_labels is not None
            and self.show_background_checkbox.isChecked()
            and 0 <= row < background_labels.shape[0]
            and 0 <= column < background_labels.shape[1]
        ):
            background_id = int(background_labels[row, column])
            if background_id > 0:
                state.selected_background_id = background_id
                state.selected_pool_ids.clear()
                self._show_correction_state()
                self._refresh_background_selection_ui()
                return

        pool_id = int(labels[row, column])
        if pool_id == 0:
            if not toggle:
                state.selected_pool_ids.clear()
        elif toggle and pool_id in state.selected_pool_ids:
            state.selected_pool_ids.remove(pool_id)
        elif toggle:
            state.selected_pool_ids.add(pool_id)
        else:
            state.selected_pool_ids = {pool_id}
        state.selected_background_id = None
        self._show_correction_state()
        self._refresh_background_selection_ui()

    def _begin_correction_drawing(self, mode: str) -> None:
        state = self.current_analysis_state
        if state is None or self._current_labels() is None:
            return
        selected_count = len(state.selected_pool_ids)
        if mode == "separate" and selected_count != 1:
            self.statusBar().showMessage("Select exactly one pool before separating")
            return
        if mode in {"freehand", "oval", "hint"} and selected_count > 1:
            self.statusBar().showMessage(
                "Select one pool to replace, or clear selection to add a pool"
            )
            return
        self._stop_region_drawing()
        self.canvas.view_box.set_correction_drawing_mode(mode)
        self._apply_region_editability()
        self._go_to_effective_peak()
        instructions = {
            "freehand": "Draw a closed freehand boundary around the pool",
            "oval": "Drag an oval or circle around the pool",
            "separate": "Draw a line fully across the selected pool",
            "hint": "Click or draw a short stroke where the pool should be",
        }
        self.statusBar().showMessage(f"{instructions[mode]}; Escape cancels")

    @Slot(object)
    def _correction_drawn(self, payload: object) -> None:
        state = self.current_analysis_state
        labels = self._current_labels()
        self._stop_correction_drawing()
        if (
            state is None
            or labels is None
            or not isinstance(payload, tuple)
            or len(payload) != 2
        ):
            return
        mode, geometry = payload
        selected = tuple(sorted(state.selected_pool_ids))
        try:
            if mode == "freehand":
                mask = polygon_mask(labels.shape, geometry)
                self._apply_drawn_pool(mask, selected, "Freehand boundary")
            elif mode == "oval":
                mask = ellipse_mask(labels.shape, geometry)
                self._apply_drawn_pool(mask, selected, "Oval boundary")
            elif mode == "separate":
                corrected = split_pool_with_line(labels, selected[0], geometry)
                self._record_correction(corrected, f"Separate pool {selected[0]}")
            elif mode == "hint":
                rectangle = self._rectangle_for_hint(geometry)
                if self._current_path is None or state.effective_peak_index is None:
                    return
                frame = read_frame(self._current_path, state.effective_peak_index)
                mask = guided_pool_mask(
                    frame, geometry, rectangle, state.segmentation_parameters
                )
                self._apply_drawn_pool(mask, selected, "Hint-guided resegmentation")
            elif mode == "background":
                self._apply_background_geometry(geometry)
        except (IndexError, TypeError, ValueError) as error:
            QMessageBox.warning(self, "Correction not applied", str(error))

    def _rectangle_for_hint(
        self, points: tuple[tuple[float, float], ...]
    ) -> RectangleBounds:
        state = self.current_analysis_state
        if state is None or not points:
            raise ValueError("Draw the hint inside an analysis rectangle.")
        center_x = float(np.mean([point[0] for point in points]))
        center_y = float(np.mean([point[1] for point in points]))
        for rectangle in state.rectangles:
            if (
                rectangle.x0 <= center_x < rectangle.x1
                and rectangle.y0 <= center_y < rectangle.y1
            ):
                return rectangle
        raise ValueError("Draw the hint inside an analysis rectangle.")

    def _apply_drawn_pool(
        self,
        mask: np.ndarray[Any, Any],
        selected: tuple[int, ...],
        description: str,
    ) -> None:
        state = self.current_analysis_state
        labels = self._current_labels()
        if state is None or labels is None:
            return
        mask = np.asarray(mask, dtype=bool) & analysis_region_mask(
            labels.shape, tuple(state.rectangles)
        )
        replace_ids = selected if len(selected) == 1 else ()
        try:
            corrected = apply_shape(labels, mask, replace_ids=replace_ids)
        except LabelOverlapError as error:
            answer = QMessageBox.warning(
                self,
                "Boundary overlaps another pool",
                f"{error}\n\nAllow this boundary to take ownership of the "
                "overlapping pixels?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.statusBar().showMessage("Correction cancelled to prevent overlap")
                return
            corrected = apply_shape(
                labels,
                mask,
                replace_ids=replace_ids,
                allow_overwrite=True,
            )
        self._record_correction(corrected, description)

    def _record_correction(
        self, corrected: np.ndarray[Any, Any], description: str
    ) -> None:
        state = self.current_analysis_state
        labels = self._current_labels()
        if state is None or labels is None:
            return
        corrected = np.asarray(corrected, dtype=np.int32)
        edit = make_edit(labels, corrected, description)
        if not len(edit.flat_indices):
            self.statusBar().showMessage("Correction made no changes")
            return
        state.corrected_labels = corrected
        state.correction_undo.append(edit)
        state.correction_redo.clear()
        existing = {int(value) for value in np.unique(corrected) if value > 0}
        state.selected_pool_ids.intersection_update(existing)
        self._invalidate_backgrounds()
        self._show_segmentation_state()
        self.statusBar().showMessage(description)

    @Slot()
    def _delete_selected_pools(self) -> None:
        state = self.current_analysis_state
        labels = self._current_labels()
        if state is None or labels is None or not state.selected_pool_ids:
            return
        selected = tuple(sorted(state.selected_pool_ids))
        corrected = delete_pools(labels, selected)
        state.selected_pool_ids.clear()
        self._record_correction(corrected, f"Delete {len(selected)} pool(s)")

    @Slot()
    def _merge_selected_pools(self) -> None:
        state = self.current_analysis_state
        labels = self._current_labels()
        if state is None or labels is None:
            return
        selected = tuple(sorted(state.selected_pool_ids))
        try:
            corrected = merge_pools(labels, selected)
        except ValueError as error:
            QMessageBox.warning(self, "Pools not merged", str(error))
            return
        state.selected_pool_ids = {selected[0]}
        self._record_correction(corrected, f"Merge {len(selected)} pools")

    @Slot()
    def _undo_correction(self) -> None:
        state = self.current_analysis_state
        labels = self._current_labels()
        if state is None or labels is None or not state.correction_undo:
            return
        edit = state.correction_undo.pop()
        edit.apply(labels, undo=True)
        state.correction_redo.append(edit)
        state.selected_pool_ids.clear()
        self._invalidate_backgrounds()
        self._show_segmentation_state()
        self.statusBar().showMessage(f"Undid: {edit.description}")

    @Slot()
    def _redo_correction(self) -> None:
        state = self.current_analysis_state
        labels = self._current_labels()
        if state is None or labels is None or not state.correction_redo:
            return
        edit = state.correction_redo.pop()
        edit.apply(labels)
        state.correction_undo.append(edit)
        state.selected_pool_ids.clear()
        self._invalidate_backgrounds()
        self._show_segmentation_state()
        self.statusBar().showMessage(f"Redid: {edit.description}")

    @Slot()
    def _reset_corrections(self) -> None:
        state = self.current_analysis_state
        if state is None or state.segmentation_result is None:
            return
        state.selected_pool_ids.clear()
        self._record_correction(
            state.segmentation_result.labels.copy(), "Reset to automatic segmentation"
        )

    def _invalidate_backgrounds(self) -> None:
        self._background_token += 1
        self._remove_background_move_roi()
        state = self.current_analysis_state
        if state is not None:
            state.background_result = None
            state.corrected_background_labels = None
            state.background_assignments.clear()
            state.selected_background_id = None
            state.background_undo.clear()
            state.background_redo.clear()
        if hasattr(self, "canvas"):
            self.canvas.clear_backgrounds()
        if hasattr(self, "background_summary_label"):
            self._show_correction_state()
            self._show_background_state()

    @Slot()
    def _start_background_detection(self) -> None:
        state = self.current_analysis_state
        pool_labels = self._current_labels()
        if (
            state is None
            or pool_labels is None
            or self._current_path is None
            or state.effective_peak_index is None
        ):
            return
        self._stop_all_drawing()
        self._remove_background_move_roi()
        self._background_token += 1
        token = self._background_token
        path = self._current_path
        peak_index = state.effective_peak_index
        rectangles = tuple(state.rectangles)
        parameters = state.background_parameters
        pools_snapshot = np.asarray(pool_labels, dtype=np.int32).copy()
        task = _FunctionTask(
            token,
            lambda: find_recording_backgrounds(
                path, peak_index, pools_snapshot, rectangles, parameters
            ),
        )
        task.signals.succeeded.connect(self._background_detection_finished)
        task.signals.failed.connect(self._background_detection_failed)
        self.run_background_button.setEnabled(False)
        self.run_background_button.setText("Finding backgrounds…")
        mode_text = "one per pool" if parameters.mode == "pool-wise" else "one common"
        self.background_summary_label.setText(
            f"Scoring {mode_text} background candidates on peak frame {peak_index + 1}…"
        )
        self.statusBar().showMessage(f"Finding backgrounds for {path.name}")
        self._analysis_thread_pool.start(task)

    @Slot(int, object)
    def _background_detection_finished(self, token: int, result: object) -> None:
        if token != self._background_token or not isinstance(result, BackgroundResult):
            return
        state = self.current_analysis_state
        if state is None:
            return
        state.background_result = result
        state.corrected_background_labels = result.labels.copy()
        state.background_assignments = {
            region.background_id: region.pool_id for region in result.regions
        }
        state.selected_background_id = None
        state.background_undo.clear()
        state.background_redo.clear()
        if result.missing_pool_ids:
            state.selected_pool_ids = {result.missing_pool_ids[0]}
        self._show_correction_state()
        self._show_background_state()
        missing_count = len(result.missing_pool_ids) + int(result.common_missing)
        self.statusBar().showMessage(
            f"Found {len(result.regions)} background region(s)"
            + (f"; {missing_count} require manual placement" if missing_count else "")
        )
        if missing_count:
            details = (
                "No valid common background was found."
                if result.common_missing
                else "No valid automatic background was found for pool(s): "
                + ", ".join(str(value) for value in result.missing_pool_ids)
                + "."
            )
            instructions = (
                "Use Draw/add to place the common background manually."
                if result.common_missing
                else "Choose a pool in the new dropdown, then use Draw/add to "
                "place its background. The first missing pool is highlighted now."
            )
            QMessageBox.information(
                self,
                "Manual background needed",
                f"{details}\n\n{instructions}",
            )

    @Slot(int, str)
    def _background_detection_failed(self, token: int, message: str) -> None:
        if token != self._background_token:
            return
        self.run_background_button.setText("Find backgrounds")
        self._show_background_state()
        self.background_summary_label.setText("Background detection failed.")
        self.statusBar().showMessage("Background detection failed")
        self._show_error("Could not find backgrounds", message)

    def _current_background_labels(self) -> np.ndarray[Any, Any] | None:
        state = self.current_analysis_state
        if state is None or state.background_result is None:
            return None
        if state.corrected_background_labels is None:
            state.corrected_background_labels = state.background_result.labels.copy()
        return state.corrected_background_labels

    @staticmethod
    def _missing_background_pool_ids(
        state: RecordingAnalysisState, pool_labels: np.ndarray[Any, Any]
    ) -> tuple[int, ...]:
        if (
            state.background_result is None
            or state.background_parameters.mode != "pool-wise"
        ):
            return ()
        background_labels = state.corrected_background_labels
        if background_labels is None:
            background_labels = state.background_result.labels
        existing_background_ids = {
            int(value) for value in np.unique(background_labels) if value > 0
        }
        assigned_pool_ids = {
            int(pool_id)
            for background_id, pool_id in state.background_assignments.items()
            if background_id in existing_background_ids and pool_id is not None
        }
        pool_ids = {int(value) for value in np.unique(pool_labels) if value > 0}
        return tuple(sorted(pool_ids - assigned_pool_ids))

    def _show_background_state(self) -> None:
        if not hasattr(self, "run_background_button"):
            return
        state = self.current_analysis_state
        pool_labels = self._current_labels()
        can_detect = state is not None and pool_labels is not None
        self.background_mode_checkbox.setEnabled(can_detect)
        self.background_advanced_button.setEnabled(can_detect)
        self.background_advanced_widget.setEnabled(can_detect)
        self.run_background_button.setEnabled(can_detect)
        self.run_background_button.setText("Find backgrounds")
        labels = self._current_background_labels()

        self.background_list.blockSignals(True)
        self.background_list.clear()
        existing_ids: set[int] = set()
        if state is not None and labels is not None:
            counts = np.bincount(labels.ravel())
            for background_id in np.flatnonzero(counts[1:]) + 1:
                value = int(background_id)
                existing_ids.add(value)
                pool_id = state.background_assignments.get(value)
                owner_text = "common" if pool_id is None else f"pool {pool_id}"
                item = QListWidgetItem(
                    f"BG {value} — {owner_text}, {int(counts[value])} px²"
                )
                item.setData(Qt.ItemDataRole.UserRole, value)
                self.background_list.addItem(item)
                if value == state.selected_background_id:
                    item.setSelected(True)
                    self.background_list.setCurrentItem(item)
            if state.selected_background_id not in existing_ids:
                state.selected_background_id = None
        self.background_list.blockSignals(False)

        has_result = state is not None and labels is not None
        has_selection = bool(
            has_result
            and state is not None
            and state.selected_background_id in existing_ids
        )
        self.background_list.setEnabled(has_result)
        self.show_background_checkbox.setEnabled(has_result)
        self.draw_background_button.setEnabled(has_result)
        self.move_background_button.setEnabled(has_selection)
        self.delete_background_button.setEnabled(has_selection)
        self.undo_background_button.setEnabled(
            bool(has_result and state and state.background_undo)
        )
        self.redo_background_button.setEnabled(
            bool(has_result and state and state.background_redo)
        )
        self.reset_background_button.setEnabled(
            bool(has_result and state and state.background_undo)
        )
        if not has_result or state is None or labels is None:
            self._set_missing_background_pools(())
            self.canvas.clear_backgrounds()
            self.background_summary_label.setText(
                "Finish pool segmentation before finding backgrounds."
                if not can_detect
                else "Choose a background mode, then click Find backgrounds."
            )
            self._show_export_state()
            return

        self.canvas.set_backgrounds(
            labels,
            ()
            if state.selected_background_id is None
            else (state.selected_background_id,),
        )
        self.canvas.background_item.setVisible(
            self.show_background_checkbox.isChecked()
        )
        if state.background_parameters.mode == "pool-wise" and pool_labels is not None:
            missing = self._missing_background_pool_ids(state, pool_labels)
            self._set_missing_background_pools(missing)
            missing_text = (
                "none" if not missing else ", ".join(str(value) for value in missing)
            )
            self.background_summary_label.setText(
                f"{len(existing_ids)} pool-wise background(s). Missing pools: "
                f"{missing_text}."
            )
        else:
            self._set_missing_background_pools(())
            self.background_summary_label.setText(
                "One common background accepted."
                if existing_ids
                else "Common background missing; use Draw/add."
            )
        self._show_export_state()

    def _set_missing_background_pools(self, pool_ids: tuple[int, ...]) -> None:
        state = self.current_analysis_state
        selected_pool_id = None
        if state is not None and len(state.selected_pool_ids) == 1:
            selected_pool_id = next(iter(state.selected_pool_ids))

        self.missing_background_combo.blockSignals(True)
        self.missing_background_combo.clear()
        self.missing_background_combo.addItem("Choose a missing pool…", None)
        selected_index = 0
        for pool_id in pool_ids:
            self.missing_background_combo.addItem(f"Pool {pool_id}", pool_id)
            if pool_id == selected_pool_id:
                selected_index = self.missing_background_combo.count() - 1
        self.missing_background_combo.setCurrentIndex(selected_index)
        self.missing_background_combo.blockSignals(False)

        has_missing = bool(pool_ids)
        self.missing_background_label.setVisible(has_missing)
        self.missing_background_combo.setVisible(has_missing)
        self.missing_background_combo.setEnabled(has_missing)

    @Slot(int)
    def _missing_background_pool_changed(self, index: int) -> None:
        state = self.current_analysis_state
        pool_id = self.missing_background_combo.itemData(index) if index >= 0 else None
        if state is None or pool_id is None:
            return
        state.selected_pool_ids = {int(pool_id)}
        state.selected_background_id = None
        self._show_correction_state()
        self._refresh_background_selection_ui()
        self.statusBar().showMessage(
            f"Pool {pool_id} needs a manual background; click Draw/add"
        )

    @Slot(bool)
    def _background_visibility_changed(self, visible: bool) -> None:
        self.canvas.background_item.setVisible(visible)

    @Slot(int)
    def _background_selection_changed(self, row: int) -> None:
        state = self.current_analysis_state
        if state is None:
            return
        item = self.background_list.item(row) if row >= 0 else None
        state.selected_background_id = (
            None if item is None else int(item.data(Qt.ItemDataRole.UserRole))
        )
        state.selected_pool_ids.clear()
        self._show_correction_state()
        self._refresh_background_selection_ui()

    def _refresh_background_selection_ui(self) -> None:
        """Refresh selection styling and actions without rebuilding the list."""
        state = self.current_analysis_state
        labels = self._current_background_labels()
        if state is None or labels is None:
            self.move_background_button.setEnabled(False)
            self.delete_background_button.setEnabled(False)
            return

        existing_ids = {int(value) for value in np.unique(labels) if value > 0}
        if state.selected_background_id not in existing_ids:
            state.selected_background_id = None

        selected_row = -1
        self.background_list.blockSignals(True)
        self.background_list.clearSelection()
        for row in range(self.background_list.count()):
            item = self.background_list.item(row)
            if int(item.data(Qt.ItemDataRole.UserRole)) == state.selected_background_id:
                selected_row = row
                break
        self.background_list.setCurrentRow(selected_row)
        self.background_list.blockSignals(False)

        has_selection = selected_row >= 0
        self.move_background_button.setEnabled(has_selection)
        self.delete_background_button.setEnabled(has_selection)
        selected_ids = (
            ()
            if state.selected_background_id is None
            else (state.selected_background_id,)
        )
        self.canvas.set_backgrounds(labels, selected_ids)
        self.canvas.background_item.setVisible(
            self.show_background_checkbox.isChecked()
        )

    @Slot()
    def _begin_background_drawing(self) -> None:
        state = self.current_analysis_state
        if state is None or self._current_background_labels() is None:
            return
        if (
            state.background_parameters.mode == "pool-wise"
            and state.selected_background_id is None
            and len(state.selected_pool_ids) != 1
        ):
            self.statusBar().showMessage(
                "Select one vesicle pool, or select an existing background to redraw"
            )
            return
        self._stop_all_drawing()
        self._remove_background_move_roi()
        self.canvas.view_box.set_correction_drawing_mode("background")
        self._apply_region_editability()
        self.statusBar().showMessage(
            "Drag where the background should be placed; Escape cancels"
        )

    def _apply_background_geometry(
        self, geometry: tuple[float, float, float, float]
    ) -> None:
        state = self.current_analysis_state
        labels = self._current_background_labels()
        pool_labels = self._current_labels()
        if (
            state is None
            or labels is None
            or pool_labels is None
            or self._metadata is None
        ):
            return
        left, top, right, bottom = (float(value) for value in geometry)
        if state.background_parameters.mode == "common":
            rectangle = clip_rectangle(
                left,
                top,
                right,
                bottom,
                image_width=self._metadata.width,
                image_height=self._metadata.height,
            )
            mask = np.zeros(labels.shape, dtype=bool)
            mask[rectangle.y0 : rectangle.y1, rectangle.x0 : rectangle.x1] = True
            background_id = state.selected_background_id or 1
            corrected = np.zeros_like(labels)
            assignments = {background_id: None}
        else:
            background_id = state.selected_background_id
            pool_id = (
                state.background_assignments.get(background_id)
                if background_id is not None
                else next(iter(state.selected_pool_ids), None)
            )
            if pool_id is None:
                raise ValueError("Select the pool that owns this background.")
            source_y, source_x = np.nonzero(pool_labels == pool_id)
            if not len(source_y):
                raise ValueError(f"Pool {pool_id} no longer exists.")
            target_x, target_y = (left + right) / 2, (top + bottom) / 2
            dx = round(target_x - float(np.mean(source_x)))
            dy = round(target_y - float(np.mean(source_y)))
            moved_x, moved_y = source_x + dx, source_y + dy
            if (
                moved_x.min() < 0
                or moved_x.max() >= labels.shape[1]
                or moved_y.min() < 0
                or moved_y.max() >= labels.shape[0]
            ):
                raise ValueError("The pool-shaped background would exceed the image.")
            mask = np.zeros(labels.shape, dtype=bool)
            mask[moved_y, moved_x] = True
            if background_id is None:
                existing_for_pool = next(
                    (
                        value
                        for value, owner in state.background_assignments.items()
                        if owner == pool_id
                    ),
                    None,
                )
                background_id = existing_for_pool or int(labels.max()) + 1
            corrected = labels.copy()
            corrected[corrected == background_id] = 0
            assignments = dict(state.background_assignments)
            assignments[background_id] = pool_id

        self._validate_background_mask(mask, background_id)
        corrected[mask] = background_id
        self._record_background_edit(corrected, assignments, "Draw background")

    def _validate_background_mask(
        self, mask: np.ndarray[Any, Any], replacing_id: int
    ) -> None:
        state = self.current_analysis_state
        pool_labels = self._current_labels()
        background_labels = self._current_background_labels()
        if state is None or pool_labels is None or background_labels is None:
            raise ValueError("Background state is unavailable.")
        allowed = analysis_region_mask(mask.shape, tuple(state.rectangles))
        if not np.any(mask) or not np.all(allowed[mask]):
            raise ValueError(
                "Backgrounds must stay completely inside an analysis region."
            )
        if np.any(pool_labels[mask] > 0):
            raise ValueError("A background cannot overlap an accepted vesicle pool.")
        conflicts = (background_labels[mask] > 0) & (
            background_labels[mask] != replacing_id
        )
        if np.any(conflicts):
            raise ValueError("Background regions cannot overlap each other.")

    @Slot()
    def _begin_background_move(self) -> None:
        state = self.current_analysis_state
        labels = self._current_background_labels()
        if state is None or labels is None or state.selected_background_id is None:
            return
        self._stop_all_drawing()
        self._remove_background_move_roi()
        yy, xx = np.nonzero(labels == state.selected_background_id)
        if not len(yy):
            return
        x0, y0 = int(xx.min()), int(yy.min())
        self._background_move_origin = (x0, y0)
        self._background_move_roi = pg.RectROI(
            (x0, y0),
            (int(xx.max()) - x0 + 1, int(yy.max()) - y0 + 1),
            movable=True,
            resizable=False,
            pen=pg.mkPen("#ff5bd7", width=3),
        )
        self._background_move_roi.setZValue(30)
        self._background_move_roi.sigRegionChangeFinished.connect(
            self._background_move_finished
        )
        self.canvas.view_box.addItem(self._background_move_roi)
        self.statusBar().showMessage(
            "Drag the magenta background box to its new location"
        )

    @Slot()
    def _background_move_finished(self) -> None:
        state = self.current_analysis_state
        labels = self._current_background_labels()
        roi = self._background_move_roi
        origin = self._background_move_origin
        if (
            state is None
            or labels is None
            or roi is None
            or origin is None
            or state.selected_background_id is None
        ):
            self._remove_background_move_roi()
            return
        dx = round(roi.pos().x() - origin[0])
        dy = round(roi.pos().y() - origin[1])
        background_id = state.selected_background_id
        yy, xx = np.nonzero(labels == background_id)
        moved_x, moved_y = xx + dx, yy + dy
        self._remove_background_move_roi()
        try:
            if (
                moved_x.min() < 0
                or moved_x.max() >= labels.shape[1]
                or moved_y.min() < 0
                or moved_y.max() >= labels.shape[0]
            ):
                raise ValueError("The moved background would exceed the image.")
            mask = np.zeros(labels.shape, dtype=bool)
            mask[moved_y, moved_x] = True
            self._validate_background_mask(mask, background_id)
            corrected = labels.copy()
            corrected[corrected == background_id] = 0
            corrected[mask] = background_id
            self._record_background_edit(
                corrected, dict(state.background_assignments), "Move background"
            )
        except ValueError as error:
            QMessageBox.warning(self, "Background not moved", str(error))

    def _remove_background_move_roi(self) -> None:
        if self._background_move_roi is not None:
            self.canvas.view_box.removeItem(self._background_move_roi)
        self._background_move_roi = None
        self._background_move_origin = None

    @Slot()
    def _delete_selected_background(self) -> None:
        state = self.current_analysis_state
        labels = self._current_background_labels()
        if state is None or labels is None or state.selected_background_id is None:
            return
        background_id = state.selected_background_id
        corrected = labels.copy()
        corrected[corrected == background_id] = 0
        assignments = dict(state.background_assignments)
        assignments.pop(background_id, None)
        state.selected_background_id = None
        self._record_background_edit(corrected, assignments, "Delete background")

    def _record_background_edit(
        self,
        corrected: np.ndarray[Any, Any],
        assignments: dict[int, int | None],
        description: str,
    ) -> None:
        state = self.current_analysis_state
        labels = self._current_background_labels()
        if state is None or labels is None:
            return
        corrected = np.asarray(corrected, dtype=np.int32)
        mask_edit = make_edit(labels, corrected, description)
        if (
            not len(mask_edit.flat_indices)
            and assignments == state.background_assignments
        ):
            self.statusBar().showMessage("Background edit made no changes")
            return
        edit = _BackgroundEdit(
            mask_edit=mask_edit,
            before_assignments=dict(state.background_assignments),
            after_assignments=dict(assignments),
            description=description,
        )
        state.corrected_background_labels = corrected
        state.background_assignments = dict(assignments)
        state.background_undo.append(edit)
        state.background_redo.clear()
        self._show_correction_state()
        self._show_background_state()
        self.statusBar().showMessage(description)

    @Slot()
    def _undo_background(self) -> None:
        state = self.current_analysis_state
        labels = self._current_background_labels()
        if state is None or labels is None or not state.background_undo:
            return
        edit = state.background_undo.pop()
        edit.mask_edit.apply(labels, undo=True)
        state.background_assignments = dict(edit.before_assignments)
        state.background_redo.append(edit)
        state.selected_background_id = None
        self._show_correction_state()
        self._show_background_state()
        self.statusBar().showMessage(f"Undid: {edit.description}")

    @Slot()
    def _redo_background(self) -> None:
        state = self.current_analysis_state
        labels = self._current_background_labels()
        if state is None or labels is None or not state.background_redo:
            return
        edit = state.background_redo.pop()
        edit.mask_edit.apply(labels)
        state.background_assignments = dict(edit.after_assignments)
        state.background_undo.append(edit)
        state.selected_background_id = None
        self._show_correction_state()
        self._show_background_state()
        self.statusBar().showMessage(f"Redid: {edit.description}")

    @Slot()
    def _reset_backgrounds(self) -> None:
        state = self.current_analysis_state
        if state is None or state.background_result is None:
            return
        assignments = {
            region.background_id: region.pool_id
            for region in state.background_result.regions
        }
        state.selected_background_id = None
        self._record_background_edit(
            state.background_result.labels.copy(), assignments, "Reset backgrounds"
        )

    def _export_readiness(self) -> tuple[bool, str]:
        state = self.current_analysis_state
        metadata = self.current_metadata
        if state is None or metadata is None or self._current_path is None:
            return False, "Open a recording before exporting."
        if state.peak_result is None or state.effective_peak_index is None:
            return False, "Calculate the recording peak before exporting."
        pool_labels = self._current_labels()
        if pool_labels is None or not np.any(pool_labels):
            return False, "Complete pool segmentation before exporting."
        background_labels = self._current_background_labels()
        if background_labels is None or not np.any(background_labels):
            return False, "Find or draw the required background region(s)."
        missing = self._missing_background_pool_ids(state, pool_labels)
        if missing:
            return (
                False,
                f"Place backgrounds for {len(missing)} red-highlighted pool(s) first.",
            )
        return True, (
            "Ready to export raw traces, ImageJ ROIs, and the display LUT using "
            "ND2 timestamps."
            if metadata.timestamps_s is not None
            else "Ready to export; the app will ask for the frame interval."
        )

    def _show_export_state(self) -> None:
        if not hasattr(self, "export_analysis_button"):
            return
        ready, message = self._export_readiness()
        self.export_analysis_button.setEnabled(ready and not self._export_running)
        if not self._export_running:
            self.export_analysis_button.setText("Export current recording…")
            self.export_summary_label.setText(message)

    @Slot()
    def _start_analysis_export(self) -> None:
        ready, message = self._export_readiness()
        if not ready:
            QMessageBox.warning(self, "Recording is not ready to export", message)
            return
        state = self.current_analysis_state
        metadata = self.current_metadata
        pool_labels = self._current_labels()
        background_labels = self._current_background_labels()
        if (
            state is None
            or metadata is None
            or self._current_path is None
            or state.peak_result is None
            or state.segmentation_result is None
            or state.background_result is None
            or state.effective_peak_index is None
            or pool_labels is None
            or background_labels is None
        ):
            return

        uniform_interval_s = None
        if metadata.timestamps_s is None:
            uniform_interval_s, provided = QInputDialog.getDouble(
                self,
                "Frame interval needed",
                "Seconds per frame:",
                1.0,
                0.000001,
                1000000.0,
                6,
            )
            if not provided:
                use_frames = QMessageBox.question(
                    self,
                    "Use frame numbers?",
                    "No frame interval was provided. Continue with one-based frame "
                    "numbers and no seconds column?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if use_frames != QMessageBox.StandardButton.Yes:
                    return
                uniform_interval_s = None

        initial_directory = self._last_export_directory or self._current_path.parent
        selected_directory = QFileDialog.getExistingDirectory(
            self,
            "Choose export folder",
            str(initial_directory),
        )
        if not selected_directory:
            return
        output_directory = Path(selected_directory).resolve()
        self._last_export_directory = output_directory
        output_paths = (
            output_directory / f"{self._current_path.stem}_analysis.xlsx",
            output_directory / f"{self._current_path.stem}_rois.zip",
            output_directory / f"{self._current_path.stem}_display.lut",
        )
        conflicts = tuple(path for path in output_paths if path.exists())
        overwrite = False
        if conflicts:
            names = "\n".join(path.name for path in conflicts)
            answer = QMessageBox.question(
                self,
                "Replace existing export files?",
                f"These files already exist:\n{names}\n\nReplace them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            overwrite = True

        request = AnalysisExportRequest(
            source_path=self._current_path,
            output_directory=output_directory,
            metadata=metadata,
            rectangles=tuple(state.rectangles),
            peak_result=state.peak_result,
            effective_peak_index=state.effective_peak_index,
            segmentation_parameters=state.segmentation_parameters,
            background_parameters=state.background_parameters,
            pool_labels=np.asarray(pool_labels, dtype=np.int32).copy(),
            automatic_pool_labels=state.segmentation_result.labels.copy(),
            background_labels=np.asarray(background_labels, dtype=np.int32).copy(),
            automatic_background_labels=state.background_result.labels.copy(),
            background_assignments=dict(state.background_assignments),
            display_lut=self.canvas.display_lut(),
            display_levels=(
                self.canvas.display_levels
                if self.canvas.display_levels is not None
                else (0.0, 1.0)
            ),
            uniform_interval_s=uniform_interval_s,
            overwrite=overwrite,
        )
        self._stop_all_drawing()
        self._export_token += 1
        token = self._export_token
        task = _FunctionTask(token, lambda: export_analysis_bundle(request))
        task.signals.succeeded.connect(self._analysis_export_finished)
        task.signals.failed.connect(self._analysis_export_failed)
        self._set_export_running(True)
        self.export_analysis_button.setText("Extracting traces and exporting…")
        self.export_summary_label.setText(
            "Streaming raw frames and writing the XLSX, ROI ZIP, and display LUT…"
        )
        self.statusBar().showMessage(f"Exporting {self._current_path.name}")
        self._analysis_thread_pool.start(task)

    def _set_export_running(self, running: bool) -> None:
        self._export_running = running
        self.recording_list.setEnabled(not running)
        self.open_file_action.setEnabled(not running)
        self.open_folder_action.setEnabled(not running)
        if hasattr(self, "analysis_controls"):
            self.analysis_controls.setEnabled(not running)

    @Slot(int, object)
    def _analysis_export_finished(self, token: int, result: object) -> None:
        if token != self._export_token or not isinstance(result, AnalysisExportResult):
            return
        self._set_export_running(False)
        self._show_background_state()
        self.export_summary_label.setText(
            f"Exported {result.frame_count} frames, {result.pool_count} pool(s), "
            f"and {result.background_count} background(s). Time source: "
            f"{result.time_source}."
        )
        self.statusBar().showMessage(f"Export complete: {result.workbook_path.name}")
        QMessageBox.information(
            self,
            "Export complete",
            f"Saved:\n{result.workbook_path}\n{result.roi_zip_path}\n{result.lut_path}",
        )

    @Slot(int, str)
    def _analysis_export_failed(self, token: int, message: str) -> None:
        if token != self._export_token:
            return
        self._set_export_running(False)
        self._show_background_state()
        self.export_summary_label.setText("Export failed; no incomplete file was kept.")
        self.statusBar().showMessage("Trace, ROI, and LUT export failed")
        self._show_error("Could not export analysis", message)

    def _clear_analysis_controls(self) -> None:
        self._stop_correction_drawing()
        self._clear_region_items()
        self.find_peak_button.setEnabled(False)
        self._update_peak_button_text()
        self.peak_summary_label.setText("Loading recording…")
        self.override_peak_checkbox.setEnabled(False)
        self.override_peak_checkbox.blockSignals(True)
        self.override_peak_checkbox.setChecked(False)
        self.override_peak_checkbox.blockSignals(False)
        self.manual_peak_spin.setEnabled(False)
        self.go_to_peak_button.setEnabled(False)
        self.run_segmentation_button.setEnabled(False)
        self.run_segmentation_button.setText("Run automatic segmentation")
        self.show_segmentation_checkbox.setEnabled(False)
        self.segmentation_summary_label.setText(
            "Calculate a recording peak before segmentation."
        )
        self.pool_list.blockSignals(True)
        self.pool_list.clear()
        self.pool_list.blockSignals(False)
        for control in (
            self.pool_list,
            self.undo_correction_button,
            self.redo_correction_button,
            self.reset_corrections_button,
            self.delete_pool_button,
            self.merge_pool_button,
            self.freehand_pool_button,
            self.oval_pool_button,
            self.separate_pool_button,
            self.hint_pool_button,
        ):
            control.setEnabled(False)
        self.correction_summary_label.setText(
            "Run automatic segmentation before making corrections."
        )
        self._set_missing_background_pools(())
        self.background_list.blockSignals(True)
        self.background_list.clear()
        self.background_list.blockSignals(False)
        for control in (
            self.background_mode_checkbox,
            self.background_advanced_button,
            self.background_advanced_widget,
            self.run_background_button,
            self.show_background_checkbox,
            self.background_list,
            self.undo_background_button,
            self.redo_background_button,
            self.reset_background_button,
            self.draw_background_button,
            self.move_background_button,
            self.delete_background_button,
        ):
            control.setEnabled(False)
        self.background_summary_label.setText(
            "Finish pool segmentation before finding backgrounds."
        )
        self.export_analysis_button.setEnabled(False)
        self.export_summary_label.setText(
            "Complete pool segmentation and background selection before exporting."
        )
        self.canvas.clear_segmentation()
        self.canvas.clear_backgrounds()

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._export_running:
            QMessageBox.information(
                self,
                "Export in progress",
                "Please wait for the current trace and ROI export to finish before "
                "closing the application.",
            )
            event.ignore()
            return
        self._stop_playback()
        self._stop_all_drawing()
        self._metadata_token += 1
        self._frame_token += 1
        self._peak_token += 1
        self._segmentation_token += 1
        self._background_token += 1
        self._export_token += 1
        self._remove_background_move_roi()
        self._thread_pool.clear()
        self._analysis_thread_pool.clear()
        self._thread_pool.waitForDone(3000)
        self._analysis_thread_pool.waitForDone(3000)
        super().closeEvent(event)


def _human_file_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")
