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
    QPointF,
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
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
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
    QTabWidget,
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
from vesicle_analysis.area_analysis import (
    AreaAnalysisResult,
    FrameAreaDetection,
    analyze_recording_areas,
    area_masks_to_rgba,
    detect_frame_pool_areas,
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
    apply_split_group_decision,
    apply_shape,
    delete_pools,
    ellipse_mask,
    guided_pool_mask,
    make_edit,
    merge_pools,
    polygon_mask,
    split_pool_with_line,
)
from vesicle_analysis.motion_correction import (
    CorrectedRecordingExportResult,
    FrameTransform,
    MotionCorrectionResult,
    apply_frame_transform,
    estimate_motion_correction,
    export_corrected_recording,
    replace_frame_transform,
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
    quality_review_to_rgba,
    segment_recording,
    split_review_to_rgba,
)
from vesicle_analysis.trace_export import (
    AnalysisExportRequest,
    AnalysisExportResult,
    TraceExtractionResult,
    export_analysis_bundle,
    extract_recording_traces,
)
from vesicle_analysis.trace_review import (
    TraceReviewDecision,
    normalize_pool_traces,
    read_review_manifest,
    write_review_manifest,
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
class _PoolReviewSnapshot:
    validity_by_pool: dict[int, bool]
    merged_split_group_ids: set[int]
    quality_review_pool_ids: set[int]
    split_review_group_ids: set[int]

    def copy(self) -> _PoolReviewSnapshot:
        return _PoolReviewSnapshot(
            validity_by_pool=dict(self.validity_by_pool),
            merged_split_group_ids=set(self.merged_split_group_ids),
            quality_review_pool_ids=set(self.quality_review_pool_ids),
            split_review_group_ids=set(self.split_review_group_ids),
        )


@dataclass(slots=True)
class _PoolEdit:
    mask_edit: MaskEdit
    review_before: _PoolReviewSnapshot
    review_after: _PoolReviewSnapshot

    @property
    def description(self) -> str:
        return self.mask_edit.description

    def apply(
        self,
        labels: np.ndarray[Any, Any],
        *,
        undo: bool = False,
    ) -> _PoolReviewSnapshot:
        self.mask_edit.apply(labels, undo=undo)
        source = self.review_before if undo else self.review_after
        return source.copy()


@dataclass(slots=True)
class RecordingAnalysisState:
    motion_mode: str | None = None
    motion_result: MotionCorrectionResult | None = None
    motion_accepted: bool = False
    motion_error: str | None = None
    rectangles: list[RectangleBounds] = field(default_factory=list)
    peak_result: PeakDetectionResult | None = None
    peak_error: str | None = None
    manual_peak_index: int | None = None
    segmentation_parameters: SegmentationParameters = field(
        default_factory=SegmentationParameters
    )
    segmentation_result: SegmentationResult | None = None
    corrected_labels: np.ndarray[Any, Any] | None = None
    pool_validity: dict[int, bool] = field(default_factory=dict)
    merged_split_group_ids: set[int] = field(default_factory=set)
    quality_review_pool_ids: set[int] = field(default_factory=set)
    split_review_group_ids: set[int] = field(default_factory=set)
    review_state_initialized: bool = False
    correction_undo: list[_PoolEdit] = field(default_factory=list)
    correction_redo: list[_PoolEdit] = field(default_factory=list)
    selected_pool_ids: set[int] = field(default_factory=set)
    area_result: AreaAnalysisResult | None = None
    area_error: str | None = None
    background_parameters: BackgroundParameters = field(
        default_factory=BackgroundParameters
    )
    background_result: BackgroundResult | None = None
    corrected_background_labels: np.ndarray[Any, Any] | None = None
    background_assignments: dict[int, int | None] = field(default_factory=dict)
    selected_background_id: int | None = None
    background_undo: list[_BackgroundEdit] = field(default_factory=list)
    background_redo: list[_BackgroundEdit] = field(default_factory=list)
    trace_result: TraceExtractionResult | None = None
    excluded_pool_reasons: dict[int, str] = field(default_factory=dict)

    def initialize_pool_review_state(self) -> None:
        result = self.segmentation_result
        if result is None:
            self.clear_pool_review_state()
            return
        self.pool_validity = {
            pool.pool_id: pool.accepted for pool in result.pools
        }
        self.merged_split_group_ids.clear()
        self.quality_review_pool_ids = {pool.pool_id for pool in result.pools}
        self.split_review_group_ids = {
            group.group_id for group in result.split_groups
        }
        self.review_state_initialized = True

    def ensure_pool_review_state(self) -> None:
        if not self.review_state_initialized:
            self.initialize_pool_review_state()

    def clear_pool_review_state(self) -> None:
        self.pool_validity.clear()
        self.merged_split_group_ids.clear()
        self.quality_review_pool_ids.clear()
        self.split_review_group_ids.clear()
        self.review_state_initialized = False

    def pool_review_snapshot(self) -> _PoolReviewSnapshot:
        self.ensure_pool_review_state()
        return _PoolReviewSnapshot(
            validity_by_pool=dict(self.pool_validity),
            merged_split_group_ids=set(self.merged_split_group_ids),
            quality_review_pool_ids=set(self.quality_review_pool_ids),
            split_review_group_ids=set(self.split_review_group_ids),
        )

    def restore_pool_review_snapshot(
        self, snapshot: _PoolReviewSnapshot
    ) -> None:
        self.pool_validity = dict(snapshot.validity_by_pool)
        self.merged_split_group_ids = set(snapshot.merged_split_group_ids)
        self.quality_review_pool_ids = set(snapshot.quality_review_pool_ids)
        self.split_review_group_ids = set(snapshot.split_review_group_ids)
        self.review_state_initialized = True

    @property
    def motion_ready(self) -> bool:
        return self.motion_mode == "none" or (
            self.motion_result is not None and self.motion_accepted
        )

    @property
    def effective_motion_correction(self) -> MotionCorrectionResult | None:
        if self.motion_accepted:
            return self.motion_result
        return None

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
class _MotionBatchResult:
    results: dict[Path, MotionCorrectionResult]
    failures: dict[Path, str]


@dataclass(slots=True)
class _AnalysisBatchExportResult:
    results: tuple[AnalysisExportResult, ...]
    failures: dict[Path, str]


def _export_analysis_batch(
    requests: tuple[AnalysisExportRequest, ...],
) -> _AnalysisBatchExportResult:
    results: list[AnalysisExportResult] = []
    failures: dict[Path, str] = {}
    for request in requests:
        try:
            results.append(export_analysis_bundle(request))
        except Exception as error:  # noqa: BLE001 - preserve other batch exports
            failures[request.source_path] = f"{type(error).__name__}: {error}"
    return _AnalysisBatchExportResult(tuple(results), failures)


@dataclass(slots=True)
class _BackgroundEdit:
    mask_edit: MaskEdit
    before_assignments: dict[int, int | None]
    after_assignments: dict[int, int | None]
    description: str


@dataclass(slots=True)
class _TraceReviewItem:
    path: Path
    metadata: RecordingMetadata
    state: RecordingAnalysisState
    pool_labels: np.ndarray[Any, Any] | None
    background_labels: np.ndarray[Any, Any] | None
    trace_readiness_error: str | None = None
    export_readiness_error: str | None = None


def _detect_peak_batch(
    requests: tuple[
        tuple[Path, tuple[RectangleBounds, ...], MotionCorrectionResult | None], ...
    ],
) -> _PeakBatchResult:
    """Calculate one independent peak per recording in a single batch action."""
    results: dict[Path, PeakDetectionResult] = {}
    failures: dict[Path, str] = {}
    for path, rectangles, correction in requests:
        try:
            results[path] = detect_recording_peak(path, rectangles, correction)
        except Exception as error:  # noqa: BLE001 - preserve other batch results
            failures[path] = f"{type(error).__name__}: {error}"
    return _PeakBatchResult(results=results, failures=failures)


def _load_motion_preview(
    path: Path, index: int, transform: FrameTransform
) -> tuple[int, FrameTransform, np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Read and correct one QC preview frame outside the GUI thread."""
    frame = read_frame(path, index)
    return index, transform, frame, apply_frame_transform(frame, transform)


def _load_area_review_frame(
    path: Path,
    index: int,
    correction: MotionCorrectionResult | None,
    pool_labels: np.ndarray[Any, Any],
    rectangles: tuple[RectangleBounds, ...],
    parameters: SegmentationParameters,
) -> tuple[int, np.ndarray[Any, Any], FrameAreaDetection]:
    frame = read_frame(path, index, correction)
    detection = detect_frame_pool_areas(frame, pool_labels, rectangles, parameters)
    return index, frame, detection


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


class _MotionWorkerSignals(QObject):
    succeeded = Signal(int, object)
    failed = Signal(int, str)
    progress = Signal(int, int, int, int, str)


class _MotionTask(QRunnable):
    def __init__(self, token: int, requests: tuple[tuple[Path, str], ...]) -> None:
        super().__init__()
        self.token = token
        self.requests = requests
        self.signals = _MotionWorkerSignals()

    @Slot()
    def run(self) -> None:
        results: dict[Path, MotionCorrectionResult] = {}
        failures: dict[Path, str] = {}
        try:
            total_recordings = len(self.requests)
            for recording_index, (path, mode) in enumerate(self.requests, start=1):
                try:
                    result = estimate_motion_correction(
                        path,
                        mode,
                        progress_callback=lambda frame, total, ri=recording_index, p=path: (
                            self.signals.progress.emit(
                                ri, total_recordings, frame, total, p.name
                            )
                        ),
                    )
                except Exception as error:  # noqa: BLE001
                    failures[path] = f"{type(error).__name__}: {error}"
                else:
                    results[path] = result
        except Exception as error:  # noqa: BLE001
            self.signals.failed.emit(self.token, f"{type(error).__name__}: {error}")
        else:
            self.signals.succeeded.emit(
                self.token, _MotionBatchResult(results=results, failures=failures)
            )


class _AreaWorkerSignals(QObject):
    succeeded = Signal(int, object)
    failed = Signal(int, str)
    progress = Signal(int, int, int)


class _AreaTask(QRunnable):
    def __init__(
        self,
        token: int,
        path: Path,
        pool_labels: np.ndarray[Any, Any],
        rectangles: tuple[RectangleBounds, ...],
        peak_frame_index: int,
        parameters: SegmentationParameters,
        correction: MotionCorrectionResult | None,
    ) -> None:
        super().__init__()
        self.token = token
        self.path = path
        self.pool_labels = pool_labels
        self.rectangles = rectangles
        self.peak_frame_index = peak_frame_index
        self.parameters = parameters
        self.correction = correction
        self.signals = _AreaWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = analyze_recording_areas(
                self.path,
                self.pool_labels,
                self.rectangles,
                self.peak_frame_index,
                self.parameters,
                self.correction,
                progress_callback=lambda frame, total: self.signals.progress.emit(
                    self.token, frame, total
                ),
            )
        except Exception as error:  # noqa: BLE001
            self.signals.failed.emit(self.token, f"{type(error).__name__}: {error}")
        else:
            self.signals.succeeded.emit(self.token, result)


class ImageCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame: np.ndarray[Any, Any] | None = None
        self._levels: tuple[float, float] | None = None
        self._pool_id_centroids: dict[int, tuple[float, float]] = {}
        self._pool_id_labels: dict[int, pg.TextItem] = {}
        self._pool_id_leaders: dict[int, pg.PlotCurveItem] = {}
        self._pool_ids_visible = True

        self.graphics = pg.GraphicsLayoutWidget()
        self.view_box = DrawingViewBox()
        self.graphics.addItem(self.view_box, row=0, col=0)
        self.view_box.invertY(True)
        self.image_item = pg.ImageItem()
        self.view_box.addItem(self.image_item)
        self.background_item = pg.ImageItem()
        self.background_item.setZValue(14)
        self.view_box.addItem(self.background_item)
        self.quality_review_item = pg.ImageItem()
        self.quality_review_item.setZValue(16)
        self.view_box.addItem(self.quality_review_item)
        self.split_review_item = pg.ImageItem()
        self.split_review_item.setZValue(17)
        self.view_box.addItem(self.split_review_item)
        self.segmentation_item = pg.ImageItem()
        self.segmentation_item.setZValue(15)
        self.view_box.addItem(self.segmentation_item)
        self.view_box.sigRangeChanged.connect(self._layout_pool_ids)
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
        self.quality_review_item.clear()
        self.split_review_item.clear()
        self.clear_pool_ids()
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

    def set_quality_review(
        self,
        candidate_labels: np.ndarray[Any, Any],
        validity_by_pool: dict[int, bool],
        quality_flags_by_pool: dict[int, tuple[str, ...]],
        review_pool_ids: set[int],
    ) -> None:
        self.quality_review_item.setImage(
            quality_review_to_rgba(
                candidate_labels,
                validity_by_pool,
                quality_flags_by_pool,
                review_pool_ids,
            ),
            autoLevels=False,
        )

    def set_split_review(
        self,
        candidate_labels: np.ndarray[Any, Any],
        result: SegmentationResult,
        merged_group_ids: set[int],
        review_group_ids: set[int],
    ) -> None:
        self.split_review_item.setImage(
            split_review_to_rgba(
                candidate_labels,
                result.split_groups,
                merged_group_ids,
                review_group_ids,
            ),
            autoLevels=False,
        )

    def clear_reviews(self) -> None:
        self.quality_review_item.clear()
        self.split_review_item.clear()

    def set_pool_ids(self, labels: np.ndarray[Any, Any]) -> None:
        self.clear_pool_ids()
        for pool_id in (int(value) for value in np.unique(labels) if value > 0):
            yy, xx = np.nonzero(labels == pool_id)
            if not len(yy):
                continue
            centroid = (float(xx.mean()), float(yy.mean()))
            label = pg.TextItem(
                text=str(pool_id),
                color=(255, 255, 255),
                anchor=(0.5, 0.5),
                fill=pg.mkBrush(0, 0, 0, 180),
                border=pg.mkPen(255, 255, 255, 150),
            )
            label.setZValue(21)
            leader = pg.PlotCurveItem(pen=pg.mkPen(255, 255, 255, 170, width=1))
            leader.setZValue(20)
            leader.hide()
            self.view_box.addItem(leader)
            self.view_box.addItem(label)
            self._pool_id_centroids[pool_id] = centroid
            self._pool_id_labels[pool_id] = label
            self._pool_id_leaders[pool_id] = leader
        self._layout_pool_ids()

    def clear_pool_ids(self) -> None:
        for item in self._pool_id_labels.values():
            self.view_box.removeItem(item)
        for item in self._pool_id_leaders.values():
            self.view_box.removeItem(item)
        self._pool_id_centroids.clear()
        self._pool_id_labels.clear()
        self._pool_id_leaders.clear()

    def set_pool_ids_visible(self, visible: bool) -> None:
        self._pool_ids_visible = visible
        if not visible:
            for item in self._pool_id_labels.values():
                item.hide()
            for item in self._pool_id_leaders.values():
                item.hide()
            return
        self._layout_pool_ids()

    @Slot()
    def _layout_pool_ids(self, *_args: object) -> None:
        if not self._pool_id_labels:
            return
        if not self._pool_ids_visible:
            return

        def rectangle(
            center: QPointF, width: float, height: float
        ) -> tuple[float, float, float, float]:
            return (
                center.x() - width / 2,
                center.y() - height / 2,
                center.x() + width / 2,
                center.y() + height / 2,
            )

        def overlaps(
            first: tuple[float, float, float, float],
            second: tuple[float, float, float, float],
        ) -> bool:
            return not (
                first[2] <= second[0]
                or second[2] <= first[0]
                or first[3] <= second[1]
                or second[3] <= first[1]
            )

        desired: dict[int, QPointF] = {
            pool_id: self.view_box.mapViewToScene(QPointF(*centroid))
            for pool_id, centroid in self._pool_id_centroids.items()
        }
        sizes = {
            pool_id: (max(18.0, 9.0 * len(str(pool_id)) + 8.0), 20.0)
            for pool_id in desired
        }
        desired_rectangles = {
            pool_id: rectangle(desired[pool_id], *sizes[pool_id]) for pool_id in desired
        }
        neighbors: dict[int, set[int]] = {pool_id: set() for pool_id in desired}
        pool_ids = sorted(desired)
        for index, first_id in enumerate(pool_ids):
            for second_id in pool_ids[index + 1 :]:
                if overlaps(
                    desired_rectangles[first_id], desired_rectangles[second_id]
                ):
                    neighbors[first_id].add(second_id)
                    neighbors[second_id].add(first_id)

        clusters: list[list[int]] = []
        unseen = set(pool_ids)
        while unseen:
            start = min(unseen)
            stack = [start]
            cluster: list[int] = []
            unseen.remove(start)
            while stack:
                current = stack.pop()
                cluster.append(current)
                for neighbor in neighbors[current]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        stack.append(neighbor)
            clusters.append(sorted(cluster))

        offsets = [(0.0, 0.0)]
        directions = (
            (1.0, 0.0),
            (0.707, 0.707),
            (0.0, 1.0),
            (-0.707, 0.707),
            (-1.0, 0.0),
            (-0.707, -0.707),
            (0.0, -1.0),
            (0.707, -0.707),
        )
        for radius in (26.0, 52.0, 78.0):
            offsets.extend((dx * radius, dy * radius) for dx, dy in directions)

        occupied: list[tuple[float, float, float, float]] = []
        for cluster in clusters:
            if len(cluster) > 10:
                for pool_id in cluster:
                    self._pool_id_labels[pool_id].hide()
                    self._pool_id_leaders[pool_id].hide()
                continue
            for pool_id in cluster:
                label = self._pool_id_labels[pool_id]
                leader = self._pool_id_leaders[pool_id]
                selected_offset = offsets[-1]
                selected_rect = desired_rectangles[pool_id]
                for offset in offsets:
                    candidate_center = desired[pool_id] + QPointF(*offset)
                    candidate_rect = rectangle(candidate_center, *sizes[pool_id])
                    if not any(
                        overlaps(candidate_rect, existing) for existing in occupied
                    ):
                        selected_offset = offset
                        selected_rect = candidate_rect
                        break
                occupied.append(selected_rect)
                scene_position = desired[pool_id] + QPointF(*selected_offset)
                view_position = self.view_box.mapSceneToView(scene_position)
                label.setPos(view_position)
                label.show()
                if selected_offset == (0.0, 0.0):
                    leader.hide()
                else:
                    centroid = self._pool_id_centroids[pool_id]
                    leader.setData(
                        [centroid[0], view_position.x()],
                        [centroid[1], view_position.y()],
                    )
                    leader.show()

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


class MotionCorrectionDialog(QDialog):
    """Review correction traces, inspect frames, and adjust individual frames."""

    def __init__(
        self,
        path: Path,
        metadata: RecordingMetadata,
        result: MotionCorrectionResult,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Review motion correction — {path.name}")
        self.setModal(True)
        self._path = path
        self._metadata = metadata
        self._result = result
        self._preview_cache: dict[
            int, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
        ] = {}
        self._preview_pending_index: int | None = None
        self._preview_loading = False
        self._preview_token = 0
        self._displayed_preview_index: int | None = None
        self._preview_initialized = False
        self._preview_shape: tuple[int, int] | None = None
        self._preview_thread_pool = QThreadPool(self)
        self._preview_thread_pool.setMaxThreadCount(1)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(90)
        self._preview_timer.timeout.connect(self._request_preview)

        outer_layout = QVBoxLayout(self)
        self.content_scroll_area = QScrollArea()
        self.content_scroll_area.setObjectName("motionCorrectionScrollArea")
        self.content_scroll_area.setWidgetResizable(True)
        self.content_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.content_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        content = QWidget()
        layout = QVBoxLayout(content)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.trace_plot = pg.PlotWidget()
        self.trace_plot.setMinimumHeight(120)
        self.trace_plot.setLabel("bottom", "Frame")
        self.trace_plot.setLabel("left", "Pixels / degrees")
        self.trace_plot.addLegend()

        self.preview_widget = pg.GraphicsLayoutWidget()
        self.preview_widget.setMinimumHeight(220)
        self.preview_widget.addLabel("Original", row=0, col=0)
        self.preview_widget.addLabel("Corrected", row=0, col=1)
        self.original_view = self.preview_widget.addViewBox(
            row=1, col=0, lockAspect=True
        )
        self.corrected_view = self.preview_widget.addViewBox(
            row=1, col=1, lockAspect=True
        )
        self.original_view.invertY(True)
        self.corrected_view.invertY(True)
        self.original_view.setDefaultPadding(0.02)
        self.corrected_view.setDefaultPadding(0.02)
        self.original_item = pg.ImageItem()
        self.corrected_item = pg.ImageItem()
        self.original_view.addItem(self.original_item)
        self.corrected_view.addItem(self.corrected_item)
        layout.addWidget(self.preview_widget, 1)

        fit_buttons = QHBoxLayout()
        fit_buttons.addStretch(1)
        fit_original = QPushButton("Fit original")
        fit_original.clicked.connect(self._fit_original_preview)
        fit_corrected = QPushButton("Fit corrected")
        fit_corrected.clicked.connect(self._fit_corrected_preview)
        fit_buttons.addWidget(fit_original)
        fit_buttons.addWidget(fit_corrected)
        layout.addLayout(fit_buttons)
        layout.addWidget(self.trace_plot)

        navigation = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, metadata.frame_count - 1)
        self.frame_slider.valueChanged.connect(self._frame_changed)
        self.frame_slider.sliderReleased.connect(self._request_preview_immediately)
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(1, metadata.frame_count)
        self.frame_spin.setPrefix("Frame ")
        self.frame_spin.valueChanged.connect(
            lambda value: self.frame_slider.setValue(value - 1)
        )
        previous_warning = QPushButton("Previous warning")
        previous_warning.clicked.connect(lambda: self._jump_warning(-1))
        next_warning = QPushButton("Next warning")
        next_warning.clicked.connect(lambda: self._jump_warning(1))
        navigation.addWidget(previous_warning)
        navigation.addWidget(self.frame_slider, 1)
        navigation.addWidget(self.frame_spin)
        navigation.addWidget(next_warning)
        layout.addLayout(navigation)

        adjustments = QGridLayout()
        self.translation_x_spin = QSpinBox()
        self.translation_x_spin.setRange(
            -result.parameters.maximum_translation_px,
            result.parameters.maximum_translation_px,
        )
        self.translation_x_spin.setSuffix(" px")
        self.translation_y_spin = QSpinBox()
        self.translation_y_spin.setRange(
            -result.parameters.maximum_translation_px,
            result.parameters.maximum_translation_px,
        )
        self.translation_y_spin.setSuffix(" px")
        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(
            -result.parameters.maximum_rotation_degrees,
            result.parameters.maximum_rotation_degrees,
        )
        self.rotation_spin.setSingleStep(0.5)
        self.rotation_spin.setDecimals(1)
        self.rotation_spin.setSuffix("°")
        self.rotation_spin.setEnabled(result.mode == "rigid")
        self.apply_motion_adjustment_button = QPushButton(
            "Apply manual adjustment to this frame"
        )
        self.apply_motion_adjustment_button.clicked.connect(self._apply_adjustment)
        adjustments.addWidget(QLabel("X translation:"), 0, 0)
        adjustments.addWidget(self.translation_x_spin, 0, 1)
        adjustments.addWidget(QLabel("Y translation:"), 0, 2)
        adjustments.addWidget(self.translation_y_spin, 0, 3)
        adjustments.addWidget(QLabel("Rotation:"), 1, 0)
        adjustments.addWidget(self.rotation_spin, 1, 1)
        adjustments.addWidget(self.apply_motion_adjustment_button, 1, 2, 1, 2)
        layout.addLayout(adjustments)

        self.frame_status = QLabel()
        self.frame_status.setWordWrap(True)
        layout.addWidget(self.frame_status)
        self.content_scroll_area.setWidget(content)
        outer_layout.addWidget(self.content_scroll_area, 1)

        self.dialog_buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.dialog_buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Accept correction"
        )
        self.dialog_buttons.accepted.connect(self.accept)
        self.dialog_buttons.rejected.connect(self.reject)
        outer_layout.addWidget(self.dialog_buttons)

        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            safe_width = max(1, int(available.width() * 0.92))
            safe_height = max(1, int(available.height() * 0.88))
            self.setMaximumSize(safe_width, safe_height)
            self.resize(min(1180, safe_width), min(850, safe_height))
        self._refresh_plot()
        initial = result.warning_indices[0] if result.warning_indices else 0
        self.frame_slider.setValue(initial)
        self._frame_changed(initial)

    @property
    def result(self) -> MotionCorrectionResult:
        return self._result

    def _refresh_plot(self) -> None:
        self.trace_plot.clear()
        frames = np.arange(1, len(self._result.transforms) + 1)
        dx = [item.translation_x_px for item in self._result.transforms]
        dy = [item.translation_y_px for item in self._result.transforms]
        angles = [item.rotation_degrees for item in self._result.transforms]
        self.trace_plot.plot(frames, dx, pen=pg.mkPen("#00cfe8", width=2), name="X px")
        self.trace_plot.plot(frames, dy, pen=pg.mkPen("#ffd166", width=2), name="Y px")
        if self._result.mode == "rigid":
            self.trace_plot.plot(
                frames,
                angles,
                pen=pg.mkPen("#ff7ad9", width=2),
                name="Rotation °",
            )
        warning_indices = self._result.warning_indices
        if warning_indices:
            self.trace_plot.plot(
                np.asarray(warning_indices) + 1,
                [dx[index] for index in warning_indices],
                pen=None,
                symbol="o",
                symbolBrush="#ff3030",
                symbolSize=9,
                name="Warning",
            )
        max_translation = max(
            max(abs(item.translation_x_px), abs(item.translation_y_px))
            for item in self._result.transforms
        )
        max_rotation = max(
            abs(item.rotation_degrees) for item in self._result.transforms
        )
        self.summary_label.setText(
            f"Mode: {self._result.mode}; reference: frame 1; maximum translation: "
            f"{max_translation} px; maximum rotation: {max_rotation:.1f}°; "
            f"warnings: {len(warning_indices)}. Red markers require inspection."
        )

    @Slot(int)
    def _frame_changed(self, index: int) -> None:
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(index + 1)
        self.frame_spin.blockSignals(False)
        transform = self._result.transforms[index]
        for spin in (
            self.translation_x_spin,
            self.translation_y_spin,
            self.rotation_spin,
        ):
            spin.blockSignals(True)
        self.translation_x_spin.setValue(transform.translation_x_px)
        self.translation_y_spin.setValue(transform.translation_y_px)
        self.rotation_spin.setValue(transform.rotation_degrees)
        editable = index != self._result.reference_frame_index
        self.translation_x_spin.setEnabled(editable)
        self.translation_y_spin.setEnabled(editable)
        self.rotation_spin.setEnabled(editable and self._result.mode == "rigid")
        self.apply_motion_adjustment_button.setEnabled(editable)
        for spin in (
            self.translation_x_spin,
            self.translation_y_spin,
            self.rotation_spin,
        ):
            spin.blockSignals(False)
        cached = self._preview_cache.get(index)
        if cached is not None:
            self._preview_timer.stop()
            self._preview_pending_index = None
            self._display_preview(index, *cached)
        else:
            self._set_frame_status(index, "Loading preview…")
            self._preview_timer.start()

    def _set_frame_status(self, index: int, preview_message: str | None = None) -> None:
        transform = self._result.transforms[index]
        status = (
            f"Frame {index + 1}: correlation {transform.correlation:.3f}; "
            f"X {transform.translation_x_px:+d} px; Y {transform.translation_y_px:+d} px; "
            f"rotation {transform.rotation_degrees:+.1f}°."
        )
        if transform.warning:
            status += f" Warning: {transform.warning}."
        if transform.manually_adjusted:
            status += " Manually adjusted."
        if preview_message:
            status += f" {preview_message}"
        self.frame_status.setText(status)

    @Slot()
    def _request_preview_immediately(self) -> None:
        self._preview_timer.stop()
        self._request_preview()

    @Slot()
    def _request_preview(self) -> None:
        index = self.frame_slider.value()
        cached = self._preview_cache.get(index)
        if cached is not None:
            self._preview_pending_index = None
            self._display_preview(index, *cached)
            return
        self._preview_pending_index = index
        self._start_pending_preview()

    def _start_pending_preview(self) -> None:
        if self._preview_loading or self._preview_pending_index is None:
            return
        index = self._preview_pending_index
        self._preview_pending_index = None
        cached = self._preview_cache.get(index)
        if cached is not None:
            if index == self.frame_slider.value():
                self._display_preview(index, *cached)
            return
        transform = self._result.transforms[index]
        self._preview_token += 1
        token = self._preview_token
        task = _FunctionTask(
            token,
            lambda: _load_motion_preview(self._path, index, transform),
        )
        task.signals.succeeded.connect(self._preview_loaded)
        task.signals.failed.connect(self._preview_failed)
        self._preview_loading = True
        self._preview_thread_pool.start(task)

    @Slot(int, object)
    def _preview_loaded(self, token: int, payload: object) -> None:
        if token != self._preview_token:
            return
        if not isinstance(payload, tuple) or len(payload) != 4:
            self._preview_loading = False
            self._set_frame_status(
                self.frame_slider.value(), "Could not load preview: invalid result."
            )
            self._start_pending_preview()
            return
        self._preview_loading = False
        index, transform, frame, corrected = payload
        if not isinstance(index, int):
            self._preview_failed(token, "Preview worker returned an invalid frame.")
            return
        if transform != self._result.transforms[index]:
            if index == self.frame_slider.value():
                self._preview_pending_index = index
            self._start_pending_preview()
            return
        self._preview_cache[index] = (frame, corrected)
        while len(self._preview_cache) > 3:
            self._preview_cache.pop(next(iter(self._preview_cache)))
        if index == self.frame_slider.value():
            self._display_preview(index, frame, corrected)
        self._start_pending_preview()

    @Slot(int, str)
    def _preview_failed(self, token: int, message: str) -> None:
        if token != self._preview_token:
            return
        self._preview_loading = False
        index = self.frame_slider.value()
        self._set_frame_status(index, f"Could not load preview: {message}")
        self._start_pending_preview()

    def _display_preview(
        self,
        index: int,
        frame: np.ndarray[Any, Any],
        corrected: np.ndarray[Any, Any],
    ) -> None:
        levels = _display_levels(frame)
        self.original_item.setImage(
            frame,
            autoLevels=False,
            levels=levels,
            autoDownsample=True,
        )
        self.corrected_item.setImage(
            corrected,
            autoLevels=False,
            levels=levels,
            autoDownsample=True,
        )
        self._displayed_preview_index = index
        self._preview_shape = frame.shape
        self._set_frame_status(index)
        if not self._preview_initialized:
            self._preview_initialized = True
            self._fit_original_preview()
            self._fit_corrected_preview()
            QTimer.singleShot(0, self._fit_original_preview)
            QTimer.singleShot(0, self._fit_corrected_preview)

    @Slot()
    def _fit_original_preview(self) -> None:
        self._fit_preview_view(self.original_view)

    @Slot()
    def _fit_corrected_preview(self) -> None:
        self._fit_preview_view(self.corrected_view)

    def _fit_preview_view(self, view: pg.ViewBox) -> None:
        if self._preview_shape is None:
            return
        height, width = self._preview_shape
        view.setRange(
            QRectF(0.0, 0.0, float(width), float(height)),
            padding=0.02,
            disableAutoRange=True,
        )

    @Slot()
    def _apply_adjustment(self) -> None:
        index = self.frame_slider.value()
        try:
            self._result = replace_frame_transform(
                self._path,
                self._result,
                index,
                translation_x_px=self.translation_x_spin.value(),
                translation_y_px=self.translation_y_spin.value(),
                rotation_degrees=(
                    self.rotation_spin.value() if self._result.mode == "rigid" else 0.0
                ),
            )
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Adjustment could not be applied", str(error))
            return
        self._preview_cache.pop(index, None)
        self._refresh_plot()
        self._frame_changed(index)

    def _jump_warning(self, direction: int) -> None:
        warnings = self._result.warning_indices
        if not warnings:
            return
        current = self.frame_slider.value()
        ordered = warnings if direction > 0 else tuple(reversed(warnings))
        target = next(
            (
                index
                for index in ordered
                if (index > current if direction > 0 else index < current)
            ),
            ordered[0],
        )
        self.frame_slider.setValue(target)


class AreaMaskReviewDialog(QDialog):
    """Optional frame-by-frame review of thresholded pool-area masks."""

    def __init__(
        self,
        path: Path,
        metadata: RecordingMetadata,
        correction: MotionCorrectionResult | None,
        pool_labels: np.ndarray[Any, Any],
        rectangles: tuple[RectangleBounds, ...],
        result: AreaAnalysisResult,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Review pool area masks — {path.name}")
        self.setModal(True)
        self._path = path
        self._correction = correction
        self._pool_labels = np.asarray(pool_labels, dtype=np.int32).copy()
        self._rectangles = rectangles
        self._result = result
        self._cache: dict[int, tuple[np.ndarray[Any, Any], FrameAreaDetection]] = {}
        self._pending_index: int | None = None
        self._loading = False
        self._token = 0
        self._displayed_index: int | None = None
        self._preview_shape: tuple[int, int] | None = None
        self._thread_pool = QThreadPool(self)
        self._thread_pool.setMaxThreadCount(1)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(90)
        self._timer.timeout.connect(self._request_preview)

        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Cyan/white pixels are detected in this frame. Yellow boundaries are "
            "the fixed accepted peak-frame masks; detection cannot extend beyond them."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        graphics = pg.GraphicsLayoutWidget()
        graphics.setMinimumHeight(320)
        self.view = graphics.addViewBox(lockAspect=True)
        self.view.invertY(True)
        self.view.setDefaultPadding(0.02)
        self.image_item = pg.ImageItem()
        self.overlay_item = pg.ImageItem()
        self.overlay_item.setZValue(10)
        self.view.addItem(self.image_item)
        self.view.addItem(self.overlay_item)
        layout.addWidget(graphics, 1)

        controls = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, metadata.frame_count - 1)
        self.frame_slider.valueChanged.connect(self._frame_changed)
        self.frame_slider.sliderReleased.connect(self._request_preview_immediately)
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(1, metadata.frame_count)
        self.frame_spin.setPrefix("Frame ")
        self.frame_spin.valueChanged.connect(
            lambda value: self.frame_slider.setValue(value - 1)
        )
        fit_button = QPushButton("Fit image")
        fit_button.clicked.connect(self._fit_image)
        controls.addWidget(self.frame_slider, 1)
        controls.addWidget(self.frame_spin)
        controls.addWidget(fit_button)
        layout.addLayout(controls)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            safe_width = max(1, int(available.width() * 0.90))
            safe_height = max(1, int(available.height() * 0.86))
            self.setMaximumSize(safe_width, safe_height)
            self.resize(min(1000, safe_width), min(760, safe_height))
        self.frame_slider.setValue(result.peak_frame_index)
        self._frame_changed(result.peak_frame_index)

    @Slot(int)
    def _frame_changed(self, index: int) -> None:
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(index + 1)
        self.frame_spin.blockSignals(False)
        cached = self._cache.get(index)
        if cached is not None:
            self._timer.stop()
            self._pending_index = None
            self._display_preview(index, *cached)
        else:
            self._set_status(index, "Loading mask…")
            self._timer.start()

    def _set_status(self, index: int, message: str | None = None) -> None:
        areas = self._result.areas_px[index]
        qc = self._result.qc_codes[index]
        missing = [
            str(pool_id)
            for pool_id, area in zip(self._result.pool_ids, areas, strict=True)
            if area == 0
        ]
        text = (
            f"Frame {index + 1}: {int(np.count_nonzero(areas))}/"
            f"{len(areas)} pools detected; {int(np.sum(areas))} detected pixels; "
            f"{int(np.count_nonzero(qc))} QC warning(s)."
        )
        if missing:
            visible = ", ".join(missing[:20])
            text += f" Not detected: {visible}"
            if len(missing) > 20:
                text += f" and {len(missing) - 20} more"
            text += "."
        if message:
            text += f" {message}"
        self.status_label.setText(text)

    @Slot()
    def _request_preview_immediately(self) -> None:
        self._timer.stop()
        self._request_preview()

    @Slot()
    def _request_preview(self) -> None:
        index = self.frame_slider.value()
        cached = self._cache.get(index)
        if cached is not None:
            self._pending_index = None
            self._display_preview(index, *cached)
            return
        self._pending_index = index
        self._start_pending_preview()

    def _start_pending_preview(self) -> None:
        if self._loading or self._pending_index is None:
            return
        index = self._pending_index
        self._pending_index = None
        self._token += 1
        token = self._token
        task = _FunctionTask(
            token,
            lambda: _load_area_review_frame(
                self._path,
                index,
                self._correction,
                self._pool_labels,
                self._rectangles,
                self._result.parameters,
            ),
        )
        task.signals.succeeded.connect(self._preview_loaded)
        task.signals.failed.connect(self._preview_failed)
        self._loading = True
        self._thread_pool.start(task)

    @Slot(int, object)
    def _preview_loaded(self, token: int, payload: object) -> None:
        if token != self._token:
            return
        self._loading = False
        if not isinstance(payload, tuple) or len(payload) != 3:
            self._preview_failed(token, "Preview worker returned an invalid result.")
            return
        index, frame, detection = payload
        if not isinstance(index, int) or not isinstance(detection, FrameAreaDetection):
            self._preview_failed(token, "Preview worker returned an invalid mask.")
            return
        self._cache[index] = (frame, detection)
        while len(self._cache) > 3:
            self._cache.pop(next(iter(self._cache)))
        if index == self.frame_slider.value():
            self._display_preview(index, frame, detection)
        self._start_pending_preview()

    @Slot(int, str)
    def _preview_failed(self, token: int, message: str) -> None:
        if token != self._token:
            return
        self._loading = False
        self._set_status(self.frame_slider.value(), f"Could not load mask: {message}")
        self._start_pending_preview()

    def _display_preview(
        self,
        index: int,
        frame: np.ndarray[Any, Any],
        detection: FrameAreaDetection,
    ) -> None:
        first_display = self._preview_shape is None
        levels = _display_levels(frame)
        self.image_item.setImage(
            frame, autoLevels=False, levels=levels, autoDownsample=True
        )
        self.overlay_item.setImage(
            area_masks_to_rgba(self._pool_labels, detection.labels),
            autoLevels=False,
            autoDownsample=True,
        )
        self._displayed_index = index
        self._preview_shape = frame.shape
        self._set_status(index)
        if index not in self._cache:
            self._cache[index] = (frame, detection)
        if first_display:
            self._fit_image()
            QTimer.singleShot(0, self._fit_image)

    @Slot()
    def _fit_image(self) -> None:
        if self._preview_shape is None:
            return
        height, width = self._preview_shape
        self.view.setRange(
            QRectF(0.0, 0.0, float(width), float(height)),
            padding=0.02,
            disableAutoRange=True,
        )


class TraceReviewDialog(QDialog):
    """Review raw/normalized pool traces and store per-pool exclusions."""

    TRACE_BATCH_SIZE = 15

    def __init__(
        self,
        items: tuple[_TraceReviewItem, ...],
        *,
        allow_batch_export: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not items:
            raise ValueError("Trace review requires at least one recording.")
        self.setWindowTitle("Review pool traces")
        self.setModal(True)
        self._items = items
        self._allow_batch_export = allow_batch_export
        self._updating_pool_list = False
        self._trace_batch_index = 0
        self._trace_token = 0
        self._loading_index: int | None = None
        self.requested_export_paths: tuple[Path, ...] = ()
        self._thread_pool = QThreadPool(self)
        self._thread_pool.setMaxThreadCount(1)

        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Review each pool before export. Unchecked pools are omitted from "
            "trace, area, and ROI exports, but remain in the workbook audit summary."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        navigation = QHBoxLayout()
        previous_button = QPushButton("Previous")
        previous_button.clicked.connect(lambda: self._step_recording(-1))
        self.recording_combo = QComboBox()
        for item in items:
            label = item.path.name
            if item.export_readiness_error is not None:
                label += " — incomplete"
            self.recording_combo.addItem(label)
        self.recording_combo.currentIndexChanged.connect(self._recording_changed)
        next_button = QPushButton("Next")
        next_button.clicked.connect(lambda: self._step_recording(1))
        navigation.addWidget(previous_button)
        navigation.addWidget(self.recording_combo, 1)
        navigation.addWidget(next_button)
        layout.addLayout(navigation)

        content = QSplitter(Qt.Orientation.Horizontal)
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_controls = QHBoxLayout()
        plot_controls.addWidget(QLabel("Trace display:"))
        self.trace_mode_combo = QComboBox()
        self.trace_mode_combo.addItem("Raw intensity", "raw")
        self.trace_mode_combo.addItem("Normalized (pool − background)", "normalized")
        self.trace_mode_combo.currentIndexChanged.connect(self._refresh_plot)
        plot_controls.addWidget(self.trace_mode_combo)
        self.previous_trace_batch_button = QPushButton("Previous trace batch")
        self.previous_trace_batch_button.clicked.connect(
            lambda: self._step_trace_batch(-1)
        )
        plot_controls.addWidget(self.previous_trace_batch_button)
        self.trace_batch_label = QLabel("Trace batch —")
        self.trace_batch_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plot_controls.addWidget(self.trace_batch_label)
        self.next_trace_batch_button = QPushButton("Next trace batch")
        self.next_trace_batch_button.clicked.connect(lambda: self._step_trace_batch(1))
        plot_controls.addWidget(self.next_trace_batch_button)
        fit_traces_button = QPushButton("Fit traces")
        fit_traces_button.clicked.connect(self._fit_trace_plot)
        plot_controls.addWidget(fit_traces_button)
        plot_controls.addStretch(1)
        plot_layout.addLayout(plot_controls)
        self.trace_plot = pg.PlotWidget()
        self.trace_plot.setLabel("bottom", "Frame")
        self.trace_plot.setLabel("left", "Mean intensity")
        self.trace_plot.showGrid(x=True, y=True, alpha=0.2)
        self.trace_plot.setMouseEnabled(x=True, y=True)
        plot_layout.addWidget(self.trace_plot, 1)
        self.trace_note = QLabel()
        self.trace_note.setWordWrap(True)
        plot_layout.addWidget(self.trace_note)
        content.addWidget(plot_panel)

        review_panel = QWidget()
        review_layout = QVBoxLayout(review_panel)
        review_layout.addWidget(QLabel("Pools included in export"))
        self.pool_list = QListWidget()
        self.pool_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.pool_list.itemChanged.connect(self._pool_check_changed)
        self.pool_list.currentRowChanged.connect(self._pool_selected)
        review_layout.addWidget(self.pool_list, 1)
        reason_form = QFormLayout()
        self.reason_combo = QComboBox()
        self.reason_combo.addItems(
            (
                "Artefact",
                "Poor segmentation",
                "Unstable background",
                "Other",
            )
        )
        self.reason_combo.currentTextChanged.connect(self._reason_preset_changed)
        self.reason_edit = QLineEdit()
        self.reason_edit.setPlaceholderText("Optional exclusion note")
        self.reason_edit.editingFinished.connect(self._reason_changed)
        reason_form.addRow("Reason preset:", self.reason_combo)
        reason_form.addRow("Reason / note:", self.reason_edit)
        review_layout.addLayout(reason_form)
        content.addWidget(review_panel)
        content.setStretchFactor(0, 1)
        content.setStretchFactor(1, 0)
        content.setSizes((760, 300))
        layout.addWidget(content, 1)

        self.recording_status = QLabel()
        self.recording_status.setWordWrap(True)
        layout.addWidget(self.recording_status)
        incomplete_count = sum(
            item.export_readiness_error is not None for item in items
        )
        self.batch_status = QLabel(
            f"{incomplete_count} incomplete recording(s) will be ignored during "
            "batch export."
            if incomplete_count
            else "All listed recordings are ready for trace review and export."
        )
        self.batch_status.setWordWrap(True)
        layout.addWidget(self.batch_status)

        file_buttons = QHBoxLayout()
        save_button = QPushButton("Save review…")
        save_button.clicked.connect(self._save_review)
        load_button = QPushButton("Load review…")
        load_button.clicked.connect(self._load_review)
        file_buttons.addWidget(save_button)
        file_buttons.addWidget(load_button)
        file_buttons.addStretch(1)
        layout.addLayout(file_buttons)

        action_buttons = QHBoxLayout()
        action_buttons.addStretch(1)
        self.export_current_button = QPushButton("Export this recording…")
        self.export_current_button.clicked.connect(self._request_current_export)
        action_buttons.addWidget(self.export_current_button)
        self.export_all_button = QPushButton("Export all ready recordings…")
        self.export_all_button.setVisible(allow_batch_export)
        self.export_all_button.clicked.connect(self._request_batch_export)
        action_buttons.addWidget(self.export_all_button)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        action_buttons.addWidget(close_button)
        layout.addLayout(action_buttons)

        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            safe_width = max(1, int(available.width() * 0.92))
            safe_height = max(1, int(available.height() * 0.88))
            self.setMaximumSize(safe_width, safe_height)
            self.resize(min(1180, safe_width), min(780, safe_height))
        self._recording_changed(0)

    def _item(self) -> _TraceReviewItem:
        return self._items[self.recording_combo.currentIndex()]

    @Slot(int)
    def _recording_changed(self, index: int) -> None:
        if index < 0:
            return
        self._trace_batch_index = 0
        item = self._items[index]
        self._populate_pool_list(item)
        self.trace_plot.clear()
        if item.trace_readiness_error is not None:
            self.recording_status.setText(
                f"Trace preview unavailable: {item.trace_readiness_error}"
            )
            self.trace_note.setText("Trace preview is unavailable for this recording.")
            self._update_export_buttons()
            return
        if item.state.trace_result is not None:
            self.recording_status.setText("Trace data loaded.")
            self._refresh_plot()
            self._update_export_buttons()
            return
        if item.pool_labels is None or item.background_labels is None:
            self.recording_status.setText("Ignored: accepted masks are unavailable.")
            self._update_export_buttons()
            return
        self._trace_token += 1
        token = self._trace_token
        self._loading_index = index
        self.recording_status.setText("Loading raw traces from the recording…")
        self.trace_note.setText("Trace preview is loading.")
        task = _FunctionTask(
            token,
            lambda: extract_recording_traces(
                item.path,
                item.pool_labels,
                item.background_labels,
                timestamps_s=item.metadata.timestamps_s,
                motion_correction=item.state.effective_motion_correction,
            ),
        )
        task.signals.succeeded.connect(self._trace_loaded)
        task.signals.failed.connect(self._trace_failed)
        self._thread_pool.start(task)
        self._update_export_buttons()

    def _populate_pool_list(self, item: _TraceReviewItem) -> None:
        self._updating_pool_list = True
        self.pool_list.clear()
        pool_ids = (
            tuple(int(value) for value in np.unique(item.pool_labels) if value > 0)
            if item.pool_labels is not None
            else ()
        )
        item.state.excluded_pool_reasons = {
            pool_id: reason
            for pool_id, reason in item.state.excluded_pool_reasons.items()
            if pool_id in pool_ids
        }
        for pool_id in pool_ids:
            excluded = pool_id in item.state.excluded_pool_reasons
            list_item = QListWidgetItem(
                f"Pool {pool_id}" + (" — excluded" if excluded else "")
            )
            list_item.setData(Qt.ItemDataRole.UserRole, pool_id)
            list_item.setFlags(list_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            list_item.setCheckState(
                Qt.CheckState.Unchecked if excluded else Qt.CheckState.Checked
            )
            self.pool_list.addItem(list_item)
        self._updating_pool_list = False
        if self.pool_list.count():
            self.pool_list.setCurrentRow(0)
        else:
            self.reason_combo.setEnabled(False)
            self.reason_edit.setEnabled(False)

    @Slot(int, object)
    def _trace_loaded(self, token: int, result: object) -> None:
        if token != self._trace_token or not isinstance(result, TraceExtractionResult):
            return
        if self._loading_index is None:
            return
        item = self._items[self._loading_index]
        item.state.trace_result = result
        self._loading_index = None
        self.recording_status.setText("Trace data loaded.")
        self._refresh_plot()
        self._update_export_buttons()

    @Slot(int, str)
    def _trace_failed(self, token: int, message: str) -> None:
        if token != self._trace_token:
            return
        self._loading_index = None
        self.recording_status.setText(f"Trace preview failed: {message}")
        self.trace_note.setText("No trace preview is available.")
        self._update_export_buttons()

    @Slot()
    def _refresh_plot(self) -> None:
        item = self._item()
        traces = item.state.trace_result
        self.trace_plot.clear()
        if traces is None:
            self._update_trace_batch_controls(0)
            return
        x_values = traces.time_s if traces.time_s is not None else traces.frame_numbers
        self.trace_plot.setLabel(
            "bottom", "Time", units="s" if traces.time_s is not None else None
        )
        mode = self.trace_mode_combo.currentData()
        values = traces.pool_traces
        available = np.ones(len(traces.pool_ids), dtype=np.bool_)
        if mode == "normalized":
            try:
                normalized = normalize_pool_traces(
                    traces.pool_ids,
                    traces.pool_traces,
                    traces.background_ids,
                    traces.background_traces,
                    background_mode=item.state.background_parameters.mode,
                    background_assignments=item.state.background_assignments,
                )
            except ValueError as error:
                self.trace_note.setText(
                    f"Displaying raw traces. Normalized traces are unavailable: {error}"
                )
                mode = "raw"
            else:
                values = normalized.values.copy()
                available = normalized.available
                values[:, ~available] = traces.pool_traces[:, ~available]
        selected = self._selected_pool_id()
        batch_count = max(
            1,
            (len(traces.pool_ids) + self.TRACE_BATCH_SIZE - 1) // self.TRACE_BATCH_SIZE,
        )
        self._trace_batch_index = min(self._trace_batch_index, batch_count - 1)
        start = self._trace_batch_index * self.TRACE_BATCH_SIZE
        stop = min(start + self.TRACE_BATCH_SIZE, len(traces.pool_ids))
        visible_columns = range(start, stop)
        self._update_trace_batch_controls(len(traces.pool_ids))
        for visible_index, column in enumerate(visible_columns):
            pool_id = traces.pool_ids[column]
            excluded = pool_id in item.state.excluded_pool_reasons
            colour = pg.intColor(
                visible_index,
                hues=max(1, stop - start),
                values=1,
                maxValue=230,
                minValue=150,
                maxHue=300,
            )
            pen = pg.mkPen(
                colour,
                width=4 if pool_id == selected else (1 if excluded else 2),
                style=Qt.PenStyle.DashLine if excluded else Qt.PenStyle.SolidLine,
            )
            curve = self.trace_plot.plot(
                x_values,
                values[:, column],
                pen=pen,
                name=f"Pool {pool_id}",
            )
            curve.setCurveClickable(True, width=12)
            curve.sigClicked.connect(
                lambda _curve, _event, value=pool_id: self._trace_clicked(value)
            )
        if mode == "raw":
            self.trace_plot.setLabel("left", "Raw mean intensity")
            self.trace_note.setText(
                "Displaying raw pool intensity traces. Dashed traces are excluded "
                "from export."
            )
        else:
            unavailable_ids = [
                str(pool_id)
                for column, (pool_id, ok) in enumerate(
                    zip(traces.pool_ids, available, strict=True)
                )
                if start <= column < stop and not ok
            ]
            self.trace_plot.setLabel("left", "Normalized intensity")
            if unavailable_ids:
                self.trace_note.setText(
                    "Normalized display is unavailable for pool(s) "
                    + ", ".join(unavailable_ids)
                    + "; their raw traces are shown instead. The first-10-frame "
                    "background-corrected baseline was zero or near zero."
                )
            else:
                self.trace_note.setText(
                    "Displaying (pool − assigned background) divided by the mean "
                    "background-corrected signal in frames 1–10. Dashed traces are excluded."
                )
        self._fit_trace_plot()
        QTimer.singleShot(0, self._fit_trace_plot)

    def _update_trace_batch_controls(self, pool_count: int) -> None:
        batch_count = max(
            1,
            (pool_count + self.TRACE_BATCH_SIZE - 1) // self.TRACE_BATCH_SIZE,
        )
        self._trace_batch_index = min(self._trace_batch_index, batch_count - 1)
        start = self._trace_batch_index * self.TRACE_BATCH_SIZE
        stop = min(start + self.TRACE_BATCH_SIZE, pool_count)
        if pool_count:
            self.trace_batch_label.setText(
                f"Pools {start + 1}–{stop} of {pool_count} "
                f"(batch {self._trace_batch_index + 1}/{batch_count})"
            )
        else:
            self.trace_batch_label.setText("No pool traces")
        self.previous_trace_batch_button.setEnabled(self._trace_batch_index > 0)
        self.next_trace_batch_button.setEnabled(
            pool_count > 0 and self._trace_batch_index + 1 < batch_count
        )

    def _step_trace_batch(self, offset: int) -> None:
        traces = self._item().state.trace_result
        if traces is None:
            return
        batch_count = max(
            1,
            (len(traces.pool_ids) + self.TRACE_BATCH_SIZE - 1) // self.TRACE_BATCH_SIZE,
        )
        target = min(max(0, self._trace_batch_index + offset), batch_count - 1)
        if target != self._trace_batch_index:
            self._trace_batch_index = target
            self._refresh_plot()

    @Slot()
    def _fit_trace_plot(self) -> None:
        self.trace_plot.getPlotItem().getViewBox().autoRange(padding=0.05)

    def _trace_clicked(self, pool_id: int) -> None:
        # The click signal is emitted while QGraphicsScene is still processing
        # mouseReleaseEvent. Selecting the list row redraws the plot and deletes
        # its curves, so defer that redraw until the scene has finished the event.
        QTimer.singleShot(0, lambda value=pool_id: self._select_clicked_trace(value))

    def _select_clicked_trace(self, pool_id: int) -> None:
        for row in range(self.pool_list.count()):
            list_item = self.pool_list.item(row)
            if int(list_item.data(Qt.ItemDataRole.UserRole)) == pool_id:
                self.pool_list.setCurrentRow(row)
                self.pool_list.scrollToItem(
                    list_item, QAbstractItemView.ScrollHint.PositionAtCenter
                )
                self.recording_status.setText(
                    f"Selected pool {pool_id} by clicking its trace."
                )
                return

    def _selected_pool_id(self) -> int | None:
        selected = self.pool_list.currentItem()
        return (
            int(selected.data(Qt.ItemDataRole.UserRole))
            if selected is not None
            else None
        )

    @Slot(QListWidgetItem)
    def _pool_check_changed(self, list_item: QListWidgetItem) -> None:
        if self._updating_pool_list:
            return
        pool_id = int(list_item.data(Qt.ItemDataRole.UserRole))
        state = self._item().state
        if list_item.checkState() == Qt.CheckState.Checked:
            state.excluded_pool_reasons.pop(pool_id, None)
        else:
            state.excluded_pool_reasons[pool_id] = self.reason_combo.currentText()
        self._populate_pool_list(self._item())
        for row in range(self.pool_list.count()):
            if int(self.pool_list.item(row).data(Qt.ItemDataRole.UserRole)) == pool_id:
                self.pool_list.setCurrentRow(row)
                break
        self._refresh_plot()
        self._update_export_buttons()

    @Slot(int)
    def _pool_selected(self, _row: int) -> None:
        pool_id = self._selected_pool_id()
        if pool_id is None:
            self.reason_combo.setEnabled(False)
            self.reason_edit.setEnabled(False)
            return
        traces = self._item().state.trace_result
        if traces is not None and pool_id in traces.pool_ids:
            column = traces.pool_ids.index(pool_id)
            target_batch = column // self.TRACE_BATCH_SIZE
            if target_batch != self._trace_batch_index:
                self._trace_batch_index = target_batch
        reason = self._item().state.excluded_pool_reasons.get(pool_id, "")
        excluded = bool(reason) or pool_id in self._item().state.excluded_pool_reasons
        self.reason_combo.setEnabled(excluded)
        self.reason_edit.setEnabled(excluded)
        self.reason_edit.blockSignals(True)
        self.reason_edit.setText(reason)
        self.reason_edit.blockSignals(False)
        if reason in {
            "Artefact",
            "Poor segmentation",
            "Unstable background",
            "Other",
        }:
            self.reason_combo.blockSignals(True)
            self.reason_combo.setCurrentText(reason)
            self.reason_combo.blockSignals(False)
        self._refresh_plot()

    @Slot(str)
    def _reason_preset_changed(self, text: str) -> None:
        pool_id = self._selected_pool_id()
        if pool_id is None or pool_id not in self._item().state.excluded_pool_reasons:
            return
        self.reason_edit.blockSignals(True)
        self.reason_edit.setText(text)
        self.reason_edit.blockSignals(False)
        self._item().state.excluded_pool_reasons[pool_id] = text

    @Slot()
    def _reason_changed(self, *_args: object) -> None:
        pool_id = self._selected_pool_id()
        if pool_id is None or pool_id not in self._item().state.excluded_pool_reasons:
            return
        text = self.reason_edit.text().strip()
        if not text:
            text = self.reason_combo.currentText()
            self.reason_edit.blockSignals(True)
            self.reason_edit.setText(text)
            self.reason_edit.blockSignals(False)
        self._item().state.excluded_pool_reasons[pool_id] = text

    def _step_recording(self, offset: int) -> None:
        count = self.recording_combo.count()
        self.recording_combo.setCurrentIndex(
            (self.recording_combo.currentIndex() + offset) % count
        )

    def _decision(self, item: _TraceReviewItem) -> TraceReviewDecision | None:
        if item.pool_labels is None:
            return None
        pool_ids = tuple(
            int(value) for value in np.unique(item.pool_labels) if value > 0
        )
        if not pool_ids:
            return None
        return TraceReviewDecision(
            source_path=str(item.path),
            source_filename=item.path.name,
            file_size_bytes=item.metadata.file_size_bytes,
            pool_ids=pool_ids,
            excluded_reasons=dict(item.state.excluded_pool_reasons),
        )

    @Slot()
    def _save_review(self) -> None:
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Save trace review",
            str(self._item().path.parent / "trace_review.json"),
            "JSON manifest (*.json)",
        )
        if not selected:
            return
        decisions = tuple(
            decision
            for item in self._items
            if (decision := self._decision(item)) is not None
        )
        try:
            write_review_manifest(Path(selected), decisions, overwrite=True)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could not save review", str(error))
            return
        self.batch_status.setText(f"Saved review decisions to {selected}")

    @Slot()
    def _load_review(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Load trace review",
            str(self._item().path.parent),
            "JSON manifest (*.json)",
        )
        if not selected:
            return
        try:
            decisions = read_review_manifest(Path(selected))
        except (OSError, ValueError, TypeError, KeyError) as error:
            QMessageBox.warning(self, "Could not load review", str(error))
            return
        applied = 0
        skipped = 0
        for item in self._items:
            decision = next(
                (
                    value
                    for value in decisions
                    if Path(value.source_path) == item.path
                    or (
                        value.source_filename == item.path.name
                        and value.file_size_bytes == item.metadata.file_size_bytes
                    )
                ),
                None,
            )
            current = self._decision(item)
            if (
                decision is None
                or current is None
                or decision.pool_ids != current.pool_ids
            ):
                skipped += 1
                continue
            item.state.excluded_pool_reasons = dict(decision.excluded_reasons)
            applied += 1
        self._populate_pool_list(self._item())
        self._refresh_plot()
        self._update_export_buttons()
        self.batch_status.setText(
            f"Loaded review: {applied} recording(s) matched; {skipped} ignored because "
            "the file or accepted pool masks did not match."
        )

    def _included_pool_ids(self, item: _TraceReviewItem) -> tuple[int, ...]:
        if item.pool_labels is None:
            return ()
        return tuple(
            int(value)
            for value in np.unique(item.pool_labels)
            if value > 0 and int(value) not in item.state.excluded_pool_reasons
        )

    def _update_export_buttons(self) -> None:
        item = self._item()
        current_ready = item.export_readiness_error is None and bool(
            self._included_pool_ids(item)
        )
        self.export_current_button.setEnabled(current_ready)
        batch_ready = any(
            value.export_readiness_error is None and self._included_pool_ids(value)
            for value in self._items
        )
        self.export_all_button.setEnabled(self._allow_batch_export and batch_ready)

    @Slot()
    def _request_current_export(self) -> None:
        item = self._item()
        if item.export_readiness_error is not None or not self._included_pool_ids(item):
            return
        self.requested_export_paths = (item.path,)
        self.accept()

    @Slot()
    def _request_batch_export(self) -> None:
        ready_exists = any(
            item.export_readiness_error is None and self._included_pool_ids(item)
            for item in self._items
        )
        if ready_exists:
            self.requested_export_paths = tuple(item.path for item in self._items)
            self.accept()


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
        self._motion_token = 0
        self._motion_running = False
        self._corrected_export_token = 0
        self._corrected_export_running = False
        self._peak_token = 0
        self._peak_batch_running = False
        self._segmentation_token = 0
        self._area_token = 0
        self._area_running = False
        self._area_path: Path | None = None
        self._background_token = 0
        self._export_token = 0
        self._export_running = False
        self._batch_skipped_count = 0
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
        self._overlay_visibility_before_review: tuple[bool, bool, bool] | None = None

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

        self.main_toolbar = self.addToolBar("Files and overlays")
        self.main_toolbar.setMovable(False)
        self.main_toolbar.addAction(self.open_file_action)
        self.main_toolbar.addAction(self.open_folder_action)
        self.main_toolbar.addSeparator()
        self.main_toolbar.addAction(self.draw_region_action)
        self.main_toolbar.addSeparator()

        self.top_show_segmentation_checkbox = QCheckBox(
            "Show accepted segmentation results"
        )
        self.top_show_segmentation_checkbox.setChecked(True)
        self.top_show_segmentation_checkbox.setEnabled(False)
        self.top_show_segmentation_checkbox.toggled.connect(
            self._segmentation_visibility_changed
        )
        self.main_toolbar.addWidget(self.top_show_segmentation_checkbox)

        self.top_show_pool_ids_checkbox = QCheckBox("Show pool IDs")
        self.top_show_pool_ids_checkbox.setChecked(True)
        self.top_show_pool_ids_checkbox.setEnabled(False)
        self.top_show_pool_ids_checkbox.toggled.connect(
            self._pool_ids_visibility_changed
        )
        self.main_toolbar.addWidget(self.top_show_pool_ids_checkbox)

        self.top_show_quality_review_checkbox = QCheckBox(
            "Show valid/invalid review"
        )
        self.top_show_quality_review_checkbox.setChecked(False)
        self.top_show_quality_review_checkbox.setEnabled(False)
        self.top_show_quality_review_checkbox.toggled.connect(
            self._quality_review_visibility_changed
        )
        self.main_toolbar.addWidget(self.top_show_quality_review_checkbox)

        self.top_show_split_review_checkbox = QCheckBox(
            "Show automatic split review"
        )
        self.top_show_split_review_checkbox.setChecked(False)
        self.top_show_split_review_checkbox.setEnabled(False)
        self.top_show_split_review_checkbox.toggled.connect(
            self._split_review_visibility_changed
        )
        self.main_toolbar.addWidget(self.top_show_split_review_checkbox)

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
        group = QGroupBox("Motion correction and analysis")
        group.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(group)

        layout.addWidget(QLabel("XY motion correction"))
        motion_help = QLabel(
            "Choose no correction, integer-pixel translation, or rigid correction "
            "with nearest-neighbor rotation. A choice must be accepted before analysis."
        )
        motion_help.setWordWrap(True)
        layout.addWidget(motion_help)
        self.motion_mode_combo = QComboBox()
        self.motion_mode_combo.addItem("Choose correction…", None)
        self.motion_mode_combo.addItem("No correction", "none")
        self.motion_mode_combo.addItem("Translation (whole pixels)", "translation")
        self.motion_mode_combo.addItem("Rigid body (nearest-neighbor)", "rigid")
        self.motion_mode_combo.currentIndexChanged.connect(self._motion_mode_changed)
        layout.addWidget(self.motion_mode_combo)

        motion_calculation_buttons = QHBoxLayout()
        self.calculate_motion_button = QPushButton("Calculate current")
        self.calculate_motion_button.clicked.connect(self._start_current_motion)
        self.calculate_all_motion_button = QPushButton("Calculate all")
        self.calculate_all_motion_button.setToolTip(
            "Apply the current correction mode to every loaded recording and "
            "calculate sequentially"
        )
        self.calculate_all_motion_button.clicked.connect(self._start_all_motion)
        motion_calculation_buttons.addWidget(self.calculate_motion_button)
        motion_calculation_buttons.addWidget(self.calculate_all_motion_button)
        layout.addLayout(motion_calculation_buttons)

        motion_result_buttons = QHBoxLayout()
        self.review_motion_button = QPushButton("Review / adjust")
        self.review_motion_button.clicked.connect(self._review_motion_correction)
        self.save_corrected_button = QPushButton("Save corrected images…")
        self.save_corrected_button.clicked.connect(self._start_corrected_export)
        motion_result_buttons.addWidget(self.review_motion_button)
        motion_result_buttons.addWidget(self.save_corrected_button)
        layout.addLayout(motion_result_buttons)
        self.motion_summary_label = QLabel(
            "Choose a correction option before drawing analysis regions."
        )
        self.motion_summary_label.setWordWrap(True)
        layout.addWidget(self.motion_summary_label)

        motion_separator = QFrame()
        motion_separator.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(motion_separator)
        layout.addWidget(QLabel("Analysis regions and recording peak"))

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

        segmentation_tabs = QTabWidget()
        basic_segmentation_page = QWidget()
        segmentation_form = QFormLayout(basic_segmentation_page)
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

        segmentation_tabs.addTab(basic_segmentation_page, "Basic")

        advanced_segmentation_page = QWidget()
        advanced_layout = QVBoxLayout(advanced_segmentation_page)
        advanced_layout.setContentsMargins(4, 4, 4, 4)
        advanced_form = QFormLayout()
        advanced_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self.segmentation_prominence_spin = QDoubleSpinBox()
        self.segmentation_prominence_spin.setRange(0.0, 10.0)
        self.segmentation_prominence_spin.setDecimals(2)
        self.segmentation_prominence_spin.setSingleStep(0.25)
        self.segmentation_prominence_spin.setValue(1.0)
        self.segmentation_prominence_spin.setSuffix(" sigma")
        self.segmentation_prominence_spin.setToolTip(
            "Minimum height of a local maximum above its surrounding saddle"
        )
        self.segmentation_prominence_spin.valueChanged.connect(
            self._segmentation_parameters_changed
        )
        advanced_form.addRow("Peak prominence:", self.segmentation_prominence_spin)

        self.segmentation_baseline_combo = QComboBox()
        self.segmentation_baseline_combo.addItem("Robust median / MAD", "robust")
        self.segmentation_baseline_combo.addItem(
            "Mean / standard deviation", "mean_std"
        )
        self.segmentation_baseline_combo.addItem("Manual thresholds only", "manual")
        self.segmentation_baseline_combo.currentIndexChanged.connect(
            self._segmentation_parameters_changed
        )
        advanced_form.addRow("Population baseline:", self.segmentation_baseline_combo)

        self.segmentation_min_area_spin = QSpinBox()
        self.segmentation_min_area_spin.setRange(1, 1000)
        self.segmentation_min_area_spin.setValue(8)
        self.segmentation_min_area_spin.setSuffix(" px²")
        self.segmentation_min_area_spin.valueChanged.connect(
            self._segmentation_parameters_changed
        )
        advanced_form.addRow("Absolute minimum area:", self.segmentation_min_area_spin)

        self.segmentation_max_area_spin = QSpinBox()
        self.segmentation_max_area_spin.setRange(1, 10000)
        self.segmentation_max_area_spin.setValue(150)
        self.segmentation_max_area_spin.setSuffix(" px²")
        self.segmentation_max_area_spin.valueChanged.connect(
            self._segmentation_parameters_changed
        )
        advanced_form.addRow("Absolute maximum area:", self.segmentation_max_area_spin)

        self.segmentation_filter_checks: dict[str, QCheckBox] = {}
        self.segmentation_multiplier_spins: dict[str, QDoubleSpinBox] = {}
        for metric, title in (
            ("area", "Area outliers"),
            ("circularity", "Low circularity"),
            ("solidity", "Low solidity"),
            ("local_contrast", "Low local contrast"),
            ("intensity_variation", "High intensity variation"),
        ):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            enabled = QCheckBox("Enable")
            enabled.setChecked(True)
            multiplier = QDoubleSpinBox()
            multiplier.setRange(0.5, 10.0)
            multiplier.setDecimals(1)
            multiplier.setSingleStep(0.5)
            multiplier.setValue(3.0)
            multiplier.setPrefix("± " if metric == "area" else "")
            multiplier.setSuffix(" × spread")
            enabled.toggled.connect(self._segmentation_parameters_changed)
            multiplier.valueChanged.connect(self._segmentation_parameters_changed)
            row_layout.addWidget(enabled)
            row_layout.addWidget(multiplier, 1)
            self.segmentation_filter_checks[metric] = enabled
            self.segmentation_multiplier_spins[metric] = multiplier
            advanced_form.addRow(f"{title}:", row_widget)

        def optional_threshold(
            maximum: float, decimals: int, step: float
        ) -> QDoubleSpinBox:
            control = QDoubleSpinBox()
            control.setRange(0.0, maximum)
            control.setDecimals(decimals)
            control.setSingleStep(step)
            control.setSpecialValueText("Off")
            control.setValue(0.0)
            control.valueChanged.connect(self._segmentation_parameters_changed)
            return control

        self.segmentation_min_circularity_spin = optional_threshold(1.0, 2, 0.05)
        advanced_form.addRow(
            "Absolute circularity minimum:", self.segmentation_min_circularity_spin
        )
        self.segmentation_min_solidity_spin = optional_threshold(1.0, 2, 0.05)
        advanced_form.addRow(
            "Absolute solidity minimum:", self.segmentation_min_solidity_spin
        )
        self.segmentation_min_contrast_spin = optional_threshold(100.0, 1, 0.5)
        advanced_form.addRow(
            "Absolute contrast minimum:", self.segmentation_min_contrast_spin
        )
        self.segmentation_max_variation_spin = optional_threshold(10.0, 2, 0.1)
        advanced_form.addRow(
            "Absolute variation maximum:", self.segmentation_max_variation_spin
        )
        advanced_layout.addLayout(advanced_form)
        self.reset_segmentation_filters_button = QPushButton("Reset filtering defaults")
        self.reset_segmentation_filters_button.clicked.connect(
            self._reset_segmentation_filter_defaults
        )
        advanced_layout.addWidget(self.reset_segmentation_filters_button)
        segmentation_tabs.addTab(advanced_segmentation_page, "Advanced filtering")
        layout.addWidget(segmentation_tabs)

        self.run_segmentation_button = QPushButton("Run automatic segmentation")
        self.run_segmentation_button.setEnabled(False)
        self.run_segmentation_button.clicked.connect(self._start_segmentation)
        layout.addWidget(self.run_segmentation_button)

        self.show_segmentation_checkbox = QCheckBox(
            "Show accepted segmentation results"
        )
        self.show_segmentation_checkbox.setChecked(True)
        self.show_segmentation_checkbox.setEnabled(False)
        self.show_segmentation_checkbox.toggled.connect(
            self._segmentation_visibility_changed
        )
        layout.addWidget(self.show_segmentation_checkbox)

        self.show_pool_ids_checkbox = QCheckBox("Show pool IDs")
        self.show_pool_ids_checkbox.setChecked(True)
        self.show_pool_ids_checkbox.setEnabled(False)
        self.show_pool_ids_checkbox.toggled.connect(self._pool_ids_visibility_changed)
        layout.addWidget(self.show_pool_ids_checkbox)

        self.show_quality_review_checkbox = QCheckBox("Show valid/invalid review")
        self.show_quality_review_checkbox.setChecked(False)
        self.show_quality_review_checkbox.setEnabled(False)
        self.show_quality_review_checkbox.toggled.connect(
            self._quality_review_visibility_changed
        )
        layout.addWidget(self.show_quality_review_checkbox)

        self.review_quality_button = QPushButton("Review valid/invalid results")
        self.review_quality_button.setCheckable(True)
        self.review_quality_button.setEnabled(False)
        self.review_quality_button.setToolTip(
            "While active, click any automatic candidate to toggle its validity"
        )
        self.review_quality_button.toggled.connect(self._quality_review_toggled)
        layout.addWidget(self.review_quality_button)

        self.quality_review_list = QListWidget()
        self.quality_review_list.setMinimumHeight(70)
        self.quality_review_list.setMaximumHeight(120)
        self.quality_review_list.itemClicked.connect(
            self._quality_review_list_clicked
        )
        layout.addWidget(self.quality_review_list)
        quality_legend = QLabel(
            "Validity: green = valid; red = invalid; orange = shape/intensity; "
            "blue = area. White edges indicate multiple filtering reasons."
        )
        quality_legend.setWordWrap(True)
        layout.addWidget(quality_legend)

        self.show_split_review_checkbox = QCheckBox("Show automatic split review")
        self.show_split_review_checkbox.setChecked(False)
        self.show_split_review_checkbox.setEnabled(False)
        self.show_split_review_checkbox.toggled.connect(
            self._split_review_visibility_changed
        )
        layout.addWidget(self.show_split_review_checkbox)

        self.review_split_button = QPushButton("Review automatic splits")
        self.review_split_button.setCheckable(True)
        self.review_split_button.setEnabled(False)
        self.review_split_button.setToolTip(
            "While active, click a proposed split to toggle the group between split and merged"
        )
        self.review_split_button.toggled.connect(self._split_review_toggled)
        layout.addWidget(self.review_split_button)

        self.split_review_list = QListWidget()
        self.split_review_list.setMinimumHeight(60)
        self.split_review_list.setMaximumHeight(110)
        self.split_review_list.itemClicked.connect(self._split_review_list_clicked)
        layout.addWidget(self.split_review_list)
        split_legend = QLabel(
            "Splits: magenta = currently split; cyan = currently merged; white "
            "edges show the original proposed pieces."
        )
        split_legend.setWordWrap(True)
        layout.addWidget(split_legend)

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

        area_separator = QFrame()
        area_separator.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(area_separator)
        layout.addWidget(QLabel("Pool area over time"))
        area_help = QLabel(
            "Measure detected pixels independently in every frame, strictly inside "
            "each accepted peak-frame pool mask. Reviewing masks is optional."
        )
        area_help.setWordWrap(True)
        layout.addWidget(area_help)
        area_buttons = QHBoxLayout()
        self.calculate_areas_button = QPushButton("Calculate areas")
        self.calculate_areas_button.clicked.connect(self._start_area_analysis)
        self.review_areas_button = QPushButton("Review masks…")
        self.review_areas_button.clicked.connect(self._review_area_masks)
        area_buttons.addWidget(self.calculate_areas_button)
        area_buttons.addWidget(self.review_areas_button)
        layout.addLayout(area_buttons)
        self.area_summary_label = QLabel(
            "Finish pool segmentation before calculating areas."
        )
        self.area_summary_label.setWordWrap(True)
        layout.addWidget(self.area_summary_label)

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
            "Review raw or normalized pool traces, optionally exclude individual "
            "pools, then export raw and normalized data to XLSX, final regions to "
            "an ImageJ-compatible ROI ZIP, and the current display gradient to LUT."
        )
        export_help.setWordWrap(True)
        layout.addWidget(export_help)
        self.review_current_traces_button = QPushButton("Review current recording…")
        self.review_current_traces_button.clicked.connect(
            lambda: self._open_trace_review(False)
        )
        layout.addWidget(self.review_current_traces_button)
        self.review_all_traces_button = QPushButton("Review all / batch export…")
        self.review_all_traces_button.clicked.connect(
            lambda: self._open_trace_review(True)
        )
        layout.addWidget(self.review_all_traces_button)
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
        self._motion_token += 1
        self._corrected_export_token += 1
        self._peak_token += 1
        self._background_token += 1
        self._export_token += 1
        self._peak_batch_running = False
        self._motion_running = False
        self._corrected_export_running = False
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
        self._corrected_export_token += 1
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
        state = self.current_analysis_state
        correction = None if state is None else state.effective_motion_correction
        self._frame_token += 1
        token = self._frame_token
        self._thread_pool.clear()
        task = _FunctionTask(token, lambda: read_frame(path, index, correction))
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

    def _restore_motion_state(self) -> None:
        state = self.current_analysis_state
        self.motion_mode_combo.blockSignals(True)
        mode = None if state is None else state.motion_mode
        index = self.motion_mode_combo.findData(mode)
        self.motion_mode_combo.setCurrentIndex(max(0, index))
        self.motion_mode_combo.blockSignals(False)
        self._show_motion_state()

    @Slot(int)
    def _motion_mode_changed(self, _index: int) -> None:
        state = self.current_analysis_state
        if state is None:
            return
        mode = self.motion_mode_combo.currentData()
        if mode == state.motion_mode:
            return
        if state.rectangles or state.peak_result is not None:
            answer = QMessageBox.warning(
                self,
                "Change motion correction?",
                "Changing motion correction invalidates the recording peak, pool "
                "segmentation, backgrounds, and exports. Analysis rectangles remain "
                "in the first-frame coordinate system. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self._restore_motion_state()
                return
        state.motion_mode = mode
        state.motion_result = None
        state.motion_error = None
        state.motion_accepted = mode == "none"
        self._clear_motion_dependents(state)
        self._restore_analysis_state()
        self._request_frame(self.frame_slider.value())

    @Slot()
    def _start_current_motion(self) -> None:
        state = self.current_analysis_state
        if self._current_path is None or state is None:
            return
        if state.motion_mode not in {"translation", "rigid"}:
            QMessageBox.information(
                self,
                "Choose a correction mode",
                "Choose Translation or Rigid body before calculating correction.",
            )
            return
        self._start_motion_requests(((self._current_path, state.motion_mode),))

    @Slot()
    def _start_all_motion(self) -> None:
        state = self.current_analysis_state
        if state is None or state.motion_mode not in {"translation", "rigid"}:
            QMessageBox.information(
                self,
                "Choose a correction mode",
                "Choose Translation or Rigid body on the current recording. That "
                "mode will be applied to every loaded recording.",
            )
            return
        if any(
            item.rectangles or item.peak_result is not None
            for item in self._analysis_states.values()
        ):
            answer = QMessageBox.warning(
                self,
                "Recalculate all corrections?",
                "This will invalidate existing peaks, segmentations, and backgrounds "
                "for every loaded recording. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        mode = state.motion_mode
        requests: list[tuple[Path, str]] = []
        for path, analysis_state in self._analysis_states.items():
            analysis_state.motion_mode = mode
            analysis_state.motion_result = None
            analysis_state.motion_accepted = False
            analysis_state.motion_error = None
            self._clear_motion_dependents(analysis_state)
            requests.append((path, mode))
        self._start_motion_requests(tuple(requests))

    def _start_motion_requests(self, requests: tuple[tuple[Path, str], ...]) -> None:
        if not requests or self._motion_running:
            return
        for path, mode in requests:
            state = self._analysis_states[path]
            state.motion_mode = mode
            state.motion_result = None
            state.motion_accepted = False
            state.motion_error = None
            self._clear_motion_dependents(state)
        self._stop_playback()
        self._stop_all_drawing()
        self._motion_token += 1
        token = self._motion_token
        task = _MotionTask(token, requests)
        task.signals.succeeded.connect(self._motion_finished)
        task.signals.failed.connect(self._motion_failed)
        task.signals.progress.connect(self._motion_progress)
        self._motion_running = True
        self.recording_list.setEnabled(False)
        self.open_file_action.setEnabled(False)
        self.open_folder_action.setEnabled(False)
        self.analysis_controls.setEnabled(False)
        self.motion_summary_label.setText(
            f"Calculating correction for {len(requests)} recording(s) against frame 1…"
        )
        self.statusBar().showMessage(
            f"Calculating motion correction for {len(requests)} recording(s)"
        )
        self._show_motion_state()
        self._analysis_thread_pool.start(task)

    @Slot(int, int, int, int, str)
    def _motion_progress(
        self,
        recording_index: int,
        recording_count: int,
        frame_index: int,
        frame_count: int,
        filename: str,
    ) -> None:
        if not self._motion_running:
            return
        message = (
            f"Correcting recording {recording_index}/{recording_count}: {filename}; "
            f"frame {frame_index}/{frame_count}"
        )
        self.motion_summary_label.setText(message)
        self.statusBar().showMessage(message)

    @Slot(int, object)
    def _motion_finished(self, token: int, payload: object) -> None:
        if token != self._motion_token or not isinstance(payload, _MotionBatchResult):
            return
        self._motion_running = False
        self.recording_list.setEnabled(True)
        self.open_file_action.setEnabled(True)
        self.open_folder_action.setEnabled(True)
        self.analysis_controls.setEnabled(True)
        for path, result in payload.results.items():
            state = self._analysis_states.get(path)
            if state is not None:
                state.motion_result = result
                state.motion_accepted = False
                state.motion_error = None
        for path, message in payload.failures.items():
            state = self._analysis_states.get(path)
            if state is not None:
                state.motion_error = message
        self._restore_motion_state()
        completed, failed = len(payload.results), len(payload.failures)
        self.statusBar().showMessage(
            f"Calculated {completed} motion correction(s)"
            + (f"; {failed} failed" if failed else "")
        )
        current = self.current_analysis_state
        if current is not None and current.motion_result is not None:
            self._review_motion_correction()

    @Slot(int, str)
    def _motion_failed(self, token: int, message: str) -> None:
        if token != self._motion_token:
            return
        self._motion_running = False
        self.recording_list.setEnabled(True)
        self.open_file_action.setEnabled(True)
        self.open_folder_action.setEnabled(True)
        self.analysis_controls.setEnabled(True)
        state = self.current_analysis_state
        if state is not None:
            state.motion_error = message
        self._show_motion_state()
        self.statusBar().showMessage("Motion correction failed")
        self._show_error("Could not calculate motion correction", message)

    @Slot()
    def _review_motion_correction(self) -> None:
        state = self.current_analysis_state
        if (
            self._current_path is None
            or self._metadata is None
            or state is None
            or state.motion_result is None
        ):
            return
        previous_result = state.motion_result
        dialog = MotionCorrectionDialog(
            self._current_path, self._metadata, previous_result, self
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self._show_motion_state()
            return
        if state.peak_result is not None and dialog.result != previous_result:
            self._clear_motion_dependents(state)
        state.motion_result = dialog.result
        state.motion_accepted = True
        state.motion_error = None
        self._show_motion_state()
        self._show_peak_state()
        self._request_frame(self.frame_slider.value())
        self.statusBar().showMessage(
            f"Accepted {state.motion_result.mode} correction for {self._current_path.name}"
        )

    def _show_motion_state(self) -> None:
        if not hasattr(self, "motion_mode_combo"):
            return
        state = self.current_analysis_state
        available = self._metadata is not None and not self._motion_running
        self.motion_mode_combo.setEnabled(available)
        calculate = (
            available
            and state is not None
            and state.motion_mode in {"translation", "rigid"}
        )
        self.calculate_motion_button.setEnabled(calculate)
        self.calculate_all_motion_button.setEnabled(calculate)
        has_result = available and state is not None and state.motion_result is not None
        self.review_motion_button.setEnabled(has_result)
        self.save_corrected_button.setEnabled(
            bool(
                has_result
                and state.motion_accepted
                and not self._corrected_export_running
            )
        )
        if state is None or state.motion_mode is None:
            message = "Choose a correction option before drawing analysis regions."
        elif state.motion_mode == "none":
            message = "No correction selected and accepted. Analysis uses original ND2 frames."
        elif state.motion_error:
            message = f"Correction failed: {state.motion_error}"
        elif state.motion_result is None:
            message = f"{state.motion_mode.title()} correction has not been calculated."
        else:
            warnings = len(state.motion_result.warning_indices)
            acceptance = "accepted" if state.motion_accepted else "not yet accepted"
            message = (
                f"{state.motion_mode.title()} correction calculated; {warnings} warning "
                f"frame(s); {acceptance}. Review the QC before analysis."
            )
        self.motion_summary_label.setText(message)
        ready = state is not None and state.motion_ready and not self._motion_running
        self.draw_region_action.setEnabled(self._metadata is not None and ready)
        self._sync_region_edit_button()
        self._update_region_actions()

    def _clear_motion_dependents(self, state: RecordingAnalysisState) -> None:
        state.peak_result = None
        state.peak_error = None
        state.manual_peak_index = None
        state.segmentation_result = None
        state.corrected_labels = None
        state.clear_pool_review_state()
        state.correction_undo.clear()
        state.correction_redo.clear()
        state.selected_pool_ids.clear()
        state.area_result = None
        state.area_error = None
        state.background_result = None
        state.corrected_background_labels = None
        state.background_assignments.clear()
        state.background_undo.clear()
        state.background_redo.clear()
        state.selected_background_id = None
        state.trace_result = None
        state.excluded_pool_reasons.clear()

    @Slot()
    def _start_corrected_export(self) -> None:
        state = self.current_analysis_state
        if (
            self._current_path is None
            or self._metadata is None
            or state is None
            or state.effective_motion_correction is None
            or self._corrected_export_running
        ):
            return
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose folder for corrected recording",
            str(self._last_export_directory or self._current_path.parent),
        )
        if not selected:
            return
        output_directory = Path(selected)
        self._last_export_directory = output_directory
        stem = f"{self._current_path.stem}_registered"
        expected = (
            output_directory / f"{stem}.ome.tif",
            output_directory / f"{stem}_transforms.csv",
            output_directory / f"{stem}_metadata.json",
        )
        overwrite = any(path.exists() for path in expected)
        if overwrite:
            answer = QMessageBox.warning(
                self,
                "Replace corrected recording files?",
                "One or more corrected output files already exist. Replace them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        path = self._current_path
        metadata = self._metadata
        correction = state.effective_motion_correction
        self._corrected_export_token += 1
        token = self._corrected_export_token
        task = _FunctionTask(
            token,
            lambda: export_corrected_recording(
                path,
                output_directory,
                metadata,
                correction,
                overwrite=overwrite,
            ),
        )
        task.signals.succeeded.connect(self._corrected_export_finished)
        task.signals.failed.connect(self._corrected_export_failed)
        self._corrected_export_running = True
        self.save_corrected_button.setEnabled(False)
        self.save_corrected_button.setText("Saving corrected images…")
        self.statusBar().showMessage(f"Saving corrected recording: {path.name}")
        self._analysis_thread_pool.start(task)

    @Slot(int, object)
    def _corrected_export_finished(self, token: int, payload: object) -> None:
        if token != self._corrected_export_token or not isinstance(
            payload, CorrectedRecordingExportResult
        ):
            return
        self._corrected_export_running = False
        self.save_corrected_button.setText("Save corrected images…")
        self._show_motion_state()
        self.statusBar().showMessage(
            f"Saved corrected recording: {payload.ome_tiff_path.name}"
        )
        QMessageBox.information(
            self,
            "Corrected recording saved",
            f"Saved {payload.frame_count} corrected frames as:\n"
            f"{payload.ome_tiff_path}\n\nTransform and QC files were saved beside it.",
        )

    @Slot(int, str)
    def _corrected_export_failed(self, token: int, message: str) -> None:
        if token != self._corrected_export_token:
            return
        self._corrected_export_running = False
        self.save_corrected_button.setText("Save corrected images…")
        self._show_motion_state()
        self.statusBar().showMessage("Corrected recording export failed")
        self._show_error("Could not save corrected recording", message)

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
        if not state.motion_ready:
            self.statusBar().showMessage(
                "Choose and accept motion correction before adding analysis regions"
            )
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
        self._restore_motion_state()
        self.manual_peak_spin.setRange(1, self._metadata.frame_count)
        self._rebuild_region_items()
        self._show_peak_state()
        self._restore_segmentation_state()
        self._show_area_state()
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
        can_edit = (
            not self._peak_batch_running
            and not self._motion_running
            and state is not None
            and state.motion_ready
        )
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
            (
                path,
                tuple(state.rectangles),
                state.effective_motion_correction,
            )
            for path, state in self._analysis_states.items()
            if state.rectangles and state.motion_ready
        )
        if not requests or self._peak_batch_running:
            return
        self._stop_playback()
        self._stop_all_drawing()
        self._peak_token += 1
        token = self._peak_token
        self._segmentation_token += 1
        for path, _rectangles, _correction in requests:
            state = self._analysis_states[path]
            state.peak_result = None
            state.peak_error = None
            state.manual_peak_index = None
            state.segmentation_result = None
            state.corrected_labels = None
            state.clear_pool_review_state()
            state.correction_undo.clear()
            state.correction_redo.clear()
            state.selected_pool_ids.clear()
            state.area_result = None
            state.area_error = None
            state.background_result = None
            state.corrected_background_labels = None
            state.background_assignments.clear()
            state.background_undo.clear()
            state.background_redo.clear()
            state.selected_background_id = None
            state.trace_result = None
            state.excluded_pool_reasons.clear()

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
        region_count = sum(
            len(rectangles) for _path, rectangles, _correction in requests
        )
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
        state = self.current_analysis_state
        self.draw_region_action.setEnabled(
            self._metadata is not None and state is not None and state.motion_ready
        )
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
        state = self.current_analysis_state
        self.draw_region_action.setEnabled(
            self._metadata is not None and state is not None and state.motion_ready
        )
        self._update_peak_button_text()
        self._update_region_actions()
        self.peak_summary_label.setText("Peak calculation failed.")
        self.statusBar().showMessage("Peak calculation failed")
        self._show_error("Could not find recording peak", message)

    def _show_peak_state(self) -> None:
        state = self.current_analysis_state
        if state is not None and not state.motion_ready:
            self.peak_summary_label.setText(
                "Choose No correction or calculate and accept motion correction first."
            )
            self.override_peak_checkbox.setEnabled(False)
            self.manual_peak_spin.setEnabled(False)
            self.go_to_peak_button.setEnabled(False)
            self._update_region_actions()
            return
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
            (
                self.segmentation_prominence_spin,
                parameters.minimum_peak_prominence_sigma,
            ),
            (self.segmentation_min_area_spin, parameters.minimum_area_px),
            (self.segmentation_max_area_spin, parameters.maximum_area_px),
            (
                self.segmentation_min_circularity_spin,
                parameters.minimum_circularity or 0.0,
            ),
            (
                self.segmentation_min_solidity_spin,
                parameters.minimum_solidity or 0.0,
            ),
            (
                self.segmentation_min_contrast_spin,
                parameters.minimum_local_contrast or 0.0,
            ),
            (
                self.segmentation_max_variation_spin,
                parameters.maximum_intensity_variation or 0.0,
            ),
        )
        for control, value in controls:
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        baseline_index = self.segmentation_baseline_combo.findData(
            parameters.baseline_mode
        )
        self.segmentation_baseline_combo.blockSignals(True)
        self.segmentation_baseline_combo.setCurrentIndex(max(0, baseline_index))
        self.segmentation_baseline_combo.blockSignals(False)
        enabled_values = {
            "area": parameters.filter_area,
            "circularity": parameters.filter_circularity,
            "solidity": parameters.filter_solidity,
            "local_contrast": parameters.filter_local_contrast,
            "intensity_variation": parameters.filter_intensity_variation,
        }
        multiplier_values = {
            "area": parameters.area_outlier_multiplier,
            "circularity": parameters.circularity_outlier_multiplier,
            "solidity": parameters.solidity_outlier_multiplier,
            "local_contrast": parameters.local_contrast_outlier_multiplier,
            "intensity_variation": parameters.intensity_variation_outlier_multiplier,
        }
        for metric, control in self.segmentation_filter_checks.items():
            control.blockSignals(True)
            control.setChecked(enabled_values[metric])
            control.blockSignals(False)
        for metric, control in self.segmentation_multiplier_spins.items():
            control.blockSignals(True)
            control.setValue(multiplier_values[metric])
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
            minimum_peak_prominence_sigma=self.segmentation_prominence_spin.value(),
            minimum_area_px=minimum_area,
            maximum_area_px=self.segmentation_max_area_spin.value(),
            baseline_mode=str(self.segmentation_baseline_combo.currentData()),
            filter_area=self.segmentation_filter_checks["area"].isChecked(),
            filter_circularity=self.segmentation_filter_checks[
                "circularity"
            ].isChecked(),
            filter_solidity=self.segmentation_filter_checks["solidity"].isChecked(),
            filter_local_contrast=self.segmentation_filter_checks[
                "local_contrast"
            ].isChecked(),
            filter_intensity_variation=self.segmentation_filter_checks[
                "intensity_variation"
            ].isChecked(),
            area_outlier_multiplier=self.segmentation_multiplier_spins["area"].value(),
            circularity_outlier_multiplier=self.segmentation_multiplier_spins[
                "circularity"
            ].value(),
            solidity_outlier_multiplier=self.segmentation_multiplier_spins[
                "solidity"
            ].value(),
            local_contrast_outlier_multiplier=self.segmentation_multiplier_spins[
                "local_contrast"
            ].value(),
            intensity_variation_outlier_multiplier=self.segmentation_multiplier_spins[
                "intensity_variation"
            ].value(),
            minimum_circularity=self.segmentation_min_circularity_spin.value() or None,
            minimum_solidity=self.segmentation_min_solidity_spin.value() or None,
            minimum_local_contrast=self.segmentation_min_contrast_spin.value() or None,
            maximum_intensity_variation=self.segmentation_max_variation_spin.value()
            or None,
        )
        self._invalidate_segmentation()

    @Slot()
    def _reset_segmentation_filter_defaults(self) -> None:
        state = self.current_analysis_state
        if state is None:
            return
        state.segmentation_parameters = SegmentationParameters()
        self._restore_segmentation_state()
        self._invalidate_segmentation()

    def _invalidate_segmentation(self) -> None:
        self._segmentation_token += 1
        self._area_token += 1
        self._background_token += 1
        state = self.current_analysis_state
        if state is not None:
            state.segmentation_result = None
            state.corrected_labels = None
            state.clear_pool_review_state()
            state.correction_undo.clear()
            state.correction_redo.clear()
            state.selected_pool_ids.clear()
            state.area_result = None
            state.area_error = None
            state.background_result = None
            state.corrected_background_labels = None
            state.background_assignments.clear()
            state.background_undo.clear()
            state.background_redo.clear()
            state.selected_background_id = None
            state.trace_result = None
            state.excluded_pool_reasons.clear()
        if hasattr(self, "canvas"):
            self.canvas.clear_segmentation()
            self.canvas.clear_reviews()
            self.canvas.clear_pool_ids()
            self.canvas.clear_backgrounds()
        if hasattr(self, "segmentation_summary_label"):
            self._show_segmentation_state()
            self._show_area_state()

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
        self._invalidate_area_analysis(state)
        self._invalidate_backgrounds()
        self._segmentation_token += 1
        token = self._segmentation_token
        path = self._current_path
        peak_index = state.effective_peak_index
        rectangles = tuple(state.rectangles)
        parameters = state.segmentation_parameters
        correction = state.effective_motion_correction
        task = _FunctionTask(
            token,
            lambda: segment_recording(
                path, peak_index, rectangles, parameters, correction
            ),
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
        state.initialize_pool_review_state()
        state.correction_undo.clear()
        state.correction_redo.clear()
        state.selected_pool_ids.clear()
        state.area_result = None
        state.area_error = None
        state.background_result = None
        state.corrected_background_labels = None
        state.background_assignments.clear()
        state.background_undo.clear()
        state.background_redo.clear()
        state.selected_background_id = None
        state.trace_result = None
        state.excluded_pool_reasons.clear()
        self.run_segmentation_button.setText("Rerun automatic segmentation")
        self._show_segmentation_state()
        self._show_area_state()
        self._go_to_effective_peak()
        self.statusBar().showMessage(
            f"Automatic segmentation accepted {len(result.accepted_pool_ids)} of "
            f"{len(result.pools)} candidate pool(s)"
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
            for control in (
                self.show_segmentation_checkbox,
                self.show_pool_ids_checkbox,
                self.show_quality_review_checkbox,
                self.show_split_review_checkbox,
                self.top_show_segmentation_checkbox,
                self.top_show_pool_ids_checkbox,
                self.top_show_quality_review_checkbox,
                self.top_show_split_review_checkbox,
                self.review_quality_button,
                self.review_split_button,
            ):
                control.setEnabled(False)
            self.review_quality_button.setChecked(False)
            self.review_split_button.setChecked(False)
            self.canvas.clear_segmentation()
            self.canvas.clear_reviews()
            self.canvas.clear_pool_ids()
            self.quality_review_list.clear()
            self.split_review_list.clear()
            self.segmentation_summary_label.setText(
                "Calculate a recording peak before segmentation."
                if not can_segment
                else "Ready. Lower the detection threshold to find dimmer pools."
            )
            self._show_correction_state()
            self._show_background_state()
            return

        result = state.segmentation_result
        state.ensure_pool_review_state()
        labels = self._current_labels()
        if labels is None:
            return
        self.run_segmentation_button.setText("Rerun automatic segmentation")
        for control in (
            self.show_segmentation_checkbox,
            self.show_pool_ids_checkbox,
            self.top_show_segmentation_checkbox,
            self.top_show_pool_ids_checkbox,
        ):
            control.setEnabled(True)
        has_quality_review = bool(state.quality_review_pool_ids)
        for control in (
            self.show_quality_review_checkbox,
            self.top_show_quality_review_checkbox,
            self.review_quality_button,
        ):
            control.setEnabled(has_quality_review)
        if not has_quality_review:
            self.review_quality_button.setChecked(False)
        has_split_review = bool(state.split_review_group_ids)
        for control in (
            self.show_split_review_checkbox,
            self.top_show_split_review_checkbox,
            self.review_split_button,
        ):
            control.setEnabled(has_split_review)
        if not has_split_review:
            self.review_split_button.setChecked(False)
        self.canvas.set_segmentation(
            labels,
            tuple(sorted(state.selected_pool_ids)),
            self._missing_background_pool_ids(state, labels),
        )
        self.canvas.segmentation_item.setVisible(
            self.show_segmentation_checkbox.isChecked()
        )
        self._refresh_review_overlays()
        self._refresh_review_lists()
        self._refresh_pool_id_overlay()
        current_count = len(np.unique(labels)) - 1
        rejected_count = sum(
            1
            for pool_id in state.quality_review_pool_ids
            if not state.pool_validity.get(pool_id, False)
        )
        self.segmentation_summary_label.setText(
            f"Found {len(result.pools)} candidate pool(s); {current_count} currently "
            f"accepted and {rejected_count} currently invalid on frame "
            f"{result.peak_frame_index + 1}.\n"
            f"Threshold: {result.parameters.seed_threshold_sigma:g} sigma; "
            f"prominence: {result.parameters.minimum_peak_prominence_sigma:g} sigma; "
            f"baseline: {result.parameters.baseline_mode}."
        )
        self._show_correction_state()
        self._show_background_state()

    @Slot(bool)
    def _segmentation_visibility_changed(self, visible: bool) -> None:
        self._synchronize_overlay_checkboxes(
            visible,
            "show_segmentation_checkbox",
            "top_show_segmentation_checkbox",
        )
        self.canvas.segmentation_item.setVisible(visible)

    @Slot(bool)
    def _pool_ids_visibility_changed(self, visible: bool) -> None:
        self._synchronize_overlay_checkboxes(
            visible,
            "show_pool_ids_checkbox",
            "top_show_pool_ids_checkbox",
        )
        self._refresh_pool_id_overlay()

    @Slot(bool)
    def _quality_review_visibility_changed(self, visible: bool) -> None:
        self._synchronize_overlay_checkboxes(
            visible,
            "show_quality_review_checkbox",
            "top_show_quality_review_checkbox",
        )
        self.canvas.quality_review_item.setVisible(visible)
        self._refresh_pool_id_overlay()

    @Slot(bool)
    def _split_review_visibility_changed(self, visible: bool) -> None:
        self._synchronize_overlay_checkboxes(
            visible,
            "show_split_review_checkbox",
            "top_show_split_review_checkbox",
        )
        self.canvas.split_review_item.setVisible(visible)
        self._refresh_pool_id_overlay()

    def _synchronize_overlay_checkboxes(
        self, visible: bool, *attribute_names: str
    ) -> None:
        for attribute_name in attribute_names:
            control = getattr(self, attribute_name, None)
            if control is None or control.isChecked() == visible:
                continue
            control.blockSignals(True)
            control.setChecked(visible)
            control.blockSignals(False)

    def _refresh_pool_id_overlay(self) -> None:
        state = self.current_analysis_state
        labels = self._current_labels()
        if state is None or labels is None:
            self.canvas.clear_pool_ids()
            return
        display_labels = np.asarray(labels, dtype=np.int32).copy()
        result = state.segmentation_result
        if result is None:
            self.canvas.clear_pool_ids()
            return
        review_pool_ids: set[int] = set()
        if self.show_quality_review_checkbox.isChecked():
            review_pool_ids.update(state.quality_review_pool_ids)
        if self.show_split_review_checkbox.isChecked():
            for group in result.split_groups:
                if group.group_id in state.split_review_group_ids:
                    review_pool_ids.update(group.pool_ids)
        if review_pool_ids:
            candidates = result.candidate_labels
            review_mask = np.isin(candidates, tuple(review_pool_ids))
            display_labels[review_mask] = candidates[review_mask]
        self.canvas.set_pool_ids(display_labels)
        self.canvas.set_pool_ids_visible(self.show_pool_ids_checkbox.isChecked())

    def _refresh_review_overlays(self) -> None:
        state = self.current_analysis_state
        if state is None or state.segmentation_result is None:
            self.canvas.clear_reviews()
            return
        result = state.segmentation_result
        state.ensure_pool_review_state()
        self.canvas.set_quality_review(
            result.candidate_labels,
            state.pool_validity,
            result.quality_flags_by_pool,
            state.quality_review_pool_ids,
        )
        self.canvas.set_split_review(
            result.candidate_labels,
            result,
            state.merged_split_group_ids,
            state.split_review_group_ids,
        )
        self.canvas.quality_review_item.setVisible(
            self.show_quality_review_checkbox.isChecked()
        )
        self.canvas.split_review_item.setVisible(
            self.show_split_review_checkbox.isChecked()
        )

    def _refresh_review_lists(self) -> None:
        state = self.current_analysis_state
        self.quality_review_list.clear()
        self.split_review_list.clear()
        if state is None or state.segmentation_result is None:
            return
        result = state.segmentation_result
        quality_flags = result.quality_flags_by_pool
        for pool_id in sorted(state.quality_review_pool_ids):
            valid = state.pool_validity.get(pool_id, False)
            flags = quality_flags.get(pool_id, ())
            detail = "valid" if valid else "invalid"
            if flags:
                detail += f" ({', '.join(flags)})"
            item = QListWidgetItem(f"Pool {pool_id} — {detail}")
            item.setData(Qt.ItemDataRole.UserRole, pool_id)
            item.setToolTip(
                "; ".join(flags) if flags else "No automatic quality warning"
            )
            self.quality_review_list.addItem(item)
        for group in result.split_groups:
            if group.group_id not in state.split_review_group_ids:
                continue
            status = (
                "merged"
                if group.group_id in state.merged_split_group_ids
                else "split"
            )
            member_text = ", ".join(str(value) for value in group.pool_ids)
            item = QListWidgetItem(
                f"Group {group.group_id} — {status}; pools {member_text}"
            )
            item.setData(Qt.ItemDataRole.UserRole, group.group_id)
            item.setToolTip(
                f"Parent pool {group.parent_pool_id}; click to toggle split/merged"
            )
            self.split_review_list.addItem(item)

    def _set_review_mode(self, mode: str, reviewing: bool) -> None:
        quality_mode = mode == "quality"
        other_button = (
            self.review_split_button if quality_mode else self.review_quality_button
        )
        overlay_checkbox = (
            self.show_quality_review_checkbox
            if quality_mode
            else self.show_split_review_checkbox
        )
        inactive_overlay_checkbox = (
            self.show_split_review_checkbox
            if quality_mode
            else self.show_quality_review_checkbox
        )
        if reviewing:
            if other_button.isChecked():
                other_button.blockSignals(True)
                other_button.setChecked(False)
                other_button.blockSignals(False)
            if self._overlay_visibility_before_review is None:
                self._overlay_visibility_before_review = (
                    self.show_segmentation_checkbox.isChecked(),
                    self.show_quality_review_checkbox.isChecked(),
                    self.show_split_review_checkbox.isChecked(),
                )
            previous = self._overlay_visibility_before_review
            self.show_segmentation_checkbox.setChecked(False)
            inactive_overlay_checkbox.setChecked(
                previous[2] if quality_mode else previous[1]
            )
            overlay_checkbox.setChecked(True)
            message = (
                "Validity review active: click any candidate to toggle valid/invalid"
                if quality_mode
                else "Split review active: click any proposed group to toggle split/merged"
            )
            self.statusBar().showMessage(message)
            return
        if (
            not other_button.isChecked()
            and self._overlay_visibility_before_review is not None
        ):
            previous = self._overlay_visibility_before_review
            self._overlay_visibility_before_review = None
            self.show_segmentation_checkbox.setChecked(previous[0])
            self.show_quality_review_checkbox.setChecked(previous[1])
            self.show_split_review_checkbox.setChecked(previous[2])
        self.statusBar().showMessage("Review finished")

    @Slot(bool)
    def _quality_review_toggled(self, reviewing: bool) -> None:
        self._set_review_mode("quality", reviewing)

    @Slot(bool)
    def _split_review_toggled(self, reviewing: bool) -> None:
        self._set_review_mode("split", reviewing)

    @Slot(QListWidgetItem)
    def _quality_review_list_clicked(self, item: QListWidgetItem) -> None:
        if not self.review_quality_button.isChecked():
            self.review_quality_button.setChecked(True)
        self._toggle_pool_validity(int(item.data(Qt.ItemDataRole.UserRole)))

    @Slot(QListWidgetItem)
    def _split_review_list_clicked(self, item: QListWidgetItem) -> None:
        if not self.review_split_button.isChecked():
            self.review_split_button.setChecked(True)
        self._toggle_split_group(int(item.data(Qt.ItemDataRole.UserRole)))

    def _current_labels(self) -> np.ndarray[Any, Any] | None:
        state = self.current_analysis_state
        if state is None or state.segmentation_result is None:
            return None
        if state.corrected_labels is None:
            state.corrected_labels = state.segmentation_result.labels.copy()
        return state.corrected_labels

    @staticmethod
    def _labels_for_state(
        state: RecordingAnalysisState,
    ) -> np.ndarray[Any, Any] | None:
        if state.segmentation_result is None:
            return None
        return (
            state.corrected_labels
            if state.corrected_labels is not None
            else state.segmentation_result.labels
        )

    @staticmethod
    def _background_labels_for_state(
        state: RecordingAnalysisState,
    ) -> np.ndarray[Any, Any] | None:
        if state.background_result is None:
            return None
        return (
            state.corrected_background_labels
            if state.corrected_background_labels is not None
            else state.background_result.labels
        )

    @staticmethod
    def _invalidate_trace_cache(
        state: RecordingAnalysisState, *, pool_identities_changed: bool = False
    ) -> None:
        state.trace_result = None
        if pool_identities_changed:
            labels = MainWindow._labels_for_state(state)
            existing = (
                {int(value) for value in np.unique(labels) if value > 0}
                if labels is not None
                else set()
            )
            state.excluded_pool_reasons = {
                pool_id: reason
                for pool_id, reason in state.excluded_pool_reasons.items()
                if pool_id in existing
            }

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
        if state.segmentation_result is not None:
            state.ensure_pool_review_state()
            self._refresh_review_overlays()
        has_quality_review = bool(state.quality_review_pool_ids)
        self.show_quality_review_checkbox.setEnabled(has_quality_review)
        self.top_show_quality_review_checkbox.setEnabled(has_quality_review)
        self.review_quality_button.setEnabled(has_quality_review)
        if not has_quality_review:
            self.review_quality_button.setChecked(False)
        has_split_review = bool(state.split_review_group_ids)
        self.show_split_review_checkbox.setEnabled(has_split_review)
        self.top_show_split_review_checkbox.setEnabled(has_split_review)
        self.review_split_button.setEnabled(has_split_review)
        if not has_split_review:
            self.review_split_button.setChecked(False)
        self._refresh_review_lists()
        self._refresh_pool_id_overlay()

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

        if (
            self.review_quality_button.isChecked()
            and state.segmentation_result is not None
        ):
            pool_id = int(state.segmentation_result.candidate_labels[row, column])
            if pool_id in state.quality_review_pool_ids:
                self._toggle_pool_validity(pool_id)
            else:
                self.statusBar().showMessage(
                    "The clicked location is not an active validity candidate"
                )
            return

        if (
            self.review_split_button.isChecked()
            and state.segmentation_result is not None
        ):
            pool_id = int(state.segmentation_result.candidate_labels[row, column])
            group = next(
                (
                    item
                    for item in state.segmentation_result.split_groups
                    if pool_id in item.pool_ids
                    and item.group_id in state.split_review_group_ids
                ),
                None,
            )
            if group is not None:
                self._toggle_split_group(group.group_id)
            else:
                self.statusBar().showMessage(
                    "The clicked location is not an active automatic split"
                )
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

    def _toggle_pool_validity(self, pool_id: int) -> None:
        state = self.current_analysis_state
        labels = self._current_labels()
        if (
            state is None
            or labels is None
            or state.segmentation_result is None
            or pool_id not in state.quality_review_pool_ids
        ):
            return
        result = state.segmentation_result
        review_after = state.pool_review_snapshot()
        review_after.validity_by_pool[pool_id] = not review_after.validity_by_pool.get(
            pool_id, False
        )
        group = next(
            (group for group in result.split_groups if pool_id in group.pool_ids),
            None,
        )
        now_valid = review_after.validity_by_pool[pool_id]
        if group is not None:
            valid_ids = {
                member_id
                for member_id in group.pool_ids
                if review_after.validity_by_pool.get(member_id, False)
            }
            try:
                corrected = apply_split_group_decision(
                    np.asarray(labels, dtype=np.int32),
                    result.candidate_labels,
                    group.pool_ids,
                    group.parent_pool_id,
                    valid_ids,
                    merged=group.group_id
                    in review_after.merged_split_group_ids,
                )
            except LabelOverlapError as error:
                QMessageBox.warning(self, "Validity not changed", str(error))
                return
            group_mask = np.isin(result.candidate_labels, group.pool_ids)
            state.selected_pool_ids = {
                int(value)
                for value in np.unique(corrected[group_mask])
                if value > 0
            }
        else:
            corrected = np.asarray(labels, dtype=np.int32).copy()
            candidate_mask = result.candidate_labels == pool_id
            if now_valid:
                conflicts = (corrected[candidate_mask] > 0) & (
                    corrected[candidate_mask] != pool_id
                )
                if np.any(conflicts):
                    QMessageBox.warning(
                        self,
                        "Candidate not made valid",
                        "A manual pool now occupies part of this candidate. Undo "
                        "that edit or rerun segmentation first.",
                    )
                    return
                corrected[candidate_mask] = pool_id
                state.selected_pool_ids = {pool_id}
            else:
                corrected[candidate_mask & (corrected == pool_id)] = 0
                state.selected_pool_ids.discard(pool_id)
        description = (
            f"Mark pool {pool_id} valid"
            if now_valid
            else f"Mark pool {pool_id} invalid"
        )
        self._record_correction(
            corrected,
            description,
            review_after=review_after,
        )

    def _toggle_split_group(self, group_id: int) -> None:
        state = self.current_analysis_state
        labels = self._current_labels()
        if (
            state is None
            or labels is None
            or state.segmentation_result is None
            or group_id not in state.split_review_group_ids
        ):
            return
        result = state.segmentation_result
        group = next(
            (item for item in result.split_groups if item.group_id == group_id),
            None,
        )
        if group is None:
            return
        review_after = state.pool_review_snapshot()
        merging = group_id not in review_after.merged_split_group_ids
        if merging:
            review_after.merged_split_group_ids.add(group_id)
        else:
            review_after.merged_split_group_ids.discard(group_id)
        valid_ids = {
            pool_id
            for pool_id in group.pool_ids
            if review_after.validity_by_pool.get(pool_id, False)
        }
        try:
            corrected = apply_split_group_decision(
                np.asarray(labels, dtype=np.int32),
                result.candidate_labels,
                group.pool_ids,
                group.parent_pool_id,
                valid_ids,
                merged=merging,
            )
        except LabelOverlapError as error:
            QMessageBox.warning(self, "Split decision not changed", str(error))
            return
        group_mask = np.isin(result.candidate_labels, group.pool_ids)
        state.selected_pool_ids = {
            int(value)
            for value in np.unique(corrected[group_mask])
            if value > 0
        }
        description = (
            f"Merge automatic split group {group_id}"
            if merging
            else f"Restore automatic split group {group_id}"
        )
        self._record_correction(
            corrected,
            description,
            review_after=review_after,
        )

    def _resolve_flagged_pool(self, pool_id: int) -> None:
        """Backward-compatible entry point for toggling candidate validity."""
        self._toggle_pool_validity(pool_id)

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
                corrected = split_pool_with_line(
                    labels,
                    selected[0],
                    geometry,
                    next_pool_id=self._next_available_pool_id(state, labels),
                )
                self._record_correction(corrected, f"Separate pool {selected[0]}")
            elif mode == "hint":
                rectangle = self._rectangle_for_hint(geometry)
                if self._current_path is None or state.effective_peak_index is None:
                    return
                frame = read_frame(
                    self._current_path,
                    state.effective_peak_index,
                    state.effective_motion_correction,
                )
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
        new_pool_id = (
            None if replace_ids else self._next_available_pool_id(state, labels)
        )
        try:
            corrected = apply_shape(
                labels,
                mask,
                replace_ids=replace_ids,
                new_pool_id=new_pool_id,
            )
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
                new_pool_id=new_pool_id,
            )
        self._record_correction(corrected, description)

    @staticmethod
    def _next_available_pool_id(
        state: RecordingAnalysisState,
        labels: np.ndarray[Any, Any],
    ) -> int:
        highest_id = int(np.max(labels)) if labels.size else 0
        if state.segmentation_result is not None:
            highest_id = max(
                highest_id,
                int(np.max(state.segmentation_result.candidate_labels)),
            )
        return highest_id + 1

    def _record_correction(
        self,
        corrected: np.ndarray[Any, Any],
        description: str,
        *,
        review_after: _PoolReviewSnapshot | None = None,
    ) -> None:
        state = self.current_analysis_state
        labels = self._current_labels()
        if state is None or labels is None:
            return
        corrected = np.asarray(corrected, dtype=np.int32)
        mask_edit = make_edit(labels, corrected, description)
        review_before = state.pool_review_snapshot()
        if review_after is None:
            review_after = review_before.copy()
            changed_ids: set[int] = set()
            if len(mask_edit.flat_indices):
                before_flat = np.asarray(labels).ravel()[mask_edit.flat_indices]
                after_flat = corrected.ravel()[mask_edit.flat_indices]
                changed_ids = {
                    int(value)
                    for value in np.unique(np.concatenate((before_flat, after_flat)))
                    if value > 0
                }
                if state.segmentation_result is not None:
                    candidate_flat = state.segmentation_result.candidate_labels.ravel()[
                        mask_edit.flat_indices
                    ]
                    changed_ids.update(
                        int(value) for value in np.unique(candidate_flat) if value > 0
                    )
            related_ids = set(changed_ids)
            if state.segmentation_result is not None:
                for group in state.segmentation_result.split_groups:
                    if related_ids.intersection(group.pool_ids):
                        related_ids.update(group.pool_ids)
                        review_after.split_review_group_ids.discard(group.group_id)
                        review_after.merged_split_group_ids.discard(group.group_id)
            review_after.quality_review_pool_ids.difference_update(related_ids)
        if not len(mask_edit.flat_indices) and review_after == review_before:
            self.statusBar().showMessage("Correction made no changes")
            return
        edit = _PoolEdit(mask_edit, review_before, review_after.copy())
        state.corrected_labels = corrected
        state.restore_pool_review_snapshot(review_after)
        state.correction_undo.append(edit)
        state.correction_redo.clear()
        existing = {int(value) for value in np.unique(corrected) if value > 0}
        state.selected_pool_ids.intersection_update(existing)
        self._invalidate_trace_cache(state, pool_identities_changed=True)
        self._invalidate_area_analysis(state)
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
        state.restore_pool_review_snapshot(edit.apply(labels, undo=True))
        state.correction_redo.append(edit)
        state.selected_pool_ids.clear()
        self._invalidate_trace_cache(state, pool_identities_changed=True)
        self._invalidate_area_analysis(state)
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
        state.restore_pool_review_snapshot(edit.apply(labels))
        state.correction_undo.append(edit)
        state.selected_pool_ids.clear()
        self._invalidate_trace_cache(state, pool_identities_changed=True)
        self._invalidate_area_analysis(state)
        self._invalidate_backgrounds()
        self._show_segmentation_state()
        self.statusBar().showMessage(f"Redid: {edit.description}")

    @Slot()
    def _reset_corrections(self) -> None:
        state = self.current_analysis_state
        if state is None or state.segmentation_result is None:
            return
        state.selected_pool_ids.clear()
        result = state.segmentation_result
        review_after = _PoolReviewSnapshot(
            validity_by_pool={pool.pool_id: pool.accepted for pool in result.pools},
            merged_split_group_ids=set(),
            quality_review_pool_ids={pool.pool_id for pool in result.pools},
            split_review_group_ids={
                group.group_id for group in result.split_groups
            },
        )
        self._record_correction(
            result.labels.copy(),
            "Reset to automatic segmentation",
            review_after=review_after,
        )

    def _invalidate_area_analysis(
        self, state: RecordingAnalysisState | None = None
    ) -> None:
        self._area_token += 1
        target = state or self.current_analysis_state
        if target is not None:
            target.area_result = None
            target.area_error = None
        if hasattr(self, "area_summary_label"):
            self._show_area_state()

    def _show_area_state(self) -> None:
        if not hasattr(self, "calculate_areas_button"):
            return
        state = self.current_analysis_state
        labels = self._current_labels()
        can_calculate = (
            state is not None
            and labels is not None
            and np.any(labels)
            and state.effective_peak_index is not None
            and not self._area_running
        )
        self.calculate_areas_button.setEnabled(can_calculate)
        has_result = state is not None and state.area_result is not None
        self.review_areas_button.setEnabled(has_result and not self._area_running)
        if self._area_running:
            return
        if state is None or labels is None or not np.any(labels):
            message = "Finish pool segmentation before calculating areas."
        elif state.area_error:
            message = f"Area calculation failed: {state.area_error}"
        elif state.area_result is None:
            message = (
                "Ready to calculate peak-mask areas and independently detected "
                "pixels for every frame."
            )
        else:
            result = state.area_result
            message = (
                f"Calculated {len(result.frame_numbers)} frame(s) for "
                f"{len(result.pool_ids)} pool(s); {result.warning_count} pool-frame "
                "QC warning(s). Mask review is optional."
            )
        self.area_summary_label.setText(message)
        self._show_export_state()

    @Slot()
    def _start_area_analysis(self) -> None:
        state = self.current_analysis_state
        labels = self._current_labels()
        if (
            self._current_path is None
            or state is None
            or labels is None
            or state.effective_peak_index is None
            or self._area_running
        ):
            return
        self._stop_playback()
        self._stop_all_drawing()
        self._area_token += 1
        token = self._area_token
        path = self._current_path
        state.area_result = None
        state.area_error = None
        task = _AreaTask(
            token,
            path,
            np.asarray(labels, dtype=np.int32).copy(),
            tuple(state.rectangles),
            state.effective_peak_index,
            state.segmentation_parameters,
            state.effective_motion_correction,
        )
        task.signals.succeeded.connect(self._area_analysis_finished)
        task.signals.failed.connect(self._area_analysis_failed)
        task.signals.progress.connect(self._area_analysis_progress)
        self._area_running = True
        self._area_path = path
        self.recording_list.setEnabled(False)
        self.open_file_action.setEnabled(False)
        self.open_folder_action.setEnabled(False)
        self.analysis_controls.setEnabled(False)
        self.calculate_areas_button.setEnabled(False)
        self.review_areas_button.setEnabled(False)
        self.area_summary_label.setText("Calculating pool areas…")
        self.statusBar().showMessage(f"Calculating pool areas for {path.name}")
        self._analysis_thread_pool.start(task)

    @Slot(int, int, int)
    def _area_analysis_progress(self, token: int, frame: int, total: int) -> None:
        if token != self._area_token or not self._area_running:
            return
        message = f"Calculating pool areas: frame {frame}/{total}"
        self.area_summary_label.setText(message)
        self.statusBar().showMessage(message)

    @Slot(int, object)
    def _area_analysis_finished(self, token: int, payload: object) -> None:
        if token != self._area_token or not isinstance(payload, AreaAnalysisResult):
            return
        path = self._area_path
        self._area_running = False
        self._area_path = None
        self.recording_list.setEnabled(True)
        self.open_file_action.setEnabled(True)
        self.open_folder_action.setEnabled(True)
        self.analysis_controls.setEnabled(True)
        state = self._analysis_states.get(path) if path is not None else None
        if state is not None:
            state.area_result = payload
            state.area_error = None
        self._show_area_state()
        self.statusBar().showMessage(
            f"Pool areas calculated with {payload.warning_count} QC warning(s)"
        )

    @Slot(int, str)
    def _area_analysis_failed(self, token: int, message: str) -> None:
        if token != self._area_token:
            return
        path = self._area_path
        self._area_running = False
        self._area_path = None
        self.recording_list.setEnabled(True)
        self.open_file_action.setEnabled(True)
        self.open_folder_action.setEnabled(True)
        self.analysis_controls.setEnabled(True)
        state = self._analysis_states.get(path) if path is not None else None
        if state is not None:
            state.area_error = message
        self._show_area_state()
        self.statusBar().showMessage("Pool area calculation failed")
        self._show_error("Could not calculate pool areas", message)

    @Slot()
    def _review_area_masks(self) -> None:
        state = self.current_analysis_state
        labels = self._current_labels()
        if (
            self._current_path is None
            or self._metadata is None
            or state is None
            or labels is None
            or state.area_result is None
        ):
            return
        dialog = AreaMaskReviewDialog(
            self._current_path,
            self._metadata,
            state.effective_motion_correction,
            labels,
            tuple(state.rectangles),
            state.area_result,
            self,
        )
        dialog.exec()

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
            self._invalidate_trace_cache(state)
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
        correction = state.effective_motion_correction
        pools_snapshot = np.asarray(pool_labels, dtype=np.int32).copy()
        task = _FunctionTask(
            token,
            lambda: find_recording_backgrounds(
                path,
                peak_index,
                pools_snapshot,
                rectangles,
                parameters,
                correction,
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
        self._invalidate_trace_cache(state)
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
        self._invalidate_trace_cache(state)
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
        self._invalidate_trace_cache(state)
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
        self._invalidate_trace_cache(state)
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

    def _recording_paths(self) -> tuple[Path, ...]:
        return tuple(
            Path(self.recording_list.item(row).data(Qt.ItemDataRole.UserRole)).resolve()
            for row in range(self.recording_list.count())
        )

    def _trace_readiness_for(self, path: Path) -> tuple[bool, str]:
        state = self._analysis_states.get(path)
        if state is None:
            return False, "Recording analysis state is unavailable."
        if not state.motion_ready:
            return False, "Motion correction has not been chosen and accepted."
        if state.peak_result is None or state.effective_peak_index is None:
            return False, "The recording peak has not been calculated."
        pool_labels = self._labels_for_state(state)
        if pool_labels is None or not np.any(pool_labels):
            return False, "Pool segmentation is incomplete."
        background_labels = self._background_labels_for_state(state)
        if background_labels is None or not np.any(background_labels):
            return False, "Required background regions are incomplete."
        missing = self._missing_background_pool_ids(state, pool_labels)
        if missing:
            return False, f"{len(missing)} pool-wise background(s) are missing."
        return True, "Ready for trace review."

    def _export_readiness_for(self, path: Path) -> tuple[bool, str]:
        state = self._analysis_states.get(path)
        trace_ready, message = self._trace_readiness_for(path)
        if not trace_ready or state is None:
            return False, message
        pool_labels = self._labels_for_state(state)
        pool_ids = (
            {int(value) for value in np.unique(pool_labels) if value > 0}
            if pool_labels is not None
            else set()
        )
        if pool_ids and pool_ids.issubset(state.excluded_pool_reasons):
            return False, "Every accepted pool is excluded from export."
        if state.area_result is None:
            return False, "Pool areas have not been calculated."
        return True, "Ready to export."

    def _export_readiness(self) -> tuple[bool, str]:
        metadata = self.current_metadata
        if metadata is None or self._current_path is None:
            return False, "Open a recording before exporting."
        ready, message = self._export_readiness_for(self._current_path)
        if not ready:
            return False, message
        return True, (
            "Ready to review/export raw and normalized traces, ImageJ ROIs, areas, "
            "and the display LUT using ND2 timestamps."
            if metadata.timestamps_s is not None
            else "Ready to export; the app will ask for the frame interval."
        )

    @Slot(bool)
    def _open_trace_review(self, all_recordings: bool = False) -> None:
        if self._export_running:
            return
        paths = (
            self._recording_paths()
            if all_recordings
            else ((self._current_path,) if self._current_path is not None else ())
        )
        items: list[_TraceReviewItem] = []
        unreadable: list[str] = []
        for path in paths:
            if path is None:
                continue
            try:
                metadata = (
                    self._metadata
                    if path == self._current_path and self._metadata is not None
                    else read_metadata(path)
                )
            except Exception as error:  # noqa: BLE001 - report all unreadable files
                unreadable.append(f"{path.name}: {type(error).__name__}: {error}")
                continue
            state = self._analysis_states[path]
            trace_ready, trace_message = self._trace_readiness_for(path)
            export_ready, export_message = self._export_readiness_for(path)
            items.append(
                _TraceReviewItem(
                    path=path,
                    metadata=metadata,
                    state=state,
                    pool_labels=(
                        None
                        if self._labels_for_state(state) is None
                        else np.asarray(
                            self._labels_for_state(state), dtype=np.int32
                        ).copy()
                    ),
                    background_labels=(
                        None
                        if self._background_labels_for_state(state) is None
                        else np.asarray(
                            self._background_labels_for_state(state), dtype=np.int32
                        ).copy()
                    ),
                    trace_readiness_error=None if trace_ready else trace_message,
                    export_readiness_error=None if export_ready else export_message,
                )
            )
        if unreadable:
            QMessageBox.warning(
                self,
                "Some recordings could not be read",
                "These recordings were ignored:\n" + "\n".join(unreadable),
            )
        if not items:
            QMessageBox.warning(
                self, "Nothing to review", "No recording could be loaded."
            )
            return
        dialog = TraceReviewDialog(
            tuple(items), allow_batch_export=all_recordings, parent=self
        )
        if (
            dialog.exec() == QDialog.DialogCode.Accepted
            and dialog.requested_export_paths
        ):
            if len(dialog.requested_export_paths) == 1:
                self._start_analysis_export_for_path(dialog.requested_export_paths[0])
            else:
                self._start_batch_analysis_export(dialog.requested_export_paths)
        self._show_export_state()

    def _show_export_state(self) -> None:
        if not hasattr(self, "export_analysis_button"):
            return
        ready, message = self._export_readiness()
        self.export_analysis_button.setEnabled(ready and not self._export_running)
        self.review_current_traces_button.setEnabled(
            self._current_path is not None and not self._export_running
        )
        self.review_all_traces_button.setEnabled(
            self.recording_list.count() > 0 and not self._export_running
        )
        if not self._export_running:
            self.export_analysis_button.setText("Export current recording…")
            self.export_summary_label.setText(message)

    @Slot()
    def _start_analysis_export(self) -> None:
        if self._current_path is not None:
            self._start_analysis_export_for_path(self._current_path)

    def _start_analysis_export_for_path(self, path: Path) -> None:
        ready, message = self._export_readiness_for(path)
        if not ready:
            QMessageBox.warning(self, "Recording is not ready to export", message)
            return
        state = self._analysis_states.get(path)
        try:
            metadata = (
                self.current_metadata
                if path == self._current_path and self.current_metadata is not None
                else read_metadata(path)
            )
        except Exception as error:  # noqa: BLE001
            self._show_error("Could not read recording metadata", str(error))
            return
        pool_labels = None if state is None else self._labels_for_state(state)
        background_labels = (
            None if state is None else self._background_labels_for_state(state)
        )
        if (
            state is None
            or metadata is None
            or state.peak_result is None
            or state.segmentation_result is None
            or state.area_result is None
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

        initial_directory = self._last_export_directory or path.parent
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
            output_directory / f"{path.stem}_analysis.xlsx",
            output_directory / f"{path.stem}_rois.zip",
            output_directory / f"{path.stem}_display.lut",
            output_directory / f"{path.stem}_manifest.json",
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
            source_path=path,
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
            area_result=state.area_result,
            display_lut=self.canvas.display_lut(),
            display_levels=(
                self.canvas.display_levels
                if self.canvas.display_levels is not None
                else (0.0, 1.0)
            ),
            included_pool_ids=tuple(
                int(value)
                for value in np.unique(pool_labels)
                if value > 0 and int(value) not in state.excluded_pool_reasons
            ),
            exclusion_reasons=dict(state.excluded_pool_reasons),
            motion_correction=state.effective_motion_correction,
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
        self._batch_skipped_count = 0
        self.export_analysis_button.setText("Extracting traces and exporting…")
        self.export_summary_label.setText(
            "Streaming raw frames and writing the XLSX, ROI ZIP, and display LUT…"
        )
        self.statusBar().showMessage(f"Exporting {path.name}")
        self._analysis_thread_pool.start(task)

    def _start_batch_analysis_export(self, paths: tuple[Path, ...]) -> None:
        ready_paths: list[Path] = []
        skipped: list[str] = []
        for path in paths:
            ready, reason = self._export_readiness_for(path)
            if ready:
                ready_paths.append(path)
            else:
                skipped.append(f"{path.name}: {reason}")
        if not ready_paths:
            QMessageBox.warning(
                self,
                "No recordings are ready",
                "Every selected recording was incomplete and has been ignored.",
            )
            return

        metadata_by_path: dict[Path, RecordingMetadata] = {}
        intervals: dict[Path, float | None] = {}
        for path in ready_paths:
            try:
                metadata = (
                    self.current_metadata
                    if path == self._current_path and self.current_metadata is not None
                    else read_metadata(path)
                )
            except Exception as error:  # noqa: BLE001
                skipped.append(f"{path.name}: metadata could not be read ({error})")
                continue
            metadata_by_path[path] = metadata
            intervals[path] = None
            if metadata.timestamps_s is None:
                interval, provided = QInputDialog.getDouble(
                    self,
                    f"Frame interval needed — {path.name}",
                    "Seconds per frame:",
                    1.0,
                    0.000001,
                    1000000.0,
                    6,
                )
                if provided:
                    intervals[path] = interval
                else:
                    use_frames = QMessageBox.question(
                        self,
                        "Use frame numbers?",
                        f"No interval was provided for {path.name}. Continue with "
                        "one-based frame numbers and no seconds column?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes,
                    )
                    if use_frames != QMessageBox.StandardButton.Yes:
                        skipped.append(f"{path.name}: no time information provided")
                        metadata_by_path.pop(path, None)
        ready_paths = [path for path in ready_paths if path in metadata_by_path]
        if not ready_paths:
            QMessageBox.warning(
                self,
                "No recordings are ready",
                "The selected recordings were ignored because required metadata "
                "or time information was unavailable.",
            )
            return

        initial_directory = self._last_export_directory or ready_paths[0].parent
        selected_directory = QFileDialog.getExistingDirectory(
            self, "Choose batch export folder", str(initial_directory)
        )
        if not selected_directory:
            return
        output_directory = Path(selected_directory).resolve()
        self._last_export_directory = output_directory
        conflicts = tuple(
            output_directory / f"{path.stem}{suffix}"
            for path in ready_paths
            for suffix in (
                "_analysis.xlsx",
                "_rois.zip",
                "_display.lut",
                "_manifest.json",
            )
            if (output_directory / f"{path.stem}{suffix}").exists()
        )
        overwrite = False
        if conflicts:
            visible = "\n".join(path.name for path in conflicts[:12])
            if len(conflicts) > 12:
                visible += f"\n…and {len(conflicts) - 12} more"
            answer = QMessageBox.question(
                self,
                "Replace existing export files?",
                f"These files already exist:\n{visible}\n\nReplace them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            overwrite = True

        display_lut = self.canvas.display_lut()
        display_levels = self.canvas.display_levels or (0.0, 1.0)
        requests: list[AnalysisExportRequest] = []
        for path in ready_paths:
            state = self._analysis_states[path]
            pool_labels = self._labels_for_state(state)
            background_labels = self._background_labels_for_state(state)
            if (
                pool_labels is None
                or background_labels is None
                or state.peak_result is None
                or state.segmentation_result is None
                or state.background_result is None
                or state.area_result is None
                or state.effective_peak_index is None
            ):
                skipped.append(f"{path.name}: analysis became incomplete")
                continue
            requests.append(
                AnalysisExportRequest(
                    source_path=path,
                    output_directory=output_directory,
                    metadata=metadata_by_path[path],
                    rectangles=tuple(state.rectangles),
                    peak_result=state.peak_result,
                    effective_peak_index=state.effective_peak_index,
                    segmentation_parameters=state.segmentation_parameters,
                    background_parameters=state.background_parameters,
                    pool_labels=np.asarray(pool_labels, dtype=np.int32).copy(),
                    automatic_pool_labels=state.segmentation_result.labels.copy(),
                    background_labels=np.asarray(
                        background_labels, dtype=np.int32
                    ).copy(),
                    automatic_background_labels=state.background_result.labels.copy(),
                    background_assignments=dict(state.background_assignments),
                    area_result=state.area_result,
                    display_lut=display_lut.copy(),
                    display_levels=display_levels,
                    included_pool_ids=tuple(
                        int(value)
                        for value in np.unique(pool_labels)
                        if value > 0 and int(value) not in state.excluded_pool_reasons
                    ),
                    exclusion_reasons=dict(state.excluded_pool_reasons),
                    motion_correction=state.effective_motion_correction,
                    uniform_interval_s=intervals[path],
                    overwrite=overwrite,
                )
            )
        if not requests:
            QMessageBox.warning(
                self, "Nothing to export", "No ready recording remains."
            )
            return

        self._stop_all_drawing()
        self._export_token += 1
        token = self._export_token
        task = _FunctionTask(token, lambda: _export_analysis_batch(tuple(requests)))
        task.signals.succeeded.connect(self._analysis_export_finished)
        task.signals.failed.connect(self._analysis_export_failed)
        self._set_export_running(True)
        self._batch_skipped_count = len(skipped)
        self.export_analysis_button.setText("Batch export running…")
        ignored_text = (
            f"; {len(skipped)} incomplete recording(s) ignored" if skipped else ""
        )
        self.export_summary_label.setText(
            f"Exporting {len(requests)} ready recording(s){ignored_text}…"
        )
        self.statusBar().showMessage(
            f"Batch export: {len(requests)} ready, {len(skipped)} ignored"
        )
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
        if token != self._export_token:
            return
        if isinstance(result, _AnalysisBatchExportResult):
            self._set_export_running(False)
            self._show_background_state()
            success_count = len(result.results)
            failure_count = len(result.failures)
            self.export_summary_label.setText(
                f"Batch export finished: {success_count} recording(s) exported; "
                f"{self._batch_skipped_count} incomplete recording(s) ignored; "
                f"{failure_count} failed."
            )
            self.statusBar().showMessage(
                f"Batch export: {success_count} exported, "
                f"{self._batch_skipped_count} ignored, {failure_count} failed"
            )
            details = "\n".join(
                f"{path.name}: {message}" for path, message in result.failures.items()
            )
            QMessageBox.information(
                self,
                "Batch export complete",
                f"Exported {success_count} recording(s); ignored "
                f"{self._batch_skipped_count} incomplete recording(s)."
                + (f"\n\nFailures:\n{details}" if details else ""),
            )
            return
        if not isinstance(result, AnalysisExportResult):
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
            f"Saved:\n{result.workbook_path}\n{result.roi_zip_path}\n"
            f"{result.lut_path}\n{result.manifest_path}",
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
        self.motion_mode_combo.setEnabled(False)
        self.calculate_motion_button.setEnabled(False)
        self.calculate_all_motion_button.setEnabled(False)
        self.review_motion_button.setEnabled(False)
        self.save_corrected_button.setEnabled(False)
        self.save_corrected_button.setText("Save corrected images…")
        self.motion_summary_label.setText("Loading recording…")
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
        for control in (
            self.show_segmentation_checkbox,
            self.show_pool_ids_checkbox,
            self.show_quality_review_checkbox,
            self.show_split_review_checkbox,
            self.top_show_segmentation_checkbox,
            self.top_show_pool_ids_checkbox,
            self.top_show_quality_review_checkbox,
            self.top_show_split_review_checkbox,
            self.review_quality_button,
            self.review_split_button,
        ):
            control.setEnabled(False)
        self.review_quality_button.setChecked(False)
        self.review_split_button.setChecked(False)
        self._overlay_visibility_before_review = None
        self.quality_review_list.clear()
        self.split_review_list.clear()
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
        self.calculate_areas_button.setEnabled(False)
        self.calculate_areas_button.setText("Calculate areas")
        self.review_areas_button.setEnabled(False)
        self.area_summary_label.setText(
            "Finish pool segmentation before calculating areas."
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
        self.review_current_traces_button.setEnabled(False)
        self.review_all_traces_button.setEnabled(self.recording_list.count() > 0)
        self.export_summary_label.setText(
            "Complete pool segmentation and background selection before exporting."
        )
        self.canvas.clear_segmentation()
        self.canvas.clear_reviews()
        self.canvas.clear_pool_ids()
        self.canvas.clear_backgrounds()

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def closeEvent(self, event: QCloseEvent) -> None:
        if (
            self._export_running
            or self._corrected_export_running
            or self._motion_running
            or self._area_running
        ):
            QMessageBox.information(
                self,
                "Analysis in progress",
                "Please wait for the current motion, area calculation, or export to "
                "finish before closing the application.",
            )
            event.ignore()
            return
        self._stop_playback()
        self._stop_all_drawing()
        self._metadata_token += 1
        self._frame_token += 1
        self._motion_token += 1
        self._corrected_export_token += 1
        self._peak_token += 1
        self._segmentation_token += 1
        self._area_token += 1
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
