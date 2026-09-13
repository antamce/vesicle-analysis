from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import nd2
import numpy as np
from numpy.typing import NDArray
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font
from roifile import ImagejRoi, roiread, roiwrite
from scipy import ndimage
from skimage.measure import find_contours

from vesicle_analysis import __version__
from vesicle_analysis.analysis import PeakDetectionResult, RectangleBounds
from vesicle_analysis.background import BackgroundParameters
from vesicle_analysis.nd2_io import RecordingMetadata
from vesicle_analysis.segmentation import SegmentationParameters


@dataclass(frozen=True, slots=True)
class TraceExtractionResult:
    frame_numbers: NDArray[np.int32]
    time_s: NDArray[np.float64] | None
    time_source: str
    pool_ids: tuple[int, ...]
    background_ids: tuple[int, ...]
    pool_traces: NDArray[np.float64]
    background_traces: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AnalysisExportRequest:
    source_path: Path
    output_directory: Path
    metadata: RecordingMetadata
    rectangles: tuple[RectangleBounds, ...]
    peak_result: PeakDetectionResult
    effective_peak_index: int
    segmentation_parameters: SegmentationParameters
    background_parameters: BackgroundParameters
    pool_labels: NDArray[np.int32]
    automatic_pool_labels: NDArray[np.int32]
    background_labels: NDArray[np.int32]
    automatic_background_labels: NDArray[np.int32]
    background_assignments: dict[int, int | None]
    display_lut: NDArray[np.uint8]
    display_levels: tuple[float, float]
    uniform_interval_s: float | None = None
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class AnalysisExportResult:
    workbook_path: Path
    roi_zip_path: Path
    lut_path: Path
    frame_count: int
    pool_count: int
    background_count: int
    time_source: str


def extract_recording_traces(
    path: Path,
    pool_labels: NDArray[np.int32],
    background_labels: NDArray[np.int32],
    *,
    timestamps_s: tuple[float, ...] | None = None,
    uniform_interval_s: float | None = None,
) -> TraceExtractionResult:
    """Stream one ND2 and calculate raw means for every fixed labelled mask."""
    path = Path(path).expanduser().resolve()
    pool_labels = np.asarray(pool_labels, dtype=np.int32)
    background_labels = np.asarray(background_labels, dtype=np.int32)
    if pool_labels.ndim != 2 or background_labels.shape != pool_labels.shape:
        raise ValueError("Pool and background labels must be matching 2D arrays.")
    if uniform_interval_s is not None and (
        not np.isfinite(uniform_interval_s) or uniform_interval_s <= 0
    ):
        raise ValueError("The uniform frame interval must be a positive number.")

    pool_ids, pool_counts = _label_ids_and_counts(pool_labels)
    background_ids, background_counts = _label_ids_and_counts(background_labels)
    if not pool_ids:
        raise ValueError("At least one accepted vesicle pool is required.")
    if not background_ids:
        raise ValueError("At least one accepted background is required.")

    with nd2.ND2File(path) as recording:
        sizes = dict(recording.sizes)
        frame_count = int(sizes.get("T", 0))
        expected_shape = (int(sizes.get("Y", 0)), int(sizes.get("X", 0)))
        if frame_count < 1 or expected_shape != pool_labels.shape:
            raise ValueError(
                "ND2 dimensions do not match the accepted pool/background masks."
            )
        unsupported = {
            axis: size
            for axis, size in sizes.items()
            if axis not in {"T", "Y", "X"} and size != 1
        }
        if unsupported:
            raise ValueError(f"Unsupported ND2 dimensions: {unsupported}.")

        pool_traces = np.empty((frame_count, len(pool_ids)), dtype=np.float64)
        background_traces = np.empty(
            (frame_count, len(background_ids)), dtype=np.float64
        )
        flat_pool_labels = pool_labels.ravel()
        flat_background_labels = background_labels.ravel()
        for frame_index in range(frame_count):
            frame = np.asarray(recording.read_frame(frame_index))
            if frame.shape != expected_shape:
                raise ValueError(
                    f"Frame {frame_index + 1} has unexpected shape {frame.shape}."
                )
            flat_frame = frame.ravel()
            pool_traces[frame_index] = _label_means(
                flat_frame, flat_pool_labels, pool_ids, pool_counts
            )
            background_traces[frame_index] = _label_means(
                flat_frame,
                flat_background_labels,
                background_ids,
                background_counts,
            )

    time_s, time_source = _build_time_axis(
        frame_count, timestamps_s, uniform_interval_s
    )
    return TraceExtractionResult(
        frame_numbers=np.arange(1, frame_count + 1, dtype=np.int32),
        time_s=time_s,
        time_source=time_source,
        pool_ids=pool_ids,
        background_ids=background_ids,
        pool_traces=pool_traces,
        background_traces=background_traces,
    )


def export_analysis_bundle(request: AnalysisExportRequest) -> AnalysisExportResult:
    """Extract traces and stage the workbook, ImageJ ROI ZIP, and display LUT."""
    output_directory = Path(request.output_directory).expanduser().resolve()
    if not output_directory.is_dir():
        raise NotADirectoryError(output_directory)
    workbook_path = output_directory / f"{request.source_path.stem}_analysis.xlsx"
    roi_zip_path = output_directory / f"{request.source_path.stem}_rois.zip"
    lut_path = output_directory / f"{request.source_path.stem}_display.lut"
    for path in (workbook_path, roi_zip_path, lut_path):
        if path.exists() and not request.overwrite:
            raise FileExistsError(path)

    traces = extract_recording_traces(
        request.source_path,
        request.pool_labels,
        request.background_labels,
        timestamps_s=request.metadata.timestamps_s,
        uniform_interval_s=request.uniform_interval_s,
    )
    staging_directory = (
        output_directory / f".{request.source_path.stem}-export-{uuid4().hex}"
    )
    staging_directory.mkdir()
    staged_workbook = staging_directory / workbook_path.name
    staged_rois = staging_directory / roi_zip_path.name
    staged_lut = staging_directory / lut_path.name
    try:
        staging_request = replace(
            request, output_directory=staging_directory, overwrite=True
        )
        write_analysis_workbook(staged_workbook, traces, staging_request)
        write_imagej_roi_zip(
            staged_rois,
            request.pool_labels,
            request.background_labels,
            request.rectangles,
            effective_peak_index=request.effective_peak_index,
            overwrite=True,
        )
        write_imagej_lut(staged_lut, request.display_lut, overwrite=True)
        for path in (workbook_path, roi_zip_path, lut_path):
            if path.exists() and not request.overwrite:
                raise FileExistsError(path)
        os.replace(staged_workbook, workbook_path)
        os.replace(staged_rois, roi_zip_path)
        os.replace(staged_lut, lut_path)
    finally:
        staged_workbook.unlink(missing_ok=True)
        staged_rois.unlink(missing_ok=True)
        staged_lut.unlink(missing_ok=True)
        staging_directory.rmdir()
    return AnalysisExportResult(
        workbook_path=workbook_path,
        roi_zip_path=roi_zip_path,
        lut_path=lut_path,
        frame_count=len(traces.frame_numbers),
        pool_count=len(traces.pool_ids),
        background_count=len(traces.background_ids),
        time_source=traces.time_source,
    )


def write_imagej_lut(
    path: Path, display_lut: NDArray[np.uint8], *, overwrite: bool = False
) -> None:
    """Write and verify a raw 768-byte ImageJ LUT (red, green, blue planes)."""
    path = Path(path).expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    lut = np.asarray(display_lut)
    if lut.shape != (256, 3):
        raise ValueError("An ImageJ LUT must contain exactly 256 RGB colors.")
    if not np.issubdtype(lut.dtype, np.integer):
        raise ValueError("ImageJ LUT color values must be integers.")
    if np.any(lut < 0) or np.any(lut > 255):
        raise ValueError("ImageJ LUT color values must be between 0 and 255.")
    lut = np.asarray(lut, dtype=np.uint8)
    payload = b"".join(lut[:, channel].tobytes() for channel in range(3))

    temporary_path = _temporary_sibling(path, ".lut")
    try:
        temporary_path.write_bytes(payload)
        verification = temporary_path.read_bytes()
        if len(verification) != 768 or verification != payload:
            raise RuntimeError("ImageJ LUT verification failed.")
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_analysis_workbook(
    path: Path,
    traces: TraceExtractionResult,
    request: AnalysisExportRequest,
) -> None:
    path = Path(path).expanduser().resolve()
    if path.exists() and not request.overwrite:
        raise FileExistsError(path)
    workbook = Workbook()
    metadata_sheet = workbook.active
    metadata_sheet.title = "Metadata"
    _write_metadata_sheet(metadata_sheet, traces, request)
    _write_trace_sheet(
        workbook.create_sheet("Pool traces"),
        traces.frame_numbers,
        traces.time_s,
        traces.pool_ids,
        traces.pool_traces,
        "Pool",
    )
    _write_trace_sheet(
        workbook.create_sheet("Background traces"),
        traces.frame_numbers,
        traces.time_s,
        traces.background_ids,
        traces.background_traces,
        "BG",
    )
    _write_regions_sheet(workbook.create_sheet("Regions"), request)
    _write_recording_mean_sheet(
        workbook.create_sheet("Recording mean"), traces, request.peak_result
    )

    temporary_path = _temporary_sibling(path, ".xlsx")
    try:
        workbook.save(temporary_path)
        verification = load_workbook(temporary_path, read_only=True, data_only=True)
        try:
            expected_sheets = [
                "Metadata",
                "Pool traces",
                "Background traces",
                "Regions",
                "Recording mean",
            ]
            if verification.sheetnames != expected_sheets:
                raise RuntimeError("Workbook verification found unexpected sheets.")
            if verification["Pool traces"].max_row != len(traces.frame_numbers) + 1:
                raise RuntimeError(
                    "Workbook verification found incomplete pool traces."
                )
        finally:
            verification.close()
        if path.exists() and not request.overwrite:
            raise FileExistsError(path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_imagej_roi_zip(
    path: Path,
    pool_labels: NDArray[np.int32],
    background_labels: NDArray[np.int32],
    rectangles: tuple[RectangleBounds, ...],
    *,
    effective_peak_index: int,
    overwrite: bool = False,
) -> None:
    path = Path(path).expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    if effective_peak_index < 0:
        raise ValueError("Effective peak index cannot be negative.")

    rois: list[ImagejRoi] = []
    expected_coordinates: dict[str, NDArray[np.float64]] = {}
    for prefix, labels in (("Pool", pool_labels), ("Background", background_labels)):
        for region_id in (int(value) for value in np.unique(labels) if value > 0):
            components, count = ndimage.label(labels == region_id)
            for component_id in range(1, count + 1):
                points = _mask_outer_contour(components == component_id)
                suffix = "" if count == 1 else f"_part_{component_id:02d}"
                name = f"{prefix}_{region_id:03d}{suffix}"
                rois.append(
                    ImagejRoi.frompoints(
                        points,
                        name=name,
                        position=effective_peak_index,
                    )
                )
                expected_coordinates[name] = points

    for index, rectangle in enumerate(rectangles, start=1):
        points = np.asarray(
            [
                (rectangle.x0, rectangle.y0),
                (rectangle.x1, rectangle.y0),
                (rectangle.x1, rectangle.y1),
                (rectangle.x0, rectangle.y1),
            ],
            dtype=np.float64,
        )
        name = f"Analysis_Region_{index:03d}"
        rois.append(
            ImagejRoi.frompoints(
                points,
                name=name,
                position=effective_peak_index,
            )
        )
        expected_coordinates[name] = points
    if not rois:
        raise ValueError("There are no regions to export.")

    temporary_path = _temporary_sibling(path, ".zip")
    try:
        roiwrite(temporary_path, rois, mode="w")
        loaded = roiread(temporary_path)
        loaded_rois = loaded if isinstance(loaded, list) else [loaded]
        if len(loaded_rois) != len(rois):
            raise RuntimeError("ROI ZIP verification found the wrong ROI count.")
        for roi in loaded_rois:
            if roi.name not in expected_coordinates:
                raise RuntimeError(f"ROI ZIP contains an unexpected ROI: {roi.name}.")
            expected = expected_coordinates[roi.name]
            actual = np.asarray(roi.coordinates(), dtype=np.float64)
            if actual.shape != expected.shape or not np.allclose(
                actual, expected, atol=0.01
            ):
                raise RuntimeError(
                    f"ROI coordinate verification failed for {roi.name}."
                )
            if roi.position != effective_peak_index + 1:
                raise RuntimeError(
                    f"ROI frame-position verification failed for {roi.name}."
                )
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _label_ids_and_counts(
    labels: NDArray[np.int32],
) -> tuple[tuple[int, ...], NDArray[np.int64]]:
    counts = np.bincount(labels.ravel())
    ids = tuple(int(value) for value in np.flatnonzero(counts[1:]) + 1)
    return ids, counts


def _label_means(
    flat_frame: NDArray[np.generic],
    flat_labels: NDArray[np.int32],
    ids: tuple[int, ...],
    counts: NDArray[np.int64],
) -> NDArray[np.float64]:
    sums = np.bincount(flat_labels, weights=flat_frame, minlength=len(counts))
    indices = np.asarray(ids, dtype=np.int64)
    return np.asarray(sums[indices] / counts[indices], dtype=np.float64)


def _build_time_axis(
    frame_count: int,
    timestamps_s: tuple[float, ...] | None,
    uniform_interval_s: float | None,
) -> tuple[NDArray[np.float64] | None, str]:
    if timestamps_s is not None:
        values = np.asarray(timestamps_s, dtype=np.float64)
        if (
            len(values) != frame_count
            or not np.all(np.isfinite(values))
            or np.any(np.diff(values) < 0)
        ):
            raise ValueError("ND2 timestamps are incomplete or invalid.")
        return values, "ND2 metadata timestamps"
    if uniform_interval_s is not None:
        return (
            np.arange(frame_count, dtype=np.float64) * uniform_interval_s,
            f"User-provided uniform interval ({uniform_interval_s:g} s/frame)",
        )
    return None, "One-based frame numbers (time interval unavailable)"


def _write_metadata_sheet(
    sheet: Any, traces: TraceExtractionResult, request: AnalysisExportRequest
) -> None:
    metadata = request.metadata
    rows: list[tuple[str, Any]] = [
        ("Source file", str(request.source_path)),
        ("Exported UTC", datetime.now(UTC).isoformat(timespec="seconds")),
        ("Application version", __version__),
        ("Frames", metadata.frame_count),
        ("Height [px]", metadata.height),
        ("Width [px]", metadata.width),
        ("Pixel type", metadata.dtype),
        ("Pixel size X [um]", metadata.pixel_size_x_um),
        ("Pixel size Y [um]", metadata.pixel_size_y_um),
        ("Exposure [ms]", metadata.exposure_ms),
        ("Time source", traces.time_source),
        ("Automatic peak frame", request.peak_result.automatic_peak_index + 1),
        ("Effective peak frame", request.effective_peak_index + 1),
        ("Analysis rectangle count", len(request.rectangles)),
        ("Accepted pool count", len(traces.pool_ids)),
        ("Accepted background count", len(traces.background_ids)),
        ("Background mode", request.background_parameters.mode),
        ("Display LUT file", f"{request.source_path.stem}_display.lut"),
        ("Display minimum", request.display_levels[0]),
        ("Display maximum", request.display_levels[1]),
        ("Display note", "Display-only; analysis used raw ND2 pixel values"),
    ]
    for name, value in asdict(request.segmentation_parameters).items():
        rows.append((f"Segmentation: {name}", value))
    for name, value in asdict(request.background_parameters).items():
        rows.append((f"Background: {name}", value))
    sheet.append(("Field", "Value"))
    for row in rows:
        sheet.append(row)
    _format_table_sheet(sheet)


def _write_trace_sheet(
    sheet: Any,
    frame_numbers: NDArray[np.int32],
    time_s: NDArray[np.float64] | None,
    region_ids: tuple[int, ...],
    values: NDArray[np.float64],
    prefix: str,
) -> None:
    headers = ["Frame"]
    if time_s is not None:
        headers.append("Time [s]")
    headers.extend(f"{prefix} {region_id}" for region_id in region_ids)
    sheet.append(headers)
    for index, frame_number in enumerate(frame_numbers):
        row: list[int | float] = [int(frame_number)]
        if time_s is not None:
            row.append(float(time_s[index]))
        row.extend(float(value) for value in values[index])
        sheet.append(row)
    _format_table_sheet(sheet)


def _write_regions_sheet(sheet: Any, request: AnalysisExportRequest) -> None:
    headers = (
        "Region",
        "Type",
        "ID",
        "Owner pool ID",
        "Analysis rectangle",
        "Area [px^2]",
        "Centroid X [px]",
        "Centroid Y [px]",
        "X min",
        "Y min",
        "X max exclusive",
        "Y max exclusive",
        "Origin",
    )
    sheet.append(headers)
    for row in _region_rows(
        "Pool",
        request.pool_labels,
        request.automatic_pool_labels,
        request.rectangles,
        {},
    ):
        sheet.append(row)
    for row in _region_rows(
        "Background",
        request.background_labels,
        request.automatic_background_labels,
        request.rectangles,
        request.background_assignments,
    ):
        sheet.append(row)
    _format_table_sheet(sheet)


def _region_rows(
    kind: str,
    labels: NDArray[np.int32],
    automatic_labels: NDArray[np.int32],
    rectangles: tuple[RectangleBounds, ...],
    assignments: dict[int, int | None],
) -> list[tuple[Any, ...]]:
    rows = []
    for region_id in (int(value) for value in np.unique(labels) if value > 0):
        yy, xx = np.nonzero(labels == region_id)
        rectangle_index = _owning_rectangle_index(xx, yy, rectangles)
        unchanged = automatic_labels.shape == labels.shape and np.array_equal(
            labels == region_id, automatic_labels == region_id
        )
        rows.append(
            (
                f"{kind} {region_id}",
                kind.casefold(),
                region_id,
                assignments.get(region_id),
                rectangle_index,
                len(xx),
                float(np.mean(xx)),
                float(np.mean(yy)),
                int(xx.min()),
                int(yy.min()),
                int(xx.max()) + 1,
                int(yy.max()) + 1,
                "automatic" if unchanged else "manual/edited",
            )
        )
    return rows


def _owning_rectangle_index(
    xx: NDArray[np.int64],
    yy: NDArray[np.int64],
    rectangles: tuple[RectangleBounds, ...],
) -> int | None:
    for index, rectangle in enumerate(rectangles, start=1):
        if np.all(
            (xx >= rectangle.x0)
            & (xx < rectangle.x1)
            & (yy >= rectangle.y0)
            & (yy < rectangle.y1)
        ):
            return index
    return None


def _write_recording_mean_sheet(
    sheet: Any, traces: TraceExtractionResult, peak_result: PeakDetectionResult
) -> None:
    mean_trace = np.asarray(peak_result.recording_mean_trace, dtype=np.float64)
    if len(mean_trace) != len(traces.frame_numbers):
        raise ValueError("The peak-detection trace has the wrong frame count.")
    headers = ["Frame"]
    if traces.time_s is not None:
        headers.append("Time [s]")
    headers.append("Analysis-region mean")
    sheet.append(headers)
    for index, frame_number in enumerate(traces.frame_numbers):
        row: list[int | float] = [int(frame_number)]
        if traces.time_s is not None:
            row.append(float(traces.time_s[index]))
        row.append(float(mean_trace[index]))
        sheet.append(row)
    _format_table_sheet(sheet)


def _format_table_sheet(sheet: Any) -> None:
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for column_cells in sheet.columns:
        values = (
            "" if cell.value is None else str(cell.value) for cell in column_cells
        )
        width = min(45, max(len(value) for value in values) + 2)
        sheet.column_dimensions[column_cells[0].column_letter].width = width


def _mask_outer_contour(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    padded = np.pad(mask.astype(np.uint8), 1)
    contours = find_contours(padded, 0.5)
    if not contours:
        raise ValueError("Could not trace an accepted ROI mask.")
    contour = max(contours, key=len)
    points = np.column_stack((contour[:, 1] - 1, contour[:, 0] - 1))
    if len(points) < 3:
        raise ValueError("An accepted ROI does not have a valid boundary.")
    return np.asarray(points, dtype=np.float64)


def _temporary_sibling(target: Path, suffix: str) -> Path:
    handle, name = tempfile.mkstemp(
        prefix=f".{target.stem}-",
        suffix=suffix,
        dir=target.parent,
    )
    os.close(handle)
    return Path(name)
