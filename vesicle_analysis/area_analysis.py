from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import nd2
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage.filters import gaussian

from vesicle_analysis.analysis import RectangleBounds
from vesicle_analysis.motion_correction import apply_frame_transform
from vesicle_analysis.nd2_io import _validate_dimensions, _validated_path
from vesicle_analysis.segmentation import SegmentationParameters

if TYPE_CHECKING:
    from vesicle_analysis.motion_correction import MotionCorrectionResult

QC_NOT_DETECTED = np.uint8(1)
QC_BELOW_MINIMUM = np.uint8(2)
QC_ABOVE_MAXIMUM = np.uint8(4)
QC_THRESHOLD_UNAVAILABLE = np.uint8(8)


@dataclass(frozen=True, slots=True)
class FrameAreaDetection:
    labels: NDArray[np.int32]
    pool_ids: tuple[int, ...]
    areas_px: NDArray[np.int32]
    thresholds: NDArray[np.float64]
    qc_codes: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class AreaAnalysisResult:
    frame_numbers: NDArray[np.int32]
    pool_ids: tuple[int, ...]
    peak_areas_px: NDArray[np.int32]
    areas_px: NDArray[np.int32]
    thresholds: NDArray[np.float64]
    qc_codes: NDArray[np.uint8]
    peak_frame_index: int
    parameters: SegmentationParameters

    def __post_init__(self) -> None:
        frame_count = len(self.frame_numbers)
        pool_count = len(self.pool_ids)
        expected = (frame_count, pool_count)
        if self.peak_areas_px.shape != (pool_count,):
            raise ValueError("Peak-area dimensions do not match the pool identifiers.")
        if self.areas_px.shape != expected:
            raise ValueError("Framewise-area dimensions are invalid.")
        if self.thresholds.shape != expected or self.qc_codes.shape != expected:
            raise ValueError("Area QC dimensions are invalid.")
        if not 0 <= self.peak_frame_index < frame_count:
            raise ValueError("Peak frame is outside the area time series.")

    @property
    def warning_count(self) -> int:
        return int(np.count_nonzero(self.qc_codes))


@dataclass(frozen=True, slots=True)
class _PoolDomain:
    pool_id: int
    owner: int
    y0: int
    y1: int
    x0: int
    x1: int
    mask: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class _PreparedAreaAnalysis:
    labels: NDArray[np.int32]
    pool_ids: tuple[int, ...]
    domains: tuple[_PoolDomain, ...]


def analyze_recording_areas(
    path: Path,
    pool_labels: NDArray[np.int32],
    rectangles: tuple[RectangleBounds, ...],
    peak_frame_index: int,
    parameters: SegmentationParameters,
    correction: MotionCorrectionResult | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> AreaAnalysisResult:
    """Measure threshold-detected pixels inside fixed accepted peak masks."""
    path = _validated_path(path)
    prepared = _prepare_area_analysis(pool_labels, rectangles)
    pool_labels = prepared.labels
    pool_ids = prepared.pool_ids
    peak_areas = _peak_areas(pool_labels, pool_ids)

    with nd2.ND2File(path) as recording:
        sizes = dict(recording.sizes)
        _validate_dimensions(sizes)
        frame_count = int(sizes["T"])
        expected_shape = (int(sizes["Y"]), int(sizes["X"]))
        if pool_labels.shape != expected_shape:
            raise ValueError("Accepted pool masks do not match the ND2 dimensions.")
        if not 0 <= peak_frame_index < frame_count:
            raise ValueError("Peak frame is outside the ND2 recording.")
        if correction is not None and (
            correction.image_shape != expected_shape
            or len(correction.transforms) != frame_count
        ):
            raise ValueError("Motion correction does not match the ND2 recording.")

        areas = np.zeros((frame_count, len(pool_ids)), dtype=np.int32)
        thresholds = np.full((frame_count, len(pool_ids)), np.nan, dtype=np.float64)
        qc_codes = np.zeros((frame_count, len(pool_ids)), dtype=np.uint8)
        for frame_index in range(frame_count):
            frame = np.asarray(recording.read_frame(frame_index))
            if frame.shape != expected_shape:
                raise ValueError(
                    f"Frame {frame_index + 1} has unexpected shape {frame.shape}."
                )
            if correction is not None:
                frame = apply_frame_transform(frame, correction.transforms[frame_index])
            detection = _detect_frame_pool_areas_prepared(
                frame, prepared, rectangles, parameters
            )
            areas[frame_index] = detection.areas_px
            thresholds[frame_index] = detection.thresholds
            qc_codes[frame_index] = detection.qc_codes
            if progress_callback is not None:
                progress_callback(frame_index + 1, frame_count)

    return AreaAnalysisResult(
        frame_numbers=np.arange(1, frame_count + 1, dtype=np.int32),
        pool_ids=pool_ids,
        peak_areas_px=peak_areas,
        areas_px=areas,
        thresholds=thresholds,
        qc_codes=qc_codes,
        peak_frame_index=peak_frame_index,
        parameters=parameters,
    )


def detect_frame_pool_areas(
    frame: NDArray[np.generic],
    pool_labels: NDArray[np.int32],
    rectangles: tuple[RectangleBounds, ...],
    parameters: SegmentationParameters,
) -> FrameAreaDetection:
    """Detect each pool only inside its accepted peak-frame mask."""
    frame = np.asarray(frame)
    prepared = _prepare_area_analysis(pool_labels, rectangles)
    if frame.ndim != 2 or frame.shape != prepared.labels.shape:
        raise ValueError("Area detection requires matching two-dimensional arrays.")
    return _detect_frame_pool_areas_prepared(frame, prepared, rectangles, parameters)


def _detect_frame_pool_areas_prepared(
    frame: NDArray[np.generic],
    prepared: _PreparedAreaAnalysis,
    rectangles: tuple[RectangleBounds, ...],
    parameters: SegmentationParameters,
) -> FrameAreaDetection:
    pool_labels = prepared.labels
    pool_ids = prepared.pool_ids
    padding = int(np.ceil(parameters.background_sigma * 4.0))
    image_height, image_width = frame.shape
    crop_x0 = max(0, min(rectangle.x0 for rectangle in rectangles) - padding)
    crop_y0 = max(0, min(rectangle.y0 for rectangle in rectangles) - padding)
    crop_x1 = min(image_width, max(rectangle.x1 for rectangle in rectangles) + padding)
    crop_y1 = min(image_height, max(rectangle.y1 for rectangle in rectangles) + padding)
    enhanced = _enhanced_image(frame[crop_y0:crop_y1, crop_x0:crop_x1], parameters)
    rectangle_thresholds = tuple(
        _rectangle_thresholds(
            enhanced,
            RectangleBounds(
                rectangle.x0 - crop_x0,
                rectangle.y0 - crop_y0,
                rectangle.x1 - crop_x0,
                rectangle.y1 - crop_y0,
            ),
            parameters,
        )
        for rectangle in rectangles
    )

    detected_labels = np.zeros(pool_labels.shape, dtype=np.int32)
    areas = np.zeros(len(pool_ids), dtype=np.int32)
    thresholds = np.full(len(pool_ids), np.nan, dtype=np.float64)
    qc_codes = np.zeros(len(pool_ids), dtype=np.uint8)
    for column, domain in enumerate(prepared.domains):
        foreground_threshold, seed_threshold, available = rectangle_thresholds[
            domain.owner
        ]
        thresholds[column] = foreground_threshold
        if not available:
            qc_codes[column] |= QC_THRESHOLD_UNAVAILABLE | QC_NOT_DETECTED
            continue
        enhanced_crop = enhanced[
            domain.y0 - crop_y0 : domain.y1 - crop_y0,
            domain.x0 - crop_x0 : domain.x1 - crop_x0,
        ]
        foreground = domain.mask & (enhanced_crop > foreground_threshold)
        seeds = domain.mask & (enhanced_crop > seed_threshold)
        if not np.any(seeds):
            qc_codes[column] |= QC_NOT_DETECTED
            continue
        components, _ = ndimage.label(foreground)
        seeded_components = np.unique(components[seeds])
        seeded_components = seeded_components[seeded_components > 0]
        detected = np.isin(components, seeded_components, assume_unique=False)
        area = int(np.count_nonzero(detected))
        areas[column] = area
        target = detected_labels[domain.y0 : domain.y1, domain.x0 : domain.x1]
        target[detected] = domain.pool_id
        if area == 0:
            qc_codes[column] |= QC_NOT_DETECTED
        elif area < parameters.minimum_area_px:
            qc_codes[column] |= QC_BELOW_MINIMUM
        elif area > parameters.maximum_area_px:
            qc_codes[column] |= QC_ABOVE_MAXIMUM

    return FrameAreaDetection(
        labels=detected_labels,
        pool_ids=pool_ids,
        areas_px=areas,
        thresholds=thresholds,
        qc_codes=qc_codes,
    )


def _prepare_area_analysis(
    pool_labels: NDArray[np.int32], rectangles: tuple[RectangleBounds, ...]
) -> _PreparedAreaAnalysis:
    labels = _validated_pool_labels(pool_labels, rectangles)
    pool_ids = _pool_ids(labels)
    owners = _pool_rectangle_indices(labels, pool_ids, rectangles)
    domains: list[_PoolDomain] = []
    for pool_id, owner in zip(pool_ids, owners, strict=True):
        yy, xx = np.nonzero(labels == pool_id)
        y0, y1 = int(yy.min()), int(yy.max()) + 1
        x0, x1 = int(xx.min()), int(xx.max()) + 1
        domains.append(
            _PoolDomain(
                pool_id=pool_id,
                owner=owner,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                mask=np.asarray(labels[y0:y1, x0:x1] == pool_id),
            )
        )
    return _PreparedAreaAnalysis(labels, pool_ids, tuple(domains))


def area_qc_text(code: int) -> str:
    messages: list[str] = []
    if code & int(QC_THRESHOLD_UNAVAILABLE):
        messages.append("threshold unavailable")
    if code & int(QC_NOT_DETECTED):
        messages.append("not detected")
    if code & int(QC_BELOW_MINIMUM):
        messages.append("below segmentation minimum")
    if code & int(QC_ABOVE_MAXIMUM):
        messages.append("above segmentation maximum")
    return "; ".join(messages) if messages else "OK"


def area_masks_to_rgba(
    peak_labels: NDArray[np.int32], detected_labels: NDArray[np.int32]
) -> NDArray[np.uint8]:
    """Render detected pixels and fixed peak boundaries for QC display only."""
    peak_labels = np.asarray(peak_labels, dtype=np.int32)
    detected_labels = np.asarray(detected_labels, dtype=np.int32)
    if peak_labels.shape != detected_labels.shape or peak_labels.ndim != 2:
        raise ValueError("Peak and detected area masks must have matching shapes.")
    rgba = np.zeros((*peak_labels.shape, 4), dtype=np.uint8)
    detected = detected_labels > 0
    rgba[detected, :3] = (0, 229, 255)
    rgba[detected, 3] = 95

    peak = peak_labels > 0
    peak_boundary = peak & ~ndimage.binary_erosion(peak)
    detected_boundary = detected & ~ndimage.binary_erosion(detected)
    rgba[peak_boundary, :3] = (255, 209, 64)
    rgba[peak_boundary, 3] = 235
    rgba[detected_boundary, :3] = (255, 255, 255)
    rgba[detected_boundary, 3] = 255
    return rgba


def _validated_pool_labels(
    pool_labels: NDArray[np.int32], rectangles: tuple[RectangleBounds, ...]
) -> NDArray[np.int32]:
    labels = np.asarray(pool_labels, dtype=np.int32)
    if labels.ndim != 2:
        raise ValueError("Accepted pool labels must be a two-dimensional array.")
    if not rectangles:
        raise ValueError("At least one analysis rectangle is required.")
    if np.any(labels < 0):
        raise ValueError("Pool labels cannot be negative.")
    if not np.any(labels > 0):
        raise ValueError("At least one accepted pool is required.")
    height, width = labels.shape
    for rectangle in rectangles:
        if rectangle.x1 > width or rectangle.y1 > height:
            raise ValueError("An analysis rectangle exceeds the image bounds.")
    return labels


def _pool_ids(labels: NDArray[np.int32]) -> tuple[int, ...]:
    return tuple(int(value) for value in np.unique(labels) if value > 0)


def _peak_areas(
    labels: NDArray[np.int32], pool_ids: tuple[int, ...]
) -> NDArray[np.int32]:
    counts = np.bincount(labels.ravel())
    return np.asarray([counts[pool_id] for pool_id in pool_ids], dtype=np.int32)


def _pool_rectangle_indices(
    labels: NDArray[np.int32],
    pool_ids: tuple[int, ...],
    rectangles: tuple[RectangleBounds, ...],
) -> tuple[int, ...]:
    owners: list[int] = []
    for pool_id in pool_ids:
        mask = labels == pool_id
        overlaps = [
            int(
                np.count_nonzero(
                    mask[rectangle.y0 : rectangle.y1, rectangle.x0 : rectangle.x1]
                )
            )
            for rectangle in rectangles
        ]
        owner = int(np.argmax(overlaps))
        if overlaps[owner] == 0:
            raise ValueError(f"Pool {pool_id} is outside every analysis rectangle.")
        owners.append(owner)
    return tuple(owners)


def _enhanced_image(
    frame: NDArray[np.generic], parameters: SegmentationParameters
) -> NDArray[np.float32]:
    values = np.asarray(frame, dtype=np.float32)
    smoothed = gaussian(
        values, sigma=parameters.smoothing_sigma, preserve_range=True
    ).astype(np.float32, copy=False)
    background = gaussian(
        values, sigma=parameters.background_sigma, preserve_range=True
    ).astype(np.float32, copy=False)
    return np.asarray(smoothed - background, dtype=np.float32)


def _rectangle_thresholds(
    enhanced: NDArray[np.float32],
    rectangle: RectangleBounds,
    parameters: SegmentationParameters,
) -> tuple[float, float, bool]:
    crop = enhanced[rectangle.y0 : rectangle.y1, rectangle.x0 : rectangle.x1]
    median = float(np.median(crop))
    noise = float(1.4826 * np.median(np.abs(crop - median)))
    if not np.isfinite(noise) or noise <= np.finfo(np.float32).eps:
        noise = float(np.std(crop, dtype=np.float64))
    if not np.isfinite(noise) or noise <= np.finfo(np.float32).eps:
        return np.nan, np.nan, False
    foreground_threshold = median + parameters.foreground_threshold_sigma * noise
    seed_threshold = median + parameters.seed_threshold_sigma * noise
    available = np.isfinite(foreground_threshold) and np.isfinite(seed_threshold)
    return float(foreground_threshold), float(seed_threshold), bool(available)
