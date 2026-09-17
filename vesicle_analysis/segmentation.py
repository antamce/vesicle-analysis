from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.measure import regionprops
from skimage.morphology import dilation, disk, remove_small_objects
from skimage.segmentation import watershed

from vesicle_analysis.analysis import RectangleBounds
from vesicle_analysis.nd2_io import read_frame

if TYPE_CHECKING:
    from vesicle_analysis.motion_correction import MotionCorrectionResult


@dataclass(frozen=True, slots=True)
class SegmentationParameters:
    background_sigma: float = 8.0
    smoothing_sigma: float = 1.2
    seed_threshold_sigma: float = 6.0
    foreground_threshold_sigma: float = 1.0
    minimum_seed_distance: int = 4
    minimum_area_px: int = 8
    maximum_area_px: int = 150

    def __post_init__(self) -> None:
        if self.background_sigma <= self.smoothing_sigma:
            raise ValueError("Background scale must exceed the smoothing scale.")
        if self.seed_threshold_sigma <= self.foreground_threshold_sigma:
            raise ValueError("Seed threshold must exceed the foreground threshold.")
        if self.minimum_seed_distance < 1:
            raise ValueError("Minimum seed distance must be at least one pixel.")
        if self.minimum_area_px < 1:
            raise ValueError("Minimum pool area must be positive.")
        if self.maximum_area_px < self.minimum_area_px:
            raise ValueError("Maximum pool area cannot be smaller than the minimum.")


@dataclass(frozen=True, slots=True)
class PoolProperties:
    pool_id: int
    analysis_region_index: int
    area_px: int
    centroid_x: float
    centroid_y: float
    x0: int
    y0: int
    x1: int
    y1: int
    peak_response: float


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    labels: NDArray[np.int32]
    pools: tuple[PoolProperties, ...]
    parameters: SegmentationParameters
    peak_frame_index: int


def segment_recording(
    path: Path,
    peak_frame_index: int,
    rectangles: tuple[RectangleBounds, ...],
    parameters: SegmentationParameters,
    correction: MotionCorrectionResult | None = None,
) -> SegmentationResult:
    frame = read_frame(path, peak_frame_index, correction)
    return segment_frame(frame, peak_frame_index, rectangles, parameters)


def segment_frame(
    frame: NDArray[np.generic],
    peak_frame_index: int,
    rectangles: tuple[RectangleBounds, ...],
    parameters: SegmentationParameters,
) -> SegmentationResult:
    if frame.ndim != 2:
        raise ValueError(f"Expected a 2D frame, received shape {frame.shape!r}.")
    if not rectangles:
        raise ValueError("At least one analysis rectangle is required.")

    image_height, image_width = frame.shape
    global_labels = np.zeros(frame.shape, dtype=np.int32)
    pools: list[PoolProperties] = []
    next_pool_id = 1

    for region_index, rectangle in enumerate(rectangles, start=1):
        if rectangle.x1 > image_width or rectangle.y1 > image_height:
            raise ValueError(f"Rectangle {rectangle} exceeds the frame bounds.")
        crop = np.asarray(
            frame[rectangle.y0 : rectangle.y1, rectangle.x0 : rectangle.x1],
            dtype=np.float32,
        )
        local_labels, enhanced = _segment_crop(crop, parameters)
        for local_id in range(1, int(local_labels.max()) + 1):
            local_mask = local_labels == local_id
            target = global_labels[
                rectangle.y0 : rectangle.y1, rectangle.x0 : rectangle.x1
            ]
            available_mask = local_mask & (target == 0)
            available_area = int(np.count_nonzero(available_mask))
            original_area = int(np.count_nonzero(local_mask))

            # Confirmed analysis-rectangle overlaps can produce the same pool
            # twice. Retain the first instance when most pixels already belong
            # to an accepted pool; otherwise keep only the non-overlapping part.
            if original_area == 0 or available_area / original_area < 0.5:
                continue
            if (
                not parameters.minimum_area_px
                <= available_area
                <= parameters.maximum_area_px
            ):
                continue

            target[available_mask] = next_pool_id
            yy, xx = np.nonzero(available_mask)
            pools.append(
                PoolProperties(
                    pool_id=next_pool_id,
                    analysis_region_index=region_index,
                    area_px=available_area,
                    centroid_x=float(xx.mean() + rectangle.x0),
                    centroid_y=float(yy.mean() + rectangle.y0),
                    x0=int(xx.min() + rectangle.x0),
                    y0=int(yy.min() + rectangle.y0),
                    x1=int(xx.max() + rectangle.x0 + 1),
                    y1=int(yy.max() + rectangle.y0 + 1),
                    peak_response=float(enhanced[available_mask].max()),
                )
            )
            next_pool_id += 1

    return SegmentationResult(
        labels=global_labels,
        pools=tuple(pools),
        parameters=parameters,
        peak_frame_index=peak_frame_index,
    )


def _segment_crop(
    crop: NDArray[np.float32], parameters: SegmentationParameters
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    smoothed = gaussian(
        crop, sigma=parameters.smoothing_sigma, preserve_range=True
    ).astype(np.float32, copy=False)
    background = gaussian(
        crop, sigma=parameters.background_sigma, preserve_range=True
    ).astype(np.float32, copy=False)
    enhanced = smoothed - background

    median = float(np.median(enhanced))
    noise = float(1.4826 * np.median(np.abs(enhanced - median)))
    if not np.isfinite(noise) or noise <= np.finfo(np.float32).eps:
        noise = float(np.std(enhanced, dtype=np.float64))
    if not np.isfinite(noise) or noise <= np.finfo(np.float32).eps:
        return np.zeros(crop.shape, dtype=np.int32), enhanced

    foreground = enhanced > (median + parameters.foreground_threshold_sigma * noise)
    foreground = remove_small_objects(
        foreground, max_size=parameters.minimum_area_px - 1
    )
    seed_coordinates = peak_local_max(
        enhanced,
        min_distance=parameters.minimum_seed_distance,
        threshold_abs=median + parameters.seed_threshold_sigma * noise,
        labels=foreground,
        exclude_border=False,
    )
    if not len(seed_coordinates):
        return np.zeros(crop.shape, dtype=np.int32), enhanced

    markers = np.zeros(crop.shape, dtype=np.int32)
    markers[tuple(seed_coordinates.T)] = np.arange(
        1, len(seed_coordinates) + 1, dtype=np.int32
    )
    labels = watershed(-enhanced, markers, mask=foreground).astype(np.int32, copy=False)

    filtered = np.zeros(labels.shape, dtype=np.int32)
    next_label = 1
    for region in regionprops(labels):
        if parameters.minimum_area_px <= region.area <= parameters.maximum_area_px:
            filtered[labels == region.label] = next_label
            next_label += 1
    return filtered, enhanced


def labels_to_rgba(
    labels: NDArray[np.int32],
    selected_ids: tuple[int, ...] = (),
    warning_ids: tuple[int, ...] = (),
) -> NDArray[np.uint8]:
    """Create a deterministic translucent overlay for display only."""
    rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
    if labels.max() == 0:
        return rgba

    palette = np.asarray(
        [
            (0, 229, 255),
            (255, 209, 102),
            (255, 99, 132),
            (126, 249, 180),
            (190, 130, 255),
            (255, 155, 66),
        ],
        dtype=np.uint8,
    )
    foreground = labels > 0
    rgba[foreground, :3] = palette[(labels[foreground] - 1) % len(palette)]
    rgba[foreground, 3] = 55
    boundaries = np.zeros(labels.shape, dtype=bool)
    boundaries[1:, :] |= (labels[1:, :] > 0) & (labels[1:, :] != labels[:-1, :])
    boundaries[:-1, :] |= (labels[:-1, :] > 0) & (labels[:-1, :] != labels[1:, :])
    boundaries[:, 1:] |= (labels[:, 1:] > 0) & (labels[:, 1:] != labels[:, :-1])
    boundaries[:, :-1] |= (labels[:, :-1] > 0) & (labels[:, :-1] != labels[:, 1:])
    rgba[boundaries, :3] = 255
    rgba[boundaries, 3] = 235
    warning = (
        np.isin(labels, warning_ids)
        if warning_ids
        else np.zeros_like(labels, dtype=bool)
    )
    if np.any(warning):
        warning_halo = dilation(warning, footprint=disk(2)) & ~foreground
        rgba[warning_halo, :3] = (255, 0, 0)
        rgba[warning_halo, 3] = 255
        rgba[warning, :3] = (255, 0, 0)
        rgba[warning, 3] = 175
        warning_boundary = boundaries & warning
        rgba[warning_boundary, :3] = (255, 32, 32)
        rgba[warning_boundary, 3] = 255
    if selected_ids:
        selected = np.isin(labels, selected_ids)
        selected_warning = selected & warning
        selected_regular = selected & ~warning
        rgba[selected_regular, :3] = (255, 221, 64)
        rgba[selected_regular, 3] = 125
        rgba[selected_warning, :3] = (255, 0, 0)
        rgba[selected_warning, 3] = 235
        selected_boundary = boundaries & selected
        rgba[selected_boundary, :3] = (255, 255, 255)
        rgba[selected_boundary, 3] = 255
        if np.any(selected_warning):
            selected_halo = dilation(selected_warning, footprint=disk(3)) & ~foreground
            rgba[selected_halo, :3] = (255, 255, 255)
            rgba[selected_halo, 3] = 255
    return rgba
