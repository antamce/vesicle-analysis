from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.measure import regionprops
from skimage.morphology import dilation, disk, h_maxima
from skimage.segmentation import watershed

from vesicle_analysis.analysis import RectangleBounds
from vesicle_analysis.nd2_io import read_frame

if TYPE_CHECKING:
    from vesicle_analysis.motion_correction import MotionCorrectionResult


FLAG_AUTO_SPLIT = "Automatically split"
FLAG_AREA = "Area outlier"
FLAG_CIRCULARITY = "Low circularity"
FLAG_SOLIDITY = "Low solidity"
FLAG_LOCAL_CONTRAST = "Low local contrast"
FLAG_INTENSITY_VARIATION = "High intensity variation"

QUALITY_FLAGS = frozenset(
    {
        FLAG_AREA,
        FLAG_CIRCULARITY,
        FLAG_SOLIDITY,
        FLAG_LOCAL_CONTRAST,
        FLAG_INTENSITY_VARIATION,
    }
)


@dataclass(frozen=True, slots=True)
class SegmentationParameters:
    background_sigma: float = 8.0
    smoothing_sigma: float = 1.2
    seed_threshold_sigma: float = 6.0
    foreground_threshold_sigma: float = 1.0
    minimum_peak_prominence_sigma: float = 1.0
    minimum_area_px: int = 8
    maximum_area_px: int = 150
    baseline_mode: Literal["robust", "mean_std", "manual"] = "robust"
    filter_area: bool = True
    filter_circularity: bool = True
    filter_solidity: bool = True
    filter_local_contrast: bool = True
    filter_intensity_variation: bool = True
    area_outlier_multiplier: float = 3.0
    circularity_outlier_multiplier: float = 3.0
    solidity_outlier_multiplier: float = 3.0
    local_contrast_outlier_multiplier: float = 3.0
    intensity_variation_outlier_multiplier: float = 3.0
    minimum_circularity: float | None = None
    minimum_solidity: float | None = None
    minimum_local_contrast: float | None = None
    maximum_intensity_variation: float | None = None

    def __post_init__(self) -> None:
        if self.background_sigma <= self.smoothing_sigma:
            raise ValueError("Background scale must exceed the smoothing scale.")
        if self.seed_threshold_sigma <= self.foreground_threshold_sigma:
            raise ValueError("Seed threshold must exceed the foreground threshold.")
        if self.minimum_peak_prominence_sigma < 0:
            raise ValueError("Minimum peak prominence cannot be negative.")
        if self.minimum_area_px < 1:
            raise ValueError("Minimum pool area must be positive.")
        if self.maximum_area_px < self.minimum_area_px:
            raise ValueError("Maximum pool area cannot be smaller than the minimum.")
        if self.baseline_mode not in {"robust", "mean_std", "manual"}:
            raise ValueError(f"Unknown segmentation baseline {self.baseline_mode!r}.")
        for name, value in (
            ("area", self.area_outlier_multiplier),
            ("circularity", self.circularity_outlier_multiplier),
            ("solidity", self.solidity_outlier_multiplier),
            ("local contrast", self.local_contrast_outlier_multiplier),
            ("intensity variation", self.intensity_variation_outlier_multiplier),
        ):
            if value <= 0:
                raise ValueError(f"The {name} outlier multiplier must be positive.")
        for name, value in (
            ("minimum circularity", self.minimum_circularity),
            ("minimum solidity", self.minimum_solidity),
        ):
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"The {name} must be between zero and one.")
        if self.minimum_local_contrast is not None and self.minimum_local_contrast < 0:
            raise ValueError("Minimum local contrast cannot be negative.")
        if (
            self.maximum_intensity_variation is not None
            and self.maximum_intensity_variation <= 0
        ):
            raise ValueError("Maximum intensity variation must be positive.")


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
    circularity: float
    solidity: float
    local_contrast: float
    intensity_variation: float
    flags: tuple[str, ...] = ()
    split_group_id: int | None = None
    accepted: bool = True


@dataclass(frozen=True, slots=True)
class SplitGroup:
    group_id: int
    parent_pool_id: int
    pool_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    labels: NDArray[np.int32]
    candidate_labels: NDArray[np.int32]
    pools: tuple[PoolProperties, ...]
    split_groups: tuple[SplitGroup, ...]
    parameters: SegmentationParameters
    peak_frame_index: int

    @property
    def accepted_pool_ids(self) -> tuple[int, ...]:
        return tuple(pool.pool_id for pool in self.pools if pool.accepted)

    @property
    def flags_by_pool(self) -> dict[int, tuple[str, ...]]:
        return {pool.pool_id: pool.flags for pool in self.pools if pool.flags}

    @property
    def quality_flags_by_pool(self) -> dict[int, tuple[str, ...]]:
        """Return filtering reasons without conflating them with split provenance."""
        return {
            pool.pool_id: tuple(
                flag for flag in pool.flags if flag in QUALITY_FLAGS
            )
            for pool in self.pools
            if QUALITY_FLAGS.intersection(pool.flags)
        }


@dataclass(frozen=True, slots=True)
class _LocalPool:
    local_id: int
    area_px: int
    centroid_x: float
    centroid_y: float
    x0: int
    y0: int
    x1: int
    y1: int
    peak_response: float
    circularity: float
    solidity: float
    local_contrast: float
    intensity_variation: float
    flags: tuple[str, ...]
    split_group_id: int | None


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
    accepted_labels = np.zeros(frame.shape, dtype=np.int32)
    candidate_labels = np.zeros(frame.shape, dtype=np.int32)
    pools: list[PoolProperties] = []
    next_pool_id = 1
    pool_group_keys: dict[int, tuple[int, int]] = {}

    for region_index, rectangle in enumerate(rectangles, start=1):
        if rectangle.x1 > image_width or rectangle.y1 > image_height:
            raise ValueError(f"Rectangle {rectangle} exceeds the frame bounds.")
        crop = np.asarray(
            frame[rectangle.y0 : rectangle.y1, rectangle.x0 : rectangle.x1],
            dtype=np.float32,
        )
        local_labels, local_pools = _segment_crop(crop, parameters)
        for local_pool in local_pools:
            local_mask = local_labels == local_pool.local_id
            target_candidates = candidate_labels[
                rectangle.y0 : rectangle.y1, rectangle.x0 : rectangle.x1
            ]
            available_mask = local_mask & (target_candidates == 0)
            available_area = int(np.count_nonzero(available_mask))
            original_area = int(np.count_nonzero(local_mask))

            # Overlapping analysis rectangles can discover the same candidate twice.
            # Keep the first when at least half of the later candidate is occupied.
            if original_area == 0 or available_area / original_area < 0.5:
                continue

            pool_id = next_pool_id
            next_pool_id += 1
            target_candidates[available_mask] = pool_id
            accepted = not bool(QUALITY_FLAGS.intersection(local_pool.flags))
            if accepted:
                target_accepted = accepted_labels[
                    rectangle.y0 : rectangle.y1, rectangle.x0 : rectangle.x1
                ]
                target_accepted[available_mask] = pool_id

            yy, xx = np.nonzero(available_mask)
            pools.append(
                PoolProperties(
                    pool_id=pool_id,
                    analysis_region_index=region_index,
                    area_px=available_area,
                    centroid_x=float(xx.mean() + rectangle.x0),
                    centroid_y=float(yy.mean() + rectangle.y0),
                    x0=int(xx.min() + rectangle.x0),
                    y0=int(yy.min() + rectangle.y0),
                    x1=int(xx.max() + rectangle.x0 + 1),
                    y1=int(yy.max() + rectangle.y0 + 1),
                    peak_response=local_pool.peak_response,
                    circularity=local_pool.circularity,
                    solidity=local_pool.solidity,
                    local_contrast=local_pool.local_contrast,
                    intensity_variation=local_pool.intensity_variation,
                    flags=local_pool.flags,
                    accepted=accepted,
                )
            )
            if local_pool.split_group_id is not None:
                pool_group_keys[pool_id] = (region_index, local_pool.split_group_id)

    grouped_ids: dict[tuple[int, int], list[int]] = {}
    for pool_id, key in pool_group_keys.items():
        grouped_ids.setdefault(key, []).append(pool_id)
    property_by_id = {pool.pool_id: pool for pool in pools}
    split_groups: list[SplitGroup] = []
    group_id_by_pool: dict[int, int] = {}
    for member_ids in grouped_ids.values():
        if len(member_ids) < 2:
            continue
        ordered = tuple(
            sorted(
                member_ids, key=lambda value: (-property_by_id[value].area_px, value)
            )
        )
        group_id = len(split_groups) + 1
        split_groups.append(
            SplitGroup(group_id=group_id, parent_pool_id=ordered[0], pool_ids=ordered)
        )
        for pool_id in ordered:
            group_id_by_pool[pool_id] = group_id

    normalized_pools: list[PoolProperties] = []
    for pool in pools:
        group_id = group_id_by_pool.get(pool.pool_id)
        flags = pool.flags
        if FLAG_AUTO_SPLIT in flags and group_id is None:
            flags = tuple(flag for flag in flags if flag != FLAG_AUTO_SPLIT)
        normalized_pools.append(replace(pool, flags=flags, split_group_id=group_id))

    return SegmentationResult(
        labels=accepted_labels,
        candidate_labels=candidate_labels,
        pools=tuple(normalized_pools),
        split_groups=tuple(split_groups),
        parameters=parameters,
        peak_frame_index=peak_frame_index,
    )


def find_seed_coordinates(
    enhanced: NDArray[np.floating],
    foreground: NDArray[np.bool_],
    median: float,
    noise: float,
    parameters: SegmentationParameters,
) -> NDArray[np.intp]:
    """Find thresholded regional maxima without imposing a spatial exclusion radius."""
    threshold = median + parameters.seed_threshold_sigma * noise
    prominence = parameters.minimum_peak_prominence_sigma * noise
    if prominence <= np.finfo(np.float32).eps:
        return peak_local_max(
            enhanced,
            min_distance=1,
            threshold_abs=threshold,
            labels=foreground,
            exclude_border=False,
        ).astype(np.intp, copy=False)

    maxima = h_maxima(enhanced, prominence)
    maxima &= foreground & (enhanced >= threshold)
    components, count = ndimage.label(maxima, structure=np.ones((3, 3), dtype=np.uint8))
    coordinates: list[tuple[int, int]] = []
    for component_id in range(1, count + 1):
        yy, xx = np.nonzero(components == component_id)
        if not len(yy):
            continue
        values = enhanced[yy, xx]
        best = int(np.argmax(values))
        coordinates.append((int(yy[best]), int(xx[best])))
    coordinates.sort()
    return np.asarray(coordinates, dtype=np.intp).reshape((-1, 2))


def _segment_crop(
    crop: NDArray[np.float32], parameters: SegmentationParameters
) -> tuple[NDArray[np.int32], tuple[_LocalPool, ...]]:
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
        return np.zeros(crop.shape, dtype=np.int32), ()

    foreground = enhanced > (median + parameters.foreground_threshold_sigma * noise)
    # Area limits are reversible flags, so no thresholded candidate is discarded here.
    seed_coordinates = find_seed_coordinates(
        enhanced, foreground, median, noise, parameters
    )
    if not len(seed_coordinates):
        return np.zeros(crop.shape, dtype=np.int32), ()

    markers = np.zeros(crop.shape, dtype=np.int32)
    markers[tuple(seed_coordinates.T)] = np.arange(
        1, len(seed_coordinates) + 1, dtype=np.int32
    )
    raw_watershed = watershed(-enhanced, markers, mask=foreground).astype(
        np.int32, copy=False
    )
    foreground_components, component_count = ndimage.label(
        foreground, structure=np.ones((3, 3), dtype=np.uint8)
    )

    local_labels = np.zeros(crop.shape, dtype=np.int32)
    split_group_by_label: dict[int, int] = {}
    next_label = 1
    next_group = 1
    for component_id in range(1, component_count + 1):
        component = foreground_components == component_id
        marker_ids = tuple(
            int(value) for value in np.unique(raw_watershed[component]) if value
        )
        if not marker_ids:
            continue
        child_masks = [raw_watershed == marker_id for marker_id in marker_ids]
        child_masks.sort(key=lambda mask: int(np.count_nonzero(mask)), reverse=True)
        valid_split = len(child_masks) > 1 and all(
            np.count_nonzero(mask) >= parameters.minimum_area_px for mask in child_masks
        )
        if valid_split:
            group_id = next_group
            next_group += 1
            for child in child_masks:
                local_labels[child] = next_label
                split_group_by_label[next_label] = group_id
                next_label += 1
        else:
            local_labels[component] = next_label
            next_label += 1

    if local_labels.max() == 0:
        return local_labels, ()

    all_candidate_pixels = local_labels > 0
    metric_values: dict[int, dict[str, float]] = {}
    properties_by_id = {region.label: region for region in regionprops(local_labels)}
    for local_id, region in properties_by_id.items():
        mask = local_labels == local_id
        area = int(region.area)
        perimeter = float(region.perimeter_crofton)
        circularity = (
            min(1.0, float(4.0 * np.pi * area / (perimeter * perimeter)))
            if perimeter > 0
            else 1.0
        )
        solidity = float(region.solidity)
        annulus = dilation(mask, footprint=disk(6)) & ~dilation(mask, footprint=disk(2))
        annulus &= ~all_candidate_pixels
        local_contrast = float("nan")
        if np.any(annulus):
            local_contrast = float(
                (
                    np.mean(smoothed[mask], dtype=np.float64)
                    - np.median(smoothed[annulus])
                )
                / noise
            )
        object_response = enhanced[mask]
        object_median = float(np.median(object_response))
        robust_spread = float(
            1.4826 * np.median(np.abs(object_response - object_median))
        )
        intensity_variation = robust_spread / max(abs(object_median), noise * 0.1)
        metric_values[local_id] = {
            "area": float(area),
            "circularity": circularity,
            "solidity": solidity,
            "local_contrast": local_contrast,
            "intensity_variation": float(intensity_variation),
        }

    flags_by_id = _quality_flags(metric_values, parameters)
    pools: list[_LocalPool] = []
    for local_id, region in properties_by_id.items():
        values = metric_values[local_id]
        mask = local_labels == local_id
        flags = list(flags_by_id[local_id])
        if local_id in split_group_by_label:
            flags.insert(0, FLAG_AUTO_SPLIT)
        min_row, min_col, max_row, max_col = region.bbox
        pools.append(
            _LocalPool(
                local_id=local_id,
                area_px=int(region.area),
                centroid_x=float(region.centroid[1]),
                centroid_y=float(region.centroid[0]),
                x0=int(min_col),
                y0=int(min_row),
                x1=int(max_col),
                y1=int(max_row),
                peak_response=float(np.max(enhanced[mask])),
                circularity=values["circularity"],
                solidity=values["solidity"],
                local_contrast=values["local_contrast"],
                intensity_variation=values["intensity_variation"],
                flags=tuple(flags),
                split_group_id=split_group_by_label.get(local_id),
            )
        )
    return local_labels, tuple(pools)


def _quality_flags(
    metric_values: Mapping[int, Mapping[str, float]],
    parameters: SegmentationParameters,
) -> dict[int, tuple[str, ...]]:
    flags: dict[int, list[str]] = {pool_id: [] for pool_id in metric_values}

    def flag(pool_id: int, reason: str) -> None:
        if reason not in flags[pool_id]:
            flags[pool_id].append(reason)

    enabled = {
        "area": parameters.filter_area,
        "circularity": parameters.filter_circularity,
        "solidity": parameters.filter_solidity,
        "local_contrast": parameters.filter_local_contrast,
        "intensity_variation": parameters.filter_intensity_variation,
    }
    reasons = {
        "area": FLAG_AREA,
        "circularity": FLAG_CIRCULARITY,
        "solidity": FLAG_SOLIDITY,
        "local_contrast": FLAG_LOCAL_CONTRAST,
        "intensity_variation": FLAG_INTENSITY_VARIATION,
    }
    low_tail = {"circularity", "solidity", "local_contrast"}
    high_tail = {"intensity_variation"}
    multipliers = {
        "area": parameters.area_outlier_multiplier,
        "circularity": parameters.circularity_outlier_multiplier,
        "solidity": parameters.solidity_outlier_multiplier,
        "local_contrast": parameters.local_contrast_outlier_multiplier,
        "intensity_variation": parameters.intensity_variation_outlier_multiplier,
    }

    for pool_id, values in metric_values.items():
        if enabled["area"] and not (
            parameters.minimum_area_px <= values["area"] <= parameters.maximum_area_px
        ):
            flag(pool_id, FLAG_AREA)
        if (
            enabled["circularity"]
            and parameters.minimum_circularity is not None
            and values["circularity"] < parameters.minimum_circularity
        ):
            flag(pool_id, FLAG_CIRCULARITY)
        if (
            enabled["solidity"]
            and parameters.minimum_solidity is not None
            and values["solidity"] < parameters.minimum_solidity
        ):
            flag(pool_id, FLAG_SOLIDITY)
        if (
            enabled["local_contrast"]
            and parameters.minimum_local_contrast is not None
            and np.isfinite(values["local_contrast"])
            and values["local_contrast"] < parameters.minimum_local_contrast
        ):
            flag(pool_id, FLAG_LOCAL_CONTRAST)
        if (
            enabled["intensity_variation"]
            and parameters.maximum_intensity_variation is not None
            and values["intensity_variation"] > parameters.maximum_intensity_variation
        ):
            flag(pool_id, FLAG_INTENSITY_VARIATION)

    if parameters.baseline_mode == "manual" or len(metric_values) < 5:
        return {pool_id: tuple(reasons_) for pool_id, reasons_ in flags.items()}

    for metric, is_enabled in enabled.items():
        if not is_enabled:
            continue
        pool_values = [
            (pool_id, float(values[metric]))
            for pool_id, values in metric_values.items()
            if np.isfinite(values[metric])
        ]
        if len(pool_values) < 5:
            continue
        sample = np.asarray([value for _, value in pool_values], dtype=np.float64)
        if parameters.baseline_mode == "robust":
            center = float(np.median(sample))
            scale = float(1.4826 * np.median(np.abs(sample - center)))
        else:
            center = float(np.mean(sample))
            scale = float(np.std(sample, ddof=1))
        if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
            continue
        lower = center - multipliers[metric] * scale
        upper = center + multipliers[metric] * scale
        for pool_id, value in pool_values:
            outlier = (
                value < lower
                if metric in low_tail
                else value > upper
                if metric in high_tail
                else value < lower or value > upper
            )
            if outlier:
                flag(pool_id, reasons[metric])

    return {pool_id: tuple(reasons_) for pool_id, reasons_ in flags.items()}


def labels_to_rgba(
    labels: NDArray[np.int32],
    selected_ids: tuple[int, ...] = (),
    warning_ids: tuple[int, ...] = (),
) -> NDArray[np.uint8]:
    """Create a deterministic translucent overlay for accepted pool masks."""
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
    boundaries = _label_boundaries(labels)
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


def flags_to_rgba(
    candidate_labels: NDArray[np.int32],
    flags_by_pool: Mapping[int, tuple[str, ...]],
) -> NDArray[np.uint8]:
    """Create the optional review overlay using a stable colour per flag class."""
    rgba = np.zeros((*candidate_labels.shape, 4), dtype=np.uint8)
    if not flags_by_pool:
        return rgba
    priority = (
        (FLAG_AUTO_SPLIT, (220, 64, 255)),
        (FLAG_LOCAL_CONTRAST, (255, 48, 48)),
        (FLAG_CIRCULARITY, (255, 145, 32)),
        (FLAG_SOLIDITY, (255, 145, 32)),
        (FLAG_INTENSITY_VARIATION, (255, 145, 32)),
        (FLAG_AREA, (64, 150, 255)),
    )
    for pool_id, flags in flags_by_pool.items():
        mask = candidate_labels == pool_id
        if not np.any(mask):
            continue
        colour = next(
            (value for name, value in priority if name in flags), (255, 255, 255)
        )
        rgba[mask, :3] = colour
        rgba[mask, 3] = 115
        boundary = _binary_boundary(mask)
        rgba[boundary, :3] = (255, 255, 255) if len(flags) > 1 else colour
        rgba[boundary, 3] = 255
    return rgba


def quality_review_to_rgba(
    candidate_labels: NDArray[np.int32],
    validity_by_pool: Mapping[int, bool],
    quality_flags_by_pool: Mapping[int, tuple[str, ...]],
    review_pool_ids: frozenset[int] | set[int],
) -> NDArray[np.uint8]:
    """Render every reviewable candidate according to its current validity."""
    rgba = np.zeros((*candidate_labels.shape, 4), dtype=np.uint8)
    invalid_priority = (
        (FLAG_LOCAL_CONTRAST, (255, 48, 48)),
        (FLAG_CIRCULARITY, (255, 145, 32)),
        (FLAG_SOLIDITY, (255, 145, 32)),
        (FLAG_INTENSITY_VARIATION, (255, 145, 32)),
        (FLAG_AREA, (64, 150, 255)),
    )
    for pool_id in sorted(review_pool_ids):
        mask = candidate_labels == pool_id
        if not np.any(mask):
            continue
        flags = quality_flags_by_pool.get(pool_id, ())
        if validity_by_pool.get(pool_id, False):
            colour = (48, 210, 96)
        else:
            colour = next(
                (value for name, value in invalid_priority if name in flags),
                (255, 48, 48),
            )
        rgba[mask, :3] = colour
        rgba[mask, 3] = 105
        boundary = _binary_boundary(mask)
        multiple_invalid_reasons = (
            not validity_by_pool.get(pool_id, False) and len(flags) > 1
        )
        rgba[boundary, :3] = (
            (255, 255, 255) if multiple_invalid_reasons else colour
        )
        rgba[boundary, 3] = 255
    return rgba


def split_review_to_rgba(
    candidate_labels: NDArray[np.int32],
    split_groups: tuple[SplitGroup, ...],
    merged_group_ids: frozenset[int] | set[int],
    review_group_ids: frozenset[int] | set[int],
) -> NDArray[np.uint8]:
    """Render original split proposals even after a group has been merged."""
    rgba = np.zeros((*candidate_labels.shape, 4), dtype=np.uint8)
    for group in split_groups:
        if group.group_id not in review_group_ids:
            continue
        colour = (
            (64, 200, 255)
            if group.group_id in merged_group_ids
            else (220, 64, 255)
        )
        for pool_id in group.pool_ids:
            mask = candidate_labels == pool_id
            if not np.any(mask):
                continue
            rgba[mask, :3] = colour
            rgba[mask, 3] = 105
            boundary = _binary_boundary(mask)
            rgba[boundary, :3] = (255, 255, 255)
            rgba[boundary, 3] = 255
    return rgba


def _label_boundaries(labels: NDArray[np.int32]) -> NDArray[np.bool_]:
    boundaries = np.zeros(labels.shape, dtype=bool)
    boundaries[1:, :] |= (labels[1:, :] > 0) & (labels[1:, :] != labels[:-1, :])
    boundaries[:-1, :] |= (labels[:-1, :] > 0) & (labels[:-1, :] != labels[1:, :])
    boundaries[:, 1:] |= (labels[:, 1:] > 0) & (labels[:, 1:] != labels[:, :-1])
    boundaries[:, :-1] |= (labels[:, :-1] > 0) & (labels[:, :-1] != labels[:, 1:])
    return boundaries


def _binary_boundary(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    return mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
