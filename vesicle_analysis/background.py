from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from vesicle_analysis.analysis import RectangleBounds
from vesicle_analysis.nd2_io import read_frame


@dataclass(frozen=True, slots=True)
class BackgroundParameters:
    mode: str = "pool-wise"
    pool_min_distance_px: int = 5
    pool_max_distance_px: int = 10
    common_size_px: int = 40
    common_clearance_px: int = 5

    def __post_init__(self) -> None:
        if self.mode not in {"pool-wise", "common"}:
            raise ValueError("Background mode must be 'pool-wise' or 'common'.")
        if self.pool_min_distance_px < 0:
            raise ValueError("Minimum pool-background distance cannot be negative.")
        if self.pool_max_distance_px < self.pool_min_distance_px:
            raise ValueError(
                "Maximum distance cannot be smaller than minimum distance."
            )
        if self.common_size_px < 1:
            raise ValueError("Common background size must be positive.")
        if self.common_clearance_px < 0:
            raise ValueError("Common-background clearance cannot be negative.")


@dataclass(frozen=True, slots=True)
class BackgroundRegion:
    background_id: int
    pool_id: int | None
    area_px: int
    mean_intensity: float
    intensity_std: float
    score: float


@dataclass(frozen=True, slots=True)
class BackgroundResult:
    labels: NDArray[np.int32]
    regions: tuple[BackgroundRegion, ...]
    missing_pool_ids: tuple[int, ...]
    common_missing: bool
    parameters: BackgroundParameters
    peak_frame_index: int


def find_recording_backgrounds(
    path: Path,
    peak_frame_index: int,
    pool_labels: NDArray[np.int32],
    rectangles: tuple[RectangleBounds, ...],
    parameters: BackgroundParameters,
) -> BackgroundResult:
    frame = read_frame(path, peak_frame_index)
    return find_backgrounds(
        frame,
        pool_labels,
        rectangles,
        parameters,
        peak_frame_index=peak_frame_index,
    )


def find_backgrounds(
    frame: NDArray[np.generic],
    pool_labels: NDArray[np.int32],
    rectangles: tuple[RectangleBounds, ...],
    parameters: BackgroundParameters,
    *,
    peak_frame_index: int,
) -> BackgroundResult:
    if frame.ndim != 2 or pool_labels.shape != frame.shape:
        raise ValueError("Background detection requires matching 2D frame and labels.")
    if not rectangles:
        raise ValueError("At least one analysis rectangle is required.")

    allowed = _analysis_mask(frame.shape, rectangles)
    values = np.asarray(frame[allowed], dtype=np.float64)
    if not values.size:
        raise ValueError("Analysis rectangles contain no pixels.")
    target_mean = float(np.mean(values))
    median = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - median)))
    if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
        scale = max(float(np.std(values)), 1.0)

    labels = np.zeros(frame.shape, dtype=np.int32)
    regions: list[BackgroundRegion] = []
    missing: list[int] = []
    occupied_pools = pool_labels > 0
    distance_to_pool = ndimage.distance_transform_edt(~occupied_pools)

    if parameters.mode == "pool-wise":
        pool_ids = [int(value) for value in np.unique(pool_labels) if value > 0]
        for pool_id in pool_ids:
            candidate = _best_pool_candidate(
                frame,
                pool_labels,
                pool_id,
                rectangles,
                labels > 0,
                distance_to_pool,
                target_mean,
                scale,
                parameters,
            )
            if candidate is None:
                missing.append(pool_id)
                continue
            mask, score = candidate
            background_id = len(regions) + 1
            labels[mask] = background_id
            candidate_values = np.asarray(frame[mask], dtype=np.float64)
            regions.append(
                BackgroundRegion(
                    background_id=background_id,
                    pool_id=pool_id,
                    area_px=int(np.count_nonzero(mask)),
                    mean_intensity=float(np.mean(candidate_values)),
                    intensity_std=float(np.std(candidate_values)),
                    score=score,
                )
            )
        common_missing = False
    else:
        candidate = _best_common_candidate(
            frame,
            rectangles,
            distance_to_pool,
            target_mean,
            scale,
            parameters,
        )
        common_missing = candidate is None
        if candidate is not None:
            mask, score = candidate
            labels[mask] = 1
            candidate_values = np.asarray(frame[mask], dtype=np.float64)
            regions.append(
                BackgroundRegion(
                    background_id=1,
                    pool_id=None,
                    area_px=int(np.count_nonzero(mask)),
                    mean_intensity=float(np.mean(candidate_values)),
                    intensity_std=float(np.std(candidate_values)),
                    score=score,
                )
            )

    return BackgroundResult(
        labels=labels,
        regions=tuple(regions),
        missing_pool_ids=tuple(missing),
        common_missing=common_missing,
        parameters=parameters,
        peak_frame_index=peak_frame_index,
    )


def _analysis_mask(
    shape: tuple[int, int], rectangles: tuple[RectangleBounds, ...]
) -> NDArray[np.bool_]:
    allowed = np.zeros(shape, dtype=bool)
    for rectangle in rectangles:
        if rectangle.x1 > shape[1] or rectangle.y1 > shape[0]:
            raise ValueError(f"Rectangle {rectangle} exceeds the frame bounds.")
        allowed[rectangle.y0 : rectangle.y1, rectangle.x0 : rectangle.x1] = True
    return allowed


def _best_pool_candidate(
    frame: NDArray[np.generic],
    pool_labels: NDArray[np.int32],
    pool_id: int,
    rectangles: tuple[RectangleBounds, ...],
    used_background: NDArray[np.bool_],
    distance_to_pool: NDArray[np.float64],
    target_mean: float,
    scale: float,
    parameters: BackgroundParameters,
) -> tuple[NDArray[np.bool_], float] | None:
    yy, xx = np.nonzero(pool_labels == pool_id)
    if not len(yy):
        return None
    center_x, center_y = float(np.mean(xx)), float(np.mean(yy))
    owner = next(
        (
            rectangle
            for rectangle in rectangles
            if rectangle.x0 <= center_x < rectangle.x1
            and rectangle.y0 <= center_y < rectangle.y1
        ),
        None,
    )
    if owner is None:
        return None

    best: tuple[NDArray[np.bool_], float] | None = None
    gaps = range(
        parameters.pool_min_distance_px,
        parameters.pool_max_distance_px + 1,
    )
    for angle_degrees in range(0, 360, 15):
        angle = np.deg2rad(angle_degrees)
        direction_x, direction_y = float(np.cos(angle)), float(np.sin(angle))
        projection = xx * direction_x + yy * direction_y
        projected_width = float(np.max(projection) - np.min(projection))
        for gap in gaps:
            shift = projected_width + gap + 1
            dx = round(direction_x * shift)
            dy = round(direction_y * shift)
            moved_x, moved_y = xx + dx, yy + dy
            if (
                moved_x.min() < owner.x0
                or moved_x.max() >= owner.x1
                or moved_y.min() < owner.y0
                or moved_y.max() >= owner.y1
            ):
                continue
            if np.any(pool_labels[moved_y, moved_x] > 0) or np.any(
                used_background[moved_y, moved_x]
            ):
                continue
            clearance = float(np.min(distance_to_pool[moved_y, moved_x]))
            if not (
                parameters.pool_min_distance_px
                <= clearance
                <= parameters.pool_max_distance_px + 1
            ):
                continue
            mask = np.zeros(frame.shape, dtype=bool)
            mask[moved_y, moved_x] = True
            score = _candidate_score(
                np.asarray(frame[moved_y, moved_x], dtype=np.float64),
                target_mean,
                scale,
            )
            distance_penalty = abs(
                clearance
                - (parameters.pool_min_distance_px + parameters.pool_max_distance_px)
                / 2
            ) / max(parameters.pool_max_distance_px + 1, 1)
            score += 0.1 * distance_penalty
            if best is None or score < best[1]:
                best = mask, score
    return best


def _best_common_candidate(
    frame: NDArray[np.generic],
    rectangles: tuple[RectangleBounds, ...],
    distance_to_pool: NDArray[np.float64],
    target_mean: float,
    scale: float,
    parameters: BackgroundParameters,
) -> tuple[NDArray[np.bool_], float] | None:
    size = parameters.common_size_px
    step = max(2, size // 8)
    best: tuple[NDArray[np.bool_], float] | None = None
    for rectangle in rectangles:
        if rectangle.width < size or rectangle.height < size:
            continue
        x_positions = _scan_positions(rectangle.x0, rectangle.x1 - size, step)
        y_positions = _scan_positions(rectangle.y0, rectangle.y1 - size, step)
        for y0 in y_positions:
            for x0 in x_positions:
                region = np.s_[y0 : y0 + size, x0 : x0 + size]
                if np.min(distance_to_pool[region]) <= parameters.common_clearance_px:
                    continue
                values = np.asarray(frame[region], dtype=np.float64).ravel()
                score = _candidate_score(values, target_mean, scale)
                if best is None or score < best[1]:
                    mask = np.zeros(frame.shape, dtype=bool)
                    mask[region] = True
                    best = mask, score
    return best


def _scan_positions(start: int, stop: int, step: int) -> tuple[int, ...]:
    positions = list(range(start, stop + 1, step))
    if not positions or positions[-1] != stop:
        positions.append(stop)
    return tuple(positions)


def _candidate_score(
    values: NDArray[np.float64], target_mean: float, scale: float
) -> float:
    mean = float(np.mean(values))
    variation = float(np.std(values)) / scale
    high_mean = max(0.0, mean - target_mean) / scale
    mean_distance = abs(mean - target_mean) / scale
    bright_threshold = target_mean + 2.5 * scale
    bright_fraction = float(np.mean(values > bright_threshold))
    bright_outlier = (
        max(0.0, float(np.percentile(values, 99)) - bright_threshold) / scale
    )
    return (
        3.0 * high_mean
        + 0.15 * mean_distance
        + variation
        + 8.0 * bright_fraction
        + 0.5 * bright_outlier
    )


def background_labels_to_rgba(
    labels: NDArray[np.int32], selected_ids: tuple[int, ...] = ()
) -> NDArray[np.uint8]:
    rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
    foreground = labels > 0
    rgba[foreground, :3] = (75, 255, 120)
    rgba[foreground, 3] = 45
    boundaries = np.zeros(labels.shape, dtype=bool)
    boundaries[1:] |= foreground[1:] & (labels[1:] != labels[:-1])
    boundaries[:-1] |= foreground[:-1] & (labels[:-1] != labels[1:])
    boundaries[:, 1:] |= foreground[:, 1:] & (labels[:, 1:] != labels[:, :-1])
    boundaries[:, :-1] |= foreground[:, :-1] & (labels[:, :-1] != labels[:, 1:])
    rgba[boundaries, :3] = (80, 255, 120)
    rgba[boundaries, 3] = 240
    if selected_ids:
        selected = np.isin(labels, selected_ids)
        rgba[selected, :3] = (255, 90, 220)
        rgba[selected, 3] = 110
    return rgba
