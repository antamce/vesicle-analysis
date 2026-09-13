from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage.draw import ellipse, line, polygon
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.morphology import dilation, disk
from skimage.segmentation import watershed

from vesicle_analysis.analysis import RectangleBounds
from vesicle_analysis.segmentation import SegmentationParameters


class LabelOverlapError(ValueError):
    """Raised when an edit would overwrite an unselected pool."""


@dataclass(frozen=True, slots=True)
class MaskEdit:
    flat_indices: NDArray[np.int64]
    before: NDArray[np.int32]
    after: NDArray[np.int32]
    description: str

    def apply(self, labels: NDArray[np.int32], *, undo: bool = False) -> None:
        labels.ravel()[self.flat_indices] = self.before if undo else self.after


def make_edit(
    before: NDArray[np.int32], after: NDArray[np.int32], description: str
) -> MaskEdit:
    if before.shape != after.shape:
        raise ValueError("Correction masks must have matching shapes.")
    changed = np.flatnonzero(before.ravel() != after.ravel()).astype(
        np.int64, copy=False
    )
    return MaskEdit(
        flat_indices=changed,
        before=before.ravel()[changed].copy(),
        after=after.ravel()[changed].copy(),
        description=description,
    )


def polygon_mask(
    shape: tuple[int, int], points: tuple[tuple[float, float], ...]
) -> NDArray[np.bool_]:
    if len(points) < 3:
        raise ValueError("Draw at least three points for a boundary.")
    xx = np.asarray([point[0] for point in points], dtype=np.float64)
    yy = np.asarray([point[1] for point in points], dtype=np.float64)
    rows, columns = polygon(yy, xx, shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rows, columns] = True
    return mask


def ellipse_mask(
    shape: tuple[int, int], bounds: tuple[float, float, float, float]
) -> NDArray[np.bool_]:
    left, top, right, bottom = bounds
    if right <= left or bottom <= top:
        raise ValueError("The oval must have positive width and height.")
    center_y = (top + bottom) / 2
    center_x = (left + right) / 2
    radius_y = (bottom - top) / 2
    radius_x = (right - left) / 2
    rows, columns = ellipse(center_y, center_x, radius_y, radius_x, shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rows, columns] = True
    return mask


def analysis_region_mask(
    shape: tuple[int, int], rectangles: tuple[RectangleBounds, ...]
) -> NDArray[np.bool_]:
    allowed = np.zeros(shape, dtype=bool)
    for rectangle in rectangles:
        allowed[rectangle.y0 : rectangle.y1, rectangle.x0 : rectangle.x1] = True
    return allowed


def apply_shape(
    labels: NDArray[np.int32],
    mask: NDArray[np.bool_],
    *,
    replace_ids: tuple[int, ...] = (),
    allow_overwrite: bool = False,
) -> NDArray[np.int32]:
    if labels.shape != mask.shape:
        raise ValueError("The new boundary does not match the label-mask shape.")
    if not np.any(mask):
        raise ValueError("The drawn boundary contains no pixels.")

    replace = {int(pool_id) for pool_id in replace_ids if pool_id > 0}
    overlapped = {int(value) for value in np.unique(labels[mask]) if value > 0}
    conflicts = overlapped - replace
    if conflicts and not allow_overwrite:
        conflict_text = ", ".join(str(value) for value in sorted(conflicts))
        raise LabelOverlapError(f"The boundary overlaps pool(s) {conflict_text}.")

    corrected = labels.copy()
    if replace:
        corrected[np.isin(corrected, tuple(replace))] = 0
        target_id = min(replace)
    else:
        target_id = int(corrected.max()) + 1
    corrected[mask] = target_id
    return corrected


def delete_pools(
    labels: NDArray[np.int32], pool_ids: tuple[int, ...]
) -> NDArray[np.int32]:
    corrected = labels.copy()
    corrected[np.isin(corrected, pool_ids)] = 0
    return corrected


def merge_pools(
    labels: NDArray[np.int32], pool_ids: tuple[int, ...]
) -> NDArray[np.int32]:
    selected = tuple(sorted({int(value) for value in pool_ids if value > 0}))
    if len(selected) < 2:
        raise ValueError("Select at least two pools to merge.")
    corrected = labels.copy()
    corrected[np.isin(corrected, selected)] = selected[0]
    return corrected


def polyline_mask(
    shape: tuple[int, int],
    points: tuple[tuple[float, float], ...],
    *,
    radius: int = 1,
) -> NDArray[np.bool_]:
    if not points:
        raise ValueError("Draw a line or click a location first.")
    mask = np.zeros(shape, dtype=bool)
    rounded = [
        (
            int(np.clip(round(y), 0, shape[0] - 1)),
            int(np.clip(round(x), 0, shape[1] - 1)),
        )
        for x, y in points
    ]
    if len(rounded) == 1:
        mask[rounded[0]] = True
    else:
        for start, end in pairwise(rounded):
            rows, columns = line(start[0], start[1], end[0], end[1])
            mask[rows, columns] = True
    if radius > 0:
        mask = dilation(mask, footprint=disk(radius))
    return mask


def split_pool_with_line(
    labels: NDArray[np.int32],
    pool_id: int,
    points: tuple[tuple[float, float], ...],
) -> NDArray[np.int32]:
    original = labels == pool_id
    if not np.any(original):
        raise ValueError(f"Pool {pool_id} no longer exists.")
    barrier = polyline_mask(labels.shape, points, radius=1)
    components, count = ndimage.label(original & ~barrier, structure=np.ones((3, 3)))
    component_ids = [
        component_id
        for component_id in range(1, count + 1)
        if np.count_nonzero(components == component_id) >= 2
    ]
    if len(component_ids) < 2:
        raise ValueError("The line must cross the selected pool from edge to edge.")

    markers = np.zeros(labels.shape, dtype=np.int32)
    for marker_id, component_id in enumerate(component_ids, start=1):
        markers[components == component_id] = marker_id
    divided = watershed(np.zeros(labels.shape, dtype=np.uint8), markers, mask=original)
    ordered = sorted(
        range(1, len(component_ids) + 1),
        key=lambda value: int(np.count_nonzero(divided == value)),
        reverse=True,
    )
    corrected = labels.copy()
    corrected[original] = 0
    next_id = int(labels.max()) + 1
    for index, component_id in enumerate(ordered):
        assigned_id = pool_id if index == 0 else next_id
        corrected[divided == component_id] = assigned_id
        if index > 0:
            next_id += 1
    return corrected


def guided_pool_mask(
    frame: NDArray[np.generic],
    points: tuple[tuple[float, float], ...],
    rectangle: RectangleBounds,
    parameters: SegmentationParameters,
) -> NDArray[np.bool_]:
    """Use a foreground hint as a forced marker in a local watershed."""
    if frame.ndim != 2:
        raise ValueError("Guided resegmentation requires a 2D peak frame.")
    hint = polyline_mask(frame.shape, points, radius=2)
    hint &= analysis_region_mask(frame.shape, (rectangle,))
    if not np.any(hint):
        raise ValueError("Draw the hint inside an analysis rectangle.")

    hint_y, hint_x = np.nonzero(hint)
    padding = max(16, round(parameters.background_sigma * 3))
    x0 = max(rectangle.x0, int(hint_x.min()) - padding)
    y0 = max(rectangle.y0, int(hint_y.min()) - padding)
    x1 = min(rectangle.x1, int(hint_x.max()) + padding + 1)
    y1 = min(rectangle.y1, int(hint_y.max()) + padding + 1)
    crop = np.asarray(frame[y0:y1, x0:x1], dtype=np.float32)
    local_hint = hint[y0:y1, x0:x1]

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
        noise = max(float(np.std(enhanced, dtype=np.float64)), 1.0)

    foreground = enhanced > (median + parameters.foreground_threshold_sigma * noise)
    foreground |= local_hint
    markers = np.zeros(crop.shape, dtype=np.int32)
    hint_coordinates = np.column_stack(np.nonzero(local_hint))
    forced = hint_coordinates[len(hint_coordinates) // 2]
    markers[tuple(forced)] = 1

    automatic_seeds = peak_local_max(
        enhanced,
        min_distance=parameters.minimum_seed_distance,
        threshold_abs=median + parameters.seed_threshold_sigma * noise,
        labels=foreground,
        exclude_border=False,
    )
    next_marker = 2
    for seed in automatic_seeds:
        if np.min(np.sum((hint_coordinates - seed) ** 2, axis=1)) <= (
            parameters.minimum_seed_distance**2
        ):
            continue
        markers[tuple(seed)] = next_marker
        next_marker += 1

    local_labels = watershed(-enhanced, markers, mask=foreground)
    candidate = local_labels == 1
    if not np.any(candidate):
        raise ValueError("The hint did not produce a pool boundary.")
    result = np.zeros(frame.shape, dtype=bool)
    result[y0:y1, x0:x1] = candidate
    return result
