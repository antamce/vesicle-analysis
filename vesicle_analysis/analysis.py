from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import nd2
import numpy as np
from numpy.typing import NDArray

from vesicle_analysis.nd2_io import _validate_dimensions, _validated_path

if TYPE_CHECKING:
    from vesicle_analysis.motion_correction import MotionCorrectionResult


@dataclass(frozen=True, slots=True)
class RectangleBounds:
    """A full-image rectangle using zero-based, half-open pixel bounds."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x0 < 0 or self.y0 < 0:
            raise ValueError("Rectangle coordinates cannot be negative.")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("Rectangle must have positive width and height.")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    def overlaps(self, other: RectangleBounds) -> bool:
        return max(self.x0, other.x0) < min(self.x1, other.x1) and max(
            self.y0, other.y0
        ) < min(self.y1, other.y1)


@dataclass(frozen=True, slots=True)
class PeakDetectionResult:
    automatic_peak_index: int
    recording_mean_trace: NDArray[np.float64]
    union_pixel_count: int


def clip_rectangle(
    left: float,
    top: float,
    right: float,
    bottom: float,
    *,
    image_width: int,
    image_height: int,
) -> RectangleBounds:
    """Snap a display rectangle outward to valid half-open pixel bounds."""
    x0 = max(0, min(image_width, int(np.floor(min(left, right)))))
    y0 = max(0, min(image_height, int(np.floor(min(top, bottom)))))
    x1 = max(0, min(image_width, int(np.ceil(max(left, right)))))
    y1 = max(0, min(image_height, int(np.ceil(max(top, bottom)))))
    return RectangleBounds(x0, y0, x1, y1)


def overlaps_any(
    candidate: RectangleBounds,
    rectangles: Iterable[RectangleBounds],
    *,
    excluded_index: int | None = None,
) -> bool:
    return any(
        index != excluded_index and candidate.overlaps(rectangle)
        for index, rectangle in enumerate(rectangles)
    )


def detect_recording_peak(
    path: Path,
    rectangles: Iterable[RectangleBounds],
    correction: MotionCorrectionResult | None = None,
) -> PeakDetectionResult:
    """Return the strict raw mean-intensity maximum over a rectangle union."""
    path = _validated_path(path)
    rectangles = tuple(rectangles)
    if not rectangles:
        raise ValueError("At least one analysis rectangle is required.")

    with nd2.ND2File(path) as recording:
        sizes = dict(recording.sizes)
        _validate_dimensions(sizes)
        height, width, frame_count = sizes["Y"], sizes["X"], sizes["T"]
        _validate_rectangles(rectangles, image_width=width, image_height=height)

        min_x = min(rectangle.x0 for rectangle in rectangles)
        min_y = min(rectangle.y0 for rectangle in rectangles)
        max_x = max(rectangle.x1 for rectangle in rectangles)
        max_y = max(rectangle.y1 for rectangle in rectangles)
        union = np.zeros((max_y - min_y, max_x - min_x), dtype=bool)
        for rectangle in rectangles:
            union[
                rectangle.y0 - min_y : rectangle.y1 - min_y,
                rectangle.x0 - min_x : rectangle.x1 - min_x,
            ] = True

        pixel_count = int(np.count_nonzero(union))
        if pixel_count == 0:
            raise ValueError("The analysis-region union contains no pixels.")

        means = np.empty(frame_count, dtype=np.float64)
        for frame_index in range(frame_count):
            frame = np.asarray(recording.read_frame(frame_index))
            if correction is not None:
                if (
                    correction.image_shape != frame.shape
                    or len(correction.transforms) != frame_count
                ):
                    raise ValueError(
                        "Motion correction does not match the ND2 recording."
                    )
                from vesicle_analysis.motion_correction import apply_frame_transform

                frame = apply_frame_transform(frame, correction.transforms[frame_index])
            crop = frame[min_y:max_y, min_x:max_x]
            means[frame_index] = np.sum(crop, where=union, dtype=np.float64)
        means /= pixel_count

    return PeakDetectionResult(
        automatic_peak_index=int(np.argmax(means)),
        recording_mean_trace=means,
        union_pixel_count=pixel_count,
    )


def _validate_rectangles(
    rectangles: Iterable[RectangleBounds], *, image_width: int, image_height: int
) -> None:
    for rectangle in rectangles:
        if rectangle.x1 > image_width or rectangle.y1 > image_height:
            raise ValueError(
                f"Rectangle {rectangle} exceeds {image_width} x {image_height} image bounds."
            )
