from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import nd2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from vesicle_analysis.motion_correction import MotionCorrectionResult


class UnsupportedRecordingError(ValueError):
    """Raised when an ND2 file does not match the supported data contract."""


@dataclass(frozen=True, slots=True)
class RecordingMetadata:
    path: Path
    frame_count: int
    height: int
    width: int
    dtype: str
    pixel_size_x_um: float
    pixel_size_y_um: float
    timestamps_s: tuple[float, ...] | None
    exposure_ms: float | None
    file_size_bytes: int

    @property
    def duration_s(self) -> float | None:
        if not self.timestamps_s or len(self.timestamps_s) < 2:
            return None
        return self.timestamps_s[-1] - self.timestamps_s[0]

    @property
    def median_interval_s(self) -> float | None:
        if not self.timestamps_s or len(self.timestamps_s) < 2:
            return None
        return float(np.median(np.diff(self.timestamps_s)))

    def timestamp_for_frame(self, index: int) -> float | None:
        if self.timestamps_s is None:
            return None
        return self.timestamps_s[index]


def discover_nd2_files(directory: Path) -> list[Path]:
    """Return ND2 files immediately inside *directory*, in stable name order."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.casefold() == ".nd2"
        ),
        key=lambda path: path.name.casefold(),
    )


def read_metadata(path: Path) -> RecordingMetadata:
    path = _validated_path(path)
    with nd2.ND2File(path) as recording:
        sizes = dict(recording.sizes)
        _validate_dimensions(sizes)
        timestamps, exposure_ms = _read_events(recording, sizes["T"])
        voxel = recording.voxel_size()
        return RecordingMetadata(
            path=path,
            frame_count=sizes["T"],
            height=sizes["Y"],
            width=sizes["X"],
            dtype=str(recording.dtype),
            pixel_size_x_um=float(voxel.x),
            pixel_size_y_um=float(voxel.y),
            timestamps_s=timestamps,
            exposure_ms=exposure_ms,
            file_size_bytes=path.stat().st_size,
        )


def read_frame(
    path: Path,
    index: int,
    correction: MotionCorrectionResult | None = None,
) -> NDArray[np.generic]:
    path = _validated_path(path)
    with nd2.ND2File(path) as recording:
        sizes = dict(recording.sizes)
        _validate_dimensions(sizes)
        if not 0 <= index < sizes["T"]:
            raise IndexError(
                f"Frame {index + 1} is outside the recording's 1-{sizes['T']} range."
            )
        # Some ND2 backends return a view into memory owned by the open file.
        # The GUI consumes frames after this context closes, so take ownership.
        frame = np.array(recording.read_frame(index), copy=True)
    if frame.shape != (sizes["Y"], sizes["X"]):
        raise UnsupportedRecordingError(
            f"Expected a 2D frame, received shape {frame.shape!r}."
        )
    if correction is not None:
        if correction.image_shape != frame.shape:
            raise ValueError("Motion correction dimensions do not match the ND2 frame.")
        if len(correction.transforms) != sizes["T"]:
            raise ValueError(
                "Motion correction frame count does not match the ND2 file."
            )
        from vesicle_analysis.motion_correction import apply_frame_transform

        frame = apply_frame_transform(frame, correction.transforms[index])
    return frame


def _validated_path(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.casefold() != ".nd2":
        raise UnsupportedRecordingError(f"Not an ND2 file: {path.name}")
    return path


def _validate_dimensions(sizes: dict[str, int]) -> None:
    required = {"T", "Y", "X"}
    if not required.issubset(sizes):
        raise UnsupportedRecordingError(
            f"Recording dimensions must include T, Y, and X; found {sizes}."
        )
    unsupported = {
        axis: size for axis, size in sizes.items() if axis not in required and size != 1
    }
    if unsupported:
        raise UnsupportedRecordingError(
            "Only one-channel 2D time series are supported in this version; "
            f"found extra dimensions {unsupported}."
        )


def _read_events(
    recording: nd2.ND2File, frame_count: int
) -> tuple[tuple[float, ...] | None, float | None]:
    try:
        events = recording.events(orient="records")
    # Event metadata is optional. Reader/parser failures must not prevent image
    # viewing when the pixel data itself remains valid.
    except Exception:  # noqa: BLE001
        return None, None

    timestamps: tuple[float, ...] | None = None
    if len(events) == frame_count and all("Time [s]" in event for event in events):
        values = np.asarray([event["Time [s]"] for event in events], dtype=float)
        if np.all(np.isfinite(values)) and np.all(np.diff(values) >= 0):
            timestamps = tuple(float(value) for value in values)

    exposures = [
        float(event["Exposure Time [ms]"])
        for event in events
        if "Exposure Time [ms]" in event
        and np.isfinite(float(event["Exposure Time [ms]"]))
    ]
    exposure_ms = float(np.median(exposures)) if exposures else None
    return timestamps, exposure_ms
