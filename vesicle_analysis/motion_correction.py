from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import nd2
import numpy as np
import tifffile
from numpy.typing import NDArray
from scipy.ndimage import rotate
from skimage.filters import gaussian
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar

from vesicle_analysis.nd2_io import (
    RecordingMetadata,
    _validate_dimensions,
    _validated_path,
)

CORRECTION_MODES = ("translation", "rigid")


@dataclass(frozen=True, slots=True)
class MotionCorrectionParameters:
    maximum_translation_px: int = 200
    maximum_rotation_degrees: float = 15.0
    registration_max_size_px: int = 512
    rotation_bins: int = 720
    qc_minimum_correlation: float = 0.20

    def __post_init__(self) -> None:
        if self.maximum_translation_px < 1:
            raise ValueError("Maximum translation must be positive.")
        if not 0 < self.maximum_rotation_degrees <= 90:
            raise ValueError("Maximum rotation must be between 0 and 90 degrees.")
        if self.registration_max_size_px < 64:
            raise ValueError("Registration size must be at least 64 pixels.")
        if self.rotation_bins < 180:
            raise ValueError("Rotation sampling must contain at least 180 bins.")
        if not -1 <= self.qc_minimum_correlation <= 1:
            raise ValueError("QC correlation threshold must be between -1 and 1.")


@dataclass(frozen=True, slots=True)
class FrameTransform:
    translation_x_px: int = 0
    translation_y_px: int = 0
    rotation_degrees: float = 0.0
    correlation: float = 1.0
    warning: str | None = None
    manually_adjusted: bool = False


@dataclass(frozen=True, slots=True)
class MotionCorrectionResult:
    mode: str
    reference_frame_index: int
    image_shape: tuple[int, int]
    transforms: tuple[FrameTransform, ...]
    parameters: MotionCorrectionParameters

    def __post_init__(self) -> None:
        if self.mode not in CORRECTION_MODES:
            raise ValueError(f"Unsupported correction mode: {self.mode}")
        if self.reference_frame_index != 0:
            raise ValueError("Motion correction currently uses the first frame.")
        if not self.transforms:
            raise ValueError("At least one frame transform is required.")

    @property
    def warning_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, transform in enumerate(self.transforms)
            if transform.warning
        )


@dataclass(frozen=True, slots=True)
class CorrectedRecordingExportResult:
    ome_tiff_path: Path
    transforms_csv_path: Path
    metadata_json_path: Path
    frame_count: int


@dataclass(frozen=True, slots=True)
class _PreparedReference:
    raw: NDArray[np.generic]
    registration: NDArray[np.float32]
    polar_spectrum: NDArray[np.float64] | None


def apply_frame_transform(
    frame: NDArray[np.generic], transform: FrameTransform
) -> NDArray[np.generic]:
    """Apply nearest-neighbor rotation and an integer translation.

    Output pixels are copied from input pixels without intensity interpolation.
    Pixels outside the source field are filled with numeric zero.
    """
    frame = np.asarray(frame)
    if frame.ndim != 2:
        raise ValueError(f"Expected a 2D frame, received shape {frame.shape!r}.")
    if transform.rotation_degrees:
        rotated = rotate(
            frame,
            angle=transform.rotation_degrees,
            reshape=False,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        )
    else:
        rotated = frame
    return translate_integer(
        rotated,
        transform.translation_x_px,
        transform.translation_y_px,
    )


def translate_integer(
    frame: NDArray[np.generic], translation_x_px: int, translation_y_px: int
) -> NDArray[np.generic]:
    """Translate a 2D image by whole pixels with zero-filled borders."""
    frame = np.asarray(frame)
    if frame.ndim != 2:
        raise ValueError(f"Expected a 2D frame, received shape {frame.shape!r}.")
    dx, dy = int(translation_x_px), int(translation_y_px)
    output = np.zeros_like(frame)
    height, width = frame.shape
    if abs(dx) >= width or abs(dy) >= height:
        return output
    source_x0, source_x1 = max(0, -dx), min(width, width - dx)
    source_y0, source_y1 = max(0, -dy), min(height, height - dy)
    target_x0, target_x1 = source_x0 + dx, source_x1 + dx
    target_y0, target_y1 = source_y0 + dy, source_y1 + dy
    output[target_y0:target_y1, target_x0:target_x1] = frame[
        source_y0:source_y1, source_x0:source_x1
    ]
    return output


def estimate_frame_transform(
    reference: NDArray[np.generic],
    moving: NDArray[np.generic],
    mode: str,
    parameters: MotionCorrectionParameters | None = None,
) -> FrameTransform:
    """Estimate one moving-to-reference transform without subpixel shifts."""
    parameters = parameters or MotionCorrectionParameters()
    if mode not in CORRECTION_MODES:
        raise ValueError(f"Unsupported correction mode: {mode}")
    reference = np.asarray(reference)
    moving = np.asarray(moving)
    if reference.ndim != 2 or moving.shape != reference.shape:
        raise ValueError("Registration requires matching two-dimensional frames.")
    prepared = _prepare_reference(reference, mode, parameters)
    return _estimate_frame_transform_prepared(prepared, moving, mode, parameters)


def _estimate_frame_transform_prepared(
    prepared: _PreparedReference,
    moving: NDArray[np.generic],
    mode: str,
    parameters: MotionCorrectionParameters,
) -> FrameTransform:
    moving = np.asarray(moving)

    warnings: list[str] = []
    correction_angle = 0.0
    if mode == "rigid":
        if prepared.polar_spectrum is None:
            raise ValueError("Rigid registration reference was not prepared.")
        try:
            measured_angle = _estimate_rotation(
                prepared.polar_spectrum, moving, parameters
            )
        except (FloatingPointError, ValueError) as error:
            warnings.append(f"rotation estimate failed ({error})")
            measured_angle = 0.0
        if abs(measured_angle) > parameters.maximum_rotation_degrees:
            warnings.append(
                f"estimated rotation {measured_angle:.1f}° exceeds the "
                f"±{parameters.maximum_rotation_degrees:g}° limit"
            )
            measured_angle = float(
                np.clip(
                    measured_angle,
                    -parameters.maximum_rotation_degrees,
                    parameters.maximum_rotation_degrees,
                )
            )
        correction_angle = -measured_angle

    rotation_transform = FrameTransform(rotation_degrees=correction_angle)
    rotated = apply_frame_transform(moving, rotation_transform)
    moving_registration = _registration_image(rotated)
    try:
        shift, _, _ = phase_cross_correlation(
            prepared.registration,
            moving_registration,
            upsample_factor=1,
            normalization="phase",
        )
        if not np.all(np.isfinite(shift)):
            raise ValueError("non-finite shift")
        dy, dx = round(float(shift[0])), round(float(shift[1]))
    except (FloatingPointError, ValueError) as error:
        warnings.append(f"translation estimate failed ({error})")
        dx, dy = 0, 0
    if (
        abs(dx) > parameters.maximum_translation_px
        or abs(dy) > parameters.maximum_translation_px
    ):
        warnings.append(
            f"estimated translation ({dx}, {dy}) px exceeds the "
            f"±{parameters.maximum_translation_px} px limit"
        )
        dx = int(
            np.clip(
                dx,
                -parameters.maximum_translation_px,
                parameters.maximum_translation_px,
            )
        )
        dy = int(
            np.clip(
                dy,
                -parameters.maximum_translation_px,
                parameters.maximum_translation_px,
            )
        )

    provisional = FrameTransform(
        translation_x_px=dx,
        translation_y_px=dy,
        rotation_degrees=correction_angle,
    )
    corrected = apply_frame_transform(moving, provisional)
    correlation = _registration_correlation_prepared(prepared.registration, corrected)
    if not np.isfinite(correlation) or correlation < parameters.qc_minimum_correlation:
        warnings.append(
            f"low registration correlation ({correlation:.3f}; expected at least "
            f"{parameters.qc_minimum_correlation:.2f})"
        )
    return replace(
        provisional,
        correlation=float(correlation),
        warning="; ".join(warnings) if warnings else None,
    )


def estimate_motion_correction(
    path: Path,
    mode: str,
    parameters: MotionCorrectionParameters | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> MotionCorrectionResult:
    """Estimate direct-to-first-frame transforms for a complete ND2 recording."""
    parameters = parameters or MotionCorrectionParameters()
    if mode not in CORRECTION_MODES:
        raise ValueError(f"Unsupported correction mode: {mode}")
    path = _validated_path(path)
    with nd2.ND2File(path) as recording:
        sizes = dict(recording.sizes)
        _validate_dimensions(sizes)
        reference = np.array(recording.read_frame(0), copy=True)
        prepared = _prepare_reference(reference, mode, parameters)
        transforms = [FrameTransform()]
        if progress_callback is not None:
            progress_callback(1, sizes["T"])
        for frame_index in range(1, sizes["T"]):
            moving = np.asarray(recording.read_frame(frame_index))
            transforms.append(
                _estimate_frame_transform_prepared(prepared, moving, mode, parameters)
            )
            if progress_callback is not None:
                progress_callback(frame_index + 1, sizes["T"])
    transforms = _add_sequence_warnings(transforms)
    return MotionCorrectionResult(
        mode=mode,
        reference_frame_index=0,
        image_shape=reference.shape,
        transforms=tuple(transforms),
        parameters=parameters,
    )


def _add_sequence_warnings(
    transforms: list[FrameTransform],
) -> list[FrameTransform]:
    reviewed = list(transforms)
    for index in range(1, len(reviewed)):
        previous, current = reviewed[index - 1], reviewed[index]
        translation_jump = max(
            abs(current.translation_x_px - previous.translation_x_px),
            abs(current.translation_y_px - previous.translation_y_px),
        )
        rotation_jump = abs(current.rotation_degrees - previous.rotation_degrees)
        messages = [current.warning] if current.warning else []
        if translation_jump > 25:
            messages.append(f"abrupt translation change ({translation_jump} px)")
        if rotation_jump > 3.0:
            messages.append(f"abrupt rotation change ({rotation_jump:.1f}°)")
        if messages != ([current.warning] if current.warning else []):
            reviewed[index] = replace(current, warning="; ".join(messages))
    return reviewed


def replace_frame_transform(
    path: Path,
    result: MotionCorrectionResult,
    frame_index: int,
    *,
    translation_x_px: int,
    translation_y_px: int,
    rotation_degrees: float,
) -> MotionCorrectionResult:
    """Return a result with one manually adjusted transform and refreshed QC."""
    if not 0 <= frame_index < len(result.transforms):
        raise IndexError(frame_index)
    if frame_index == result.reference_frame_index and (
        translation_x_px or translation_y_px or rotation_degrees
    ):
        raise ValueError("Reference frame 1 must remain untransformed.")
    if (
        abs(translation_x_px) > result.parameters.maximum_translation_px
        or abs(translation_y_px) > result.parameters.maximum_translation_px
    ):
        raise ValueError("Manual translation exceeds the configured limit.")
    if result.mode == "translation" and rotation_degrees:
        raise ValueError("Translation-only correction cannot contain rotation.")
    if abs(rotation_degrees) > result.parameters.maximum_rotation_degrees:
        raise ValueError("Manual rotation exceeds the configured limit.")
    path = _validated_path(path)
    with nd2.ND2File(path) as recording:
        reference = np.asarray(recording.read_frame(result.reference_frame_index))
        moving = np.asarray(recording.read_frame(frame_index))
    adjusted = FrameTransform(
        translation_x_px=int(translation_x_px),
        translation_y_px=int(translation_y_px),
        rotation_degrees=float(rotation_degrees),
        manually_adjusted=True,
    )
    correlation = registration_correlation(
        reference, apply_frame_transform(moving, adjusted)
    )
    warning = None
    if correlation < result.parameters.qc_minimum_correlation:
        warning = (
            f"low registration correlation ({correlation:.3f}; expected at least "
            f"{result.parameters.qc_minimum_correlation:.2f})"
        )
    adjusted = replace(adjusted, correlation=float(correlation), warning=warning)
    transforms = list(result.transforms)
    transforms[frame_index] = adjusted
    return replace(result, transforms=tuple(transforms))


def registration_correlation(
    reference: NDArray[np.generic], corrected: NDArray[np.generic]
) -> float:
    reference_values = _registration_image(reference)
    return _registration_correlation_prepared(reference_values, corrected)


def _registration_correlation_prepared(
    reference_registration: NDArray[np.float32],
    corrected: NDArray[np.generic],
) -> float:
    reference_values = reference_registration.astype(np.float64, copy=False)
    corrected_values = _registration_image(corrected).astype(np.float64, copy=False)
    denominator = float(
        np.sqrt(
            np.sum(reference_values * reference_values, dtype=np.float64)
            * np.sum(corrected_values * corrected_values, dtype=np.float64)
        )
    )
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(
        np.sum(reference_values * corrected_values, dtype=np.float64) / denominator
    )


def export_corrected_recording(
    source_path: Path,
    output_directory: Path,
    metadata: RecordingMetadata,
    result: MotionCorrectionResult,
    *,
    overwrite: bool = False,
) -> CorrectedRecordingExportResult:
    """Write a lossless corrected OME-BigTIFF and transform/QC sidecars."""
    source_path = _validated_path(source_path)
    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    if len(result.transforms) != metadata.frame_count:
        raise ValueError("Transform count does not match recording frame count.")
    stem = f"{source_path.stem}_registered"
    final_tiff = output_directory / f"{stem}.ome.tif"
    final_csv = output_directory / f"{stem}_transforms.csv"
    final_json = output_directory / f"{stem}_metadata.json"
    final_paths = (final_tiff, final_csv, final_json)
    existing = [path for path in final_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(existing[0])

    staging = output_directory / f".{stem}-{uuid4().hex}"
    staging.mkdir()
    try:
        staged_tiff = staging / final_tiff.name
        staged_csv = staging / final_csv.name
        staged_json = staging / final_json.name
        dtype = np.dtype(metadata.dtype)

        def corrected_frames() -> Any:
            with nd2.ND2File(source_path) as recording:
                for index, transform in enumerate(result.transforms):
                    frame = np.asarray(recording.read_frame(index))
                    yield apply_frame_transform(frame, transform)

        ome_metadata: dict[str, Any] = {
            "axes": "TYX",
            "PhysicalSizeX": metadata.pixel_size_x_um,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": metadata.pixel_size_y_um,
            "PhysicalSizeYUnit": "µm",
        }
        if metadata.median_interval_s is not None:
            ome_metadata["TimeIncrement"] = metadata.median_interval_s
            ome_metadata["TimeIncrementUnit"] = "s"
        tifffile.imwrite(
            staged_tiff,
            data=corrected_frames(),
            shape=(metadata.frame_count, metadata.height, metadata.width),
            dtype=dtype,
            bigtiff=True,
            ome=True,
            photometric="minisblack",
            compression="zlib",
            metadata=ome_metadata,
        )
        _write_transform_csv(staged_csv, result)
        staged_json.write_text(
            json.dumps(
                {
                    "source_file": source_path.name,
                    "mode": result.mode,
                    "reference_frame_one_based": result.reference_frame_index + 1,
                    "shape_tyx": [
                        metadata.frame_count,
                        metadata.height,
                        metadata.width,
                    ],
                    "dtype": metadata.dtype,
                    "pixel_size_x_um": metadata.pixel_size_x_um,
                    "pixel_size_y_um": metadata.pixel_size_y_um,
                    "timestamps_s": metadata.timestamps_s,
                    "exposure_ms": metadata.exposure_ms,
                    "parameters": asdict(result.parameters),
                    "zero_filled_borders": True,
                    "intensity_interpolation": "none; nearest-neighbor pixel assignment",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _verify_corrected_export(staged_tiff, staged_csv, staged_json, metadata)
        for staged_path, final_path in zip(
            (staged_tiff, staged_csv, staged_json), final_paths, strict=True
        ):
            os.replace(staged_path, final_path)
    finally:
        if staging.exists():
            for remaining in staging.iterdir():
                if remaining.is_file():
                    remaining.unlink()
            staging.rmdir()
    return CorrectedRecordingExportResult(
        ome_tiff_path=final_tiff,
        transforms_csv_path=final_csv,
        metadata_json_path=final_json,
        frame_count=metadata.frame_count,
    )


def _registration_image(frame: NDArray[np.generic]) -> NDArray[np.float32]:
    values = np.asarray(frame, dtype=np.float32)
    smoothed = gaussian(values, sigma=1.0, preserve_range=True)
    background = gaussian(values, sigma=8.0, preserve_range=True)
    enhanced = np.asarray(smoothed - background, dtype=np.float32)
    enhanced -= float(np.mean(enhanced, dtype=np.float64))
    scale = float(np.std(enhanced, dtype=np.float64))
    if np.isfinite(scale) and scale > np.finfo(np.float32).eps:
        enhanced /= scale
    window = np.outer(
        np.hanning(enhanced.shape[0]), np.hanning(enhanced.shape[1])
    ).astype(np.float32)
    return enhanced * window


def _prepare_reference(
    reference: NDArray[np.generic],
    mode: str,
    parameters: MotionCorrectionParameters,
) -> _PreparedReference:
    reference = np.asarray(reference)
    registration = _registration_image(reference)
    polar = _polar_spectrum(reference, parameters) if mode == "rigid" else None
    return _PreparedReference(reference, registration, polar)


def _polar_spectrum(
    frame: NDArray[np.generic], parameters: MotionCorrectionParameters
) -> NDArray[np.float64]:
    small = _decimate(frame, parameters.registration_max_size_px)
    registration = _registration_image(small)
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(registration))))
    radius = max(8, min(small.shape) // 2 - 2)
    return warp_polar(
        spectrum,
        radius=radius,
        output_shape=(parameters.rotation_bins, radius),
        scaling="linear",
        order=1,
        preserve_range=True,
    )


def _estimate_rotation(
    reference_polar: NDArray[np.float64],
    moving: NDArray[np.generic],
    parameters: MotionCorrectionParameters,
) -> float:
    moving_polar = _polar_spectrum(moving, parameters)
    shift, _, _ = phase_cross_correlation(
        reference_polar,
        moving_polar,
        upsample_factor=1,
        normalization=None,
    )
    angle = float(shift[0]) * 360.0 / parameters.rotation_bins
    # Fourier magnitude is invariant to a 180-degree rotation. Map that
    # ambiguity into the expected small-rotation interval around zero.
    return float((angle + 90.0) % 180.0 - 90.0)


def _decimate(frame: NDArray[np.generic], maximum_size: int) -> NDArray[np.generic]:
    stride = max(1, int(np.ceil(max(frame.shape) / maximum_size)))
    return np.asarray(frame)[::stride, ::stride]


def _write_transform_csv(path: Path, result: MotionCorrectionResult) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "frame",
                "translation_x_px",
                "translation_y_px",
                "rotation_degrees",
                "correlation",
                "warning",
                "manually_adjusted",
            )
        )
        for index, transform in enumerate(result.transforms, start=1):
            writer.writerow(
                (
                    index,
                    transform.translation_x_px,
                    transform.translation_y_px,
                    transform.rotation_degrees,
                    transform.correlation,
                    transform.warning or "",
                    transform.manually_adjusted,
                )
            )


def _verify_corrected_export(
    tiff_path: Path,
    csv_path: Path,
    json_path: Path,
    metadata: RecordingMetadata,
) -> None:
    with tifffile.TiffFile(tiff_path) as tif:
        if not tif.is_ome:
            raise ValueError("Corrected image validation failed: file is not OME-TIFF.")
        if not tif.is_bigtiff:
            raise ValueError("Corrected image validation failed: file is not BigTIFF.")
        if tif.series[0].shape != (
            metadata.frame_count,
            metadata.height,
            metadata.width,
        ):
            raise ValueError(
                "Corrected image validation failed: unexpected dimensions."
            )
        if np.dtype(tif.series[0].dtype) != np.dtype(metadata.dtype):
            raise ValueError("Corrected image validation failed: unexpected dtype.")
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        if sum(1 for _ in csv.reader(stream)) != metadata.frame_count + 1:
            raise ValueError("Transform CSV validation failed.")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("shape_tyx") != [
        metadata.frame_count,
        metadata.height,
        metadata.width,
    ]:
        raise ValueError("Correction metadata validation failed.")
