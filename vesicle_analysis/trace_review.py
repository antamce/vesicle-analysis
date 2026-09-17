from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class NormalizedTraceResult:
    pool_ids: tuple[int, ...]
    values: NDArray[np.float64]
    available: NDArray[np.bool_]
    baselines: NDArray[np.float64]
    background_ids: tuple[int, ...]
    baseline_frame_count: int


@dataclass(frozen=True, slots=True)
class TraceReviewDecision:
    source_path: str
    source_filename: str
    file_size_bytes: int
    pool_ids: tuple[int, ...]
    excluded_reasons: dict[int, str]


def normalize_pool_traces(
    pool_ids: tuple[int, ...],
    pool_traces: NDArray[np.float64],
    background_ids: tuple[int, ...],
    background_traces: NDArray[np.float64],
    *,
    background_mode: str,
    background_assignments: dict[int, int | None],
    baseline_frame_count: int = 10,
) -> NormalizedTraceResult:
    """Calculate (pool-background)/mean(first baseline frames)."""
    pools = np.asarray(pool_traces, dtype=np.float64)
    backgrounds = np.asarray(background_traces, dtype=np.float64)
    if pools.ndim != 2 or pools.shape[1] != len(pool_ids):
        raise ValueError("Pool trace dimensions do not match the pool identifiers.")
    if backgrounds.ndim != 2 or backgrounds.shape[1] != len(background_ids):
        raise ValueError(
            "Background trace dimensions do not match the background identifiers."
        )
    if backgrounds.shape[0] != pools.shape[0]:
        raise ValueError("Pool and background traces have different frame counts.")
    if baseline_frame_count < 1:
        raise ValueError("At least one baseline frame is required.")
    if background_mode not in {"pool-wise", "common"}:
        raise ValueError(f"Unsupported background mode: {background_mode}")

    background_columns = {value: index for index, value in enumerate(background_ids)}
    selected_background_ids: list[int] = []
    if background_mode == "common":
        if len(background_ids) != 1:
            raise ValueError("Common-background normalization requires one background.")
        selected_background_ids = [background_ids[0]] * len(pool_ids)
    else:
        owner_to_background = {
            owner: background_id
            for background_id, owner in background_assignments.items()
            if owner is not None
        }
        for pool_id in pool_ids:
            background_id = owner_to_background.get(pool_id)
            if background_id not in background_columns:
                raise ValueError(f"Pool {pool_id} has no exported background trace.")
            selected_background_ids.append(background_id)

    corrected = np.empty_like(pools)
    for column, background_id in enumerate(selected_background_ids):
        corrected[:, column] = (
            pools[:, column] - backgrounds[:, background_columns[background_id]]
        )
    used_frames = min(baseline_frame_count, len(corrected))
    baselines = np.mean(corrected[:used_frames], axis=0, dtype=np.float64)
    local_scale = np.max(np.abs(corrected[:used_frames]), axis=0)
    tolerance = np.maximum(1e-12, local_scale * 1e-9)
    available = np.isfinite(baselines) & (np.abs(baselines) > tolerance)
    values = np.full_like(corrected, np.nan)
    values[:, available] = corrected[:, available] / baselines[available]
    return NormalizedTraceResult(
        pool_ids=pool_ids,
        values=values,
        available=available,
        baselines=baselines,
        background_ids=tuple(selected_background_ids),
        baseline_frame_count=used_frames,
    )


def write_review_manifest(
    path: Path,
    decisions: tuple[TraceReviewDecision, ...],
    *,
    overwrite: bool = False,
) -> Path:
    path = Path(path).expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "vesicle-analysis-trace-review",
        "version": 1,
        "normalization": (
            "(pool - assigned background) / mean(pool - assigned background, "
            "frames 1-10)"
        ),
        "recordings": [asdict(decision) for decision in decisions],
    }
    temporary = path.parent / f".{path.name}-{uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        verification = json.loads(temporary.read_text(encoding="utf-8"))
        if verification.get("format") != payload["format"]:
            raise RuntimeError("Trace-review manifest verification failed.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_review_manifest(path: Path) -> tuple[TraceReviewDecision, ...]:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "vesicle-analysis-trace-review":
        raise ValueError("This is not a Vesicle Analysis trace-review manifest.")
    if payload.get("version") != 1 or not isinstance(payload.get("recordings"), list):
        raise ValueError("Unsupported trace-review manifest version.")
    decisions: list[TraceReviewDecision] = []
    for item in payload["recordings"]:
        if not isinstance(item, dict):
            raise TypeError("Trace-review manifest contains an invalid recording.")
        pool_ids = tuple(int(value) for value in item.get("pool_ids", ()))
        excluded = {
            int(pool_id): str(reason)
            for pool_id, reason in item.get("excluded_reasons", {}).items()
        }
        if not set(excluded).issubset(pool_ids):
            raise ValueError("Manifest exclusions do not match its pool identifiers.")
        decisions.append(
            TraceReviewDecision(
                source_path=str(item["source_path"]),
                source_filename=str(item["source_filename"]),
                file_size_bytes=int(item["file_size_bytes"]),
                pool_ids=pool_ids,
                excluded_reasons=excluded,
            )
        )
    return tuple(decisions)
