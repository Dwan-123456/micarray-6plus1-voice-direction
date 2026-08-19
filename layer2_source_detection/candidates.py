from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from common.angle import circular_distance_deg

from .configuration import DirectionScanConfig


def robust_z_sigmoid(raw_scores: np.ndarray, config: DirectionScanConfig) -> np.ndarray:
    raw = np.asarray(raw_scores)
    if raw.shape != (360,) or not np.isfinite(raw).all():
        raise ValueError("raw_scores必须为finite [360]")
    median = np.median(raw)
    mad = np.median(np.abs(raw - median))
    scale = max(1.4826 * float(mad), 1e-6)
    z_score = (raw - median) / scale
    logits = np.clip(
        config.normalization_alpha * (z_score - config.normalization_beta),
        -80.0,
        80.0,
    )
    return np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=np.float32)


def rank_candidate_indices(normalized_scores: np.ndarray, config: DirectionScanConfig) -> tuple[int, ...]:
    scores = np.asarray(normalized_scores)
    if scores.shape != (360,) or not np.isfinite(scores).all():
        raise ValueError("normalized_scores必须为finite [360]")
    tiled = np.tile(scores, 3)
    peaks, _ = find_peaks(
        tiled,
        prominence=config.peak_prominence,
        plateau_size=(1, None),
    )
    local = {
        int(index - 360)
        for index in peaks
        if 360 <= index < 720 and scores[index - 360] >= config.direction_threshold
    }
    ranked = sorted(local, key=lambda index: (-float(scores[index]), index))
    selected: list[int] = []
    for index in ranked:
        if any(circular_distance_deg(index, other) < config.min_peak_distance_deg for other in selected):
            continue
        selected.append(index)
    return tuple(selected)


def select_candidate_indices(normalized_scores: np.ndarray, config: DirectionScanConfig) -> tuple[int, ...]:
    """Return the formal Layer 2 candidates, capped before DTO construction."""

    return rank_candidate_indices(normalized_scores, config)[: config.max_candidates]
