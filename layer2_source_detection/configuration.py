from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.config import Layer2Config, ProjectConfig


@dataclass(frozen=True, slots=True)
class DirectionScanConfig:
    scanner_backend: str
    angle_step_deg: float
    frequency_min_hz: float
    frequency_max_hz: float
    n_fft: int
    window: str
    win_length: int
    hop_length: int
    context_ms: int
    covariance_shrinkage: float
    diagonal_loading: float
    eigenvalue_floor: float
    min_valid_frequency_bins: int
    direction_threshold: float
    peak_prominence: float
    min_peak_distance_deg: float
    max_candidates: int
    effective_order_limit: int

    @classmethod
    def from_project(cls, config: ProjectConfig) -> "DirectionScanConfig":
        return cls.from_layer2(config.layer2)

    @classmethod
    def from_layer2(cls, config: Layer2Config) -> "DirectionScanConfig":
        values = config.model_dump(
            exclude={"probability_gate", "music", "direction_kalman", "direction_id_tracking"}
        )
        return cls(**values)

    def __post_init__(self) -> None:
        if self.scanner_backend != "frequency_normalized_music":
            raise ValueError("Layer 2 DOA backend must be frequency_normalized_music")
        if self.angle_step_deg != 1.0:
            raise ValueError("MUSIC scan step must be one degree")
        if (self.frequency_min_hz, self.frequency_max_hz) != (2_000.0, 4_000.0):
            raise ValueError("Layer 2 MUSIC band is fixed at 2000..4000 Hz")
        if (self.n_fft, self.win_length, self.hop_length, self.window) != (1024, 960, 480, "hann_periodic"):
            raise ValueError("MUSIC STFT must be 1024/960/480 periodic Hann")
        if self.context_ms not in {160, 200, 240, 320}:
            raise ValueError("MUSIC context_ms must be 160, 200, 240, or 320")
        finite = (
            self.direction_threshold, self.peak_prominence, self.min_peak_distance_deg,
            self.covariance_shrinkage, self.diagonal_loading, self.eigenvalue_floor,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValueError("Layer 2配置必须全部finite")
        if not 0 <= self.direction_threshold <= 1:
            raise ValueError("MUSIC direction threshold is invalid")
        if self.peak_prominence < 0 or self.min_peak_distance_deg != 50.0:
            raise ValueError("prominence或NMS角距无效")
        if self.max_candidates != 3:
            raise ValueError("Layer 2 max_candidates is fixed at 3")
        if self.effective_order_limit not in {1, 2, 3}:
            raise ValueError("effective MUSIC order limit must be 1, 2, or 3")
        if not 0 <= self.covariance_shrinkage < 1 or self.diagonal_loading <= 0:
            raise ValueError("MUSIC covariance regularization is invalid")
        if self.eigenvalue_floor <= 0:
            raise ValueError("MUSIC eigensolver configuration is invalid")
        if self.min_valid_frequency_bins < 1:
            raise ValueError("MUSIC frequency quality configuration is invalid")
