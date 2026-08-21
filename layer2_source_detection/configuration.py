from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.config import Layer2Config, ProjectConfig


@dataclass(frozen=True, slots=True)
class DirectionScanConfig:
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
    mdl_max_age_ms: int
    min_valid_frequency_bins: int
    min_cross_frequency_consistency: float
    direction_threshold: float
    peak_prominence: float
    min_peak_distance_deg: float
    max_candidates: int
    effective_order_limit: int
    dpd_rank1_enabled: bool
    dpd_min_eigenvalue_ratio: float
    dpd_min_plane_wave_fit: float
    dpd_min_frequency_support_ratio: float
    dpd_angle_tolerance_deg: int
    dpd_min_cluster_frequency_bins: int
    dpd_frequency_subbands: int
    dpd_min_cluster_subbands: int
    dpd_min_circular_concentration: float
    dpd_peak_fusion_distance_deg: float
    dpd_peak_fusion_min_normalized_score: float
    noise_whitening_enabled: bool
    noise_covariance_shrinkage: float

    @classmethod
    def from_project(cls, config: ProjectConfig) -> "DirectionScanConfig":
        if config.layer2.scanner_backend != "frequency_normalized_music":
            raise ValueError("Layer 2 scanner_backend must be frequency_normalized_music")
        return cls.from_layer2(config.layer2)

    @classmethod
    def from_layer2(cls, config: Layer2Config) -> "DirectionScanConfig":
        values = config.model_dump(
            exclude={"probability_gate", "music", "direction_kalman", "direction_id_tracking", "scanner_backend"}
        )
        return cls(**values)

    def __post_init__(self) -> None:
        if self.angle_step_deg != 1.0:
            raise ValueError("MUSIC scan step must be one degree")
        if (self.frequency_min_hz, self.frequency_max_hz) != (2_000.0, 4_000.0):
            raise ValueError("Layer 2 MUSIC band is fixed at 2000..4000 Hz")
        if (self.n_fft, self.win_length, self.hop_length, self.window) != (1024, 960, 480, "hann_periodic"):
            raise ValueError("MUSIC STFT must be 1024/960/480 periodic Hann")
        if self.context_ms not in {160, 240, 320}:
            raise ValueError("MUSIC context_ms must be 160, 240, or 320")
        finite = (
            self.direction_threshold, self.peak_prominence, self.min_peak_distance_deg,
            self.covariance_shrinkage, self.diagonal_loading, self.eigenvalue_floor,
            self.min_cross_frequency_consistency, self.dpd_min_eigenvalue_ratio,
            self.dpd_min_plane_wave_fit, self.dpd_min_frequency_support_ratio,
            self.dpd_min_circular_concentration,
            self.dpd_peak_fusion_distance_deg,
            self.dpd_peak_fusion_min_normalized_score,
            self.noise_covariance_shrinkage,
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
        if type(self.dpd_rank1_enabled) is not bool or type(self.noise_whitening_enabled) is not bool:
            raise TypeError("DPD/whitening switches must be bool")
        if (
            self.dpd_min_eigenvalue_ratio <= 1
            or not 0 <= self.dpd_min_plane_wave_fit <= 1
            or not 0 < self.dpd_min_frequency_support_ratio <= 1
            or not 1 <= self.dpd_angle_tolerance_deg <= 45
            or self.dpd_min_cluster_frequency_bins < 1
            or self.dpd_frequency_subbands < 1
            or not 1 <= self.dpd_min_cluster_subbands <= self.dpd_frequency_subbands
            or not 0 <= self.dpd_min_circular_concentration <= 1
            or not 0 < self.dpd_peak_fusion_distance_deg <= self.min_peak_distance_deg
            or not 0 <= self.dpd_peak_fusion_min_normalized_score <= 1
        ):
            raise ValueError("DPD rank-1 quality configuration is invalid")
        if (
            not 0 <= self.noise_covariance_shrinkage <= 1
        ):
            raise ValueError("MUSIC noise covariance configuration is invalid")
        if not 0 <= self.covariance_shrinkage < 1 or self.diagonal_loading <= 0:
            raise ValueError("MUSIC covariance regularization is invalid")
        if self.eigenvalue_floor <= 0 or not 1 <= self.mdl_max_age_ms <= 100:
            raise ValueError("MUSIC eigensolver/MDL configuration is invalid")
        if self.min_valid_frequency_bins < 1 or not 0 <= self.min_cross_frequency_consistency <= 1:
            raise ValueError("MUSIC frequency quality configuration is invalid")
