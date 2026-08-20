from __future__ import annotations

from dataclasses import dataclass

from common.config import ProjectConfig


@dataclass(frozen=True, slots=True)
class StftSettings:
    n_fft: int
    win_length: int
    hop_length: int
    center: bool
    pad_mode: str
    normalized: bool
    onesided: bool
    window_samples: int
    window_hops: int
    frame_count: int

    @classmethod
    def from_project(cls, config: ProjectConfig) -> "StftSettings":
        value = config.stft
        spec = config.downstream_audio_window
        return cls(
            value.n_fft, value.win_length, value.hop_length, value.center, value.pad_mode,
            value.normalized, value.onesided, spec.samples, spec.decision_hops, spec.stft_frames,
        )


@dataclass(frozen=True, slots=True)
class SpatialSeparationConfig:
    frequency_min_hz: float
    frequency_max_hz: float
    rho_lcmv_max: float
    rho_soft_null_max: float
    loading_retry_factors: tuple[float, ...]
    noise_covariance_shrinkage: float
    uncertainty_loading_multiplier: float
    alias_guard_hz: float
    alias_loading_multiplier: float
    soft_null_strength: float
    condition_number_limit: float
    constraint_tolerance: float
    min_frequency_gain: float
    constant_beamwidth_fnbw_deg: float
    constant_beamwidth_design_grid_deg: float
    constant_beamwidth_regularization: float
    constant_beamwidth_min_wng_db: float

    @classmethod
    def from_project(cls, config: ProjectConfig) -> "SpatialSeparationConfig":
        value = config.layer3
        return cls(
            value.frequency_min_hz, value.frequency_max_hz,
            value.rho_lcmv_max, value.rho_soft_null_max,
            value.loading_retry_factors, value.noise_covariance_shrinkage,
            value.uncertainty_loading_multiplier, value.alias_guard_hz,
            value.alias_loading_multiplier, value.soft_null_strength,
            value.condition_number_limit, value.constraint_tolerance,
            value.min_frequency_gain,
            value.constant_beamwidth_fnbw_deg,
            value.constant_beamwidth_design_grid_deg,
            value.constant_beamwidth_regularization,
            value.constant_beamwidth_min_wng_db,
        )


@dataclass(frozen=True, slots=True)
class FeatureSettings:
    preprocessing_version: str
    first_bin: int
    last_bin_inclusive: int
    log_epsilon: float
    frame_count: int

    @classmethod
    def from_project(cls, config: ProjectConfig) -> "FeatureSettings":
        value = config.feature
        return cls(
            value.preprocessing_version,
            value.first_bin,
            value.last_bin_inclusive,
            value.log_epsilon,
            config.downstream_audio_window.stft_frames,
        )
