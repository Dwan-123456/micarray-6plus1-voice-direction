from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.config import ProjectConfig, SourceCountingConfig


@dataclass(frozen=True, slots=True)
class SourceCounterConfig:
    enabled: bool
    backend: str
    context_ms: int
    frequency_min_hz: float
    frequency_max_hz: float
    n_fft: int
    win_length: int
    hop_length: int
    angle_step_deg: float
    lag_oversampling: int
    activity_rms_threshold_dbfs: float
    first_peak_threshold: float
    first_peak_z_threshold: float
    residual_peak_threshold: float
    residual_peak_z_threshold: float
    residual_ratio_threshold: float
    min_peak_distance_deg: float
    deemphasis_strength: float
    deemphasis_width_samples: float
    coactivity_frame_threshold: float
    coactivity_required_frames: int
    persistence_window_frames: int
    persistence_required_frames: int

    @classmethod
    def from_project(cls, config: ProjectConfig) -> "SourceCounterConfig":
        return cls.from_source_counting(config.source_counting)

    @classmethod
    def from_source_counting(cls, config: SourceCountingConfig) -> "SourceCounterConfig":
        return cls(**config.model_dump(exclude={"music_order_from_source_count"}))

    def __post_init__(self) -> None:
        if self.backend != "incremental_gcc_phat_deemphasis_v1":
            raise ValueError("unsupported source-counting backend")
        if self.context_ms != 160:
            raise ValueError("source counter context must be 160 ms")
        if (self.n_fft, self.win_length, self.hop_length) != (1024, 960, 480):
            raise ValueError("source counter STFT must be 1024/960/480")
        if (self.frequency_min_hz, self.frequency_max_hz) != (2_000.0, 4_000.0):
            raise ValueError("source counter band must be 2000..4000 Hz")
        if self.angle_step_deg != 1.0 or self.lag_oversampling != 4:
            raise ValueError("source counter grid must be 1 degree and 4x lag oversampling")
        finite = (
            self.activity_rms_threshold_dbfs,
            self.first_peak_threshold,
            self.first_peak_z_threshold,
            self.residual_peak_threshold,
            self.residual_peak_z_threshold,
            self.residual_ratio_threshold,
            self.min_peak_distance_deg,
            self.deemphasis_strength,
            self.deemphasis_width_samples,
            self.coactivity_frame_threshold,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValueError("source counter configuration must be finite")
        if not 0 <= self.first_peak_threshold <= 1 or not 0 <= self.residual_peak_threshold <= 1:
            raise ValueError("source counter peak thresholds must be in [0,1]")
        if self.first_peak_z_threshold < 0 or self.residual_peak_z_threshold < 0:
            raise ValueError("source counter robust-z thresholds must be non-negative")
        if not 0 <= self.residual_ratio_threshold <= 1:
            raise ValueError("source counter residual ratio must be in [0,1]")
        if self.min_peak_distance_deg != 50.0:
            raise ValueError("source counter angular separation is fixed at 50 degrees")
        if not 0 <= self.deemphasis_strength <= 1 or self.deemphasis_width_samples <= 0:
            raise ValueError("source counter de-emphasis configuration is invalid")
        if not 0 <= self.coactivity_frame_threshold <= 1:
            raise ValueError("source counter coactivity threshold must be in [0,1]")
        if self.coactivity_required_frames != 3:
            raise ValueError("source counter coactivity requires three frames")
        if (self.persistence_window_frames, self.persistence_required_frames) != (3, 2):
            raise ValueError("source counter persistence must be two of three decisions")
