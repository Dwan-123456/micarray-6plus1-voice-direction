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
    remove_channel_mean: bool
    gcc_interpolation: int
    phat_epsilon: float
    normalization_backend: str
    normalization_alpha: float
    normalization_beta: float
    direction_threshold: float
    peak_prominence: float
    min_peak_distance_deg: float
    max_candidates: int
    iterative_peak_search_enabled: bool
    iterative_max_sources: int
    iterative_suppression_strength: float
    iterative_phase_power: float
    iterative_pair_phase_threshold: float
    iterative_min_pair_support: int
    iterative_min_frequency_support: int
    iterative_min_remaining_weight_ratio: float
    iterative_min_residual_peak_ratio: float

    @classmethod
    def from_project(cls, config: ProjectConfig) -> "DirectionScanConfig":
        if config.layer2.scanner_backend != "srp_phat":
            raise ValueError("本版本Layer 2 scanner_backend必须为srp_phat")
        return cls.from_layer2(config.layer2)

    @classmethod
    def from_layer2(cls, config: Layer2Config) -> "DirectionScanConfig":
        values = config.model_dump(
            exclude={
                "probability_gate",
                "direction_kalman",
                "direction_id_tracking",
                "scanner_backend",
                "music",
            }
        )
        return cls(**values)

    def __post_init__(self) -> None:
        if self.angle_step_deg != 1.0:
            raise ValueError("v0.2扫描间隔固定为1度")
        if (self.frequency_min_hz, self.frequency_max_hz) != (2_000.0, 4_000.0):
            raise ValueError("Layer 2 SRP-PHAT band is fixed at 2000..4000 Hz")
        if self.n_fft != 2_048 or self.window != "hann_periodic" or not self.remove_channel_mean:
            raise ValueError("v0.2前端固定为2048点、periodic Hann和逐通道去均值")
        if self.gcc_interpolation != 16 or self.phat_epsilon <= 0:
            raise ValueError("v0.2要求16倍GCC插值及正PHAT epsilon")
        if self.normalization_backend != "robust_z_sigmoid":
            raise ValueError("v0.2归一化固定为robust_z_sigmoid")
        finite = (
            self.normalization_alpha, self.normalization_beta, self.direction_threshold,
            self.peak_prominence, self.min_peak_distance_deg, self.phat_epsilon,
            self.iterative_suppression_strength, self.iterative_phase_power,
            self.iterative_pair_phase_threshold, self.iterative_min_remaining_weight_ratio,
            self.iterative_min_residual_peak_ratio,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValueError("Layer 2配置必须全部finite")
        if self.normalization_alpha <= 0 or not 0 <= self.direction_threshold <= 1:
            raise ValueError("归一化alpha和方向阈值无效")
        if self.peak_prominence < 0 or self.min_peak_distance_deg != 45.0:
            raise ValueError("prominence或NMS角距无效")
        if self.max_candidates != 3:
            raise ValueError("Layer 2 max_candidates is fixed at 3")
        if type(self.iterative_peak_search_enabled) is not bool:
            raise ValueError("iterative_peak_search_enabled must be bool")
        if self.iterative_max_sources != 2:
            raise ValueError("iterative_max_sources is fixed at 2 for v1")
        if not 0 < self.iterative_suppression_strength < 1 or self.iterative_phase_power < 1:
            raise ValueError("iterative suppression parameters are invalid")
        if not 0 <= self.iterative_pair_phase_threshold <= 1:
            raise ValueError("iterative pair phase threshold is invalid")
        if not 1 <= self.iterative_min_pair_support <= 21:
            raise ValueError("iterative minimum pair support is invalid")
        if self.iterative_min_frequency_support < 1:
            raise ValueError("iterative minimum frequency support is invalid")
        if not 0 < self.iterative_min_remaining_weight_ratio <= 1:
            raise ValueError("iterative remaining weight ratio is invalid")
        if not 0 < self.iterative_min_residual_peak_ratio <= 1:
            raise ValueError("iterative residual peak ratio is invalid")
