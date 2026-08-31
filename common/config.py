from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HardwareConfig(StrictModel):
    physical_mic_count: Literal[7]
    ring_radius_m: float = Field(gt=0)
    speed_of_sound_mps: float = Field(gt=0)
    geometry_version: str
    hardware_calibration_status: Literal["unverified", "verified"]
    hardware_calibration_report_hash: str | None


class DeviceConfig(StrictModel):
    sample_rate: Literal[48000]
    device_channels: Literal[8]
    pcm_format: Literal["s16-le"]
    layout: Literal["interleaved"]
    block_size_samples: Literal[960]
    physical_channel_map: tuple[int, ...]
    hardware_mix_channel: Literal[6]
    logical_channel_map: tuple[int, ...]
    device_name: str
    host_api: str
    serial_enabled: bool
    serial_port: str
    serial_baud: int = Field(gt=0)
    serial_required: bool


class CalibrationAssetConfig(StrictModel):
    uri: str
    version: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CalibrationConfig(StrictModel):
    version: str
    correction_model: Literal["gain_polarity_integer_delay_v1"]
    gains: tuple[float, ...]
    polarity: tuple[int, ...]
    delay_samples: tuple[int, ...]
    fractional_delay_asset: CalibrationAssetConfig | None
    frequency_response_asset: CalibrationAssetConfig | None


class TimingConfig(StrictModel):
    decision_hop_samples: Literal[960]
    doa_window_samples: Literal[1920]
    context_samples: Literal[7680]
    timestamp_tolerance_ms: float = Field(ge=0)


class Layer1ImcraConfig(StrictModel):
    enabled: bool
    algorithm_version: Literal["cohen_imcra_2003_l1_v11"]
    hop_samples: Literal[960]
    n_fft: Literal[960]
    window: Literal["hann_periodic"]
    output_frequency_min_hz: Literal[0.0]
    output_frequency_max_hz: Literal[10000.0]
    frequency_min_hz: Literal[250.0]
    frequency_max_hz: Literal[3400.0]
    frequency_smoothing_half_width: Literal[1]
    spectrum_smoothing: Literal[0.77]
    noise_smoothing: Literal[0.66]
    prior_snr_smoothing: Literal[0.81]
    minimum_subwindow_frames: Literal[5]
    minimum_history_subwindows: Literal[10]
    minimum_bias: Literal[1.66]
    gamma0: Literal[4.6]
    gamma1: Literal[3.0]
    zeta0: Literal[1.67]
    bias_compensation: Literal[1.47]
    warmup_seconds: float = Field(gt=0)
    eps: float = Field(gt=0)


class Layer1PreDenoiseConfig(StrictModel):
    enabled: bool
    algorithm_version: Literal["imcra_wiener_wola_v4"]
    frame_samples: Literal[960]
    hop_samples: Literal[480]
    n_fft: Literal[960]
    frequency_max_hz: Literal[10000.0]
    window: Literal["sqrt_hann_50pct"]
    minimum_gain_db: float
    gain_smoothing: float


class Layer2ProbabilityGateConfig(StrictModel):
    backend: Literal["current_20ms_v1"]
    threshold: float = Field(ge=0, le=1)


class Layer2MusicPreparationConfig(StrictModel):
    context_ms: Literal[160, 200, 240, 320]
    comparison_context_ms: tuple[int, ...]
    max_history_ms: Literal[320]


class Layer2DirectionIdTrackingConfig(StrictModel):
    backend: Literal["circular_imm_jpda_v1"]
    association_gate_deg: float
    association_chi2: float
    max_velocity_dps: float
    confirmation_observations: int
    confirmation_window_ms: int
    tentative_ttl_ms: int
    coasting_ttl_ms: int
    probability_detect: float
    probability_track: float
    probability_new: float
    probability_false: float
    minimum_association_probability: float
    minimum_birth_probability: float
    confirmation_existence_probability: float
    deletion_existence_probability: float
    survival_probability_per_second: float
    measurement_std_deg: float
    stationary_angle_std_deg: float
    stationary_velocity_std_dps: float
    stationary_velocity_half_life_seconds: float
    moving_angle_std_deg: float
    moving_velocity_std_dps: float
    moving_velocity_half_life_seconds: float
    stationary_to_moving_probability: float
    moving_to_stationary_probability: float
    prediction_freeze_std_deg: float
    duplicate_birth_guard_deg: float
    max_active_tracks: Literal[4]


class Layer2Config(StrictModel):
    probability_gate: Layer2ProbabilityGateConfig
    music: Layer2MusicPreparationConfig
    direction_id_tracking: Layer2DirectionIdTrackingConfig
    scanner_backend: Literal["frequency_normalized_music"]
    angle_step_deg: Literal[1.0]
    frequency_min_hz: Literal[2000.0]
    frequency_max_hz: Literal[4000.0]
    n_fft: Literal[1024]
    win_length: Literal[960]
    hop_length: Literal[480]
    window: Literal["hann_periodic"]
    context_ms: Literal[160, 200, 240, 320]
    covariance_shrinkage: float
    diagonal_loading: float
    eigenvalue_floor: float
    min_valid_frequency_bins: int
    direction_threshold: float
    peak_prominence: float
    min_peak_distance_deg: Literal[50.0]
    max_candidates: Literal[3]
    effective_order_limit: Literal[1, 2, 3]


class SourceCountingConfig(StrictModel):
    enabled: bool = True
    music_order_from_source_count: bool = False
    backend: Literal["incremental_gcc_phat_deemphasis_v1"] = (
        "incremental_gcc_phat_deemphasis_v1"
    )
    context_ms: Literal[160] = 160
    frequency_min_hz: Literal[2000.0] = 2000.0
    frequency_max_hz: Literal[4000.0] = 4000.0
    n_fft: Literal[1024] = 1024
    win_length: Literal[960] = 960
    hop_length: Literal[480] = 480
    angle_step_deg: Literal[1.0] = 1.0
    lag_oversampling: Literal[4] = 4
    activity_rms_threshold_dbfs: float = -70.0
    first_peak_threshold: float = Field(default=0.16, ge=0, le=1)
    first_peak_z_threshold: float = Field(default=2.0, ge=0)
    residual_peak_threshold: float = Field(default=0.07, ge=0, le=1)
    residual_peak_z_threshold: float = Field(default=2.0, ge=0)
    residual_ratio_threshold: float = Field(default=0.09, ge=0, le=1)
    min_peak_distance_deg: Literal[50.0] = 50.0
    deemphasis_strength: float = Field(default=0.90, ge=0, le=1)
    deemphasis_width_samples: float = Field(default=0.25, gt=0)
    coactivity_frame_threshold: float = Field(default=0.08, ge=0, le=1)
    coactivity_required_frames: Literal[3] = 3
    persistence_window_frames: Literal[3] = 3
    persistence_required_frames: Literal[2] = 2


class RuntimeConfig(StrictModel):
    capture_handoff_blocks: int = Field(ge=1)
    l2_queue_windows: int = Field(ge=1)
    overflow_policy: Literal["drop_oldest"]
    graceful_shutdown_timeout_seconds: float = Field(gt=0)
    adaptive_fallback_enabled: bool = True
    adaptive_maximum_period_ms: int = Field(default=200, ge=20)
    adaptive_overload_threshold_ms: float = Field(default=20.0, gt=0)
    adaptive_recovery_threshold_ms: float = Field(default=12.0, gt=0)
    adaptive_recovery_stable_ms: int = Field(default=5_000, ge=20)

    @model_validator(mode="after")
    def validate_adaptive_fallback(self) -> "RuntimeConfig":
        if self.adaptive_maximum_period_ms % 20:
            raise ValueError("adaptive maximum period must be a multiple of 20 ms")
        if self.adaptive_recovery_threshold_ms >= self.adaptive_overload_threshold_ms:
            raise ValueError("adaptive recovery threshold must be below overload threshold")
        return self


class DevUiConfig(StrictModel):
    start_fullscreen: bool
    stale_after_ms: int
    l1_meter_refresh_hz: int
    polar_refresh_hz: int
    snapshot_mailbox_capacity: int


class ProjectConfig(StrictModel):
    schema_version: Literal["project_config_v1.4"]
    hardware: HardwareConfig
    device: DeviceConfig
    calibration: CalibrationConfig
    timing: TimingConfig
    layer1_imcra: Layer1ImcraConfig
    layer1_pre_denoise: Layer1PreDenoiseConfig
    layer2: Layer2Config
    source_counting: SourceCountingConfig = Field(default_factory=SourceCountingConfig)
    runtime: RuntimeConfig
    dev_test_ui: DevUiConfig

    @model_validator(mode="after")
    def validate_contract(self) -> "ProjectConfig":
        vectors = (self.device.physical_channel_map, self.calibration.gains,
                   self.calibration.polarity, self.calibration.delay_samples)
        if any(len(value) != 7 for value in vectors):
            raise ValueError("L1 mapping/calibration vectors require seven entries")
        if self.device.logical_channel_map != (*self.device.physical_channel_map, 6):
            raise ValueError("logical map must be physical microphones followed by HardwareMix")
        if tuple(sorted(self.device.logical_channel_map)) != tuple(range(8)):
            raise ValueError("logical map must be a permutation of 0..7")
        if self.layer1_pre_denoise.enabled and not self.layer1_imcra.enabled:
            raise ValueError("pre-denoise requires IMCRA")
        if self.layer2.context_ms != self.layer2.music.context_ms:
            raise ValueError("MUSIC context values must agree")
        if self.source_counting.frequency_min_hz >= self.source_counting.frequency_max_hz:
            raise ValueError("source-counting frequency range is invalid")
        if self.source_counting.persistence_required_frames > self.source_counting.persistence_window_frames:
            raise ValueError("source-counting persistence requirement exceeds its history")
        return self


def _canonical(value: BaseModel) -> bytes:
    return json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()


def calibration_config_hash(value: CalibrationConfig) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def config_hash(config: ProjectConfig) -> str:
    return hashlib.sha256(_canonical(config)).hexdigest()


def load_config(path: str | Path, *, environ: dict[str, str] | None = None) -> ProjectConfig:
    # Kept for source compatibility with earlier callers. v1.4 deliberately
    # does not read environment overrides into the minimal runtime schema.
    del environ
    with Path(path).open("r", encoding="utf-8") as handle:
        return ProjectConfig.model_validate(yaml.safe_load(handle))
