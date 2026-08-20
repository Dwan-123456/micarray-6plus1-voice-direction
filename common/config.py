from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(StrictModel):
    data_root: str
    models_root: str


class HardwareConfig(StrictModel):
    physical_mic_count: int
    ring_radius_m: float
    speed_of_sound_mps: float
    geometry_version: str
    hardware_calibration_status: Literal["unverified", "verified"]
    hardware_calibration_report_hash: str | None

    @model_validator(mode="after")
    def validate_calibration_report_hash(self) -> "HardwareConfig":
        value = self.hardware_calibration_report_hash
        if value is not None and (
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("hardware_calibration_report_hash必须为小写SHA-256或null")
        return self


class DeviceConfig(StrictModel):
    sample_rate: Literal[48000]
    device_channels: Literal[8]
    pcm_format: Literal["s16-le"]
    layout: Literal["interleaved"]
    block_size_samples: int = Field(gt=0)
    physical_channel_map: tuple[int, ...]
    hardware_mix_channel: int
    logical_channel_map: tuple[int, ...]
    device_name: str
    host_api: str
    serial_enabled: bool
    serial_port: str
    serial_baud: int = Field(gt=0)
    serial_required: bool
    light_service_url: str | None


class CalibrationAssetConfig(StrictModel):
    uri: str = Field(min_length=1)
    version: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CalibrationConfig(StrictModel):
    version: str = Field(min_length=1)
    correction_model: Literal["gain_polarity_integer_delay_v1"]
    gains: tuple[float, ...]
    polarity: tuple[int, ...]
    delay_samples: tuple[int, ...]
    fractional_delay_asset: CalibrationAssetConfig | None = None
    frequency_response_asset: CalibrationAssetConfig | None = None


@dataclass(frozen=True, slots=True)
class DownstreamAudioWindowSpec:
    """Single derived L3/L4/Test-UI audio-window contract."""

    duration_ms: int
    samples: int
    decision_hops: int
    stft_frames: int
    resampled_16k_samples: int


class TimingConfig(StrictModel):
    decision_hop_samples: Literal[960]
    doa_window_samples: Literal[1920]
    context_samples: Literal[7680]
    downstream_audio_window_ms: Literal[80, 160] = 80
    timestamp_tolerance_ms: float = Field(ge=0)


class Layer1ImcraConfig(StrictModel):
    enabled: bool
    algorithm_version: Literal["cohen_imcra_2003_l1_v2"]
    hop_samples: Literal[960]
    n_fft: Literal[2048]
    window: Literal["hann_periodic"]
    output_frequency_min_hz: Literal[0.0]
    output_frequency_max_hz: Literal[8000.0]
    frequency_min_hz: Literal[500.0]
    frequency_max_hz: Literal[4000.0]
    frequency_smoothing_half_width: Literal[1]
    spectrum_smoothing: Literal[0.9]
    noise_smoothing: Literal[0.85]
    prior_snr_smoothing: Literal[0.92]
    minimum_subwindow_frames: Literal[15]
    minimum_history_subwindows: Literal[8]
    minimum_bias: Literal[1.66]
    gamma0: Literal[4.6]
    gamma1: Literal[3.0]
    zeta0: Literal[1.67]
    bias_compensation: Literal[1.47]
    warmup_seconds: float = Field(gt=0)
    eps: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_imcra(self) -> "Layer1ImcraConfig":
        if self.frequency_max_hz <= self.frequency_min_hz:
            raise ValueError("L1 IMCRA frequency band must be increasing")
        if not self.output_frequency_min_hz < self.frequency_min_hz < self.frequency_max_hz < self.output_frequency_max_hz:
            raise ValueError("L1 IMCRA output and Gate frequency bands are inconsistent")
        if self.minimum_history_subwindows * self.minimum_subwindow_frames != 120:
            raise ValueError("Cohen IMCRA minimum window must contain 120 frames")
        return self


class Layer1PreDenoiseConfig(StrictModel):
    enabled: bool = False
    algorithm_version: Literal["imcra_wiener_wola_v2"]
    frame_samples: Literal[1920]
    hop_samples: Literal[960]
    n_fft: Literal[2048]
    window: Literal["sqrt_hann_50pct"]
    minimum_gain_db: float = Field(ge=-60.0, le=0.0)
    gain_smoothing: float = Field(ge=0.0, lt=1.0)


class Layer2ProbabilityGateConfig(StrictModel):
    backend: Literal["mean_2x20ms_v1"]
    threshold: float = Field(default=0.60, ge=0, le=1)


class Layer2MusicPreparationConfig(StrictModel):
    context_ms: Literal[160, 240, 320]
    comparison_context_ms: tuple[Literal[160, 240, 320], ...]
    max_history_ms: Literal[320]

    @model_validator(mode="after")
    def validate_history_candidates(self) -> "Layer2MusicPreparationConfig":
        if self.comparison_context_ms != (160, 240, 320):
            raise ValueError("MUSIC首轮历史比较必须固定包含160/240/320 ms")
        if self.context_ms not in self.comparison_context_ms:
            raise ValueError("music.context_ms必须来自comparison_context_ms")
        return self


class Layer2DirectionKalmanConfig(StrictModel):
    enabled: bool = False
    backend: Literal["circular_kalman_v1", "damped_circular_kalman_v2"]
    process_noise_scale: float = Field(ge=0.02, le=10.0)
    measurement_noise_scale: float = Field(ge=0.02, le=10.0)
    process_angle_std_deg: float = Field(gt=0)
    process_velocity_std_dps: float = Field(gt=0)
    measurement_std_deg: float = Field(gt=0)
    max_missed_windows: int = Field(ge=0)
    velocity_half_life_seconds: float = Field(default=0.5, gt=0)
    max_velocity_dps: float = Field(default=60.0, gt=0, le=360)
    prediction_freeze_std_deg: float = Field(default=float("inf"), gt=0)


class Layer2DirectionIdTrackingConfig(StrictModel):
    backend: Literal["global_assignment_v1"]
    association_gate_deg: float = Field(gt=0, le=180)
    max_velocity_dps: float = Field(gt=0, le=360)
    confirmation_observations: int = Field(ge=1)
    confirmation_window_ms: int = Field(gt=0)
    coasting_ttl_ms: int = Field(gt=0)
    miss_cost: float = Field(gt=0)
    birth_cost: float = Field(gt=0)


class Layer2Config(StrictModel):
    probability_gate: Layer2ProbabilityGateConfig
    music: Layer2MusicPreparationConfig
    direction_kalman: Layer2DirectionKalmanConfig
    direction_id_tracking: Layer2DirectionIdTrackingConfig
    scanner_backend: Literal["frequency_normalized_music"]
    angle_step_deg: Literal[1.0]
    frequency_min_hz: Literal[2000.0]
    frequency_max_hz: Literal[4000.0]
    n_fft: Literal[1024]
    win_length: Literal[960]
    hop_length: Literal[480]
    window: Literal["hann_periodic"]
    context_ms: Literal[160, 240, 320]
    covariance_shrinkage: float = Field(ge=0, lt=1)
    diagonal_loading: float = Field(gt=0)
    eigenvalue_floor: float = Field(gt=0)
    mdl_max_age_ms: int = Field(gt=0, le=100)
    min_valid_frequency_bins: int = Field(gt=0)
    min_cross_frequency_consistency: float = Field(ge=0, le=1)
    direction_threshold: float
    peak_prominence: float
    min_peak_distance_deg: Literal[45.0]
    max_candidates: Literal[3]
    effective_order_limit: Literal[1, 2, 3]
    dpd_rank1_enabled: bool = False
    dpd_min_eigenvalue_ratio: float = Field(gt=1)
    dpd_min_plane_wave_fit: float = Field(ge=0, le=1)
    dpd_min_frequency_support_ratio: float = Field(gt=0, le=1)
    dpd_angle_tolerance_deg: int = Field(gt=0, le=45)
    dpd_min_cluster_frequency_bins: int = Field(ge=1)
    dpd_frequency_subbands: int = Field(ge=1)
    dpd_min_cluster_subbands: int = Field(ge=1)
    dpd_min_circular_concentration: float = Field(ge=0, le=1)
    noise_whitening_enabled: bool = False
    noise_covariance_shrinkage: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_direction_postprocessing(self) -> "Layer2Config":
        for value in (
            self.direction_kalman.process_noise_scale,
            self.direction_kalman.measurement_noise_scale,
        ):
            if value != 0.02 and abs(value * 10.0 - round(value * 10.0)) > 1.0e-9:
                raise ValueError("Layer 2 Kalman Q/R scales must use 0.1 steps (or the 0.02 minimum)")
        if type(self.dpd_rank1_enabled) is not bool or type(self.noise_whitening_enabled) is not bool:
            raise TypeError("Layer 2 DPD/whitening switches must be bool")
        if self.dpd_min_cluster_subbands > self.dpd_frequency_subbands:
            raise ValueError("DPD minimum cluster subbands cannot exceed configured subbands")
        return self

class StftConfig(StrictModel):
    n_fft: int
    win_length: int
    hop_length: int
    window: str
    center: bool
    pad_mode: str
    normalized: bool
    onesided: bool
    return_complex: bool


class Layer3Config(StrictModel):
    main_backend: str
    fallback_backend: str
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


class FeatureConfig(StrictModel):
    preprocessing_version: str
    frequency_min_hz: float
    frequency_max_hz: float
    first_bin: int
    last_bin_inclusive: int
    log_epsilon: float
    normalization: str


class Layer4ModelConfig(StrictModel):
    model_id: str
    backend: Literal["nvidia_marblenet_continuous_v2", "nvidia_marblenet_window_v1"]
    model_artifact: str
    role: Literal["primary", "shadow"]
    enabled: bool = True


class Layer4InputGainCompensationConfig(StrictModel):
    enabled: bool
    algorithm_version: Literal["imcra_probability_rms_v1"]
    target_rms_dbfs: float
    no_compensation_probability: float = Field(ge=0.0, le=1.0)
    full_compensation_probability: float = Field(ge=0.0, le=1.0)
    peak_ceiling_dbfs: float = Field(le=0.0)
    silence_floor_dbfs: float
    time_interpolation: Literal["linear_db"]

    @model_validator(mode="after")
    def validate_gain_compensation(self) -> "Layer4InputGainCompensationConfig":
        if self.no_compensation_probability >= self.full_compensation_probability:
            raise ValueError("Layer4 gain probability breakpoints must increase")
        if self.silence_floor_dbfs >= self.target_rms_dbfs:
            raise ValueError("Layer4 silence floor must be below the target RMS")
        return self


class Layer4Config(StrictModel):
    primary_model_id: str
    models: tuple[Layer4ModelConfig, ...]
    allow_mock: bool
    voice_probability_limit: float
    input_gain_compensation: Layer4InputGainCompensationConfig
    continuous_context_ms: int = Field(default=3_200, ge=60)

    @model_validator(mode="after")
    def validate_models(self) -> "Layer4Config":
        enabled = tuple(item for item in self.models if item.enabled)
        ids = tuple(item.model_id for item in enabled)
        if not enabled or len(ids) != len(set(ids)):
            raise ValueError("Layer4 enabled model ids must be non-empty and unique")
        primary = tuple(item for item in enabled if item.role == "primary")
        if len(primary) != 1 or primary[0].model_id != self.primary_model_id:
            raise ValueError("Layer4 must define exactly one enabled primary model")
        if not 0.0 <= self.voice_probability_limit <= 1.0:
            raise ValueError("Layer4 voice probability threshold must be in [0,1]")
        if self.continuous_context_ms % 20:
            raise ValueError("Layer4 continuous context must use complete 20 ms hops")
        return self


class RuntimeConfig(StrictModel):
    mode: Literal["development", "production"]
    preferred_device: str
    allow_cpu_fallback: bool
    max_candidate_batch: Literal[3]
    capture_handoff_blocks: int = Field(gt=0)
    # Kept for configuration compatibility.  The staged runtime uses the
    # per-stage capacities below rather than one shared serial queue.
    processing_queue_windows: int = Field(gt=0)
    # Normal deployments change only this shared value. Per-stage fields stay
    # optional for focused tests and exceptional diagnostic profiles.
    stage_queue_windows: int = Field(default=1_000, gt=0, le=10_000)
    l2_queue_windows: int | None = Field(default=None, gt=0, le=10_000)
    l3_queue_windows: int | None = Field(default=None, gt=0, le=10_000)
    l4_queue_windows: int | None = Field(default=None, gt=0, le=10_000)
    completion_queue_windows: int = Field(default=8, gt=0, le=128)
    max_inflight_windows: int | None = Field(default=None, gt=0, le=30_003)
    compute_cache_max_bytes: int = Field(default=67_108_864, ge=8_388_608)
    overflow_policy: Literal["drop_oldest"] = "drop_oldest"
    graceful_shutdown_timeout_seconds: float = Field(default=10.0, gt=0.0, le=120.0)

    @model_validator(mode="after")
    def validate_pipeline_capacity(self) -> "RuntimeConfig":
        for name in ("l2_queue_windows", "l3_queue_windows", "l4_queue_windows"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, self.stage_queue_windows)
        assert self.l2_queue_windows is not None
        assert self.l3_queue_windows is not None
        assert self.l4_queue_windows is not None
        # Joiner retains every admitted window until L2/L3/L4 are terminal.
        # Its hard bound must therefore cover all stage queues plus one active
        # item per worker; otherwise a valid configured backlog could fail at
        # registration before the advertised queue policy is reached.
        minimum = self.l2_queue_windows + self.l3_queue_windows + self.l4_queue_windows + 3
        if self.max_inflight_windows is None:
            object.__setattr__(self, "max_inflight_windows", minimum)
        assert self.max_inflight_windows is not None
        if self.max_inflight_windows < minimum:
            raise ValueError(
                "runtime.max_inflight_windows must cover all staged queues and active workers "
                f"(minimum {minimum})"
            )
        return self


class DevUiConfig(StrictModel):
    start_fullscreen: bool
    stale_after_ms: int
    performance_bar_height_px: int
    performance_refresh_hz: int
    performance_window_count: int
    sample_rate_window_seconds: int
    l1_meter_refresh_hz: int
    polar_refresh_hz: int
    waveform_refresh_hz: int
    snapshot_mailbox_capacity: int
    scratch_root: str
    autoplay: bool
    loop_gap_ms: int
    follow_latest_window: bool
    preview_volume: float
    preview_peak_dbfs: float
    preview_fade_ms: int
    default_selected_backend: str
    minimum_listening_track_seconds: float = Field(ge=0.0)


class RuntimeRecordingConfig(StrictModel):
    mode: Literal["off", "manual", "continuous", "event"]
    chunk_seconds: int = Field(gt=0)
    audio_queue_seconds: int = Field(gt=0)
    result_queue_capacity: int = Field(gt=0, le=256)
    record_native_8ch: bool
    record_logical_8ch: bool
    record_physical_7ch: bool
    record_physical_float32: bool
    record_results_jsonl: bool
    record_spatial_response: bool
    record_hotmaps: bool
    record_imcra: bool
    record_noise_spectrum: bool
    retention_days: int = Field(gt=0)
    max_storage_gb: int = Field(gt=0)
    min_free_storage_gb: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_storage_budget(self) -> "RuntimeRecordingConfig":
        if self.min_free_storage_gb >= self.max_storage_gb:
            raise ValueError("min_free_storage_gb必须小于max_storage_gb")
        return self


class EventRecordingConfig(StrictModel):
    pre_roll_seconds: int = Field(ge=0)
    post_roll_seconds: int = Field(ge=0)


class RecordingConfig(StrictModel):
    runtime: RuntimeRecordingConfig
    event: EventRecordingConfig
    trash_retention_days: int = Field(gt=0)


class PrivacyConfig(StrictModel):
    local_only: bool
    automatic_upload: bool


class ProjectConfig(StrictModel):
    schema_version: Literal["project_config_v1"]
    paths: PathsConfig
    hardware: HardwareConfig
    device: DeviceConfig
    calibration: CalibrationConfig
    timing: TimingConfig
    layer1_imcra: Layer1ImcraConfig
    layer1_pre_denoise: Layer1PreDenoiseConfig
    layer2: Layer2Config
    stft: StftConfig
    layer3: Layer3Config
    feature: FeatureConfig
    layer4: Layer4Config
    runtime: RuntimeConfig
    dev_test_ui: DevUiConfig
    recording: RecordingConfig
    privacy: PrivacyConfig

    @property
    def downstream_audio_window(self) -> DownstreamAudioWindowSpec:
        duration_ms = self.timing.downstream_audio_window_ms
        samples = duration_ms * self.device.sample_rate // 1_000
        return DownstreamAudioWindowSpec(
            duration_ms=duration_ms,
            samples=samples,
            decision_hops=samples // self.timing.decision_hop_samples,
            stft_frames=1 + samples // self.stft.hop_length,
            resampled_16k_samples=samples * 16_000 // self.device.sample_rate,
        )

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "ProjectConfig":
        count = self.hardware.physical_mic_count
        if count != 7 or any(
            len(values) != count
            for values in (
                self.device.physical_channel_map,
                self.calibration.gains,
                self.calibration.polarity,
                self.calibration.delay_samples,
            )
        ):
            raise ValueError("物理通道、映射和校准数组必须全部为7项")
        channel_map = self.device.physical_channel_map
        if len(set(channel_map)) != len(channel_map) or any(not 0 <= item < 8 for item in channel_map):
            raise ValueError("physical_channel_map必须唯一且位于0..7")
        if self.device.hardware_mix_channel != 6:
            raise ValueError("HardwareMix固定来自Host CH6")
        if self.device.logical_channel_map != (*channel_map, self.device.hardware_mix_channel):
            raise ValueError("logical_channel_map必须为physical_channel_map后追加HardwareMix")
        if len(self.device.logical_channel_map) != 8 or len(set(self.device.logical_channel_map)) != 8:
            raise ValueError("logical_channel_map必须是Host 0..7的完整排列")
        if not all(value > 0 for value in self.calibration.gains) or not all(
            value in (-1, 1) for value in self.calibration.polarity
        ):
            raise ValueError("gain必须>0，polarity只能为±1")
        if any(value < 0 for value in self.calibration.delay_samples) or min(self.calibration.delay_samples) != 0:
            raise ValueError("delay必须为非负整数且最小值为0")
        if not self.stft.hop_length <= self.stft.win_length <= self.stft.n_fft:
            raise ValueError("STFT必须满足hop<=win<=n_fft")
        if self.layer1_imcra.hop_samples != self.timing.decision_hop_samples:
            raise ValueError("L1 IMCRA hop必须等于decision hop")
        if (
            self.layer1_pre_denoise.hop_samples != self.timing.decision_hop_samples
            or self.layer1_pre_denoise.frame_samples != 2 * self.layer1_pre_denoise.hop_samples
            or self.layer1_pre_denoise.n_fft < self.layer1_pre_denoise.frame_samples
        ):
            raise ValueError("L1预降噪必须使用40 ms窗、20 ms步长且FFT长度不能小于窗长")
        if self.layer1_pre_denoise.enabled and not self.layer1_imcra.enabled:
            raise ValueError("L1预降噪启用时必须同时启用IMCRA")
        if self.runtime.max_candidate_batch < self.layer2.max_candidates:
            raise ValueError("max_candidate_batch不能小于max_candidates")
        duration_ms = self.timing.downstream_audio_window_ms
        if duration_ms not in {80, 160} or duration_ms <= 0 or duration_ms % 20:
            raise ValueError("downstream_audio_window_ms第一阶段只能为80或160且必须为20 ms整数倍")
        spec = self.downstream_audio_window
        if spec.samples > self.timing.context_samples:
            raise ValueError("downstream audio window不能超过DecisionWindow直接上下文")
        if spec.samples % self.timing.decision_hop_samples:
            raise ValueError("downstream audio window必须包含完整decision hops")
        if not self.stft.center or spec.stft_frames != 1 + spec.samples // self.stft.hop_length:
            raise ValueError("downstream STFT frame derivation与当前center=true配置不一致")
        if spec.resampled_16k_samples not in {1280, 2560}:
            raise ValueError("Layer4模型不支持所选downstream audio window")
        if (self.layer3.main_backend, self.layer3.fallback_backend) != (
            "imcra_spatial_separation",
            "das",
        ):
            raise ValueError("Layer3后端必须为imcra_spatial_separation/das")
        if not (
            0 <= self.layer3.frequency_min_hz < self.layer3.frequency_max_hz <= self.device.sample_rate / 2
        ):
            raise ValueError("Layer3频带边界必须严格递增并位于Nyquist以内")
        if not 0 < self.layer3.rho_lcmv_max < self.layer3.rho_soft_null_max < 1:
            raise ValueError("Layer3空间相关度阈值必须满足0<LCMV<soft-null<1")
        if not 0 <= self.layer3.noise_covariance_shrinkage <= 1:
            raise ValueError("Layer3噪声协方差收缩系数必须位于[0,1]")
        if not 0 < self.layer3.min_frequency_gain <= 1:
            raise ValueError("Layer3最小频点增益必须位于(0,1]")
        if not (
            0 < self.layer3.constant_beamwidth_fnbw_deg < 180
            and 0 < self.layer3.constant_beamwidth_design_grid_deg <= 5
            and self.layer3.constant_beamwidth_regularization > 0
            and -30 <= self.layer3.constant_beamwidth_min_wng_db <= 10
        ):
            raise ValueError("Layer3恒定波束宽度对照参数无效")
        if (
            not self.layer3.loading_retry_factors
            or any(value <= 0 for value in self.layer3.loading_retry_factors)
            or self.layer3.uncertainty_loading_multiplier < 0
            or self.layer3.alias_loading_multiplier < 1
            or not self.layer3.frequency_min_hz < self.layer3.alias_guard_hz < self.layer3.frequency_max_hz
            or self.layer3.soft_null_strength <= 0
            or self.layer3.condition_number_limit < 1
            or self.layer3.constraint_tolerance <= 0
        ):
            raise ValueError("Layer3 loading、混叠保护、soft-null或数值稳定参数无效")
        if self.runtime.mode == "production":
            if (
                self.layer4.allow_mock
                or self.runtime.allow_cpu_fallback
                or self.hardware.hardware_calibration_status != "verified"
            ):
                raise ValueError("production禁止Mock/CPU fallback并要求verified校准")
        return self


_OVERRIDES = {
    "MIC_DEVICE_NAME": ("device", "device_name"),
    "MIC_HOST_API": ("device", "host_api"),
    "MIC_SERIAL_PORT": ("device", "serial_port"),
    "MIC_SERIAL_REQUIRED": ("device", "serial_required"),
    "MIC_LIGHT_SERVICE_URL": ("device", "light_service_url"),
}


def load_config(path: str | Path = "config/config.yaml", *, environ: dict[str, str] | None = None) -> ProjectConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    env = os.environ if environ is None else environ
    for variable, (section, field) in _OVERRIDES.items():
        if variable not in env:
            continue
        value: object = env[variable]
        if variable == "MIC_SERIAL_REQUIRED":
            value = str(value).strip().lower() in {"1", "true", "yes", "on"}
        if variable == "MIC_LIGHT_SERVICE_URL" and not str(value).strip():
            value = None
        raw[section][field] = value
    return ProjectConfig.model_validate(raw)


def config_hash(config: ProjectConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def calibration_config_hash(calibration: CalibrationConfig) -> str:
    """Hash the versioned correction payload, independent of verification state/report."""

    payload = json.dumps(
        calibration.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
