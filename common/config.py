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
    """Single derived L3/L5/Test-UI audio-window contract."""

    duration_ms: int
    samples: int
    decision_hops: int
    stft_frames: int
    resampled_16k_samples: int


class TimingConfig(StrictModel):
    decision_hop_samples: Literal[960]
    doa_window_samples: Literal[1920]
    context_samples: Literal[7680]
    downstream_audio_window_ms: Literal[40, 80, 160] = 40
    timestamp_tolerance_ms: float = Field(ge=0)


class Layer1ImcraConfig(StrictModel):
    enabled: bool
    algorithm_version: Literal["cohen_imcra_2003_l1_v4"]
    hop_samples: Literal[960]
    n_fft: Literal[2048]
    window: Literal["hann_periodic"]
    output_frequency_min_hz: Literal[0.0]
    output_frequency_max_hz: Literal[10000.0]
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
    algorithm_version: Literal["imcra_wiener_wola_v3"]
    frame_samples: Literal[1920]
    hop_samples: Literal[960]
    n_fft: Literal[2048]
    frequency_max_hz: Literal[10000.0]
    window: Literal["sqrt_hann_50pct"]
    minimum_gain_db: float = Field(ge=-60.0, le=0.0)
    gain_smoothing: float = Field(ge=0.0, lt=1.0)


class Layer1SpeakerCountConfig(StrictModel):
    enabled: bool = False
    algorithm_version: Literal["countnet_crnn_5s_100ms_v3"]
    model_id: str = Field(min_length=1)
    model_artifact: str = Field(min_length=1)
    model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    input_sample_rate: Literal[16000]
    context_seconds: Literal[5]
    inference_hop_ms: Literal[100]
    output_classes: Literal[3]
    queue_blocks: int = Field(default=25, ge=5)
    input_level_target_dbfs: float = Field(default=-20.0, ge=-30.0, le=-12.0)
    input_level_floor_dbfs: float = Field(default=-70.0, ge=-100.0, le=-40.0)
    maximum_input_gain_db: float = Field(default=30.0, ge=0.0, le=40.0)

    @model_validator(mode="after")
    def validate_input_level_adapter(self) -> "Layer1SpeakerCountConfig":
        if self.input_level_floor_dbfs >= self.input_level_target_dbfs:
            raise ValueError("CountNet input level floor must be below its target")
        return self


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


class Layer2DirectionIdTrackingConfig(StrictModel):
    backend: Literal["circular_imm_jpda_v1"]
    association_gate_deg: float = Field(gt=0, le=180)
    association_chi2: float = Field(default=20.0, gt=0)
    max_velocity_dps: float = Field(gt=0, le=360)
    confirmation_observations: int = Field(ge=1)
    confirmation_window_ms: int = Field(gt=0)
    tentative_ttl_ms: int = Field(default=500, gt=0)
    coasting_ttl_ms: int = Field(gt=0)
    probability_detect: float = Field(default=0.85, ge=0, le=1)
    probability_track: float = Field(default=0.80, ge=0, le=1)
    probability_new: float = Field(default=0.10, ge=0, le=1)
    probability_false: float = Field(default=0.10, ge=0, le=1)
    minimum_association_probability: float = Field(default=0.20, ge=0, le=1)
    minimum_birth_probability: float = Field(default=0.45, ge=0, le=1)
    confirmation_existence_probability: float = Field(default=0.70, ge=0, le=1)
    deletion_existence_probability: float = Field(default=0.05, ge=0, le=1)
    survival_probability_per_second: float = Field(default=0.97**50, ge=0, le=1)
    measurement_std_deg: float = Field(default=5.0, gt=0)
    stationary_angle_std_deg: float = Field(default=0.35, gt=0)
    stationary_velocity_std_dps: float = Field(default=3.0, gt=0)
    stationary_velocity_half_life_seconds: float = Field(default=0.15, gt=0)
    moving_angle_std_deg: float = Field(default=1.25, gt=0)
    moving_velocity_std_dps: float = Field(default=15.0, gt=0)
    moving_velocity_half_life_seconds: float = Field(default=0.5, gt=0)
    stationary_to_moving_probability: float = Field(default=0.02, ge=0, le=1)
    moving_to_stationary_probability: float = Field(default=0.05, ge=0, le=1)
    prediction_freeze_std_deg: float = Field(default=25.0, gt=0)
    duplicate_birth_guard_deg: float = Field(default=15.0, gt=0, le=180)
    max_active_tracks: Literal[4] = 4

    @model_validator(mode="after")
    def validate_tracker_probabilities(self) -> "Layer2DirectionIdTrackingConfig":
        if abs(self.probability_track + self.probability_new + self.probability_false - 1.0) > 1e-9:
            raise ValueError("ID追踪track/new/false概率之和必须为1")
        return self


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
    context_ms: Literal[160, 240, 320]
    covariance_shrinkage: float = Field(ge=0, lt=1)
    diagonal_loading: float = Field(gt=0)
    eigenvalue_floor: float = Field(gt=0)
    min_valid_frequency_bins: int = Field(gt=0)
    direction_threshold: float
    peak_prominence: float
    min_peak_distance_deg: Literal[50.0]
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
    dpd_peak_fusion_distance_deg: float = Field(gt=0, le=50)
    dpd_peak_fusion_min_normalized_score: float = Field(ge=0, le=1)
    noise_whitening_enabled: bool = False
    noise_covariance_shrinkage: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_direction_postprocessing(self) -> "Layer2Config":
        if type(self.dpd_rank1_enabled) is not bool or type(self.noise_whitening_enabled) is not bool:
            raise TypeError("Layer 2 DPD/whitening switches must be bool")
        if self.dpd_min_cluster_subbands > self.dpd_frequency_subbands:
            raise ValueError("DPD minimum cluster subbands cannot exceed configured subbands")
        if self.dpd_peak_fusion_distance_deg > self.min_peak_distance_deg:
            raise ValueError("DPD peak fusion distance cannot exceed circular NMS distance")
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


class FeatureConfig(StrictModel):
    preprocessing_version: str
    frequency_min_hz: float
    frequency_max_hz: float
    first_bin: int
    last_bin_inclusive: int
    log_epsilon: float
    normalization: str


class Layer5ModelConfig(StrictModel):
    model_id: str
    backend: Literal["nvidia_marblenet_continuous_v2", "nvidia_marblenet_window_v1"]
    model_artifact: str
    role: Literal["primary", "shadow"]
    enabled: bool = True


class Layer5InputGainCompensationConfig(StrictModel):
    enabled: bool
    algorithm_version: Literal["imcra_probability_rms_v1"]
    target_rms_dbfs: float
    no_compensation_probability: float = Field(ge=0.0, le=1.0)
    full_compensation_probability: float = Field(ge=0.0, le=1.0)
    peak_ceiling_dbfs: float = Field(le=0.0)
    silence_floor_dbfs: float
    time_interpolation: Literal["linear_db"]

    @model_validator(mode="after")
    def validate_gain_compensation(self) -> "Layer5InputGainCompensationConfig":
        if self.no_compensation_probability >= self.full_compensation_probability:
            raise ValueError("Layer5 gain probability breakpoints must increase")
        if self.silence_floor_dbfs >= self.target_rms_dbfs:
            raise ValueError("Layer5 silence floor must be below the target RMS")
        return self


class Layer5Config(StrictModel):
    primary_model_id: str
    models: tuple[Layer5ModelConfig, ...]
    allow_mock: bool
    voice_probability_limit: float
    input_gain_compensation: Layer5InputGainCompensationConfig
    continuous_context_ms: int = Field(default=3_200, ge=60)

    @model_validator(mode="after")
    def validate_models(self) -> "Layer5Config":
        enabled = tuple(item for item in self.models if item.enabled)
        ids = tuple(item.model_id for item in enabled)
        if not enabled or len(ids) != len(set(ids)):
            raise ValueError("Layer5 enabled model ids must be non-empty and unique")
        primary = tuple(item for item in enabled if item.role == "primary")
        if len(primary) != 1 or primary[0].model_id != self.primary_model_id:
            raise ValueError("Layer5 must define exactly one enabled primary model")
        if not 0.0 <= self.voice_probability_limit <= 1.0:
            raise ValueError("Layer5 voice probability threshold must be in [0,1]")
        if self.continuous_context_ms % 20:
            raise ValueError("Layer5 continuous context must use complete 20 ms hops")
        return self


class Layer4StreamingConfig(StrictModel):
    enabled: bool = True
    chunk_seconds: int = Field(default=10, ge=3, le=15)
    overlap_seconds: int = Field(default=1, ge=1, le=14)
    queue_chunks: int = Field(default=2, ge=1, le=64)

    @model_validator(mode="after")
    def validate_streaming_window(self) -> "Layer4StreamingConfig":
        # Integer seconds are always exact multiples of the authoritative
        # 20 ms timeline hop.  Keep overlap strictly smaller so every admitted
        # chunk advances the downstream stream.
        if self.overlap_seconds >= self.chunk_seconds:
            raise ValueError("Layer4 streaming overlap must be shorter than its chunk")
        return self


class Layer4Config(StrictModel):
    enabled: bool = True
    default_backend: Literal["mossformer2_ss_16k", "tiger_speech_16k"]
    mossformer2_artifact: str
    tiger_artifact: str
    streaming: Layer4StreamingConfig

    @model_validator(mode="after")
    def validate_artifacts(self) -> "Layer4Config":
        if any(not value.strip() for value in (
            self.mossformer2_artifact, self.tiger_artifact,
        )):
            raise ValueError("Layer4 model artifact paths must be non-empty")
        return self


class Layer6Config(StrictModel):
    enabled: bool = True
    campplus_artifact: str
    dnsmos_artifact: str
    maximum_speakers: int = Field(default=5, ge=1, le=5)
    speaker_similarity_threshold: float = Field(default=0.62, ge=0.0, le=1.0)
    secondary_candidate_match_gap_max: float = Field(default=0.20, ge=0.0, le=1.0)
    secondary_candidate_match_min: float = Field(default=0.50, ge=0.0, le=1.0)
    secondary_candidate_mos_min: float = Field(default=0.30, ge=0.0, le=1.0)
    maximum_internal_silence_ms: int = Field(default=2_000, ge=0)
    clustering_backend: Literal["complete_link", "multistage"] = "multistage"
    multistage_l: int = Field(default=30, ge=1)
    multistage_u1: int = Field(default=100, ge=2)
    multistage_u2: int = Field(default=600, ge=3)
    multistage_fallback_distance: float = Field(default=0.38, gt=0.0, le=2.0)
    # A long capture can produce more unique 2 s voiceprint windows than the
    # clustering horizon needs at once.  Bound the content-addressed cache so
    # a 30+ minute session trades old-cache recomputation for stable RAM.
    embedding_cache_max_segments: int = Field(default=600, ge=1, le=100_000)

    @model_validator(mode="after")
    def validate_layer6(self) -> "Layer6Config":
        if not self.campplus_artifact.strip() or not self.dnsmos_artifact.strip():
            raise ValueError("Layer6 model artifact paths must be non-empty")
        if self.maximum_internal_silence_ms % 20:
            raise ValueError("Layer6 maximum_internal_silence_ms must align to 20 ms")
        if self.multistage_l > self.multistage_u1:
            raise ValueError("Layer6 multi-stage L must not exceed U1")
        if not self.maximum_speakers < self.multistage_u1 < self.multistage_u2:
            raise ValueError("Layer6 multi-stage limits must satisfy maximum_speakers < U1 < U2")
        return self


class RuntimeConfig(StrictModel):
    mode: Literal["development", "production"]
    # Legacy fallback retained for old launch profiles. New profiles select
    # the compute device independently for each accelerated layer.
    preferred_device: str
    l3_device: Literal["cpu", "cuda"] | None = None
    l4_device: Literal["cpu", "cuda"] | None = None
    l5_device: Literal["cpu", "cuda"] | None = None
    torch_cpu_threads: int = Field(default=1, gt=0, le=64)
    allow_cpu_fallback: bool
    max_candidate_batch: Literal[3]
    capture_handoff_blocks: int = Field(gt=0)
    # Kept for configuration compatibility.  The staged runtime uses the
    # per-stage capacities below rather than one shared serial queue.
    processing_queue_windows: int = Field(gt=0)
    # Normal deployments change only this shared value. Per-stage fields stay
    # optional for focused tests and exceptional diagnostic profiles.
    stage_queue_windows: int = Field(default=100, gt=0, le=10_000)
    l2_queue_windows: int | None = Field(default=None, gt=0, le=10_000)
    l3_queue_windows: int | None = Field(default=None, gt=0, le=10_000)
    l5_queue_windows: int | None = Field(default=None, gt=0, le=10_000)
    l3_cuda_microbatch_windows: int = Field(default=4, gt=0, le=16)
    l3_cuda_batch_wait_ms: float = Field(default=0.0, ge=0.0, le=20.0)
    completion_queue_windows: int = Field(default=8, gt=0, le=128)
    max_inflight_windows: int | None = Field(default=None, gt=0, le=30_003)
    compute_cache_max_bytes: int = Field(default=67_108_864, ge=8_388_608)
    # These are observable long-session guardrails.  Audio evidence itself is
    # spooled to disk; this budget describes the intended resident L4-L6
    # working set, while the disk reserve drives status/alerting on that spool
    # volume without hard-coding the progressive chunk duration.
    layer456_resident_memory_budget_bytes: int = Field(
        default=134_217_728, ge=16_777_216, le=4_294_967_296,
    )
    layer456_spool_min_free_bytes: int = Field(
        default=5_368_709_120, ge=0, le=1_099_511_627_776,
    )
    overflow_policy: Literal["drop_oldest"] = "drop_oldest"
    graceful_shutdown_timeout_seconds: float = Field(default=10.0, gt=0.0, le=120.0)

    @model_validator(mode="after")
    def validate_pipeline_capacity(self) -> "RuntimeConfig":
        for name in ("l2_queue_windows", "l3_queue_windows", "l5_queue_windows"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, self.stage_queue_windows)
        assert self.l2_queue_windows is not None
        assert self.l3_queue_windows is not None
        assert self.l5_queue_windows is not None
        # Joiner retains every admitted window until L2/L3/L5 are terminal.
        # Its hard bound must therefore cover all stage queues plus one active
        # item per worker; otherwise a valid configured backlog could fail at
        # registration before the advertised queue policy is reached.
        minimum = self.l2_queue_windows + self.l3_queue_windows + self.l5_queue_windows + 3
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
    schema_version: Literal["project_config_v2"]
    paths: PathsConfig
    hardware: HardwareConfig
    device: DeviceConfig
    calibration: CalibrationConfig
    timing: TimingConfig
    layer1_imcra: Layer1ImcraConfig
    layer1_pre_denoise: Layer1PreDenoiseConfig
    layer1_speaker_count: Layer1SpeakerCountConfig
    layer2: Layer2Config
    stft: StftConfig
    layer3: Layer3Config
    layer4: Layer4Config
    feature: FeatureConfig
    layer5: Layer5Config
    layer6: Layer6Config
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
        if duration_ms not in {40, 80, 160} or duration_ms <= 0 or duration_ms % 20:
            raise ValueError("downstream_audio_window_ms只能为40、80或160且必须为20 ms整数倍")
        spec = self.downstream_audio_window
        if spec.samples > self.timing.context_samples:
            raise ValueError("downstream audio window不能超过DecisionWindow直接上下文")
        if spec.samples % self.timing.decision_hop_samples:
            raise ValueError("downstream audio window必须包含完整decision hops")
        if not self.stft.center or spec.stft_frames != 1 + spec.samples // self.stft.hop_length:
            raise ValueError("downstream STFT frame derivation与当前center=true配置不一致")
        if spec.resampled_16k_samples not in {640, 1280, 2560}:
            raise ValueError("Layer5模型不支持所选downstream audio window")
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
                self.layer5.allow_mock
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
