from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from common.data_types import (
    CandidateDirection, ImcraHopSnapshot, PipelineStatus, SpatialResponse,
    TrackedDirection,
)
from layer2_source_detection.music import MusicDiagnostics
from layer2_source_detection.probability_gate import ProbabilityGateDecision
from layer4_voice_classifier.contracts import Layer4Result


def _readonly(value: object, dtype: object, name: str):
    raw = np.asarray(value, dtype=dtype)
    if raw.shape != (8,):
        raise ValueError(f"{name}必须为[8]")
    return np.frombuffer(np.ascontiguousarray(raw).tobytes(), dtype=raw.dtype)


@dataclass(frozen=True, slots=True)
class L1MeterSnapshot:
    session_id: str
    stream_epoch: int
    end_sample: int
    sequence_id: int
    rms_dbfs: NDArray[np.float32]
    peak_dbfs: NDArray[np.float32]
    clipped: NDArray[np.bool_]
    light_state: str
    recording_state: str
    imcra_hop: ImcraHopSnapshot | None = None
    pre_denoise_enabled: bool = False
    pre_denoise_mean_gain_db: float = 0.0
    calibration_status: str = "unverified"
    calibration_version: str = "unversioned"
    calibration_hash: str = "0" * 64

    def __post_init__(self) -> None:
        if not self.session_id or min(self.stream_epoch, self.end_sample, self.sequence_id) < 0:
            raise ValueError("L1MeterSnapshot标识无效")
        if self.light_state not in {"on", "off", "unknown", "error"}:
            raise ValueError("light_state无效")
        if self.recording_state not in {"idle", "recording", "paused", "finalizing", "complete", "error"}:
            raise ValueError("recording_state无效")
        if type(self.pre_denoise_enabled) is not bool or not np.isfinite(self.pre_denoise_mean_gain_db):
            raise ValueError("L1 pre-denoise meter state is invalid")
        if self.calibration_status not in {"verified", "unverified"}:
            raise ValueError("L1 calibration status must be verified or unverified")
        if not self.calibration_version or len(self.calibration_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.calibration_hash
        ):
            raise ValueError("L1 calibration version/hash is invalid")
        rms, peak = _readonly(self.rms_dbfs, np.float32, "rms_dbfs"), _readonly(self.peak_dbfs, np.float32, "peak_dbfs")
        if not np.isfinite(rms).all() or not np.isfinite(peak).all():
            raise ValueError("meter数值必须finite")
        object.__setattr__(self, "rms_dbfs", rms)
        object.__setattr__(self, "peak_dbfs", peak)
        object.__setattr__(self, "clipped", _readonly(self.clipped, np.bool_, "clipped"))
        if self.imcra_hop is not None and (
            self.imcra_hop.session_id != self.session_id
            or self.imcra_hop.stream_epoch != self.stream_epoch
            or self.imcra_hop.end_sample != self.end_sample
        ):
            raise ValueError("L1 meter IMCRA snapshot must align with the meter block")


@dataclass(frozen=True, slots=True)
class BeamformPreview:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    theta_deg: float
    waveform: NDArray[np.float32]
    runtime_backend: str
    fallback_reason: str | None = None
    diagnostics: tuple[str, ...] = ()
    track_id: int | None = None
    track_state: str | None = None

    def __post_init__(self) -> None:
        waveform = np.asarray(self.waveform)
        if (
            waveform.shape not in {(1_920,), (3_840,), (7_680,)}
            or waveform.dtype != np.float32
            or not np.isfinite(waveform).all()
        ):
            raise ValueError("BeamformPreview waveform必须为finite float32 [1920、3840或7680]")
        object.__setattr__(self, "waveform", np.frombuffer(np.ascontiguousarray(waveform).tobytes(), dtype=np.float32))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        if self.track_id is not None and (type(self.track_id) is not int or self.track_id <= 0):
            raise ValueError("BeamformPreview track_id must be a positive integer")
        if self.track_state is not None and self.track_state not in {
            "tentative", "confirmed", "coasting"
        }:
            raise ValueError("BeamformPreview track_state is invalid")


@dataclass(frozen=True, slots=True)
class TrackedAudioSnapshot:
    """Test-UI-only identity and bounded disk-cache status."""

    session_id: str
    stream_epoch: int
    track_id: int
    state: str
    theta_deg: float
    score: float
    audio_sample_count: int
    sample_rate: int = 48_000
    waveform_envelope: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.session_id or min(self.stream_epoch, self.track_id, self.audio_sample_count) < 0:
            raise ValueError("tracked audio identity is invalid")
        if self.state not in {"active", "coasting", "ended"}:
            raise ValueError("tracked audio state is invalid")
        if self.sample_rate != 48_000 or not 0.0 <= self.theta_deg < 360.0:
            raise ValueError("tracked audio angle/sample rate is invalid")
        if not np.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("tracked audio score must be in [0,1]")
        envelope = tuple(float(item) for item in self.waveform_envelope)
        if any(not np.isfinite(item) or item < 0.0 for item in envelope):
            raise ValueError("tracked audio waveform envelope must be finite and non-negative")
        object.__setattr__(self, "waveform_envelope", envelope)

    @property
    def duration_seconds(self) -> float:
        return self.audio_sample_count / self.sample_rate


@dataclass(frozen=True, slots=True)
class AlgorithmPerformanceSnapshot:
    session_id: str
    stream_epoch: int
    window_id: int | None
    warmup_buffered_samples: int
    warmup_required_samples: int
    configured_sample_rate_hz: int
    observed_sample_rate_hz: float | None
    compute_time_ms_current: float | None
    compute_time_ms_p50: float | None
    compute_time_ms_p95: float | None
    latency_ms_current: float | None
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    metrics_source: str | None = None
    metrics_unavailable_reason: str | None = "CNN model unavailable"
    model_version: str | None = None
    dataset_version: str | None = None
    evaluation_threshold: float | None = None
    l2_time_ms_last_second_avg: float | None = None
    l3_time_ms_last_second_avg: float | None = None
    l4_time_ms_last_second_avg: float | None = None
    l2_refresh_hz_last_second: float = 0.0
    l3_refresh_hz_last_second: float = 0.0
    l4_refresh_hz_last_second: float = 0.0
    processed_windows_last_second: int = 0
    dropped_windows_last_second: int = 0
    drop_rate_last_second: float = 0.0

    def __post_init__(self) -> None:
        if self.warmup_required_samples <= 0 or self.configured_sample_rate_hz <= 0:
            raise ValueError("性能快照采样参数无效")
        values = (
            self.compute_time_ms_current,
            self.compute_time_ms_p50,
            self.compute_time_ms_p95,
            self.latency_ms_current,
            self.latency_ms_p50,
            self.latency_ms_p95,
            self.l2_time_ms_last_second_avg,
            self.l3_time_ms_last_second_avg,
            self.l4_time_ms_last_second_avg,
            self.l2_refresh_hz_last_second,
            self.l3_refresh_hz_last_second,
            self.l4_refresh_hz_last_second,
        )
        if any(value is not None and (not np.isfinite(value) or value < 0) for value in values):
            raise ValueError("性能时间必须为非负finite或None")
        if (
            type(self.processed_windows_last_second) is not int
            or self.processed_windows_last_second < 0
            or type(self.dropped_windows_last_second) is not int
            or self.dropped_windows_last_second < 0
        ):
            raise ValueError("上一秒窗口计数必须为非负整数")
        if (
            not np.isfinite(self.drop_rate_last_second)
            or not 0.0 <= self.drop_rate_last_second <= 1.0
        ):
            raise ValueError("上一秒丢窗率必须位于[0,1]")
        if (
            self.compute_time_ms_current is not None
            and self.latency_ms_current is not None
            and self.latency_ms_current < self.compute_time_ms_current
        ):
            raise ValueError("Latency必须不小于Compute")


@dataclass(frozen=True, slots=True)
class DevUiFrame:
    l1: L1MeterSnapshot | None
    spatial_response: SpatialResponse | None
    candidates: tuple[CandidateDirection, ...]
    previews: tuple[BeamformPreview, ...]
    tracked_audio: tuple[TrackedAudioSnapshot, ...]
    gate_decision: ProbabilityGateDecision | None
    gate_threshold: float | None
    gate_config_revision: int | None
    direction_threshold: float | None
    direction_kalman_enabled: bool | None
    direction_kalman_q_scale: float | None
    direction_kalman_r_scale: float | None
    scan_config_revision: int | None
    pipeline_status: PipelineStatus
    performance: AlgorithmPerformanceSnapshot
    published_monotonic: float
    spatial_published_monotonic: float | None
    search_diagnostics: MusicDiagnostics | None
    missing_reasons: Mapping[str, str]
    l4_result: Layer4Result | None = None
    directions: tuple[TrackedDirection, ...] = ()
    active_tracks: tuple[TrackedDirection, ...] = ()

    def __post_init__(self) -> None:
        directions = tuple(self.directions)
        active_tracks = tuple(self.active_tracks) or directions
        if any((item.session_id, item.stream_epoch) != (
            self.pipeline_status.session_id, self.pipeline_status.stream_epoch
        ) for item in (*directions, *active_tracks)):
            raise ValueError("TrackedDirection rows must match the pipeline stream")
        direction_identities = {
            (item.session_id, item.stream_epoch, item.window_id, item.decision_sample)
            for item in directions
        }
        if len(direction_identities) > 1:
            raise ValueError("DevUiFrame directions must belong to one authoritative L2 window")
        direction_identity = next(iter(direction_identities), None)
        if len({item.track_id for item in active_tracks}) != len(active_tracks):
            raise ValueError("active_tracks must contain unique authoritative L2 IDs")
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "active_tracks", active_tracks)
        if not np.isfinite(self.published_monotonic):
            raise ValueError("published_monotonic必须finite")
        if self.spatial_published_monotonic is not None and not np.isfinite(
            self.spatial_published_monotonic
        ):
            raise ValueError("spatial_published_monotonic must be finite when present")
        pipeline_stream = (
            self.pipeline_status.session_id,
            self.pipeline_status.stream_epoch,
        )
        if any((item.session_id, item.stream_epoch) != pipeline_stream for item in self.candidates):
            raise ValueError("DevUiFrame candidates must match the pipeline stream")
        if self.l1 is not None and (self.l1.session_id, self.l1.stream_epoch) != pipeline_stream:
            raise ValueError("DevUiFrame L1 snapshot must match the pipeline stream")
        if self.spatial_response is None and self.candidates:
            raise ValueError("没有SpatialResponse时不能携带候选")
        if len(self.candidates) > 3:
            raise ValueError("DevUiFrame cannot publish more than 3 Layer 2 candidates")
        if self.spatial_response is None and self.previews and direction_identity is None:
            raise ValueError(
                "DevUiFrame prediction-only L3 previews require authoritative L2 directions"
            )
        if self.spatial_response is None and self.search_diagnostics is not None:
            raise ValueError("DevUiFrame cannot publish search diagnostics without an SRP response")
        if self.spatial_response is not None:
            identity = (
                self.spatial_response.session_id,
                self.spatial_response.stream_epoch,
                self.spatial_response.window_id,
                self.spatial_response.decision_sample,
            )
            response_identity = (
                *identity,
                self.spatial_response.doa_start_sample,
                self.spatial_response.doa_end_sample,
            )
            if identity[:2] != pipeline_stream:
                raise ValueError("DevUiFrame SRP response must match the pipeline stream")
            for candidate in self.candidates:
                candidate_identity = (
                    candidate.session_id,
                    candidate.stream_epoch,
                    candidate.window_id,
                    candidate.decision_sample,
                    candidate.doa_start_sample,
                    candidate.doa_end_sample,
                )
                if candidate_identity != response_identity:
                    raise ValueError("DevUiFrame不能混合不同window")
            for preview in self.previews:
                if (preview.session_id, preview.stream_epoch, preview.window_id, preview.decision_sample) != identity:
                    raise ValueError("DevUiFrame预览不能混合不同window")
        elif direction_identity is not None:
            if any(
                (item.session_id, item.stream_epoch, item.window_id, item.decision_sample)
                != direction_identity
                for item in self.previews
            ):
                raise ValueError("prediction-only L3 previews must match the authoritative L2 window")
            direction_ids = tuple(item.track_id for item in directions)
            preview_ids = tuple(item.track_id for item in self.previews)
            if self.previews and (
                any(item is None for item in preview_ids) or preview_ids != direction_ids
            ):
                raise ValueError("prediction-only L3 previews must preserve authoritative L2 IDs")
            if any(item.track_state not in {"confirmed", "coasting"} for item in directions):
                raise ValueError("prediction-only L3 previews require formal L2 IDs")
        for track in self.tracked_audio:
            if (track.session_id, track.stream_epoch) != pipeline_stream:
                raise ValueError("DevUiFrame tracked audio must match the pipeline stream")
        if self.l4_result is not None:
            if self.l4_result.primary_model_id not in {
                item.model_id for item in self.l4_result.predictions
            }:
                raise ValueError("DevUiFrame L4 primary model is missing")
            for detection in self.l4_result.detections:
                if (detection.session_id, detection.stream_epoch) != pipeline_stream:
                    raise ValueError("DevUiFrame L4 result must match the pipeline stream")
                if self.spatial_response is not None and (
                    detection.window_id != self.spatial_response.window_id
                    or detection.decision_sample != self.spatial_response.decision_sample
                ):
                    raise ValueError("DevUiFrame L4 result must match the SRP window")
                if self.spatial_response is None and direction_identity is not None and (
                    detection.session_id,
                    detection.stream_epoch,
                    detection.window_id,
                    detection.decision_sample,
                ) != direction_identity:
                    raise ValueError("prediction-only L4 result must match the authoritative L2 window")
            if self.spatial_response is None and self.l4_result.detections:
                if direction_identity is None:
                    raise ValueError("prediction-only L4 result requires authoritative L2 directions")
                if tuple(item.track_id for item in self.l4_result.detections) != tuple(
                    item.track_id for item in directions
                ):
                    raise ValueError("prediction-only L4 result must preserve authoritative L2 IDs")
        if self.gate_decision is not None:
            gate_identity = (
                self.gate_decision.session_id,
                self.gate_decision.stream_epoch,
                self.gate_decision.window_id,
                self.gate_decision.decision_sample,
            )
            if self.spatial_response is not None and gate_identity != identity:
                raise ValueError("DevUiFrame gate decision must match the SRP window")
            if self.spatial_response is None and direction_identity is not None and (
                gate_identity != direction_identity
            ):
                raise ValueError("prediction-only directions must match the Gate window")
            if gate_identity[:2] != pipeline_stream:
                raise ValueError("DevUiFrame gate decision must match the pipeline stream")
        if self.gate_decision is not None:
            if self.spatial_response is None and self.gate_decision.allow_srp:
                raise ValueError("an open/bypassed Gate requires an SRP response")
            if self.spatial_response is not None and not self.gate_decision.allow_srp:
                raise ValueError("a blocked Gate cannot publish an SRP response")
        scan_fields = (
            self.direction_threshold,
            self.direction_kalman_enabled,
            self.direction_kalman_q_scale,
            self.direction_kalman_r_scale,
            self.scan_config_revision,
        )
        if any(value is not None for value in scan_fields) and not all(value is not None for value in scan_fields):
            raise ValueError("DevUiFrame scan settings must be published atomically")
        gate_fields = (self.gate_threshold, self.gate_config_revision)
        if any(value is not None for value in gate_fields) and not all(value is not None for value in gate_fields):
            raise ValueError("DevUiFrame Gate settings must be published atomically")
        if self.gate_decision is not None and (
            self.gate_threshold != self.gate_decision.threshold
            or self.gate_config_revision != self.gate_decision.config_revision
        ):
            raise ValueError("DevUiFrame Gate settings must match the applied decision")
        has_l2_result = self.spatial_response is not None or self.gate_decision is not None
        if has_l2_result and self.direction_threshold is None:
            raise ValueError("DevUiFrame Layer 2 results require applied scan settings")
        if self.direction_threshold is not None and (
            type(self.direction_threshold) not in {float, int}
            or not np.isfinite(self.direction_threshold)
            or not 0.0 <= self.direction_threshold <= 1.0
        ):
            raise ValueError("DevUiFrame direction threshold must be finite and in [0,1]")
        if self.direction_kalman_enabled is not None and type(self.direction_kalman_enabled) is not bool:
            raise ValueError("DevUiFrame Kalman setting must be bool")
        for name, value in (
            ("Q", self.direction_kalman_q_scale),
            ("R", self.direction_kalman_r_scale),
        ):
            if value is not None and (
                not np.isfinite(value)
                or not 0.02 <= value <= 10.0
                or (value != 0.02 and abs(value * 10.0 - round(value * 10.0)) > 1.0e-9)
            ):
                raise ValueError(f"DevUiFrame Kalman {name} scale is invalid")
        if self.scan_config_revision is not None and (
            type(self.scan_config_revision) is not int or self.scan_config_revision < 0
        ):
            raise ValueError("DevUiFrame scan config revision must be a non-negative int")
        if self.spatial_response is not None and self.search_diagnostics is None:
            raise ValueError("DevUiFrame SRP response requires search diagnostics")
        if self.search_diagnostics is not None and (
            self.scan_config_revision != self.search_diagnostics.config_revision
            or self.search_diagnostics.mode != "frequency_normalized_music"
        ):
            raise ValueError("DevUiFrame MUSIC diagnostics/config revision mismatch")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "previews", tuple(self.previews))
        object.__setattr__(self, "tracked_audio", tuple(self.tracked_audio))
        object.__setattr__(self, "missing_reasons", dict(self.missing_reasons))

    @property
    def age_ms(self) -> float:
        return max(0.0, (monotonic() - self.published_monotonic) * 1_000.0)
