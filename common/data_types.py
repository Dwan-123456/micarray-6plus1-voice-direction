from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


_IMCRA_FREQUENCIES_HZ = np.fft.rfftfreq(2048, 1.0 / 48_000).astype(np.float32)
_IMCRA_FREQUENCIES_HZ = _IMCRA_FREQUENCIES_HZ[
    _IMCRA_FREQUENCIES_HZ <= 8_000.0
]
_IMCRA_BIN_COUNT = int(_IMCRA_FREQUENCIES_HZ.size)


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class CalibrationAssetIdentity:
    """Versioned identity for a future calibration asset; payloads stay outside audio DTOs."""

    uri: str
    version: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.uri or not self.version or not _valid_sha256(self.sha256):
            raise ValueError("calibration asset requires uri, version, and lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class CalibrationMetadata:
    """Immutable L1 correction boundary propagated with every calibrated stream."""

    status: str
    version: str
    calibration_hash: str
    correction_model: str
    report_hash: str | None = None
    fractional_delay_asset: CalibrationAssetIdentity | None = None
    frequency_response_asset: CalibrationAssetIdentity | None = None

    def __post_init__(self) -> None:
        if self.status not in {"verified", "unverified"}:
            raise ValueError("calibration status must be verified or unverified")
        if not self.version or not self.correction_model or not _valid_sha256(self.calibration_hash):
            raise ValueError("calibration metadata requires version, model, and lowercase SHA-256")
        if self.report_hash is not None and not _valid_sha256(self.report_hash):
            raise ValueError("calibration report hash must be lowercase SHA-256 or None")
        for asset in (self.fractional_delay_asset, self.frequency_response_asset):
            if asset is not None and not isinstance(asset, CalibrationAssetIdentity):
                raise TypeError("calibration extension identity has an invalid type")

    @classmethod
    def unverified_identity(cls) -> "CalibrationMetadata":
        payload = b"unverified_identity_gain_polarity_integer_delay_v1"
        return cls(
            "unverified",
            "unverified_identity_v1",
            hashlib.sha256(payload).hexdigest(),
            "gain_polarity_integer_delay_v1",
        )


def _readonly_float32(value: object, shape: tuple[int | None, ...], name: str) -> NDArray[np.float32]:
    raw = np.asarray(value)
    if raw.ndim != len(shape) or any(
        expected is not None and raw.shape[index] != expected for index, expected in enumerate(shape)
    ):
        raise ValueError(f"{name} shape必须为 {shape}，实际为 {raw.shape}")
    if raw.size == 0 or not np.isfinite(raw).all():
        raise ValueError(f"{name}必须非空且全部为有限数值")
    # An immutable bytes owner makes the promise stronger than setflags(False)
    # on an array whose original owner could later re-enable writes.
    return np.frombuffer(np.ascontiguousarray(raw, dtype=np.float32).tobytes(), dtype=np.float32).reshape(raw.shape)


@dataclass(frozen=True, slots=True)
class ImcraHopSnapshot:
    """One L1 IMCRA result aligned to an exact 20 ms audio hop."""

    session_id: str
    stream_epoch: int
    start_sample: int
    end_sample: int
    source_sequence_ids: tuple[int, ...]
    algorithm_version: str
    state: str
    frequencies_hz: NDArray[np.float32]
    noise_psd: NDArray[np.float32]
    smoothed_psd: NDArray[np.float32]
    conditional_smoothed_psd: NDArray[np.float32]
    minimum_psd: NDArray[np.float32]
    conditional_minimum_psd: NDArray[np.float32]
    spp: NDArray[np.float32]
    speech_absence_probability: NDArray[np.float32]
    posterior_snr: NDArray[np.float32]
    prior_snr: NDArray[np.float32]
    noise_features: NDArray[np.float32]
    noise_level_db: NDArray[np.float32]
    source_probability_per_mic: NDArray[np.float32]
    array_source_probability_20ms: float | None

    def __post_init__(self) -> None:
        if not self.session_id or min(
            self.stream_epoch, self.start_sample
        ) < 0:
            raise ValueError("IMCRA hop identity is invalid")
        if self.end_sample - self.start_sample != 960:
            raise ValueError("IMCRA hop must span exactly 960 samples")
        sequence_ids = tuple(dict.fromkeys(self.source_sequence_ids))
        if not sequence_ids or any(value < 0 for value in sequence_ids):
            raise ValueError("IMCRA source_sequence_ids are invalid")
        if not self.algorithm_version or self.state not in {"warming_up", "ready", "invalid"}:
            raise ValueError("IMCRA estimator/state is invalid")
        frequencies = _readonly_float32(self.frequencies_hz, (_IMCRA_BIN_COUNT,), "frequencies_hz")
        if not np.array_equal(frequencies, _IMCRA_FREQUENCIES_HZ):
            raise ValueError("IMCRA frequency axis must be the 48 kHz/2048-point 0-8000 Hz bins")
        spectral_shape = (7, _IMCRA_BIN_COUNT)
        noise_psd = _readonly_float32(self.noise_psd, spectral_shape, "noise_psd")
        smoothed_psd = _readonly_float32(self.smoothed_psd, spectral_shape, "smoothed_psd")
        conditional_smoothed_psd = _readonly_float32(
            self.conditional_smoothed_psd, spectral_shape, "conditional_smoothed_psd"
        )
        minimum_psd = _readonly_float32(self.minimum_psd, spectral_shape, "minimum_psd")
        conditional_minimum_psd = _readonly_float32(
            self.conditional_minimum_psd, spectral_shape, "conditional_minimum_psd"
        )
        spp = _readonly_float32(self.spp, spectral_shape, "spp")
        absence = _readonly_float32(
            self.speech_absence_probability, spectral_shape, "speech_absence_probability"
        )
        posterior_snr = _readonly_float32(self.posterior_snr, spectral_shape, "posterior_snr")
        prior_snr = _readonly_float32(self.prior_snr, spectral_shape, "prior_snr")
        features = _readonly_float32(self.noise_features, (7, 4), "noise_features")
        if any(np.any(item < 0.0) for item in (
            noise_psd, smoothed_psd, conditional_smoothed_psd, minimum_psd,
            conditional_minimum_psd, posterior_snr, prior_snr,
        )):
            raise ValueError("IMCRA PSD state must be non-negative")
        if np.any((spp < 0.0) | (spp > 1.0)) or np.any((absence < 0.0) | (absence > 1.0)):
            raise ValueError("IMCRA probabilities must be in [0,1]")
        noise = _readonly_float32(self.noise_level_db, (7,), "noise_level_db")
        probability = _readonly_float32(
            self.source_probability_per_mic, (7,), "source_probability_per_mic"
        )
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError("per-microphone IMCRA probabilities must be in [0,1]")
        array_probability = self.array_source_probability_20ms
        if array_probability is not None and (
            not np.isfinite(array_probability) or not 0.0 <= array_probability <= 1.0
        ):
            raise ValueError("array IMCRA probability must be finite and in [0,1]")
        if self.state == "ready" and array_probability is None:
            raise ValueError("ready IMCRA hop requires an array probability")
        if self.state != "ready" and array_probability is not None:
            raise ValueError("non-ready IMCRA hop cannot publish an array probability")
        object.__setattr__(self, "source_sequence_ids", sequence_ids)
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "noise_psd", noise_psd)
        object.__setattr__(self, "smoothed_psd", smoothed_psd)
        object.__setattr__(self, "conditional_smoothed_psd", conditional_smoothed_psd)
        object.__setattr__(self, "minimum_psd", minimum_psd)
        object.__setattr__(self, "conditional_minimum_psd", conditional_minimum_psd)
        object.__setattr__(self, "spp", spp)
        object.__setattr__(self, "speech_absence_probability", absence)
        object.__setattr__(self, "posterior_snr", posterior_snr)
        object.__setattr__(self, "prior_snr", prior_snr)
        object.__setattr__(self, "noise_features", features)
        object.__setattr__(self, "noise_level_db", noise)
        object.__setattr__(self, "source_probability_per_mic", probability)
        if array_probability is not None:
            object.__setattr__(self, "array_source_probability_20ms", float(array_probability))


@dataclass(frozen=True, slots=True)
class IngestedAudioBlock:
    session_id: str
    stream_epoch: int
    start_sample: int
    end_sample: int
    sample_rate: int
    sequence_id: int
    timestamp: float
    samples: NDArray[np.float32]
    native_samples: NDArray[np.float32] | None = None
    hotmap: object | None = None
    noise_spectrum: object | None = None
    imcra_hop: ImcraHopSnapshot | None = None
    calibration: CalibrationMetadata = field(default_factory=CalibrationMetadata.unverified_identity)

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id不能为空")
        if self.stream_epoch < 0 or self.start_sample < 0 or self.sequence_id < 0:
            raise ValueError("epoch、sample index和sequence_id不能为负")
        if self.sample_rate != 48_000 or not np.isfinite(self.timestamp):
            raise ValueError("IngestedAudioBlock必须为48 kHz且timestamp有限")
        samples = _readonly_float32(self.samples, (None, 8), "samples")
        if self.end_sample - self.start_sample != samples.shape[0]:
            raise ValueError("sample边界与samples长度不一致")
        native = None
        if self.native_samples is not None:
            native = _readonly_float32(self.native_samples, (samples.shape[0], 8), "native_samples")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "native_samples", native)
        object.__setattr__(self, "timestamp", float(self.timestamp))
        if not isinstance(self.calibration, CalibrationMetadata):
            raise TypeError("IngestedAudioBlock calibration must be CalibrationMetadata")
        if self.imcra_hop is not None:
            hop = self.imcra_hop
            if (
                hop.session_id != self.session_id
                or hop.stream_epoch != self.stream_epoch
                or hop.start_sample != self.start_sample
                or hop.end_sample != self.end_sample
                or hop.source_sequence_ids != (self.sequence_id,)
            ):
                raise ValueError("IMCRA hop must align exactly with its audio block")


@dataclass(frozen=True, slots=True)
class DecisionWindow:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    doa_start_sample: int
    doa_end_sample: int
    context_start_sample: int
    context_end_sample: int
    sample_rate: int
    samples: NDArray[np.float32]
    source_sequence_ids: tuple[int, ...]
    imcra_hops: tuple[ImcraHopSnapshot, ...] = ()
    calibration: CalibrationMetadata = field(default_factory=CalibrationMetadata.unverified_identity)

    def __post_init__(self) -> None:
        if not self.session_id or min(self.stream_epoch, self.window_id, self.context_start_sample) < 0:
            raise ValueError("DecisionWindow标识和边界无效")
        if self.sample_rate != 48_000:
            raise ValueError("DecisionWindow采样率必须为48000")
        if not isinstance(self.calibration, CalibrationMetadata):
            raise TypeError("DecisionWindow calibration must be CalibrationMetadata")
        if self.doa_end_sample != self.context_end_sample or self.context_end_sample != self.decision_sample:
            raise ValueError("DOA、context与decision endpoint必须相同")
        if self.doa_end_sample - self.doa_start_sample != 1_920:
            raise ValueError("DOA窗口必须为1920 samples")
        if self.context_end_sample - self.context_start_sample != 15_360:
            raise ValueError("context窗口必须为15360 samples")
        if not self.source_sequence_ids or any(value < 0 for value in self.source_sequence_ids):
            raise ValueError("source_sequence_ids不能为空或包含负值")
        raw_samples = np.asarray(self.samples)
        if raw_samples.shape != (15_360, 8):
            raise ValueError(f"samples shape必须为 (15360, 8)，实际为 {raw_samples.shape}")
        object.__setattr__(
            self,
            "samples",
            _readonly_float32(self.samples, raw_samples.shape, "samples"),
        )
        object.__setattr__(self, "source_sequence_ids", tuple(dict.fromkeys(self.source_sequence_ids)))
        hops = tuple(self.imcra_hops)
        if any(
            hop.session_id != self.session_id
            or hop.stream_epoch != self.stream_epoch
            or hop.start_sample < self.context_start_sample
            or hop.end_sample > self.context_end_sample
            for hop in hops
        ):
            raise ValueError("DecisionWindow IMCRA hops must align with its stream and context")
        if tuple(hop.end_sample for hop in hops) != tuple(
            sorted({hop.end_sample for hop in hops})
        ):
            raise ValueError("DecisionWindow IMCRA hops must be unique and chronological")
        object.__setattr__(self, "imcra_hops", hops)

    @property
    def physical_samples(self) -> NDArray[np.float32]:
        """Continuous calibrated MIC0..MIC5+Center history; HardwareMix is excluded."""

        return self.samples[:, :7]

    @property
    def hardware_mix(self) -> NDArray[np.float32]:
        return self.samples[:, 7]

    @property
    def available_history_samples(self) -> int:
        return self.context_end_sample - self.context_start_sample

    @property
    def rolling_update_start_sample(self) -> int:
        """Start of the newest 20 ms that incremental L2 state should append."""

        return self.decision_sample - 960

    @property
    def rolling_state_key(self) -> tuple[str, int, int]:
        return self.session_id, self.stream_epoch, self.decision_sample

    def is_contiguous_successor_of(self, previous: "DecisionWindow") -> bool:
        return (
            (self.session_id, self.stream_epoch) == (previous.session_id, previous.stream_epoch)
            and self.decision_sample == previous.decision_sample + 960
        )

    def physical_history(self, context_ms: int) -> NDArray[np.float32]:
        """Return one configured MUSIC comparison history without HardwareMix."""

        if context_ms not in {160, 240, 320}:
            raise ValueError("MUSIC context_ms must be one of 160/240/320")
        sample_count = context_ms * 48
        return self.physical_samples[-sample_count:]

    def physical_history_start_sample(self, context_ms: int) -> int:
        if context_ms not in {160, 240, 320}:
            raise ValueError("MUSIC context_ms must be one of 160/240/320")
        return self.decision_sample - context_ms * 48


def _readonly_exact_float32(value: object, shape: tuple[int, ...], name: str) -> NDArray[np.float32]:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype != np.float32:
        raise ValueError(f"{name}必须为float32 {shape}")
    if not raw.flags.c_contiguous or not np.isfinite(raw).all():
        raise ValueError(f"{name}必须C-contiguous且全部finite")
    return np.frombuffer(raw.tobytes(), dtype=np.float32).reshape(shape)


@dataclass(frozen=True, slots=True)
class SpatialResponse:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    doa_start_sample: int
    doa_end_sample: int
    theta_degrees: NDArray[np.float32]
    raw_scores: NDArray[np.float32]
    normalized_scores: NDArray[np.float32]
    model_order: ModelOrderEstimate | None = None
    valid_frequency_bins: int | None = None
    numerical_status: str | None = None
    algorithm_version: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id or min(self.stream_epoch, self.window_id, self.doa_start_sample) < 0:
            raise ValueError("SpatialResponse标识或边界无效")
        if self.doa_end_sample != self.decision_sample or self.doa_end_sample - self.doa_start_sample != 1_920:
            raise ValueError("SpatialResponse的DOA边界无效")
        theta = _readonly_exact_float32(self.theta_degrees, (360,), "theta_degrees")
        raw = _readonly_exact_float32(self.raw_scores, (360,), "raw_scores")
        normalized = _readonly_exact_float32(self.normalized_scores, (360,), "normalized_scores")
        if not np.array_equal(theta, np.arange(360, dtype=np.float32)):
            raise ValueError("theta_degrees必须严格为0..359")
        if np.any((normalized < 0.0) | (normalized > 1.0)):
            raise ValueError("normalized_scores必须位于[0,1]")
        if self.model_order is not None and not isinstance(self.model_order, ModelOrderEstimate):
            raise TypeError("SpatialResponse model_order must be ModelOrderEstimate")
        if self.valid_frequency_bins is not None and self.valid_frequency_bins < 0:
            raise ValueError("SpatialResponse valid_frequency_bins must be non-negative")
        if self.numerical_status is not None and self.numerical_status not in {
            "warming_up", "ready", "degraded", "failed"
        }:
            raise ValueError("SpatialResponse numerical_status is invalid")
        if self.algorithm_version is not None and not self.algorithm_version:
            raise ValueError("SpatialResponse algorithm_version cannot be empty")
        object.__setattr__(self, "theta_degrees", theta)
        object.__setattr__(self, "raw_scores", raw)
        object.__setattr__(self, "normalized_scores", normalized)


@dataclass(frozen=True, slots=True)
class CandidateDirection:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    doa_start_sample: int
    doa_end_sample: int
    theta_deg: float
    raw_score: float
    normalized_score: float

    def __post_init__(self) -> None:
        values = (self.theta_deg, self.raw_score, self.normalized_score)
        if not self.session_id or min(self.stream_epoch, self.window_id, self.doa_start_sample) < 0:
            raise ValueError("CandidateDirection标识或边界无效")
        if self.doa_end_sample != self.decision_sample or self.doa_end_sample - self.doa_start_sample != 1_920:
            raise ValueError("CandidateDirection的DOA边界无效")
        if not all(np.isfinite(value) for value in values):
            raise ValueError("CandidateDirection分数与角度必须finite")
        if not 0.0 <= self.theta_deg < 360.0 or not 0.0 <= self.normalized_score <= 1.0:
            raise ValueError("CandidateDirection角度或归一化分数越界")


@dataclass(frozen=True, slots=True)
class TrackedDirection:
    """Authoritative Layer-2 direction-track result.

    ``track_id`` identifies a spatial direction trajectory within one session;
    it is deliberately not a person or speaker identity.
    """

    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    doa_start_sample: int
    doa_end_sample: int
    track_id: int
    rank: int
    measured_theta_deg: float | None
    theta_deg: float
    raw_score: float
    normalized_score: float
    track_state: str
    is_observed: bool
    is_new_track: bool
    first_seen_sample: int
    last_observed_sample: int
    missed_samples: int
    kalman_applied: bool
    is_noise_interference: bool = False

    def __post_init__(self) -> None:
        if not self.session_id or min(
            self.stream_epoch, self.window_id, self.doa_start_sample,
            self.track_id, self.rank, self.first_seen_sample,
            self.last_observed_sample, self.missed_samples,
        ) < 0:
            raise ValueError("TrackedDirection identity/lifetime is invalid")
        if self.track_id == 0 or self.rank == 0:
            raise ValueError("TrackedDirection track_id and rank are one-based")
        if self.doa_end_sample != self.decision_sample or self.doa_end_sample - self.doa_start_sample != 1_920:
            raise ValueError("TrackedDirection DOA boundary is invalid")
        values = (self.theta_deg, self.raw_score, self.normalized_score)
        if self.measured_theta_deg is not None:
            values += (self.measured_theta_deg,)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("TrackedDirection angle/score must be finite")
        if not 0.0 <= self.theta_deg < 360.0 or (
            self.measured_theta_deg is not None and not 0.0 <= self.measured_theta_deg < 360.0
        ):
            raise ValueError("TrackedDirection angle must be in [0,360)")
        if not 0.0 <= self.normalized_score <= 1.0:
            raise ValueError("TrackedDirection normalized_score must be in [0,1]")
        if self.track_state not in {"tentative", "confirmed", "coasting"}:
            raise ValueError("TrackedDirection state is invalid")
        if self.is_observed != (self.measured_theta_deg is not None):
            raise ValueError("TrackedDirection observed state is inconsistent")
        if self.last_observed_sample > self.decision_sample:
            raise ValueError("TrackedDirection observation cannot be in the future")
        if self.missed_samples != self.decision_sample - self.last_observed_sample:
            raise ValueError("TrackedDirection missed_samples must use absolute samples")
        if type(self.is_noise_interference) is not bool:
            raise TypeError("TrackedDirection noise-interference flag must be bool")


@dataclass(frozen=True, slots=True)
class ModelOrderEstimate:
    estimated_sources: int
    valid_frequency_bins: int
    snapshot_count: int
    cross_frequency_consistency: float
    mdl_age_samples: int
    status: str

    def __post_init__(self) -> None:
        if not 0 <= self.estimated_sources <= 6:
            raise ValueError("MUSIC model order must be 0..6")
        if min(self.valid_frequency_bins, self.snapshot_count, self.mdl_age_samples) < 0:
            raise ValueError("MUSIC model order counts must be non-negative")
        if not np.isfinite(self.cross_frequency_consistency) or not 0.0 <= self.cross_frequency_consistency <= 1.0:
            raise ValueError("MUSIC cross-frequency consistency must be in [0,1]")
        if self.status not in {"warming_up", "ready", "degraded", "failed"}:
            raise ValueError("MUSIC model-order status is invalid")


def _readonly_exact_complex64(value: object, shape: tuple[int, ...], name: str) -> NDArray[np.complex64]:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype != np.complex64:
        raise ValueError(f"{name}必须为complex64 {shape}")
    if not raw.flags.c_contiguous or not np.isfinite(raw.real).all() or not np.isfinite(raw.imag).all():
        raise ValueError(f"{name}必须C-contiguous且全部finite")
    return np.frombuffer(raw.tobytes(), dtype=np.complex64).reshape(shape)


@dataclass(frozen=True, slots=True)
class DirectionalSignal:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    context_start_sample: int
    context_end_sample: int
    theta_deg: float
    sample_rate: int
    beamformer_backend: str
    fallback_reason: str | None
    stft_complex: NDArray[np.complex64]
    track_id: int | None = None

    def __post_init__(self) -> None:
        if not self.session_id or min(self.stream_epoch, self.window_id, self.context_start_sample) < 0:
            raise ValueError("DirectionalSignal标识或边界无效")
        if self.context_end_sample != self.decision_sample or self.context_end_sample - self.context_start_sample != 15_360:
            raise ValueError("DirectionalSignal上下文边界无效")
        if self.sample_rate != 48_000 or not np.isfinite(self.theta_deg) or not 0 <= self.theta_deg < 360:
            raise ValueError("DirectionalSignal采样率或角度无效")
        if self.beamformer_backend not in {
            "frequency_hybrid", "imcra_spatial_separation", "das", "ds_baseline",
            "constant_beamwidth_baseline",
        }:
            raise ValueError("DirectionalSignal后端无效")
        if self.track_id is not None and (type(self.track_id) is not int or self.track_id <= 0):
            raise ValueError("DirectionalSignal track_id must be a positive integer")
        object.__setattr__(self, "stft_complex", _readonly_exact_complex64(self.stft_complex, (513, 33), "stft_complex"))


@dataclass(frozen=True, slots=True)
class SpectrogramFeature:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    context_start_sample: int
    context_end_sample: int
    theta_deg: float
    beamformer_backend: str
    preprocessing_version: str
    spectrogram: NDArray[np.float32]
    track_id: int | None = None

    def __post_init__(self) -> None:
        if not self.session_id or min(self.stream_epoch, self.window_id, self.context_start_sample) < 0:
            raise ValueError("SpectrogramFeature标识或边界无效")
        if self.context_end_sample != self.decision_sample or self.context_end_sample - self.context_start_sample != 15_360:
            raise ValueError("SpectrogramFeature上下文边界无效")
        if not np.isfinite(self.theta_deg) or not 0 <= self.theta_deg < 360:
            raise ValueError("SpectrogramFeature角度无效")
        if self.track_id is not None and (type(self.track_id) is not int or self.track_id <= 0):
            raise ValueError("SpectrogramFeature track_id must be a positive integer")
        object.__setattr__(self, "spectrogram", _readonly_exact_float32(self.spectrogram, (33, 169), "spectrogram"))


@dataclass(frozen=True, slots=True)
class EnhancedAudio:
    """Public Layer-3 output for one candidate direction."""

    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    context_start_sample: int
    context_end_sample: int
    theta_deg: float
    sample_rate: int
    algorithm: str
    fallback_reason: str | None
    diagnostics: tuple[str, ...]
    enhanced_audio: NDArray[np.float32]
    track_id: int | None = None

    def __post_init__(self) -> None:
        if not self.session_id or min(self.stream_epoch, self.window_id, self.context_start_sample) < 0:
            raise ValueError("EnhancedAudio标识或边界无效")
        if self.context_end_sample != self.decision_sample or self.context_end_sample - self.context_start_sample != 15_360:
            raise ValueError("EnhancedAudio上下文边界无效")
        if self.sample_rate != 48_000 or not np.isfinite(self.theta_deg) or not 0 <= self.theta_deg < 360:
            raise ValueError("EnhancedAudio采样率或角度无效")
        if not self.algorithm:
            raise ValueError("EnhancedAudio算法标识不能为空")
        if self.track_id is not None and (type(self.track_id) is not int or self.track_id <= 0):
            raise ValueError("EnhancedAudio track_id must be a positive integer")
        waveform = _readonly_exact_float32(self.enhanced_audio, (15_360,), "enhanced_audio")
        object.__setattr__(self, "diagnostics", tuple(str(item) for item in self.diagnostics))
        object.__setattr__(self, "enhanced_audio", waveform)


@dataclass(frozen=True, slots=True)
class PipelineStatus:
    state: str
    session_id: str
    stream_epoch: int
    buffered_samples: int
    required_samples: int
    message: str

    def __post_init__(self) -> None:
        if self.state not in {"stopped", "warming_up", "running", "degraded", "error"}:
            raise ValueError("未知PipelineStatus状态")
        if not self.session_id or self.stream_epoch < 0 or self.buffered_samples < 0 or self.required_samples <= 0:
            raise ValueError("PipelineStatus字段无效")
