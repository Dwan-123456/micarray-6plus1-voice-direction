from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np
from numpy.typing import NDArray


L4_MODEL_SAMPLE_RATE = 16_000
L4_MATCH_FREQUENCY_MIN_HZ = 1_000.0
L4_MATCH_FREQUENCY_MAX_HZ = 4_000.0
L3_SAMPLE_RATE = 48_000
L3_HOP_SAMPLES = 960


def _readonly_float32_1d(value: NDArray[np.float32], name: str) -> NDArray[np.float32]:
    array = np.asarray(value)
    if (
        array.ndim != 1
        or array.dtype != np.float32
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be finite C-contiguous float32 mono audio")
    result = np.frombuffer(array.tobytes(), dtype=np.float32)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class Layer4LongAudioInput:
    """One sealed, overlap-free Layer 3 track on the authoritative timeline."""

    asset_id: str
    sha256: str
    session_id: str
    stream_epoch: int
    track_id: int
    theta_deg: float
    start_sample: int
    sample_rate: Literal[48000]
    waveform: NDArray[np.float32]
    l2_direction_counts: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.asset_id or not self.session_id:
            raise ValueError("Layer 4 input requires asset and session identities")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("Layer 4 input sha256 must be lowercase hexadecimal")
        if self.stream_epoch < 0 or self.track_id <= 0 or self.start_sample < 0:
            raise ValueError("Layer 4 input timeline and track identity are invalid")
        if not np.isfinite(self.theta_deg) or not 0.0 <= self.theta_deg < 360.0:
            raise ValueError("Layer 4 input angle must be in [0,360)")
        if self.sample_rate != L3_SAMPLE_RATE:
            raise ValueError("Layer 4 accepts sealed Layer 3 audio at 48 kHz")
        waveform = _readonly_float32_1d(self.waveform, "Layer 4 input")
        if not len(waveform) or len(waveform) % L3_HOP_SAMPLES:
            raise ValueError("Layer 4 input must contain complete 20 ms Layer 3 hops")
        counts = tuple((int(sample), int(count)) for sample, count in self.l2_direction_counts)
        if not counts:
            raise ValueError("Layer 4 input requires aligned L2 direction-count history")
        if any(
            sample < self.start_sample or sample > self.end_sample or count not in {0, 1, 2, 3}
            for sample, count in counts
        ):
            raise ValueError("Layer 4 L2 direction counts must be 0..3 on the source timeline")
        if any(right[0] <= left[0] for left, right in zip(counts, counts[1:])):
            raise ValueError("Layer 4 L2 direction-count history must be strictly ordered")
        if max(count for _, count in counts) not in {1, 2, 3}:
            raise ValueError("Layer 4 requires at least one L2 direction in its time range")
        object.__setattr__(self, "waveform", waveform)
        object.__setattr__(self, "l2_direction_counts", counts)

    @property
    def end_sample(self) -> int:
        return self.start_sample + len(self.waveform)


@dataclass(frozen=True, slots=True)
class SpeakerCountDecision:
    asset_id: str
    speaker_count: Literal[1, 2]
    confidence: float
    classifier_id: str
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.asset_id or not self.classifier_id:
            raise ValueError("speaker-count decision requires asset and classifier identities")
        if type(self.speaker_count) is not int or self.speaker_count not in {1, 2}:
            raise ValueError("legal Layer 4 inputs contain one or two speakers")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("speaker-count confidence must be in [0,1]")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class Layer4SeparationRequest:
    """A user-triggered offline request admitted only after the source asset is sealed."""

    request_id: str
    source: Layer4LongAudioInput
    speaker_count: SpeakerCountDecision
    backend: Literal["mossformer2_ss_16k", "tiger_speech_16k"]

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("Layer 4 separation request requires request_id")
        if self.speaker_count.asset_id != self.source.asset_id:
            raise ValueError("speaker-count decision must describe the Layer 4 source asset")
        if self.speaker_count.speaker_count != 2:
            raise ValueError("one-speaker audio bypasses Layer 4 and proceeds to Layer 5")


@dataclass(frozen=True, slots=True)
class Layer4CandidatePair:
    """Exactly two anonymous 16 kHz outputs from one separation backend invocation."""

    request_id: str
    model_id: str
    model_revision: str
    sample_rate: Literal[16000]
    sources: tuple[NDArray[np.float32], NDArray[np.float32]]

    def __post_init__(self) -> None:
        if not self.request_id or not self.model_id or not self.model_revision:
            raise ValueError("Layer 4 candidates require request and model provenance")
        if self.sample_rate != L4_MODEL_SAMPLE_RATE or len(self.sources) != 2:
            raise ValueError("Layer 4 backend must return exactly two 16 kHz sources")
        sources = tuple(
            _readonly_float32_1d(value, f"Layer 4 candidate {index}")
            for index, value in enumerate(self.sources)
        )
        if not len(sources[0]) or len(sources[0]) != len(sources[1]):
            raise ValueError("Layer 4 candidate sources must be non-empty and equal length")
        object.__setattr__(self, "sources", sources)


@dataclass(frozen=True, slots=True)
class Layer4PrimarySelection:
    """The one separated source that inherits the parent Layer 2 ID and angle."""

    request_id: str
    parent_asset_id: str
    session_id: str
    stream_epoch: int
    track_id: int
    theta_deg: float
    sample_rate: Literal[16000]
    selected_source_index: Literal[0, 1]
    candidate_scores: tuple[float, float]
    score_margin: float
    matching_algorithm: Literal[
        "l3_bf_1_4khz_complex_coherence_v3",
        "l3_bf_1_4khz_cross_track_penalty_v4",
    ]
    waveform: NDArray[np.float32]
    used_reference_fallback: bool = False
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id or not self.parent_asset_id or not self.session_id:
            raise ValueError("Layer 4 selection requires complete provenance")
        if self.stream_epoch < 0 or self.track_id <= 0:
            raise ValueError("Layer 4 selection must preserve a valid parent track")
        if not np.isfinite(self.theta_deg) or not 0.0 <= self.theta_deg < 360.0:
            raise ValueError("Layer 4 selection must preserve a valid parent angle")
        if self.sample_rate != L4_MODEL_SAMPLE_RATE or self.selected_source_index not in {0, 1}:
            raise ValueError("Layer 4 selection must identify one 16 kHz candidate")
        scores = tuple(float(value) for value in self.candidate_scores)
        if len(scores) != 2 or any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in scores):
            raise ValueError("Layer 4 matching scores must contain two finite values in [0,1]")
        expected_margin = abs(scores[0] - scores[1])
        if not np.isclose(self.score_margin, expected_margin, atol=1e-7, rtol=0.0):
            raise ValueError("Layer 4 score margin must equal the candidate score difference")
        if type(self.used_reference_fallback) is not bool:
            raise ValueError("Layer 4 reference fallback flag must be bool")
        if self.used_reference_fallback != (self.fallback_reason is not None):
            raise ValueError("Layer 4 reference fallback requires exactly one reason")
        object.__setattr__(self, "candidate_scores", scores)
        object.__setattr__(self, "waveform", _readonly_float32_1d(self.waveform, "Layer 4 selection"))


@dataclass(frozen=True, slots=True)
class Layer4ProcessedAudio:
    """L4 terminal audio held for listening until the user sends it to L5."""

    request_id: str
    source: Layer4LongAudioInput
    speaker_count: SpeakerCountDecision
    path: Literal["single_speaker_bypass", "two_speaker_separation"]
    selected: Layer4PrimarySelection | None
    output_asset_id: str
    output_sha256: str
    waveform_16k: NDArray[np.float32]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.request_id or not self.output_asset_id:
            raise ValueError("processed L4 audio requires request and asset identities")
        if self.speaker_count.asset_id != self.source.asset_id:
            raise ValueError("processed L4 speaker count must describe its source")
        if (self.path == "single_speaker_bypass") != (self.selected is None):
            raise ValueError("only single-speaker L4 audio may omit a selection")
        if len(self.output_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.output_sha256
        ):
            raise ValueError("processed L4 output sha256 must be lowercase hexadecimal")
        waveform = _readonly_float32_1d(self.waveform_16k, "processed L4 waveform")
        if len(waveform) * 3 != len(self.source.waveform) or len(waveform) % 320:
            raise ValueError("processed L4 audio must preserve complete 16 kHz 20 ms hops")
        object.__setattr__(self, "waveform_16k", waveform)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class Layer4OfflineResult:
    """Auditable terminal result for one sealed L3 asset."""

    request_id: str
    source: Layer4LongAudioInput
    speaker_count: SpeakerCountDecision
    path: Literal["single_speaker_bypass", "two_speaker_separation"]
    selected: Layer4PrimarySelection | None
    l5_probability: float
    l5_is_voice: bool
    l5_model_id: str
    l5_probabilities_20ms: tuple[float, ...]
    l5_is_voice_20ms: tuple[bool, ...]
    output_asset_id: str
    output_sha256: str
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.request_id or not self.output_asset_id or not self.l5_model_id:
            raise ValueError("offline L4 result requires request, asset and L5 identities")
        if self.speaker_count.asset_id != self.source.asset_id:
            raise ValueError("offline L4 speaker count must describe its source")
        if (self.path == "single_speaker_bypass") != (self.selected is None):
            raise ValueError("only the single-speaker path may omit an L4 selection")
        if self.path == "single_speaker_bypass" and self.speaker_count.speaker_count != 1:
            raise ValueError("single-speaker bypass requires a one-speaker decision")
        if self.path == "two_speaker_separation" and self.speaker_count.speaker_count != 2:
            raise ValueError("two-speaker separation requires a two-speaker decision")
        if not np.isfinite(self.l5_probability) or not 0.0 <= self.l5_probability <= 1.0:
            raise ValueError("offline L5 probability must be in [0,1]")
        if type(self.l5_is_voice) is not bool:
            raise ValueError("offline L5 decision must be bool")
        probabilities = tuple(float(value) for value in self.l5_probabilities_20ms)
        decisions = tuple(self.l5_is_voice_20ms)
        expected_hops = len(self.source.waveform) // L3_HOP_SAMPLES
        if len(probabilities) != expected_hops or any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities
        ):
            raise ValueError("offline L5 requires one probability per 20 ms source hop")
        if len(decisions) != expected_hops or any(type(value) is not bool for value in decisions):
            raise ValueError("offline L5 requires one bool decision per 20 ms source hop")
        if len(self.output_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.output_sha256):
            raise ValueError("offline output sha256 must be lowercase hexadecimal")
        object.__setattr__(self, "l5_probabilities_20ms", probabilities)
        object.__setattr__(self, "l5_is_voice_20ms", decisions)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
