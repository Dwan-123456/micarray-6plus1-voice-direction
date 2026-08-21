from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np
from numpy.typing import NDArray


L4_MODEL_SAMPLE_RATE = 16_000
L4_MATCH_FREQUENCY_MIN_HZ = 2_000.0
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
        object.__setattr__(self, "waveform", waveform)

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
    matching_algorithm: Literal["l3_bf_2_4khz_magnitude_cosine_v1"]
    waveform: NDArray[np.float32]

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
        object.__setattr__(self, "candidate_scores", scores)
        object.__setattr__(self, "waveform", _readonly_float32_1d(self.waveform, "Layer 4 selection"))
