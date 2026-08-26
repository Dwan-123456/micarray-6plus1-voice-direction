from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from common.disk_audio import DiskAudioView, DiskFrameSeries, is_trusted_disk_audio


MAX_SESSION_SPEAKERS = 100


def speaker_label(speaker_id: int) -> str:
    """Return stable spreadsheet-style labels from Speaker A through CV."""

    if type(speaker_id) is not int or not 1 <= speaker_id <= MAX_SESSION_SPEAKERS:
        raise ValueError("L6 speaker ID must be a session-local value in 1..100")
    value = speaker_id
    suffix = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        suffix = chr(65 + remainder) + suffix
    return f"Speaker {suffix}"


def _audio(value: NDArray[np.float32], name: str) -> NDArray[np.float32]:
    array = np.asarray(value)
    if (
        array.ndim != 1
        or array.dtype != np.float32
        or not array.flags.c_contiguous
        or (not is_trusted_disk_audio(value) and not np.isfinite(array).all())
    ):
        raise ValueError(f"{name} must be finite C-contiguous float32 mono audio")
    if isinstance(value, DiskAudioView) and not value.flags.writeable:
        return value
    result = np.frombuffer(array.tobytes(), dtype=np.float32)
    result.flags.writeable = False
    return result


def _vector(value: NDArray[np.float32], name: str) -> NDArray[np.float32]:
    result = _audio(value, name)
    if not len(result):
        raise ValueError(f"{name} cannot be empty")
    return result


@dataclass(frozen=True, slots=True)
class Layer6QualityScore:
    voice: float
    speaker: float
    mos: float
    snr: float
    continuity: float
    total: float
    dnsmos_sig: float
    dnsmos_bak: float
    dnsmos_ovrl: float

    def __post_init__(self) -> None:
        normalized = (self.voice, self.speaker, self.mos, self.snr, self.continuity, self.total)
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in normalized):
            raise ValueError("L6 normalized quality scores must be in [0,1]")
        if any(not np.isfinite(value) or not 1.0 <= value <= 5.0 for value in (
            self.dnsmos_sig, self.dnsmos_bak, self.dnsmos_ovrl,
        )):
            raise ValueError("L6 DNSMOS scores must be in [1,5]")


@dataclass(frozen=True, slots=True)
class Layer6Fragment:
    """One complete L4 A/B track assigned to one clustered voiceprint."""
    fragment_id: str
    source_asset_id: str
    source_track_id: int
    source_theta_deg: float
    branch_index: int
    selected_for_parent: bool
    start_sample_48k: int
    end_sample_48k: int
    waveform_16k: NDArray[np.float32]
    voice_probabilities_20ms: tuple[float, ...]
    voice_is_active_20ms: tuple[bool, ...]
    embedding: NDArray[np.float32]
    speaker_id: int
    match_score: float
    mos_score: float
    speaker_similarity: float

    def __post_init__(self) -> None:
        if not self.fragment_id or not self.source_asset_id or self.source_track_id <= 0:
            raise ValueError("L6 fragment identity is invalid")
        if self.branch_index not in {0, 1} or type(self.selected_for_parent) is not bool:
            raise ValueError("L6 branch provenance is invalid")
        if self.start_sample_48k < 0 or self.end_sample_48k <= self.start_sample_48k:
            raise ValueError("L6 fragment timeline is invalid")
        waveform = _audio(self.waveform_16k, "L6 fragment waveform")
        if isinstance(self.voice_probabilities_20ms, DiskFrameSeries):
            if self.voice_probabilities_20ms.dtype != np.dtype(np.float32):
                raise ValueError("L6 disk probabilities must use float32")
            probabilities = self.voice_probabilities_20ms
        else:
            probabilities = tuple(float(value) for value in self.voice_probabilities_20ms)
        if isinstance(self.voice_is_active_20ms, DiskFrameSeries):
            if self.voice_is_active_20ms.dtype != np.dtype(np.bool_):
                raise ValueError("L6 disk decisions must use bool")
            decisions = self.voice_is_active_20ms
        else:
            decisions = tuple(self.voice_is_active_20ms)
        if len(waveform) * 3 != self.end_sample_48k - self.start_sample_48k or len(waveform) % 320:
            raise ValueError("L6 fragment waveform must preserve its complete 20 ms timeline")
        if len(probabilities) != len(waveform) // 320:
            raise ValueError("L6 fragment probabilities must align to 20 ms audio")
        if len(decisions) != len(probabilities) or (
            not isinstance(decisions, DiskFrameSeries)
            and any(type(value) is not bool for value in decisions)
        ):
            raise ValueError("L6 fragment voice decisions must align to 20 ms audio")
        if type(self.speaker_id) is not int or not 1 <= self.speaker_id <= MAX_SESSION_SPEAKERS:
            raise ValueError("L6 speaker IDs are session-local values 1..100")
        if any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0
            for value in (self.match_score, self.mos_score, self.speaker_similarity)
        ):
            raise ValueError("L6 complete-track scores must be in [0,1]")
        object.__setattr__(self, "waveform_16k", waveform)
        object.__setattr__(self, "voice_probabilities_20ms", probabilities)
        object.__setattr__(self, "voice_is_active_20ms", decisions)
        object.__setattr__(self, "embedding", _vector(self.embedding, "L6 speaker embedding"))


@dataclass(frozen=True, slots=True)
class Layer6SpeakerAudio:
    speaker_id: int
    label: str
    sample_rate: int
    start_sample_48k: int
    end_sample_48k: int
    waveform_16k: NDArray[np.float32]
    source_track_ids: tuple[int, ...]
    fragment_ids: tuple[str, ...]
    mean_quality: float

    def __post_init__(self) -> None:
        waveform = _audio(self.waveform_16k, "L6 speaker output")
        if self.label != speaker_label(self.speaker_id):
            raise ValueError("L6 speaker label is invalid")
        if (
            self.sample_rate != 16_000
            or self.start_sample_48k < 0
            or self.end_sample_48k <= self.start_sample_48k
            or not len(waveform)
            or len(waveform) % 320
        ):
            raise ValueError("L6 speaker output or source timeline bounds are invalid")
        if len(waveform) * 3 > self.end_sample_48k - self.start_sample_48k:
            raise ValueError("L6 silence compression cannot lengthen the source timeline")
        if not np.isfinite(self.mean_quality) or not 0.0 <= self.mean_quality <= 1.0:
            raise ValueError("L6 speaker mean quality must be in [0,1]")
        object.__setattr__(self, "waveform_16k", waveform)


@dataclass(frozen=True, slots=True)
class Layer6Result:
    session_id: str
    speaker_count: int
    outputs: tuple[Layer6SpeakerAudio, ...]
    fragments: tuple[Layer6Fragment, ...]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            not self.session_id
            or self.speaker_count != len(self.outputs)
            or not 0 <= self.speaker_count <= MAX_SESSION_SPEAKERS
        ):
            raise ValueError("L6 result speaker count is invalid")
        if tuple(item.speaker_id for item in self.outputs) != tuple(range(1, self.speaker_count + 1)):
            raise ValueError("L6 outputs must use ordered session-local speaker IDs")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
