from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from layer4_voice_classifier.gain_compensation import InputGainCompensationDiagnostic


@dataclass(frozen=True, slots=True)
class TrackAudioWindow:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    track_id: int
    theta_deg: float
    waveform: NDArray[np.float32]
    probabilities_20ms: tuple[float | None, ...]
    processing_mode: str

    def __post_init__(self) -> None:
        waveform = np.asarray(self.waveform)
        if (
            not self.session_id
            or min(self.stream_epoch, self.window_id, self.decision_sample) < 0
            or type(self.track_id) is not int
            or self.track_id <= 0
        ):
            raise ValueError("invalid track-audio window identity")
        if not np.isfinite(self.theta_deg) or not 0.0 <= self.theta_deg < 360.0:
            raise ValueError("track-audio theta must be finite and in [0,360)")
        if (
            waveform.ndim != 1
            or len(waveform) not in {3_840, 7_680}
            or waveform.dtype != np.float32
            or not waveform.flags.c_contiguous
            or not np.isfinite(waveform).all()
        ):
            raise ValueError("track-audio window must be finite contiguous float32 [3840 or 7680]")
        expected = len(waveform) // 960
        if len(self.probabilities_20ms) != expected or any(
            value is not None and (not np.isfinite(value) or not 0.0 <= value <= 1.0)
            for value in self.probabilities_20ms
        ):
            raise ValueError("track-audio probabilities must align to every 20 ms hop")
        if not self.processing_mode:
            raise ValueError("track-audio processing mode must be non-empty")
        object.__setattr__(self, "waveform", np.frombuffer(waveform.tobytes(), np.float32))
        object.__setattr__(self, "probabilities_20ms", tuple(self.probabilities_20ms))


@dataclass(frozen=True, slots=True)
class TrackAudioHop:
    session_id: str
    stream_epoch: int
    track_id: int
    start_sample: int
    end_sample: int
    waveform: NDArray[np.float32] | None
    probability: float | None
    observed: bool

    def __post_init__(self) -> None:
        if (
            not self.session_id
            or min(self.stream_epoch, self.start_sample) < 0
            or self.end_sample - self.start_sample != 960
            or type(self.track_id) is not int
            or self.track_id <= 0
        ):
            raise ValueError("invalid continuous track hop identity/timeline")
        if type(self.observed) is not bool:
            raise ValueError("track hop observed flag must be bool")
        if self.waveform is None:
            if self.observed:
                raise ValueError("an observed track hop requires audio")
            return
        waveform = np.asarray(self.waveform)
        if (
            waveform.shape != (960,)
            or waveform.dtype != np.float32
            or not waveform.flags.c_contiguous
            or not np.isfinite(waveform).all()
        ):
            raise ValueError("continuous track hop must be finite contiguous float32 [960]")
        object.__setattr__(self, "waveform", np.frombuffer(waveform.tobytes(), np.float32))


@dataclass(frozen=True, slots=True)
class ContinuousTrackAudio:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    track_id: int
    theta_deg: float
    effective_start_sample: int
    effective_end_sample: int
    waveform: NDArray[np.float32]
    probabilities_20ms: tuple[float | None, ...]
    gain_diagnostic: InputGainCompensationDiagnostic
    processing_mode: str

    def __post_init__(self) -> None:
        waveform = np.asarray(self.waveform)
        if (
            waveform.ndim != 1
            or len(waveform) < 960
            or len(waveform) % 960
            or waveform.dtype != np.float32
            or not waveform.flags.c_contiguous
            or not np.isfinite(waveform).all()
        ):
            raise ValueError("continuous L4 audio must contain complete finite 20 ms hops")
        if self.effective_end_sample - self.effective_start_sample != len(waveform):
            raise ValueError("continuous L4 audio range must match its waveform")
        if len(self.probabilities_20ms) != len(waveform) // 960:
            raise ValueError("continuous L4 probabilities must align with waveform hops")
        object.__setattr__(self, "waveform", np.frombuffer(waveform.tobytes(), np.float32))
        object.__setattr__(self, "probabilities_20ms", tuple(self.probabilities_20ms))


@dataclass(frozen=True, slots=True)
class TrackAudioBatch:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    emitted_hops: tuple[TrackAudioHop, ...]
    continuous_audio: tuple[ContinuousTrackAudio, ...]
    active_track_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        identities = {
            (item.session_id, item.stream_epoch, item.window_id, item.decision_sample)
            for item in self.continuous_audio
        }
        if identities and identities != {
            (self.session_id, self.stream_epoch, self.window_id, self.decision_sample)
        }:
            raise ValueError("continuous track batch identity mismatch")
        ids = tuple(item.track_id for item in self.continuous_audio)
        if len(ids) != len(set(ids)):
            raise ValueError("continuous track batch IDs must be unique")
        object.__setattr__(self, "emitted_hops", tuple(self.emitted_hops))
        object.__setattr__(self, "continuous_audio", tuple(self.continuous_audio))
        object.__setattr__(self, "active_track_ids", tuple(self.active_track_ids))
