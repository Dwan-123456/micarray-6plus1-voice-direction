from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .gain_compensation import InputGainCompensationDiagnostic


@dataclass(frozen=True, slots=True)
class Layer4AudioSegment:
    """Enhanced mono audio for one candidate, independent of L3/UI DTOs."""

    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    track_id: int
    theta_deg: float
    sample_rate: int
    waveform: NDArray[np.float32]
    array_source_probabilities_20ms: tuple[float | None, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.session_id
            or min(self.stream_epoch, self.window_id, self.decision_sample) < 0
            or type(self.track_id) is not int
            or self.track_id <= 0
        ):
            raise ValueError("invalid L4 audio stream/window identity")
        if not np.isfinite(self.theta_deg) or not 0.0 <= self.theta_deg < 360.0:
            raise ValueError("L4 audio theta_deg must be finite and in [0,360)")
        if self.sample_rate != 48_000:
            raise ValueError("L4 audio must be sampled at 48 kHz")
        waveform = np.asarray(self.waveform)
        if (
            waveform.shape != (15_360,)
            or waveform.dtype != np.float32
            or not waveform.flags.c_contiguous
            or not np.isfinite(waveform).all()
        ):
            raise ValueError("L4 audio must be finite C-contiguous float32 [15360]")
        probabilities = self.array_source_probabilities_20ms or (None,) * 16
        if len(probabilities) != 16 or any(
            value is not None and (not np.isfinite(value) or not 0.0 <= value <= 1.0)
            for value in probabilities
        ):
            raise ValueError("L4 audio requires 16 finite aligned IMCRA probabilities or missing values")
        object.__setattr__(self, "waveform", np.frombuffer(waveform.tobytes(), dtype=np.float32))
        object.__setattr__(
            self,
            "array_source_probabilities_20ms",
            tuple(None if value is None else float(value) for value in probabilities),
        )


@dataclass(frozen=True, slots=True)
class VoiceDetection:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    track_id: int
    theta_deg: float
    probability: float
    is_voice: bool
    model_id: str

    def __post_init__(self) -> None:
        if (
            not self.session_id
            or min(self.stream_epoch, self.window_id, self.decision_sample) < 0
            or type(self.track_id) is not int
            or self.track_id <= 0
        ):
            raise ValueError("invalid L4 stream/window identity")
        if not np.isfinite(self.theta_deg) or not 0.0 <= self.theta_deg < 360.0:
            raise ValueError("theta_deg must be finite and in [0,360)")
        if not np.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be finite and in [0,1]")
        if type(self.is_voice) is not bool or not self.model_id:
            raise ValueError("is_voice must be bool and model_id must be non-empty")


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    model_id: str
    probabilities: NDArray[np.float32]
    latency_ms: float
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        values = np.asarray(self.probabilities, dtype=np.float32)
        if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("model probabilities must be a finite float32 vector in [0,1]")
        if not self.model_id or not np.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("invalid model prediction metadata")
        values = np.frombuffer(np.ascontiguousarray(values).tobytes(), dtype=np.float32)
        object.__setattr__(self, "probabilities", values)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class Layer4Result:
    detections: tuple[VoiceDetection, ...]
    predictions: tuple[ModelPrediction, ...]
    primary_model_id: str
    threshold: float
    input_gain_compensation: tuple[InputGainCompensationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if not self.primary_model_id or not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("invalid L4 result configuration")
        model_ids = tuple(item.model_id for item in self.predictions)
        if len(model_ids) != len(set(model_ids)) or self.primary_model_id not in model_ids:
            raise ValueError("L4 predictions must contain one unique primary model")
        primary = next(item for item in self.predictions if item.model_id == self.primary_model_id)
        if len(primary.probabilities) != len(self.detections):
            raise ValueError("primary probabilities and detections must have equal length")
        identities = {
            (item.session_id, item.stream_epoch, item.window_id, item.decision_sample)
            for item in self.detections
        }
        if len(identities) > 1:
            raise ValueError("L4 detections must belong to one WindowKey")
        track_ids = tuple(item.track_id for item in self.detections)
        if len(track_ids) > 3 or len(track_ids) != len(set(track_ids)):
            raise ValueError("L4 detections require 0..3 unique track IDs")
        if self.input_gain_compensation and len(self.input_gain_compensation) != len(self.detections):
            raise ValueError("gain-compensation diagnostics must align with detections")
        object.__setattr__(self, "detections", tuple(self.detections))
        object.__setattr__(self, "predictions", tuple(self.predictions))
        object.__setattr__(self, "input_gain_compensation", tuple(self.input_gain_compensation))

    @property
    def voice_direction_count(self) -> int:
        return sum(item.is_voice for item in self.detections)

    @property
    def track_ids(self) -> tuple[int, ...]:
        return tuple(item.track_id for item in self.detections)
