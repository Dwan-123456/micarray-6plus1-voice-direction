from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from common.data_types import DecisionWindow, DirectionalSignal, EnhancedAudio, TrackedDirection
from common.geometry import MicGeometry
from common.window_key import WindowKey

from .configuration import SpatialSeparationConfig, StftSettings


L3_MODE_OPTIMIZED = "optimized"
L3_MODE_DS_BASELINE = "ds_baseline"
L3_MODE_CONSTANT_BEAMWIDTH = "constant_beamwidth_baseline"
L3_PROCESSING_MODES = frozenset((
    L3_MODE_OPTIMIZED,
    L3_MODE_DS_BASELINE,
    L3_MODE_CONSTANT_BEAMWIDTH,
))


class Layer3Error(RuntimeError):
    pass


class Beamformer(Protocol):
    def process_batch(
        self, window: DecisionWindow, directions: tuple[TrackedDirection, ...], geometry: MicGeometry,
        config: SpatialSeparationConfig, stft: StftSettings,
    ) -> tuple[DirectionalSignal, ...]: ...


def validate_l3_directions(
    window_key: WindowKey,
    directions: tuple[TrackedDirection, ...],
) -> tuple[TrackedDirection, ...]:
    values = tuple(directions)
    if len(values) > 3:
        raise Layer3Error("L3 accepts zero to three tracked directions")
    if any(not isinstance(item, TrackedDirection) for item in values):
        raise Layer3Error("L3 accepts only public L2 TrackedDirection inputs")
    if any(item.window_key != window_key for item in values):
        raise Layer3Error("L3 directions must belong to the exact DecisionWindow WindowKey")
    track_ids = tuple(item.track_id for item in values)
    ranks = tuple(item.rank for item in values)
    if len(set(track_ids)) != len(track_ids):
        raise Layer3Error("L3 direction track_ids must be unique within one window")
    if len(set(ranks)) != len(ranks):
        raise Layer3Error("L3 direction original ranks must be unique within one window")
    if any(not item.eligible_for_l3 for item in values):
        raise Layer3Error(
            "L3 cannot process a long-coasting track without explicit short-prediction permission"
        )
    return values


@dataclass(frozen=True, slots=True)
class Layer3Output:
    window_key: WindowKey
    enhanced_audio: tuple[EnhancedAudio, ...]

    def __post_init__(self) -> None:
        outputs = tuple(self.enhanced_audio)
        if not isinstance(self.window_key, WindowKey):
            raise TypeError("Layer3Output requires a WindowKey")
        if any(item.window_key != self.window_key for item in outputs):
            raise ValueError("Layer3 output audio must belong to its exact WindowKey")
        if len({item.track_id for item in outputs}) != len(outputs):
            raise ValueError("Layer3 output track_ids must be unique within one window")
        if len({item.rank for item in outputs}) != len(outputs):
            raise ValueError("Layer3 output original ranks must be unique within one window")
        object.__setattr__(self, "enhanced_audio", outputs)

    @property
    def session_id(self) -> str:
        return self.window_key.session_id

    @property
    def stream_epoch(self) -> int:
        return self.window_key.stream_epoch

    @property
    def window_id(self) -> int:
        return self.window_key.window_id

    @property
    def decision_sample(self) -> int:
        return self.window_key.decision_sample
