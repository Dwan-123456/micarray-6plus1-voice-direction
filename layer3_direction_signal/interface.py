from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from common.data_types import CandidateDirection, DecisionWindow, DirectionalSignal, EnhancedAudio
from common.geometry import MicGeometry

from .configuration import SpatialSeparationConfig, StftSettings


L3_MODE_OPTIMIZED = "optimized"
L3_MODE_DS_BASELINE = "ds_baseline"
L3_MODE_LOADED_MVDR = "loaded_mvdr_baseline"
L3_MODE_SUBBAND_ROBUST = "subband_robust_baseline"
L3_PROCESSING_MODES = frozenset((
    L3_MODE_OPTIMIZED,
    L3_MODE_DS_BASELINE,
    L3_MODE_LOADED_MVDR,
    L3_MODE_SUBBAND_ROBUST,
))


class Layer3Error(RuntimeError):
    pass


class Beamformer(Protocol):
    def process_batch(
        self, window: DecisionWindow, candidates: tuple[CandidateDirection, ...], geometry: MicGeometry,
        config: SpatialSeparationConfig, stft: StftSettings,
    ) -> tuple[DirectionalSignal, ...]: ...


@dataclass(frozen=True, slots=True)
class Layer3Output:
    enhanced_audio: tuple[EnhancedAudio, ...]

    def __post_init__(self) -> None:
        outputs = tuple(self.enhanced_audio)
        identities = {
            (item.session_id, item.stream_epoch, item.window_id, item.decision_sample)
            for item in outputs
        }
        if len(identities) > 1:
            raise ValueError("Layer3输出必须属于同一窗口")
        if len({item.theta_deg for item in outputs}) != len(outputs):
            raise ValueError("Layer3输出方向不能重复")
        track_ids = tuple(item.track_id for item in outputs)
        if any(item is not None for item in track_ids):
            if any(item is None for item in track_ids) or len(set(track_ids)) != len(track_ids):
                raise ValueError("Layer3 public track IDs must be complete and unique")
        object.__setattr__(self, "enhanced_audio", outputs)
