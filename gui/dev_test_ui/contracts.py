from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import numpy as np
from numpy.typing import NDArray

from common.data_types import CandidateDirection, ImcraHopSnapshot, SpatialResponse, TrackedDirection
from layer2_source_detection.music import MusicDiagnostics
from layer2_source_detection.probability_gate import ProbabilityGateDecision


def _eight(value: object, dtype: object) -> np.ndarray:
    raw = np.asarray(value, dtype=dtype)
    if raw.shape != (8,) or not np.isfinite(raw.astype(np.float64)).all():
        raise ValueError("L1 meter arrays must contain eight finite values")
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
    imcra_hop: ImcraHopSnapshot | None = None
    pre_denoise_enabled: bool = False
    pre_denoise_mean_gain_db: float = 0.0
    calibration_status: str = "unverified"
    calibration_version: str = "unversioned"
    calibration_hash: str = "0" * 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "rms_dbfs", _eight(self.rms_dbfs, np.float32))
        object.__setattr__(self, "peak_dbfs", _eight(self.peak_dbfs, np.float32))
        object.__setattr__(self, "clipped", _eight(self.clipped, np.bool_))


@dataclass(frozen=True, slots=True)
class L2DevUiSnapshot:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    spatial_response: SpatialResponse | None
    candidates: tuple[CandidateDirection, ...]
    gate_decision: ProbabilityGateDecision
    gate_threshold: float
    gate_config_revision: int
    direction_threshold: float
    direction_id_tracking_enabled: bool
    scan_config_revision: int
    search_diagnostics: MusicDiagnostics | None
    directions: tuple[TrackedDirection, ...]
    active_tracks: tuple[TrackedDirection, ...]
    published_monotonic: float
    missing_reason: str | None = None
    processing_period_ms: int = 20
    reused_output: bool = False
    queue_wait_ms: float = 0.0

    @property
    def age_ms(self) -> float:
        return max(0.0, (monotonic() - self.published_monotonic) * 1000.0)
