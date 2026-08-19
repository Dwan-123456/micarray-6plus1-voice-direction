"""Layer 2: specification-locked 360-degree SRP-PHAT candidate scanning."""

from .candidates import robust_z_sigmoid, select_candidate_indices
from .configuration import DirectionScanConfig
from .direction_smoothing import (
    DirectionSmoother,
    DirectionSmoothingConfig,
    DirectionSmoothingError,
    circular_delta_deg,
)
from .circular_kalman import CircularKalmanConfig, CircularKalmanFilter
from .direction_id_tracking import DirectionIdTracker, DirectionIdTrackingConfig
from .interface import DetailedDirectionScanner, DirectionScanError, DirectionScanner
from .iterative import CandidateSearchDiagnostics, CandidateSearchEvidence
from .pipeline import Layer2ExecutionState, Layer2Pipeline, Layer2PipelineResult
from .probability_gate import (
    ProbabilityGate,
    ProbabilityGateDecision,
    ProbabilityGateState,
    SourceProbability20ms,
    SourceProbabilityState,
)
from .srp_phat import SrpPhatScanner

__all__ = [
    "DirectionScanConfig", "DirectionScanError", "DirectionScanner", "DetailedDirectionScanner",
    "SrpPhatScanner", "Layer2Pipeline", "Layer2PipelineResult", "Layer2ExecutionState",
    "ProbabilityGate", "ProbabilityGateDecision", "ProbabilityGateState",
    "SourceProbability20ms", "SourceProbabilityState",
    "robust_z_sigmoid", "select_candidate_indices",
    "CandidateSearchDiagnostics", "CandidateSearchEvidence",
    "DirectionSmoother", "DirectionSmoothingConfig", "DirectionSmoothingError",
    "CircularKalmanConfig", "CircularKalmanFilter",
    "DirectionIdTracker", "DirectionIdTrackingConfig",
    "circular_delta_deg",
]
