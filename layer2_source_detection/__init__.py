"""Layer 2 v1.1: specification-locked 360-degree SRP-PHAT direction processing."""

LAYER2_PUBLIC_VERSION = "1.1"
__version__ = LAYER2_PUBLIC_VERSION

from .candidates import robust_z_sigmoid, select_candidate_indices
from .configuration import DirectionScanConfig
from .direction_smoothing import (
    DirectionSmoother,
    DirectionSmoothingConfig,
    DirectionSmoothingError,
    circular_delta_deg,
)
from .circular_kalman import CircularKalmanConfig, CircularKalmanFilter
from .circular_kalman_v2 import CircularKalmanFilterV2, CircularKalmanV2Config
from .direction_id_tracking import DirectionIdTracker, DirectionIdTrackingConfig
from .direction_id_tracking_v2 import DirectionIdTrackerV2, DirectionIdTrackingV2Config
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
    "LAYER2_PUBLIC_VERSION",
    "DirectionScanConfig", "DirectionScanError", "DirectionScanner", "DetailedDirectionScanner",
    "SrpPhatScanner", "Layer2Pipeline", "Layer2PipelineResult", "Layer2ExecutionState",
    "ProbabilityGate", "ProbabilityGateDecision", "ProbabilityGateState",
    "SourceProbability20ms", "SourceProbabilityState",
    "robust_z_sigmoid", "select_candidate_indices",
    "CandidateSearchDiagnostics", "CandidateSearchEvidence",
    "DirectionSmoother", "DirectionSmoothingConfig", "DirectionSmoothingError",
    "CircularKalmanConfig", "CircularKalmanFilter",
    "CircularKalmanV2Config", "CircularKalmanFilterV2",
    "DirectionIdTracker", "DirectionIdTrackingConfig",
    "DirectionIdTrackingV2Config", "DirectionIdTrackerV2",
    "circular_delta_deg",
]
