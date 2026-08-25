"""Layer 2 1.1: rolling NormMUSIC and authoritative direction IDs."""

from .configuration import DirectionScanConfig
from .interface import DetailedDirectionScanner, DirectionScanError, DirectionScanner
from .pipeline import Layer2ExecutionState, Layer2Pipeline, Layer2PipelineResult
from .probability_gate import (
    ProbabilityGate,
    ProbabilityGateDecision,
    ProbabilityGateState,
    SourceProbability20ms,
    SourceProbabilityState,
)
from .music import MusicDiagnostics, MusicStateDiagnostic, RollingNormMusicScanner
from .global_tracker import GlobalDirectionTracker

LAYER2_PUBLIC_VERSION = "1.1"
__version__ = LAYER2_PUBLIC_VERSION

__all__ = [
    "LAYER2_PUBLIC_VERSION",
    "DirectionScanConfig", "DirectionScanError", "DirectionScanner", "DetailedDirectionScanner",
    "RollingNormMusicScanner", "MusicDiagnostics", "MusicStateDiagnostic",
    "Layer2Pipeline", "Layer2PipelineResult", "Layer2ExecutionState",
    "GlobalDirectionTracker",
    "ProbabilityGate", "ProbabilityGateDecision", "ProbabilityGateState",
    "SourceProbability20ms", "SourceProbabilityState",
]
