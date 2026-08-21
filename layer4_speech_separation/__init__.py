"""Offline two-speaker separation contracts for sealed Layer 3 track audio."""

from .contracts import (
    L4_MATCH_FREQUENCY_MAX_HZ,
    L4_MATCH_FREQUENCY_MIN_HZ,
    L4_MODEL_SAMPLE_RATE,
    Layer4CandidatePair,
    Layer4LongAudioInput,
    Layer4OfflineResult,
    Layer4PrimarySelection,
    Layer4SeparationRequest,
    SpeakerCountDecision,
)
from .interfaces import Layer4SeparationBackend, SpeakerCountClassifier
from .matching import BandMagnitudeMatcher
from .models import DirectionCountSpeakerClassifier, MossFormer2Backend, TigerBackend, TorchScriptSeparationBackend
from .resampling import Layer4Resampler

__all__ = [
    "BandMagnitudeMatcher",
    "L4_MATCH_FREQUENCY_MAX_HZ",
    "L4_MATCH_FREQUENCY_MIN_HZ",
    "L4_MODEL_SAMPLE_RATE",
    "Layer4CandidatePair",
    "Layer4LongAudioInput",
    "Layer4OfflineResult",
    "Layer4PrimarySelection",
    "Layer4SeparationBackend",
    "Layer4SeparationRequest",
    "SpeakerCountClassifier",
    "SpeakerCountDecision",
    "Layer4Resampler",
    "MossFormer2Backend",
    "TigerBackend",
    "TorchScriptSeparationBackend",
    "DirectionCountSpeakerClassifier",
]
