"""Offline and bounded-streaming separation for Layer 3 track audio."""

from .contracts import (
    L4_MATCH_FREQUENCY_MAX_HZ,
    L4_MATCH_FREQUENCY_MIN_HZ,
    L4_MODEL_SAMPLE_RATE,
    Layer4CandidatePair,
    Layer4LongAudioInput,
    Layer4OfflineResult,
    Layer4ProcessedAudio,
    Layer4PrimarySelection,
    Layer4SeparationRequest,
    SpeakerCountDecision,
)
from .interfaces import Layer4SeparationBackend, SpeakerCountClassifier
from .matching import BandMagnitudeMatcher
from .models import DirectionCountSpeakerClassifier, MossFormer2Backend, TigerBackend, TorchScriptSeparationBackend
from .resampling import Layer4Resampler
from .streaming import (
    L4_STREAM_BATCH_SAMPLES_48K,
    L4_STREAM_OVERLAP_SAMPLES_48K,
    Layer4StreamInputChunk,
    Layer4StreamOutputChunk,
    Layer4StreamSession,
)

__all__ = [
    "BandMagnitudeMatcher",
    "L4_MATCH_FREQUENCY_MAX_HZ",
    "L4_MATCH_FREQUENCY_MIN_HZ",
    "L4_MODEL_SAMPLE_RATE",
    "L4_STREAM_BATCH_SAMPLES_48K",
    "L4_STREAM_OVERLAP_SAMPLES_48K",
    "Layer4CandidatePair",
    "Layer4LongAudioInput",
    "Layer4OfflineResult",
    "Layer4ProcessedAudio",
    "Layer4PrimarySelection",
    "Layer4SeparationBackend",
    "Layer4SeparationRequest",
    "Layer4StreamInputChunk",
    "Layer4StreamOutputChunk",
    "Layer4StreamSession",
    "SpeakerCountClassifier",
    "SpeakerCountDecision",
    "Layer4Resampler",
    "MossFormer2Backend",
    "TigerBackend",
    "TorchScriptSeparationBackend",
    "DirectionCountSpeakerClassifier",
]
