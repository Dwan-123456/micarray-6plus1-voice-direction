"""Offline L6 speaker clustering, quality selection and timeline stitching."""

from .contracts import (
    MAX_SESSION_SPEAKERS,
    Layer6Fragment,
    Layer6QualityScore,
    Layer6Result,
    Layer6SpeakerAudio,
    speaker_label,
)
from .models import CampPlusEmbedder, DnsMosScorer
from .multistage import MultiStageSnapshot, MultiStageVoiceprintClusterer, SegmentEvidence
from .pipeline import OfflineLayer6Pipeline

__all__ = [
    "CampPlusEmbedder", "DnsMosScorer", "Layer6Fragment", "Layer6QualityScore",
    "Layer6Result", "Layer6SpeakerAudio", "MAX_SESSION_SPEAKERS", "MultiStageSnapshot",
    "MultiStageVoiceprintClusterer", "OfflineLayer6Pipeline", "SegmentEvidence",
    "speaker_label",
]
