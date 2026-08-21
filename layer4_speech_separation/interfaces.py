from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .contracts import Layer4CandidatePair, Layer4LongAudioInput, SpeakerCountDecision


class SpeakerCountClassifier(Protocol):
    """Future classifier. Legal output is deliberately limited to one or two speakers."""

    classifier_id: str

    def classify(self, source: Layer4LongAudioInput) -> SpeakerCountDecision: ...


class Layer4SeparationBackend(Protocol):
    """Common adapter boundary for MossFormer2 and TIGER; no implementation is bundled yet."""

    model_id: str
    model_revision: str
    sample_rate: int
    source_count: int

    def separate(self, request_id: str, waveform_16k: NDArray[np.float32]) -> Layer4CandidatePair: ...
