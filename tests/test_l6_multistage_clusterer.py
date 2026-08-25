from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from layer4_speech_separation import (
    Layer4LongAudioInput,
    Layer4OfflineResult,
    SpeakerCountDecision,
)
from layer6_speaker_consolidation import (
    MultiStageVoiceprintClusterer,
    OfflineLayer6Pipeline,
    SegmentEvidence,
)


def _config(**overrides: object) -> SimpleNamespace:
    values = {
        "maximum_speakers": 3,
        "speaker_similarity_threshold": 0.62,
        "secondary_candidate_match_gap_max": 0.20,
        "secondary_candidate_match_min": 0.50,
        "secondary_candidate_mos_min": 0.30,
        "maximum_internal_silence_ms": 2_000,
        "clustering_backend": "multistage",
        "multistage_l": 4,
        "multistage_u1": 6,
        "multistage_u2": 10,
        "multistage_fallback_distance": 0.38,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _SequenceBackend:
    def __init__(self, outputs: tuple[tuple[int, ...], ...]) -> None:
        self.outputs = outputs
        self.calls = 0

    def streaming_predict(self, _embedding: np.ndarray) -> np.ndarray:
        output = self.outputs[self.calls]
        self.calls += 1
        return np.asarray(output, dtype=np.int32)


def _evidence(index: int, vector: tuple[float, ...]) -> SegmentEvidence:
    return SegmentEvidence(
        f"evidence-{index}",
        f"track-{index}",
        np.asarray(vector, dtype=np.float32),
        32_000,
    )


def test_multistage_adapter_keeps_history_corrections_and_deduplicates() -> None:
    backend = _SequenceBackend(((0,), (0, 1), (0, 0, 1)))
    clusterer = MultiStageVoiceprintClusterer(_config(), backend=backend)

    first = clusterer.update((_evidence(0, (1.0, 0.0)), _evidence(1, (0.0, 1.0))))
    corrected = clusterer.update((_evidence(2, (0.9, 0.1)),))
    duplicate = clusterer.update((_evidence(2, (0.9, 0.1)),))

    assert first.labels_by_evidence_id == {"evidence-0": 0, "evidence-1": 1}
    assert corrected.labels_by_evidence_id == {
        "evidence-0": 0,
        "evidence-1": 0,
        "evidence-2": 1,
    }
    assert duplicate == corrected
    assert backend.calls == 3
    assert corrected.stage == "fallback_ahc"


def test_multistage_adapter_reports_algorithm_stages() -> None:
    outputs = tuple(tuple(0 for _ in range(count)) for count in range(1, 9))
    clusterer = MultiStageVoiceprintClusterer(
        _config(multistage_l=3, multistage_u1=5, multistage_u2=9),
        backend=_SequenceBackend(outputs),
    )

    assert clusterer.update(tuple(_evidence(i, (1.0, float(i + 1))) for i in range(2))).stage == "fallback_ahc"
    assert clusterer.update((_evidence(2, (1.0, 3.0)),)).stage == "spectral"
    assert clusterer.update(tuple(_evidence(i, (1.0, float(i + 1))) for i in range(3, 8))).stage == "preclustered_spectral"


def test_multistage_adapter_rejects_changed_evidence() -> None:
    clusterer = MultiStageVoiceprintClusterer(
        _config(),
        backend=_SequenceBackend(((0,),)),
    )
    clusterer.update((_evidence(0, (1.0, 0.0)),))

    with pytest.raises(ValueError, match="changed in place"):
        clusterer.update((_evidence(0, (0.0, 1.0)),))


class _SignEmbedder:
    def embed_batch(self, waveforms: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        return tuple(
            np.asarray((1.0, 0.0) if float(np.mean(value)) >= 0.0 else (0.0, 1.0), dtype=np.float32)
            for value in waveforms
        )


def _result(track_id: int, value: float, frames: int) -> Layer4OfflineResult:
    waveform = np.full(frames * 320, value, dtype=np.float32)
    source_waveform = np.zeros(frames * 960, dtype=np.float32)
    source = Layer4LongAudioInput(
        f"source-{track_id}",
        hashlib.sha256(source_waveform.tobytes()).hexdigest(),
        "session",
        0,
        track_id,
        float(track_id * 60),
        0,
        48_000,
        source_waveform,
        ((0, 2),),
    )
    decision = SpeakerCountDecision(source.asset_id, 2, 1.0, "test", {})
    active = (True,) * frames
    return Layer4OfflineResult(
        f"request-{track_id}",
        source,
        decision,
        "two_speaker_separation",
        None,
        0.9,
        True,
        "l5",
        (0.9,) * frames,
        active,
        f"source-{track_id}:branch-0",
        hashlib.sha256(waveform.tobytes()).hexdigest(),
        {
            "candidate_match_score": 0.9,
            "mos_score": 0.8,
            "output_waveform_16k": waveform,
        },
        "candidate_0",
    )


def test_layer6_multistage_streaming_only_submits_completed_two_second_evidence() -> None:
    pipeline = OfflineLayer6Pipeline(_SignEmbedder(), _config())

    early = pipeline.process_streaming((_result(1, 0.2, 50),))
    complete = pipeline.process_streaming((_result(1, 0.2, 100),))
    repeated = pipeline.process_streaming((_result(1, 0.2, 100),))

    assert early.speaker_count == 0
    assert early.metadata["multistage"]["evidence_count"] == 0
    assert complete.speaker_count == 1
    assert complete.metadata["multistage"]["evidence_count"] == 1
    assert repeated.metadata["multistage"]["evidence_count"] == 1


def test_layer6_multistage_final_accepts_short_tail_evidence() -> None:
    pipeline = OfflineLayer6Pipeline(_SignEmbedder(), _config())

    final = pipeline.process_streaming((_result(1, 0.2, 50),), final=True)

    assert final.speaker_count == 1
    assert final.metadata["multistage"]["evidence_count"] == 1

