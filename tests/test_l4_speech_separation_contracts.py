from __future__ import annotations

import hashlib

import numpy as np
import pytest

from layer4_speech_separation import (
    BandMagnitudeMatcher,
    Layer4CandidatePair,
    Layer4LongAudioInput,
    Layer4SeparationRequest,
    SpeakerCountDecision,
)


def _parent(waveform: np.ndarray | None = None) -> Layer4LongAudioInput:
    value = np.zeros(48_000, dtype=np.float32) if waveform is None else waveform
    return Layer4LongAudioInput(
        asset_id="asset-1",
        sha256=hashlib.sha256(value.tobytes()).hexdigest(),
        session_id="session-1",
        stream_epoch=0,
        track_id=7,
        theta_deg=359.0,
        start_sample=0,
        sample_rate=48_000,
        waveform=np.ascontiguousarray(value, dtype=np.float32),
    )


def test_layer4_requires_sealed_complete_l3_hops_and_two_speaker_admission() -> None:
    parent = _parent()
    decision = SpeakerCountDecision("asset-1", 2, 0.9, "future-count-v1", {})
    request = Layer4SeparationRequest("request-1", parent, decision, "tiger_speech_16k")
    assert request.source.track_id == 7 and request.source.theta_deg == 359.0
    with pytest.raises(ValueError, match="bypasses Layer 4"):
        Layer4SeparationRequest(
            "request-2",
            parent,
            SpeakerCountDecision("asset-1", 1, 0.9, "future-count-v1", {}),
            "mossformer2_ss_16k",
        )
    with pytest.raises(ValueError, match="complete 20 ms"):
        _parent(np.zeros(48_001, dtype=np.float32))


def test_two_candidates_are_required_and_selection_preserves_parent_id_and_angle() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    dominant = np.sin(2 * np.pi * 2_700 * time).astype(np.float32)
    interferer = np.sin(2 * np.pi * 3_500 * time + 0.4).astype(np.float32)
    reference = np.ascontiguousarray(dominant + 0.1 * interferer, dtype=np.float32)
    candidates = Layer4CandidatePair(
        request_id="request-1",
        model_id="stub",
        model_revision="1",
        sample_rate=16_000,
        sources=(np.ascontiguousarray(interferer), np.ascontiguousarray(dominant)),
    )
    selected = BandMagnitudeMatcher().select(
        parent=_parent(), reference_16k=reference, candidates=candidates
    )
    assert selected.selected_source_index == 1
    assert selected.track_id == 7 and selected.theta_deg == 359.0
    assert selected.candidate_scores[1] > selected.candidate_scores[0]
    assert selected.matching_algorithm == "l3_bf_2_4khz_magnitude_cosine_v1"
    assert not selected.waveform.flags.writeable


def test_layer4_candidate_contract_rejects_non_pair_or_misaligned_outputs() -> None:
    audio = np.zeros(16_000, dtype=np.float32)
    with pytest.raises(ValueError, match="equal length"):
        Layer4CandidatePair(
            "request", "model", "rev", 16_000,
            (audio, np.zeros(15_999, dtype=np.float32)),
        )
