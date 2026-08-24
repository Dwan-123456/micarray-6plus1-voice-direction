from __future__ import annotations

import hashlib

import numpy as np
import pytest

from layer4_speech_separation import (
    BandMagnitudeMatcher,
    L4_MATCH_FREQUENCY_MAX_HZ,
    L4_MATCH_FREQUENCY_MIN_HZ,
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
        l2_direction_counts=((48_000, 2),),
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
    dominant = np.sin(2 * np.pi * 1_500 * time).astype(np.float32)
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
    assert L4_MATCH_FREQUENCY_MIN_HZ == 1_000.0
    assert L4_MATCH_FREQUENCY_MAX_HZ == 4_000.0
    assert selected.matching_algorithm == "l3_bf_1_4khz_complex_coherence_v3"
    assert not selected.waveform.flags.writeable


def test_layer4_candidate_contract_rejects_non_pair_or_misaligned_outputs() -> None:
    audio = np.zeros(16_000, dtype=np.float32)
    with pytest.raises(ValueError, match="equal length"):
        Layer4CandidatePair(
            "request", "model", "rev", 16_000,
            (audio, np.zeros(15_999, dtype=np.float32)),
        )


def test_matcher_breaks_equal_out_of_band_candidates_with_zero() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    reference = np.ascontiguousarray(np.sin(2 * np.pi * 2_600 * time), dtype=np.float32)
    out_of_band = np.ascontiguousarray(np.sin(2 * np.pi * 900 * time), dtype=np.float32)
    candidates = Layer4CandidatePair(
        "request", "model", "rev", 16_000, (out_of_band, out_of_band.copy()),
    )
    result = BandMagnitudeMatcher().select(
        parent=_parent(), reference_16k=reference, candidates=candidates,
    )
    assert result.candidate_scores[0] == result.candidate_scores[1]
    assert result.selected_source_index == 0


def test_matcher_includes_zero_padded_final_partial_frame() -> None:
    sample_rate = 16_000
    reference = np.zeros(701, dtype=np.float32)
    time = np.arange(189, dtype=np.float64) / sample_rate
    reference[-189:] = np.sin(2 * np.pi * 3_000 * time)
    wrong = np.zeros_like(reference)
    right = reference.copy()
    result = BandMagnitudeMatcher().select(
        parent=_parent(), reference_16k=np.ascontiguousarray(reference),
        candidates=Layer4CandidatePair(
            "request", "model", "rev", 16_000,
            (np.ascontiguousarray(wrong), np.ascontiguousarray(right)),
        ),
    )
    assert result.selected_source_index == 1
    assert result.candidate_scores[1] > result.candidate_scores[0]


def test_matcher_rejects_same_magnitude_spectrum_with_wrong_time_signature() -> None:
    rng = np.random.default_rng(20260821)
    reference = rng.normal(0.0, 0.1, 16_000).astype(np.float32)
    spectrum = np.fft.rfft(reference)
    phases = np.exp(1j * rng.uniform(-np.pi, np.pi, len(spectrum)))
    wrong = np.fft.irfft(np.abs(spectrum) * phases, n=len(reference)).astype(np.float32)
    right = np.ascontiguousarray(reference * np.float32(0.4))
    result = BandMagnitudeMatcher().select(
        parent=_parent(), reference_16k=np.ascontiguousarray(reference),
        candidates=Layer4CandidatePair(
            "request", "model", "rev", 16_000,
            (np.ascontiguousarray(wrong), right),
        ),
    )
    assert result.selected_source_index == 1
    assert result.candidate_scores[1] > 0.99
    assert result.candidate_scores[0] < 0.5


def test_matcher_returns_l3_reference_when_track_is_too_short_for_reliable_separation() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    reference = np.ascontiguousarray(np.sin(2 * np.pi * 2_800 * time), dtype=np.float32)
    candidates = Layer4CandidatePair(
        "request", "model", "rev", sample_rate,
        (np.ascontiguousarray(-reference), np.ascontiguousarray(reference * 0.5)),
    )
    result = BandMagnitudeMatcher().select(
        parent=_parent(), reference_16k=reference, candidates=candidates,
    )
    assert result.used_reference_fallback is True
    assert result.fallback_reason == "shorter_than_2_seconds"
    np.testing.assert_array_equal(result.waveform, reference)


def test_matcher_returns_l3_reference_when_candidate_identity_is_ambiguous() -> None:
    sample_rate = 16_000
    time = np.arange(2 * sample_rate, dtype=np.float64) / sample_rate
    reference = np.ascontiguousarray(np.sin(2 * np.pi * 3_100 * time), dtype=np.float32)
    candidates = Layer4CandidatePair(
        "request", "model", "rev", sample_rate,
        (np.ascontiguousarray(reference * 0.4), np.ascontiguousarray(reference * 0.8)),
    )
    result = BandMagnitudeMatcher().select(
        parent=_parent(), reference_16k=reference, candidates=candidates,
    )
    assert result.used_reference_fallback is True
    assert result.fallback_reason == "ambiguous_candidate_scores"
    np.testing.assert_array_equal(result.waveform, reference)
