from __future__ import annotations

import hashlib
from types import SimpleNamespace
import wave

import numpy as np
import pytest
import torch

from gui.dev_test_ui.offline_l6_store import OfflineLayer6UiStore
from layer4_speech_separation import (
    Layer4LongAudioInput,
    Layer4OfflineResult,
    SpeakerCountDecision,
)
from layer6_speaker_consolidation import (
    Layer6Fragment,
    Layer6Result,
    Layer6SpeakerAudio,
    OfflineLayer6Pipeline,
    speaker_label,
)
from layer6_speaker_consolidation.pipeline import _cluster, _track_similarity
from layer6_speaker_consolidation.campplus import CAMPPlus
from layer6_speaker_consolidation.matching import (
    LogisticCalibration,
    fit_logistic_calibration,
    hungarian_track_features,
)


class _Embedder:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []
        self.batch_calls: list[tuple[int, ...]] = []

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        self.calls.append(waveform)
        if float(np.mean(waveform)) >= 0:
            return np.array([1.0, 0.0], np.float32)
        return np.array([0.0, 1.0], np.float32)

    def embed_batch(self, waveforms: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        self.batch_calls.append(tuple(len(waveform) for waveform in waveforms))
        return tuple(self.embed(waveform) for waveform in waveforms)


class _DistinctEmbedder:
    def __init__(self) -> None:
        self.index = 0

    def embed(self, _waveform: np.ndarray) -> np.ndarray:
        embedding = np.eye(4, dtype=np.float32)[self.index]
        self.index += 1
        return embedding


def test_campplus_batch_matches_independent_forward_passes() -> None:
    torch.manual_seed(7)
    model = CAMPPlus(80, 16).eval()
    features = torch.randn(3, 199, 80)

    with torch.inference_mode():
        batched = model(features)
        independent = torch.cat(tuple(model(row[None]) for row in features), dim=0)

    assert torch.allclose(batched, independent, atol=1e-5, rtol=1e-5)


def _sha(audio: np.ndarray) -> str:
    return hashlib.sha256(audio.tobytes()).hexdigest()


def _result(
    track_id: int,
    branch: int,
    waveform: np.ndarray,
    decisions: tuple[bool, ...],
    *,
    start_sample_48k: int = 0,
    match: float = 0.8,
    mos: float = 0.6,
) -> Layer4OfflineResult:
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    source_audio = np.zeros(len(waveform) * 3, np.float32)
    source = Layer4LongAudioInput(
        f"asset-{track_id}",
        _sha(source_audio),
        "session",
        0,
        track_id,
        float(track_id * 45 % 360),
        start_sample_48k,
        48_000,
        source_audio,
        ((start_sample_48k, 2),),
    )
    count = SpeakerCountDecision(source.asset_id, 2, 1.0, "test", {})
    probabilities = tuple(0.9 if active else 0.1 for active in decisions)
    return Layer4OfflineResult(
        f"request-{track_id}",
        source,
        count,
        "two_speaker_separation",
        None,
        max(probabilities),
        any(decisions),
        "l5",
        probabilities,
        decisions,
        f"asset-{track_id}:branch-{branch}",
        _sha(waveform),
        {
            "candidate_match_score": match,
            "mos_score": mos,
            "dnsmos_sig": 4.0,
            "dnsmos_bak": 4.0,
            "dnsmos_ovrl": 4.0,
            "l5_threshold": 0.5,
            "output_waveform_16k": waveform,
        },
        ("candidate_0", "candidate_1")[branch],
    )


def _pair(
    track_id: int,
    *,
    a_value: float = 0.2,
    b_value: float = 0.1,
    frames: int = 100,
    a_match: float = 0.8,
    b_match: float = 0.7,
    a_mos: float = 0.6,
    b_mos: float = 0.5,
) -> tuple[Layer4OfflineResult, Layer4OfflineResult]:
    decisions = (True,) * frames
    return (
        _result(
            track_id,
            0,
            np.full(frames * 320, a_value, np.float32),
            decisions,
            match=a_match,
            mos=a_mos,
        ),
        _result(
            track_id,
            1,
            np.full(frames * 320, b_value, np.float32),
            decisions,
            match=b_match,
            mos=b_mos,
        ),
    )


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        maximum_speakers=3,
        speaker_similarity_threshold=0.62,
        secondary_candidate_match_gap_max=0.20,
        secondary_candidate_match_min=0.50,
        secondary_candidate_mos_min=0.30,
        maximum_internal_silence_ms=2_000,
    )


def test_l6_accepts_single_speaker_bypass_as_the_only_a_track() -> None:
    frames = 100
    waveform = np.full(frames * 320, 0.2, np.float32)
    source_audio = np.zeros(frames * 960, np.float32)
    source = Layer4LongAudioInput(
        "single-source",
        _sha(source_audio),
        "session",
        0,
        7,
        45.0,
        0,
        48_000,
        source_audio,
        ((0, 1),),
    )
    decisions = (True,) * frames
    result = Layer4OfflineResult(
        "single-request",
        source,
        SpeakerCountDecision(source.asset_id, 1, 1.0, "test", {}),
        "single_speaker_bypass",
        None,
        0.9,
        True,
        "l5",
        (0.9,) * frames,
        decisions,
        "single-output",
        _sha(waveform),
        {
            "mos_score": 0.7,
            "dnsmos_sig": 4.0,
            "dnsmos_bak": 4.0,
            "dnsmos_ovrl": 4.0,
            "l5_threshold": 0.5,
            "output_waveform_16k": waveform,
        },
        "merged",
    )
    embedder = _Embedder()

    consolidated = OfflineLayer6Pipeline(embedder, _config()).process((result,))

    assert consolidated.speaker_count == 1
    assert len(consolidated.fragments) == 1
    assert consolidated.fragments[0].branch_index == 0
    assert consolidated.fragments[0].match_score == 1.0
    assert len(embedder.calls) == 1


def test_l6_embeds_only_l5_voice_from_a_and_b_passing_all_three_gates() -> None:
    embedder = _Embedder()
    results = (
        *_pair(1, b_match=0.70, b_mos=0.40),
        *_pair(2, a_match=0.90, b_match=0.60, b_mos=0.80),
        *_pair(3, a_match=0.70, b_match=0.50, b_mos=0.80),
        *_pair(4, a_match=0.70, b_match=0.60, b_mos=0.30),
        *_pair(5, a_match=0.71, b_match=0.51, b_mos=0.31),
    )

    result = OfflineLayer6Pipeline(embedder, _config()).process(results)

    assert len(embedder.calls) == 7
    assert all(len(waveform) == 32_000 for waveform in embedder.calls)
    assert result.metadata["extracted_audio_ids"] == (
        "asset-1:branch-0",
        "asset-1:branch-1",
        "asset-2:branch-0",
        "asset-3:branch-0",
        "asset-4:branch-0",
        "asset-5:branch-0",
        "asset-5:branch-1",
    )


def test_l6_concatenates_only_l5_voice_frames_for_one_track_embedding() -> None:
    embedder = _Embedder()
    decisions = (False,) * 10 + (True,) * 30 + (False,) * 10
    frames = np.full((50, 320), -0.5, np.float32)
    frames[10:40] = 0.25
    source = _result(1, 0, frames.reshape(-1), decisions)

    result = OfflineLayer6Pipeline(embedder, _config()).process((source,))

    assert result.speaker_count == 1
    assert len(embedder.calls) == 1
    assert len(embedder.calls[0]) == 9_600
    assert np.allclose(embedder.calls[0], 0.25)
    assert result.metadata["voiceprint_voice_sample_counts"] == {
        "asset-1:branch-0": 9_600,
    }


def test_l6_batches_two_second_segments_per_track() -> None:
    embedder = _Embedder()
    source = _result(
        1,
        0,
        np.full(80_000, 0.25, np.float32),
        (True,) * 250,
    )

    result = OfflineLayer6Pipeline(embedder, _config()).process((source,))

    assert embedder.batch_calls == [(32_000, 32_000, 16_000)]
    assert result.metadata["voiceprint_segment_counts"] == {
        "asset-1:branch-0": 3,
    }
    assert result.metadata["voiceprint_retained_segment_counts"] == {
        "asset-1:branch-0": 3,
    }


def test_l6_clusters_complete_tracks_and_records_one_voiceprint_to_many_audio() -> None:
    embedder = _Embedder()
    result = OfflineLayer6Pipeline(embedder, _config()).process((
        *_pair(1, a_value=0.2, b_value=0.1, frames=200),
        *_pair(2, a_value=-0.2, b_value=-0.1, b_match=0.4, frames=200),
    ))

    assert result.speaker_count == 2
    assert tuple(item.label for item in result.outputs) == ("Speaker A", "Speaker B")
    assert len(result.fragments) == 3
    assert result.metadata["voiceprint_audio_ids"] == {
        1: ("asset-1:branch-0", "asset-1:branch-1"),
        2: ("asset-2:branch-0",),
    }
    matrix = np.asarray(result.metadata["pairwise_similarity_matrix"])
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)


def test_l6_output_contract_supports_one_hundred_session_speakers() -> None:
    waveform = np.zeros(320, dtype=np.float32)
    outputs = tuple(
        Layer6SpeakerAudio(
            speaker_id,
            speaker_label(speaker_id),
            16_000,
            0,
            960,
            waveform,
            (speaker_id,),
            (f"fragment-{speaker_id}",),
            0.5,
        )
        for speaker_id in range(1, 101)
    )
    fragment = Layer6Fragment(
        "fragment-100",
        "asset-100",
        100,
        0.0,
        0,
        True,
        0,
        960,
        waveform,
        (0.9,),
        (True,),
        np.asarray((1.0, 0.0), dtype=np.float32),
        100,
        0.5,
        0.5,
        0.5,
    )

    result = Layer6Result("session", 100, outputs, (fragment,), {})

    assert result.outputs[-1].label == "Speaker CV"
    assert result.fragments[0].speaker_id == 100
    with pytest.raises(ValueError, match="1..100"):
        speaker_label(101)


def test_l6_forces_four_distinct_complete_track_voiceprints_down_to_three() -> None:
    inputs = tuple(
        _result(
            track_id,
            0,
            np.full(8_000, 0.1 * track_id, np.float32),
            (True,) * 25,
        )
        for track_id in range(1, 5)
    )

    result = OfflineLayer6Pipeline(_DistinctEmbedder(), _config()).process(inputs)

    assert result.speaker_count == 3
    assert len(result.metadata["voiceprint_audio_ids"]) == 3


def test_l6_overlap_keeps_the_audio_with_higher_l4_mos() -> None:
    decisions = (True,) * 200
    low = _result(
        1, 0, np.full(64_000, 0.2, np.float32), decisions, match=0.95, mos=0.40,
    )
    high = _result(
        2, 0, np.full(64_000, 0.8, np.float32), decisions, match=0.60, mos=0.90,
    )

    result = OfflineLayer6Pipeline(_Embedder(), _config()).process((low, high))

    assert result.speaker_count == 1
    assert np.allclose(result.outputs[0].waveform_16k, 0.8)
    assert result.outputs[0].fragment_ids == (
        "asset-1:branch-0", "asset-2:branch-0",
    )


def test_l6_track_similarity_requires_repeated_one_to_one_segment_evidence() -> None:
    left = np.eye(4, dtype=np.float32)
    repeated = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
    ], dtype=np.float32)
    one_match = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
    ], dtype=np.float32)

    repeated_score, _, repeated_required = _track_similarity(left, repeated)
    one_score, _, one_required = _track_similarity(left, one_match)

    assert repeated_required == one_required == 2
    assert repeated_score == 1.0
    assert one_score == 0.0


def test_l6_track_similarity_allows_one_segment_short_track_to_merge() -> None:
    short = np.array([[1.0, 0.0]], dtype=np.float32)
    established = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.8, 0.2],
    ], dtype=np.float32)

    score, matched, required = _track_similarity(short, established)

    assert required == matched == 1
    assert score == 1.0


def test_l6_hungarian_matching_avoids_greedy_pairing_trap() -> None:
    left = np.eye(2, dtype=np.float32)
    right = np.array([
        [0.90, 0.85],
        [0.80, 0.10],
    ], dtype=np.float32)

    features = hungarian_track_features(
        left,
        right,
        threshold=0.62,
        minimum_match_count=2,
        required_coverage=0.30,
    )

    assert features.decision_score == pytest.approx(0.80)
    assert features.median == pytest.approx(0.825)
    assert features.q25 == pytest.approx(0.8125)
    assert features.coverage_above_threshold == 1.0
    assert features.matched_count == features.required_count == 2


def test_l6_logistic_calibration_returns_interpretable_probability() -> None:
    features = hungarian_track_features(
        np.eye(2, dtype=np.float32),
        np.eye(2, dtype=np.float32),
        threshold=0.62,
        minimum_match_count=2,
        required_coverage=0.30,
    )
    calibration = LogisticCalibration(
        feature_names=("median", "coverage_above_threshold"),
        coefficients=(2.0, 1.0),
        intercept=-1.0,
    )

    assert calibration.predict(features) == pytest.approx(1.0 / (1.0 + np.exp(-2.0)))


def test_l6_logistic_calibration_can_be_fitted_from_labeled_pairs() -> None:
    same = hungarian_track_features(
        np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32),
        threshold=0.62, minimum_match_count=2, required_coverage=0.30,
    )
    different = hungarian_track_features(
        np.eye(2, dtype=np.float32), -np.eye(2, dtype=np.float32),
        threshold=0.62, minimum_match_count=2, required_coverage=0.30,
    )
    calibration = fit_logistic_calibration(
        (same, same, different, different),
        (True, True, False, False),
        feature_names=("median", "coverage_above_threshold"),
        l2_regularization=0.1,
    )

    assert calibration.predict(same) > calibration.predict(different)


def test_l6_complete_link_does_not_allow_one_track_to_bridge_two_people() -> None:
    similarities = np.array([
        [1.0, 0.8, 0.5],
        [0.8, 1.0, 0.8],
        [0.5, 0.8, 1.0],
    ], dtype=np.float32)

    assignments = _cluster(similarities, 0.62, 3)

    assert assignments[1] == assignments[2]
    assert assignments[0] != assignments[1]


def test_l6_trims_edge_silence_and_caps_internal_silence_at_two_seconds() -> None:
    decisions = (
        (False,) * 50
        + (True,) * 25
        + (False,) * 150
        + (True,) * 25
        + (False,) * 50
    )
    frames = np.zeros((len(decisions), 320), np.float32)
    frames[50:75] = 0.25
    frames[225:250] = 0.25
    a = _result(1, 0, frames.reshape(-1), decisions, match=0.8, mos=0.7)

    result = OfflineLayer6Pipeline(_Embedder(), _config()).process((a,))

    output = result.outputs[0].waveform_16k.reshape(-1, 320)
    assert len(output) == 25 + 100 + 25
    assert np.allclose(output[:25], 0.25)
    assert not np.any(output[25:125])
    assert np.allclose(output[125:], 0.25)


def test_l6_returns_zero_speakers_when_selected_tracks_have_no_l5_voice() -> None:
    embedder = _Embedder()
    silent = _result(
        1,
        0,
        np.zeros(3_200, np.float32),
        (False,) * 10,
        match=0.8,
        mos=0.7,
    )

    result = OfflineLayer6Pipeline(embedder, _config()).process((silent,))

    assert result.speaker_count == 0
    assert result.outputs == ()
    assert len(embedder.calls) == 0
    assert result.metadata["insufficient_voice_audio_ids"] == ("asset-1:branch-0",)


def test_l6_skips_tracks_with_less_than_half_a_second_of_l5_voice() -> None:
    embedder = _Embedder()
    short = _result(
        1,
        0,
        np.full(24 * 320, 0.25, np.float32),
        (True,) * 24,
        match=0.8,
        mos=0.7,
    )

    result = OfflineLayer6Pipeline(embedder, _config()).process((short,))

    assert result.speaker_count == 0
    assert len(embedder.calls) == 0
    assert result.metadata["insufficient_voice_audio_ids"] == ("asset-1:branch-0",)


def test_l6_ui_displays_each_silence_compressed_voiceprint_directly() -> None:
    first = Layer6SpeakerAudio(
        1,
        "Speaker A",
        16_000,
        0,
        9_600,
        np.full(640, 0.25, np.float32),
        (7,),
        ("a", "b"),
        0.8,
    )
    second = Layer6SpeakerAudio(
        2,
        "Speaker B",
        16_000,
        0,
        9_600,
        np.full(960, -0.25, np.float32),
        (9,),
        ("c",),
        0.7,
    )
    store = OfflineLayer6UiStore()
    try:
        store.set_result(Layer6Result("session", 2, (first, second), (), {}))
        for speaker_id, frames in ((1, 640), (2, 960)):
            path = store.audio_path(speaker_id)
            assert path is not None
            with wave.open(str(path), "rb") as reader:
                assert reader.getframerate() == 16_000
                assert reader.getnframes() == frames
        snapshots = store.snapshots()
        assert tuple(item.audio_sample_count for item in snapshots) == (1_920, 2_880)
        assert snapshots[0].display_label == (
            "声纹 1 · Speaker A · 关联音轨 2 · 来源L2 ID 7 · MOS 0.80"
        )
        assert snapshots[1].display_label == (
            "声纹 2 · Speaker B · 关联音轨 1 · 来源L2 ID 9 · MOS 0.70"
        )
    finally:
        store.close()
