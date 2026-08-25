from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np

from app.realtime_layer456 import CachingEmbeddingBackend, IncrementalLayer456Processor
from layer4_speech_separation import Layer4CandidatePair, Layer4LongAudioInput
from layer5_voice_classifier.gain_compensation import InputGainCompensationSettings


class _Backend:
    sample_rate = 16_000
    source_count = 2
    backend = "mossformer2_ss_16k"
    model_id = "fake-mf2"
    model_revision = "test"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def separate(self, request_id, waveform_16k):
        self.calls.append(len(waveform_16k))
        return Layer4CandidatePair(
            request_id,
            self.model_id,
            self.model_revision,
            16_000,
            (
                np.ascontiguousarray(waveform_16k * np.float32(0.8)),
                np.ascontiguousarray(-waveform_16k * np.float32(0.4)),
            ),
        )


class _RankBackend(_Backend):
    def separate(self, request_id, waveform_16k):
        self.calls.append(len(waveform_16k))
        return Layer4CandidatePair(
            request_id,
            self.model_id,
            self.model_revision,
            16_000,
            (
                np.ascontiguousarray(np.zeros_like(waveform_16k)),
                np.ascontiguousarray(waveform_16k),
            ),
        )


class _Layer5:
    threshold = 0.5
    input_gain_compensation = InputGainCompensationSettings(enabled=False)

    def __init__(self) -> None:
        self.calls: list[int] = []

    def process_long_audio_20ms(self, item):
        self.calls.append(len(item.waveform))
        count = len(item.waveform) // 320
        probabilities = np.full(count, 0.9, dtype=np.float32)
        return SimpleNamespace(
            summary_probability=0.9,
            summary_is_voice=True,
            model_id="fake-l5",
            probabilities_20ms=probabilities,
            is_voice_20ms=(True,) * count,
            threshold=self.threshold,
            metadata={"backend": "fake-l5"},
        )


class _Quality:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def score(self, waveform):
        self.calls.append(len(waveform))
        return 4.0, 4.0, 4.0


class _Embedder:
    def __init__(self) -> None:
        self.calls = 0
        self.lengths: list[int] = []

    def embed_batch(self, waveforms):
        self.calls += len(waveforms)
        self.lengths.extend(len(waveform) for waveform in waveforms)
        outputs = []
        for waveform in waveforms:
            sign = 1.0 if float(np.mean(waveform)) >= 0.0 else -1.0
            outputs.append(np.asarray((sign, 1.0), dtype=np.float32) / np.sqrt(2.0))
        return tuple(outputs)

    def embed(self, waveform):
        return self.embed_batch((waveform,))[0]


_L6 = SimpleNamespace(
    clustering_backend="multistage",
    multistage_l=30,
    multistage_u1=100,
    multistage_u2=600,
    multistage_fallback_distance=0.38,
    maximum_speakers=3,
    speaker_similarity_threshold=0.62,
    secondary_candidate_match_gap_max=1.0,
    secondary_candidate_match_min=0.0,
    secondary_candidate_mos_min=0.0,
    maximum_internal_silence_ms=2_000,
)


def _source(
    index: int,
    count: int = 2,
    *,
    duration_seconds: int = 10,
    track_id: int = 1,
):
    start = index * duration_seconds * 48_000
    samples = duration_seconds * 48_000
    time = np.arange(samples, dtype=np.float32) / np.float32(48_000)
    waveform = np.ascontiguousarray(0.02 * np.sin(2.0 * np.pi * 1_200.0 * time), dtype=np.float32)
    return Layer4LongAudioInput(
        asset_id=f"session:epoch0:track{track_id}:start{start}",
        sha256=hashlib.sha256(waveform.tobytes()).hexdigest(),
        session_id="session",
        stream_epoch=0,
        track_id=track_id,
        theta_deg=20.0,
        start_sample=start,
        sample_rate=48_000,
        waveform=waveform,
        l2_direction_counts=tuple(
            (sample, count) for sample in range(start + 960, start + samples + 1, 960)
        ),
    )


def _processor(backend=None, embedder=None, layer5=None, quality=None, **kwargs):
    return IncrementalLayer456Processor(
        backend=backend or _Backend(),
        layer5=layer5 or _Layer5(),
        quality_scorer=quality or _Quality(),
        embedder=embedder or _Embedder(),
        layer6_config=_L6,
        **kwargs,
    )


def test_two_source_pipeline_publishes_l5_stable_watermarks_and_flushes_20_seconds():
    backend = _Backend()
    layer5 = _Layer5()
    quality = _Quality()
    processor = _processor(backend, layer5=layer5, quality=quality)
    first = processor.push(_source(0))
    assert first is not None
    assert first.valid_through_sample_48k == 355_200  # 9.0 s L4 - 1.6 s L5 hold.
    assert len(first.l4_processed) == len(first.l5_results) == 2
    assert all(len(item.waveform_16k) == 118_400 for item in first.l4_processed)

    second = processor.push(_source(1))
    assert second is not None
    assert second.valid_through_sample_48k == 835_200  # 17.4 s.
    assert all(len(item.l5_probabilities_20ms) == 870 for item in second.l5_results)
    assert second.l6_result.metadata["realtime_l6_reused"] is False

    final = processor.finalize()
    assert final is not None and final.is_final
    assert final.valid_through_sample_48k == 20 * 48_000
    assert all(len(item.waveform_16k) == 20 * 16_000 for item in final.l4_processed)
    assert backend.calls == [10 * 16_000, 11 * 16_000]
    assert sorted(set(layer5.calls)) == [67_200, 144_000, 211_200]
    assert quality.calls == [144_320, 144_320, 144_320, 144_320]
    assert all(
        item.metadata["dnsmos_scope"]
        == "periodic_30s_plus_final_tail_9_02s"
        for item in final.l4_processed
    )
    assert all(
        item.metadata["dnsmos_complete_branch"] is False
        for item in final.l4_processed
    )
    assert final.l6_result.metadata["canonical"] is False
    assert final.l6_result.metadata["realtime_tail_flushed"] is True
    assert final.l6_result.metadata["finality_scope"] == "realtime_preview_tail_flushed"
    assert final.l6_result.metadata["retained_commit_dtos"] == 0
    assert processor.layer6._streaming_clusterer is None


def test_one_source_bypasses_mf2_and_can_upgrade_by_replaying_when_two_sources_appear():
    backend = _Backend()
    processor = _processor(backend)
    first = processor.push(_source(0, 1))
    assert first is not None and len(first.l4_processed) == 1
    assert backend.calls == []

    revised = processor.push(_source(1, 2))
    assert revised is not None
    assert len(revised.l4_processed) == 2
    assert revised.valid_through_sample_48k == 835_200
    assert backend.calls == [10 * 16_000, 11 * 16_000]
    assert revised.l6_result.metadata["realtime_l6_state_reset"] is True


def test_voiceprint_embeddings_are_cached_across_revisions():
    embedder = _Embedder()
    processor = _processor(embedder=embedder)
    first = processor.push(_source(0))
    first_calls = embedder.calls
    second = processor.push(_source(1))
    second_calls = embedder.calls
    assert first_calls > 0
    # L6 now refreshes with every L4 block. Previously embedded, unchanged
    # 2 s evidence remains cached; only newly completed/changed evidence is
    # sent to CAMPPlus.
    assert first_calls < second_calls <= first_calls * 2
    assert first is not None and second is not None
    assert (
        second.l6_result.metadata["multistage"]["evidence_count"]
        > first.l6_result.metadata["multistage"]["evidence_count"]
    )
    assert processor.cached_embedder.cached_segments == second_calls
    final = processor.finalize()
    assert final is not None
    # Six complete 2 s segments from the first revision remain cache hits; the
    # final batch computes only changed/new complete-track evidence.
    assert embedder.calls < first_calls + 20
    assert final.l6_result.metadata["cached_voiceprint_segments"] == embedder.calls


def test_abort_discards_multistage_state_without_flushing_audio_tails():
    processor = _processor()
    snapshot = processor.push(_source(0))

    assert snapshot is not None
    assert processor.layer6._streaming_clusterer is not None
    processor.abort()

    assert processor.layer6._streaming_clusterer is None
    assert processor._last_l6_result is None
    assert processor._finalized is True


def test_stable_branch_identity_is_separate_from_cumulative_candidate_rank():
    processor = _processor(_RankBackend())
    snapshot = processor.push(_source(0))
    assert snapshot is not None
    by_kind = {item.output_kind: item for item in snapshot.l4_processed}
    assert by_kind["candidate_0"].metadata["candidate_rank"] == 0
    assert by_kind["candidate_0"].metadata["stable_branch_id"] == 1
    assert by_kind["candidate_1"].metadata["stable_branch_id"] == 0
    assert (
        by_kind["candidate_0"].metadata["candidate_match_score"]
        > by_kind["candidate_1"].metadata["candidate_match_score"]
    )


def test_long_one_to_two_replay_degrades_preview_instead_of_replaying_unbounded_history():
    backend = _Backend()
    processor = _processor(backend, max_replay_samples_48k=10 * 48_000)
    processor.push(_source(0, 1))
    revised = processor.push(_source(1, 2))
    assert revised is not None
    assert backend.calls == [10 * 16_000]
    assert revised.valid_through_sample_48k == 10 * 48_000 + 355_200
    assert all(item.metadata["realtime_preview_degraded"] for item in revised.l4_processed)
    assert all(
        item.metadata["realtime_preview_degraded_reason"]
        == "one_to_two_replay_limit_exceeded"
        for item in revised.l4_processed
    )
    assert processor.retained_state_samples["source_48k"] == 10 * 48_000


def test_long_odd_chunk_sequence_has_linear_state_and_throttled_l6():
    quality = _Quality()
    processor = _processor(
        quality=quality,
        chunk_samples_48k=3 * 48_000,
        overlap_samples_48k=48_000,
        l6_interval_samples_48k=10 * 48_000,
    )
    real_l6 = processor.layer6

    class _CountingL6:
        def __init__(self):
            self.calls = 0

        def process_streaming(self, results, *, final=False):
            self.calls += 1
            return real_l6.process_streaming(results, final=final)

        def reset_streaming(self):
            real_l6.reset_streaming()

    counting_l6 = _CountingL6()
    processor.layer6 = counting_l6
    retained = []
    for index in range(8):
        snapshot = processor.push(_source(index, 1, duration_seconds=3))
        assert snapshot is not None
        retained.append(dict(processor.retained_state_samples))
    final = processor.finalize()
    assert final is not None
    assert [item["source_48k"] for item in retained] == [
        (index + 1) * 3 * 48_000 for index in range(8)
    ]
    assert all(item["retained_commit_dtos"] == 0 for item in retained)
    assert all(
        right["branch_16k"] - left["branch_16k"] <= 3 * 16_000
        for left, right in zip(retained, retained[1:])
    )
    assert counting_l6.calls <= 4
    assert quality.calls == [144_320, 144_320]


def test_l6_refresh_interval_defaults_to_l4_chunk_size():
    processor = _processor(
        chunk_samples_48k=4 * 48_000,
        overlap_samples_48k=48_000,
    )

    assert processor.l6_interval_samples_48k == 4 * 48_000


def test_l6_refreshes_after_each_complete_l4_chunk():
    processor = _processor(
        chunk_samples_48k=4 * 48_000,
        overlap_samples_48k=48_000,
    )
    real_l6 = processor.layer6

    class _CountingL6:
        def __init__(self):
            self.calls = 0

        def process_streaming(self, results, *, final=False):
            self.calls += 1
            return real_l6.process_streaming(results, final=final)

        def reset_streaming(self):
            real_l6.reset_streaming()

    counting_l6 = _CountingL6()
    processor.layer6 = counting_l6

    first = processor.push(_source(0, duration_seconds=4))
    second = processor.push(_source(1, duration_seconds=4))

    assert first is not None and second is not None
    assert counting_l6.calls == 2
    assert second.l6_result.metadata["realtime_l6_reused"] is False
    assert (
        second.l6_result.metadata["realtime_l6_interval_samples_48k"]
        == 4 * 48_000
    )


def test_campplus_two_second_evidence_crosses_odd_3_5_15_second_chunks():
    for chunk_seconds in (3, 5, 15):
        embedder = _Embedder()
        processor = _processor(
            embedder=embedder,
            chunk_samples_48k=chunk_seconds * 48_000,
            overlap_samples_48k=48_000,
            l6_interval_samples_48k=48_000,
        )
        processor.push(_source(0, duration_seconds=chunk_seconds))
        processor.push(_source(1, duration_seconds=chunk_seconds))
        final = processor.finalize()
        assert final is not None
        expected_samples = 2 * chunk_seconds * 16_000
        assert all(
            count == expected_samples
            for count in final.l6_result.metadata["voiceprint_voice_sample_counts"].values()
        )
        assert any(length == 2 * 16_000 for length in embedder.lengths)
        assert all(length >= 8_000 for length in embedder.lengths)


def test_non_overlapping_tracks_keep_independent_final_watermarks():
    processor = _processor()
    first = processor.push(
        _source(0, 1, track_id=1),
        is_final_chunk=True,
    )
    second = processor.push(
        _source(1, 1, track_id=2),
        is_final_chunk=True,
    )
    final = processor.finalize()

    assert first is not None
    assert second is not None
    assert final is not None and final.is_final
    assert final.valid_through_sample_48k == 20 * 48_000
    assert {
        (item.source.track_id, item.source.start_sample, len(item.source.waveform))
        for item in final.l4_processed
    } == {
        (1, 0, 10 * 48_000),
        (2, 10 * 48_000, 10 * 48_000),
    }
    assert set(
        final.l6_result.metadata["realtime_l5_track_watermarks_48k"].values()
    ) == {10 * 48_000, 20 * 48_000}


def test_pending_track_does_not_hide_an_already_stable_preview():
    processor = _processor()
    assert processor.push(_source(0, 1, track_id=1)) is not None

    snapshot = processor.push(
        _source(10, 2, duration_seconds=1, track_id=2),
    )

    assert snapshot is not None
    assert {item.source.track_id for item in snapshot.l4_processed} == {1}
    assert snapshot.l6_result.metadata["realtime_pending_track_count"] == 1


def test_session_embedding_cache_does_not_thrash_after_4096_segments():
    embedder = _Embedder()
    cache = CachingEmbeddingBackend(embedder)
    segments = tuple(
        np.asarray((float(index),), dtype=np.float32)
        for index in range(4_100)
    )

    cache.embed_batch(segments)
    first_calls = embedder.calls
    cache.embed_batch(segments)

    assert first_calls == 4_100
    assert embedder.calls == first_calls
    assert cache.cached_segments == 4_100
