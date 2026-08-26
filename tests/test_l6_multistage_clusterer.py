from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

import layer6_speaker_consolidation.pipeline as pipeline_module
from common.disk_audio import DiskAudioView
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


def test_multistage_snapshot_uses_compact_immutable_historical_labels() -> None:
    backend = _SequenceBackend(((0,), (0, 1), (1, 1, 0)))
    clusterer = MultiStageVoiceprintClusterer(_config(), backend=backend)

    first = clusterer.update((_evidence(0, (1.0, 0.0)),))
    second = clusterer.update((_evidence(1, (0.0, 1.0)),))
    corrected = clusterer.update((_evidence(2, (0.9, 0.1)),))

    assert not isinstance(corrected.labels_by_evidence_id, dict)
    assert dict(first.labels_by_evidence_id) == {"evidence-0": 0}
    assert dict(second.labels_by_evidence_id) == {
        "evidence-0": 0,
        "evidence-1": 1,
    }
    assert dict(corrected.labels_by_evidence_id) == {
        "evidence-0": 1,
        "evidence-1": 1,
        "evidence-2": 0,
    }


def test_multistage_track_assignment_updates_only_corrected_votes() -> None:
    backend = _SequenceBackend(((0,), (0, 1), (1, 1, 0)))
    clusterer = MultiStageVoiceprintClusterer(_config(), backend=backend)
    evidence = (
        SegmentEvidence("a-0", "track-a", np.asarray((1.0, 0.0)), 32_000),
        SegmentEvidence("a-1", "track-a", np.asarray((0.9, 0.1)), 16_000),
        SegmentEvidence("b-0", "track-b", np.asarray((0.0, 1.0)), 32_000),
    )

    clusterer.update((evidence[0],))
    before = clusterer.update((evidence[1],))
    corrected = clusterer.update((evidence[2],))

    assert before.assignments_by_track_key == {"track-a": 0}
    assert corrected.assignments_by_track_key == {
        "track-a": 1,
        "track-b": 0,
    }


def test_multistage_incremental_track_votes_match_full_history_reference() -> None:
    rng = np.random.default_rng(731)
    outputs: list[tuple[int, ...]] = []
    current = np.empty(0, dtype=np.int32)
    for count in range(1, 61):
        current = np.concatenate((current, rng.integers(0, 3, size=1)))
        corrections = rng.choice(count, size=min(count, 7), replace=False)
        current[corrections] = rng.integers(0, 3, size=len(corrections))
        outputs.append(tuple(int(value) for value in current))
    clusterer = MultiStageVoiceprintClusterer(
        _config(), backend=_SequenceBackend(tuple(outputs)),
    )
    evidence: list[SegmentEvidence] = []

    for index in range(60):
        item = SegmentEvidence(
            f"evidence-{index}",
            f"track-{index % 3}",
            np.asarray((1.0, float(index + 1)), dtype=np.float32),
            (index % 4 + 1) * 8_000,
        )
        evidence.append(item)
        snapshot = clusterer.update((item,))
        expected: dict[str, int] = {}
        for track_index in range(3):
            weighted: dict[int, int] = {}
            first: dict[int, int] = {}
            track_items = [
                value for value in evidence
                if value.track_key == f"track-{track_index}"
            ]
            for local_index, value in enumerate(track_items):
                label = snapshot.labels_by_evidence_id[value.evidence_id]
                weighted[label] = weighted.get(label, 0) + value.weight_samples_16k
                first.setdefault(label, local_index)
            if weighted:
                expected[f"track-{track_index}"] = max(
                    weighted,
                    key=lambda label: (weighted[label], -first[label], -label),
                )
        assert snapshot.assignments_by_track_key == expected


def test_streaming_raw_assignment_prefers_incremental_track_vote() -> None:
    class ForbiddenLabels(dict[str, int]):
        def __getitem__(self, key: str) -> int:
            raise AssertionError(f"rescanned historical label {key}")

    state = SimpleNamespace(
        track_key="track-a",
        segment_sample_counts=[32_000] * 10_000,
    )
    snapshot = pipeline_module.MultiStageSnapshot(
        ForbiddenLabels(),
        10_000,
        1,
        "test",
        {"track-a": 2},
    )

    assert pipeline_module.OfflineLayer6Pipeline._streaming_raw_assignment(
        state, snapshot,
    ) == 2


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


def test_multistage_adapter_merges_fallback_overflow_to_five_clusters() -> None:
    outputs = tuple(tuple(range(count)) for count in range(1, 7))
    clusterer = MultiStageVoiceprintClusterer(
        _config(maximum_speakers=5, multistage_u1=6),
        backend=_SequenceBackend(outputs),
    )

    snapshot = clusterer.update(tuple(
        _evidence(index, (1.0, float(index + 1)))
        for index in range(6)
    ))

    assert snapshot.cluster_count == 5
    assert len(set(snapshot.labels_by_evidence_id.values())) == 5


class _SignEmbedder:
    def embed_batch(self, waveforms: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        return tuple(
            np.asarray((1.0, 0.0) if float(np.mean(value)) >= 0.0 else (0.0, 1.0), dtype=np.float32)
            for value in waveforms
        )


class _RecordingSignEmbedder(_SignEmbedder):
    def __init__(self) -> None:
        self.batch_lengths: list[tuple[int, ...]] = []

    def embed_batch(self, waveforms: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        self.batch_lengths.append(tuple(len(value) for value in waveforms))
        return super().embed_batch(waveforms)


def _result(
    track_id: int,
    value: float,
    frames: int,
    *,
    session_id: str = "session",
    stream_epoch: int = 0,
    decisions: tuple[bool, ...] | None = None,
    start_sample_48k: int = 0,
    match_score: float = 0.9,
    mos_score: float = 0.8,
    output_asset_id: str | None = None,
) -> Layer4OfflineResult:
    waveform = np.full(frames * 320, value, dtype=np.float32)
    source_waveform = np.zeros(frames * 960, dtype=np.float32)
    active = (True,) * frames if decisions is None else tuple(decisions)
    assert len(active) == frames
    source = Layer4LongAudioInput(
        f"source-{track_id}",
        hashlib.sha256(source_waveform.tobytes()).hexdigest(),
        session_id,
        stream_epoch,
        track_id,
        float(track_id * 60),
        start_sample_48k,
        48_000,
        source_waveform,
        ((start_sample_48k, 2),),
    )
    decision = SpeakerCountDecision(source.asset_id, 2, 1.0, "test", {})
    return Layer4OfflineResult(
        f"request-{track_id}",
        source,
        decision,
        "two_speaker_separation",
        None,
        0.9,
        any(active),
        "l5",
        tuple(0.9 if is_active else 0.1 for is_active in active),
        active,
        output_asset_id or f"source-{track_id}:branch-0",
        hashlib.sha256(waveform.tobytes()).hexdigest(),
        {
            "candidate_match_score": match_score,
            "mos_score": mos_score,
            "stable_branch_id": 0,
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


def test_layer6_multistage_accepts_short_tail_only_for_finalized_track() -> None:
    pipeline = OfflineLayer6Pipeline(_SignEmbedder(), _config())
    results = (_result(1, 0.2, 50), _result(2, -0.2, 50))

    snapshot = pipeline.process_streaming(
        results,
        finalized_track_keys=frozenset({("session", 0, 1)}),
    )

    assert snapshot.metadata["multistage"]["evidence_count"] == 1
    assert snapshot.metadata["voiceprint_audio_ids"] == {
        1: ("source-1:branch-0",),
    }


def test_streaming_l6_never_calls_the_offline_full_history_path(monkeypatch) -> None:
    pipeline = OfflineLayer6Pipeline(_SignEmbedder(), _config())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("streaming L6 entered the full-history path")

    monkeypatch.setattr(pipeline, "_process", forbidden)
    monkeypatch.setattr(pipeline, "_merge_speaker", forbidden)
    monkeypatch.setattr(pipeline_module, "_pairwise_similarities", forbidden)

    result = pipeline.process_streaming((_result(1, 0.2, 100),))

    assert result.speaker_count == 1
    assert result.metadata["streaming_incremental"] is True
    assert result.metadata["pairwise_diagnostics_available"] is False
    assert result.metadata["pairwise_similarity_matrix"] == ()
    assert isinstance(result.outputs[0].waveform_16k, DiskAudioView)


def test_streaming_l6_embeds_only_new_complete_two_second_evidence() -> None:
    embedder = _RecordingSignEmbedder()
    pipeline = OfflineLayer6Pipeline(embedder, _config())

    first = pipeline.process_streaming((_result(1, 0.2, 150),))  # 3 s
    second = pipeline.process_streaming((_result(1, 0.2, 300),))  # 6 s cumulative
    repeated = pipeline.process_streaming((_result(1, 0.2, 300),))

    assert embedder.batch_lengths == [(32_000,), (32_000, 32_000)]
    assert first.metadata["multistage"]["evidence_count"] == 1
    assert first.metadata["streaming_pending_voice_samples_16k"] == {
        "session:epoch0:track1:stable0": 16_000,
    }
    assert second.metadata["multistage"]["evidence_count"] == 3
    assert second.metadata["streaming_new_evidence_count"] == 2
    assert np.array_equal(
        repeated.outputs[0].waveform_16k,
        second.outputs[0].waveform_16k,
    )
    assert repeated.outputs[0].end_sample_48k == second.outputs[0].end_sample_48k
    assert repeated.metadata["incremental_changed_speaker_ids"] == ()
    assert repeated.metadata["incremental_append_only_speaker_ids"] == ()


@pytest.mark.parametrize("chunk_seconds", (3, 5, 15))
def test_streaming_l6_two_second_boundaries_ignore_variable_l4_chunks(
    chunk_seconds: int,
) -> None:
    embedder = _RecordingSignEmbedder()
    pipeline = OfflineLayer6Pipeline(embedder, _config())
    total_seconds = 15
    latest = None
    for end_seconds in range(chunk_seconds, total_seconds + 1, chunk_seconds):
        latest = _result(1, 0.2, end_seconds * 50)
        pipeline.process_streaming((latest,))
    assert latest is not None

    final = pipeline.process_streaming((latest,), final=True)
    evidence_ids = tuple(pipeline._streaming_snapshot.labels_by_evidence_id)  # type: ignore[union-attr]

    assert evidence_ids == tuple(
        f"session:epoch0:track1:stable0:segment{index}"
        for index in range(8)
    )
    assert [length for batch in embedder.batch_lengths for length in batch] == [
        *([32_000] * 7),
        16_000,
    ]
    assert final.metadata["multistage"]["evidence_count"] == 8
    assert final.metadata["streaming_pending_voice_samples_16k"] == {
        "session:epoch0:track1:stable0": 0,
    }


@pytest.mark.parametrize(
    ("frames", "expected_evidence", "expected_length"),
    ((24, 0, None), (25, 1, 8_000)),
)
def test_streaming_l6_final_tail_requires_at_least_half_a_second_and_is_once_only(
    frames: int,
    expected_evidence: int,
    expected_length: int | None,
) -> None:
    embedder = _RecordingSignEmbedder()
    pipeline = OfflineLayer6Pipeline(embedder, _config())
    source = _result(1, 0.2, frames)

    pipeline.process_streaming((source,))
    first_final = pipeline.process_streaming((source,), final=True)
    repeated_final = pipeline.process_streaming((source,), final=True)

    assert first_final.metadata["multistage"]["evidence_count"] == expected_evidence
    assert repeated_final.metadata["multistage"]["evidence_count"] == expected_evidence
    lengths = [length for batch in embedder.batch_lengths for length in batch]
    assert lengths == ([] if expected_length is None else [expected_length])


def test_streaming_l6_resets_automatically_on_epoch_and_session_change() -> None:
    embedder = _RecordingSignEmbedder()
    pipeline = OfflineLayer6Pipeline(embedder, _config())

    epoch_zero = pipeline.process_streaming((_result(1, 0.2, 100),))
    epoch_one = pipeline.process_streaming((
        _result(1, 0.2, 100, stream_epoch=1),
    ))
    next_session = pipeline.process_streaming((
        _result(1, 0.2, 100, session_id="next", stream_epoch=0),
    ))

    assert epoch_zero.metadata["multistage"]["evidence_count"] == 1
    assert epoch_one.metadata["multistage"]["evidence_count"] == 1
    assert next_session.metadata["multistage"]["evidence_count"] == 1
    assert pipeline._streaming_identity == ("next", 0)
    assert tuple(pipeline._streaming_snapshot.labels_by_evidence_id) == (  # type: ignore[union-attr]
        "next:epoch0:track1:stable0:segment0",
    )
    assert embedder.batch_lengths == [(32_000,), (32_000,), (32_000,)]


def test_streaming_l6_reuses_result_when_only_new_silence_arrives() -> None:
    embedder = _RecordingSignEmbedder()
    pipeline = OfflineLayer6Pipeline(embedder, _config())
    voiced = pipeline.process_streaming((_result(1, 0.2, 100),))
    silence_extension = _result(
        1,
        0.2,
        200,
        decisions=(True,) * 100 + (False,) * 100,
    )

    reused = pipeline.process_streaming((silence_extension,))

    assert reused.outputs[0].end_sample_48k == silence_extension.source.end_sample
    assert reused.fragments[0].end_sample_48k == silence_extension.source.end_sample
    assert np.array_equal(
        reused.outputs[0].waveform_16k,
        voiced.outputs[0].waveform_16k,
    )
    assert reused.metadata["incremental_changed_speaker_ids"] == ()
    assert reused.metadata["incremental_append_only_speaker_ids"] == ()
    assert embedder.batch_lengths == [(32_000,)]
    state = next(iter(pipeline._streaming_states.values()))
    assert state.consumed_frames == 200


def test_streaming_l6_reports_exact_append_only_wav_updates() -> None:
    pipeline = OfflineLayer6Pipeline(_SignEmbedder(), _config())

    initial = pipeline.process_streaming((_result(1, 0.2, 100),))
    materializer = pipeline._streaming_materializers[1]
    extended = pipeline.process_streaming((_result(1, 0.2, 200),))

    assert initial.metadata["incremental_changed_speaker_ids"] == ()
    assert initial.metadata["incremental_append_only_speaker_ids"] == (1,)
    assert extended.metadata["incremental_changed_speaker_ids"] == ()
    assert extended.metadata["incremental_append_only_speaker_ids"] == (1,)
    assert np.array_equal(
        extended.outputs[0].waveform_16k[:len(initial.outputs[0].waveform_16k)],
        initial.outputs[0].waveform_16k,
    )
    assert pipeline._streaming_materializers[1].store is materializer.store
    assert pipeline._streaming_materializers[1].spool is materializer.spool
    assert materializer.spool.resident_bytes == 0


def test_streaming_l6_reports_historical_speaker_rewrites(monkeypatch) -> None:
    class CorrectingClusterer:
        def __init__(self, _config: object) -> None:
            self.evidence: list[SegmentEvidence] = []

        def update(self, evidence: tuple[SegmentEvidence, ...]):
            known = {item.evidence_id for item in self.evidence}
            self.evidence.extend(
                item for item in evidence if item.evidence_id not in known
            )
            labels = {
                item.evidence_id: (
                    index if len(self.evidence) <= 2 else 0
                )
                for index, item in enumerate(self.evidence)
            }
            return pipeline_module.MultiStageSnapshot(
                labels,
                len(self.evidence),
                len(set(labels.values())),
                "test",
            )

    monkeypatch.setattr(
        pipeline_module,
        "MultiStageVoiceprintClusterer",
        CorrectingClusterer,
    )
    pipeline = OfflineLayer6Pipeline(_SignEmbedder(), _config())

    initial = pipeline.process_streaming((
        _result(1, 0.2, 100),
        _result(2, -0.2, 100),
    ))
    corrected = pipeline.process_streaming((
        _result(1, 0.2, 200),
        _result(2, -0.2, 200),
    ))

    assert initial.speaker_count == 2
    assert corrected.speaker_count == 1
    assert corrected.metadata["incremental_changed_speaker_ids"] == (1, 2)
    assert corrected.metadata["incremental_append_only_speaker_ids"] == ()


def test_streaming_l6_disk_materializer_matches_offline_silence_merge() -> None:
    decisions = (True,) * 100 + (False,) * 150 + (True,) * 50
    source = _result(1, 0.2, 300, decisions=decisions)
    offline = OfflineLayer6Pipeline(_SignEmbedder(), _config()).process((source,))
    streaming_pipeline = OfflineLayer6Pipeline(_SignEmbedder(), _config())

    streaming = streaming_pipeline.process_streaming((source,), final=True)

    assert isinstance(streaming.outputs[0].waveform_16k, DiskAudioView)
    assert len(streaming.outputs[0].waveform_16k) == 250 * 320
    assert np.array_equal(
        streaming.outputs[0].waveform_16k,
        offline.outputs[0].waveform_16k,
    )


def test_streaming_l6_materializer_uses_bounded_ten_second_chunks(monkeypatch) -> None:
    pipeline = OfflineLayer6Pipeline(_SignEmbedder(), _config())
    frame_counts: list[int] = []
    original = pipeline._streaming_mix_chunk

    def recording_mix(items, start, end):
        frame_counts.append((end - start) // 960)
        return original(items, start, end)

    monkeypatch.setattr(pipeline, "_streaming_mix_chunk", recording_mix)

    pipeline.process_streaming((_result(1, 0.2, 750),), final=True)

    assert frame_counts == [500, 250]


def test_streaming_l6_materializer_reserve_and_reset_lifecycle() -> None:
    pipeline = OfflineLayer6Pipeline(
        _SignEmbedder(),
        _config(),
        spool_min_free_bytes=123,
    )
    result = pipeline.process_streaming((_result(1, 0.2, 100),))
    materializer = pipeline._streaming_materializers[1]

    assert materializer.store.minimum_free_bytes == 123
    pipeline.reset_streaming()
    assert pipeline._streaming_materializers == {}
    assert materializer.store._retired is True
    assert np.asarray(result.outputs[0].waveform_16k).shape == (32_000,)


def test_streaming_l6_materializer_failure_retries_without_duplicate_audio(
    monkeypatch,
) -> None:
    embedder = _RecordingSignEmbedder()
    pipeline = OfflineLayer6Pipeline(embedder, _config())
    pipeline.process_streaming((_result(1, 0.2, 100),))
    original = pipeline._streaming_mix_chunk
    failed = False

    def fail_once(items, start, end):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected materializer write failure")
        return original(items, start, end)

    monkeypatch.setattr(pipeline, "_streaming_mix_chunk", fail_once)
    extension = _result(1, 0.2, 200)
    with pytest.raises(OSError, match="injected"):
        pipeline.process_streaming((extension,))

    recovered = pipeline.process_streaming((extension,))

    assert len(recovered.outputs[0].waveform_16k) == 64_000
    assert recovered.metadata["incremental_append_only_speaker_ids"] == (1,)
    assert embedder.batch_lengths == [(32_000,), (32_000,)]


def test_streaming_l6_async_same_speaker_tracks_advance_without_rebuild(
    monkeypatch,
) -> None:
    class OneSpeakerClusterer:
        def __init__(self, _config: object) -> None:
            self.evidence: dict[str, SegmentEvidence] = {}

        def update(self, evidence: tuple[SegmentEvidence, ...]):
            self.evidence.update((item.evidence_id, item) for item in evidence)
            labels = {evidence_id: 0 for evidence_id in self.evidence}
            return pipeline_module.MultiStageSnapshot(
                labels,
                len(labels),
                1 if labels else 0,
                "test",
            )

    monkeypatch.setattr(
        pipeline_module,
        "MultiStageVoiceprintClusterer",
        OneSpeakerClusterer,
    )
    pipeline = OfflineLayer6Pipeline(_SignEmbedder(), _config())
    initial = pipeline.process_streaming((
        _result(1, 0.2, 150),
        _result(2, 0.4, 100),
    ))
    materializer_store = pipeline._streaming_materializers[1].store

    def forbidden_rebuild(*_args, **_kwargs):
        raise AssertionError("stable asynchronous tracks rebuilt speaker history")

    monkeypatch.setattr(pipeline, "_new_streaming_materializer", forbidden_rebuild)
    for fast_frames, slow_frames in ((250, 150), (350, 200), (450, 250)):
        snapshot = pipeline.process_streaming((
            _result(1, 0.2, fast_frames),
            _result(2, 0.4, slow_frames),
        ))
        assert snapshot.metadata["incremental_changed_speaker_ids"] == ()
        assert pipeline._streaming_materializers[1].store is materializer_store
        assert pipeline._streaming_materializers[1].processed_end_sample_48k == (
            slow_frames * 960
        )

    assert initial.metadata["streaming_materialized_through_sample_48k"] == {
        1: 100 * 960,
    }


def test_streaming_l6_silence_final_advances_timeline_without_rewriting_wav() -> None:
    pipeline = OfflineLayer6Pipeline(_SignEmbedder(), _config())
    voiced = pipeline.process_streaming((_result(1, 0.2, 100),))
    final_source = _result(
        1,
        0.2,
        200,
        decisions=(True,) * 100 + (False,) * 100,
    )

    final = pipeline.process_streaming(
        (final_source,),
        finalized_track_keys=frozenset({("session", 0, 1)}),
    )

    assert final.metadata["recording_end_sample_48k"] == 200 * 960
    assert final.outputs[0].end_sample_48k == 200 * 960
    assert final.fragments[0].end_sample_48k == 200 * 960
    assert final.metadata["incremental_changed_speaker_ids"] == ()
    assert final.metadata["incremental_append_only_speaker_ids"] == ()
    assert np.array_equal(
        final.outputs[0].waveform_16k,
        voiced.outputs[0].waveform_16k,
    )


def test_streaming_l6_priority_exactly_matches_legacy_merge_order() -> None:
    fragments = (
        SimpleNamespace(
            mos_score=0.8,
            match_score=0.9,
            speaker_similarity=0.7,
            branch_index=0,
            fragment_id="z",
        ),
        SimpleNamespace(
            mos_score=0.8,
            match_score=0.9,
            speaker_similarity=0.9,
            branch_index=1,
            fragment_id="y",
        ),
        SimpleNamespace(
            mos_score=0.8,
            match_score=0.9,
            speaker_similarity=0.7,
            branch_index=0,
            fragment_id="a",
        ),
    )

    ordered = sorted(
        ((object(), fragment) for fragment in fragments),
        key=OfflineLayer6Pipeline._streaming_fragment_priority,
    )

    assert [fragment.fragment_id for _, fragment in ordered] == ["y", "a", "z"]
    assert OfflineLayer6Pipeline._streaming_fragment_priority(ordered[0]) == (
        -0.8,
        -0.9,
        -0.9,
        1,
        "y",
    )


def test_streaming_l6_two_speaker_materialization_is_atomic_and_retryable(
    monkeypatch,
) -> None:
    class SeparateTrackClusterer:
        def __init__(self, _config: object) -> None:
            self.evidence: dict[str, SegmentEvidence] = {}

        def update(self, evidence: tuple[SegmentEvidence, ...]):
            self.evidence.update((item.evidence_id, item) for item in evidence)
            labels = {
                evidence_id: (0 if ":track1:" in item.track_key else 1)
                for evidence_id, item in self.evidence.items()
            }
            return pipeline_module.MultiStageSnapshot(
                labels,
                len(labels),
                len(set(labels.values())),
                "test",
            )

    monkeypatch.setattr(
        pipeline_module,
        "MultiStageVoiceprintClusterer",
        SeparateTrackClusterer,
    )
    embedder = _RecordingSignEmbedder()
    pipeline = OfflineLayer6Pipeline(embedder, _config())
    initial = pipeline.process_streaming((
        _result(1, 0.2, 100),
        _result(2, -0.2, 100),
    ))
    previous_materializers = dict(pipeline._streaming_materializers)
    original_mix = pipeline._streaming_mix_chunk
    failed = False

    def fail_second_speaker(items, start, end):
        nonlocal failed
        if not failed and items[0][1].speaker_id == 2:
            failed = True
            raise OSError("injected second-speaker failure")
        return original_mix(items, start, end)

    monkeypatch.setattr(pipeline, "_streaming_mix_chunk", fail_second_speaker)
    extension = (
        _result(1, 0.2, 200),
        _result(2, -0.2, 200),
    )
    with pytest.raises(OSError, match="second-speaker"):
        pipeline.process_streaming(extension)

    assert pipeline._streaming_materializers == previous_materializers
    assert tuple(state.consumed_frames for state in pipeline._streaming_states.values()) == (
        100,
        100,
    )
    assert tuple(len(output.waveform_16k) for output in initial.outputs) == (
        32_000,
        32_000,
    )

    recovered = pipeline.process_streaming(extension)

    assert tuple(len(output.waveform_16k) for output in recovered.outputs) == (
        64_000,
        64_000,
    )
    assert recovered.metadata["incremental_append_only_speaker_ids"] == (1, 2)
    assert embedder.batch_lengths == [
        (32_000, 32_000),
        (32_000, 32_000),
    ]


def test_streaming_l6_start_rebase_rotates_only_branch_evidence_generation() -> None:
    pipeline = OfflineLayer6Pipeline(_SignEmbedder(), _config())
    first = pipeline.process_streaming((
        _result(1, 0.2, 100, start_sample_48k=480_000),
    ))
    rebased = pipeline.process_streaming((
        _result(1, 0.2, 700, start_sample_48k=0),
    ))

    evidence_ids = tuple(pipeline._streaming_snapshot.labels_by_evidence_id)  # type: ignore[union-attr]
    state = pipeline._streaming_states[("session", 0, 1, 0)]
    assert first.metadata["multistage"]["evidence_count"] == 1
    assert rebased.metadata["multistage"]["evidence_count"] == 8
    assert evidence_ids[0] == "session:epoch0:track1:stable0:segment0"
    assert evidence_ids[1:] == tuple(
        f"session:epoch0:track1:stable0:generation1:segment{index}"
        for index in range(7)
    )
    assert state.generation == 1
    assert state.start_sample_48k == 0
    assert rebased.fragments[0].start_sample_48k == 0
    assert rebased.metadata["incremental_changed_speaker_ids"] == (1,)
