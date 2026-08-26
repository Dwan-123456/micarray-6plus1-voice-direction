from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from app.final_layer456 import plan_final_reuse, track_load_diagnostics
from app.realtime_postprocessing import RealtimePostprocessingSnapshot
from app.runtime import ApplicationRuntime
from layer4_speech_separation import (
    Layer4LongAudioInput,
    Layer4OfflineResult,
    Layer4ProcessedAudio,
    SpeakerCountDecision,
)
from layer6_speaker_consolidation import Layer6Result


def _source(*, track_id: int = 1, start: int = 0, seconds: int = 4):
    waveform = np.full(seconds * 48_000, 0.01 * track_id, dtype=np.float32)
    return Layer4LongAudioInput(
        asset_id=f"sealed:track{track_id}",
        sha256=hashlib.sha256(waveform.tobytes()).hexdigest(),
        session_id="session",
        stream_epoch=0,
        track_id=track_id,
        theta_deg=30.0,
        start_sample=start,
        sample_rate=48_000,
        waveform=waveform,
        l2_direction_counts=tuple(
            (sample, 1)
            for sample in range(start + 960, start + len(waveform) + 1, 960)
        ),
    )


def _realtime_values(
    source: Layer4LongAudioInput,
    *,
    degraded: bool = False,
    track_final: bool = True,
):
    realtime_source = Layer4LongAudioInput(
        asset_id=f"{source.asset_id}:realtime",
        sha256=source.sha256,
        session_id=source.session_id,
        stream_epoch=source.stream_epoch,
        track_id=source.track_id,
        theta_deg=source.theta_deg,
        start_sample=source.start_sample,
        sample_rate=48_000,
        waveform=np.ascontiguousarray(source.waveform, dtype=np.float32),
        l2_direction_counts=source.l2_direction_counts,
    )
    waveform_16k = np.ascontiguousarray(source.waveform[::3], dtype=np.float32)
    decision = SpeakerCountDecision(
        realtime_source.asset_id, 1, 1.0, "l2-stream", {"streaming": True},
    )
    output_hash = hashlib.sha256(waveform_16k.tobytes()).hexdigest()
    metadata = {
        "backend": "mossformer2_ss_16k",
        "realtime_preview_degraded": degraded,
        "dnsmos_complete_branch": False,
        "dnsmos_finalized_without_full_rerun": True,
        "realtime_l4_valid_through_sample_48k": source.end_sample,
        "realtime_l5_valid_through_sample_48k": source.end_sample,
        "l5_threshold": 0.7,
        "mos_score": 0.5,
        "realtime_track_final": track_final,
    }
    processed = Layer4ProcessedAudio(
        "realtime", realtime_source, decision, "single_speaker_bypass", None,
        "realtime-output", output_hash, waveform_16k, metadata,
    )
    frames = len(waveform_16k) // 320
    result = Layer4OfflineResult(
        "realtime", realtime_source, decision, "single_speaker_bypass", None,
        0.9, True, "l5", (0.9,) * frames, (True,) * frames,
        "realtime-output", output_hash,
        {**metadata, "output_waveform_16k": waveform_16k},
    )
    return processed, result


def _snapshot(*values, is_final: bool = True):
    l4 = tuple(value[0] for value in values)
    l5 = tuple(value[1] for value in values)
    end_sample = max(item.source.end_sample for item in l4)
    l6 = Layer6Result(
        "session",
        0,
        (),
        (),
        {
            "canonical": False,
            "realtime_provisional": True,
            "realtime_tail_flushed": is_final,
            "streaming_final": is_final,
            "realtime_l6_valid_through_sample_48k": end_sample,
        },
    )
    return RealtimePostprocessingSnapshot(
        "session", 3, is_final,
        end_sample,
        len(values), l4, l5, l6,
    )


def test_exact_final_snapshot_promotes_without_recomputing_audio_models():
    source = _source()
    snapshot = _snapshot(_realtime_values(source))

    plan = plan_final_reuse(
        snapshot, (source,), backend_id="mossformer2_ss_16k",
    )

    assert plan.exact_fast_path
    assert plan.missing_sources == ()
    assert plan.reused_track_keys == (("session", 0, 1),)
    assert plan.reused_l4[0].source is source
    assert plan.reused_l4[0].metadata["canonical"] is True
    assert plan.reused_l4[0].metadata["realtime_provisional"] is False
    assert plan.reused_l5[0].source is source
    assert plan.reused_l5[0].metadata["output_waveform_16k"] is plan.reused_l4[0].waveform_16k


def test_duplicate_realtime_branch_is_not_promoted_as_an_exact_track() -> None:
    source = _source()
    snapshot = _snapshot(_realtime_values(source))
    snapshot = replace(
        snapshot,
        l4_processed=snapshot.l4_processed + snapshot.l4_processed,
    )

    plan = plan_final_reuse(
        snapshot, (source,), backend_id="mossformer2_ss_16k",
    )

    assert not plan.exact_fast_path
    assert plan.missing_sources == (source,)
    assert plan.rejected == ((('session', 0, 1), "l4_branch_set_incomplete"),)


def test_degraded_track_is_the_only_track_selected_for_fallback():
    good = _source(track_id=1)
    degraded = _source(track_id=2)
    snapshot = _snapshot(
        _realtime_values(good),
        _realtime_values(degraded, degraded=True),
    )

    plan = plan_final_reuse(
        snapshot, (good, degraded), backend_id="mossformer2_ss_16k",
    )

    assert plan.reused_track_keys == (("session", 0, 1),)
    assert plan.missing_sources == (degraded,)
    assert plan.rejected == ((('session', 0, 2), "realtime_preview_degraded"),)


def test_nonfinal_checkpoint_reuses_only_individually_finalized_tracks():
    completed = _source(track_id=1)
    unfinished = _source(track_id=2)
    checkpoint = _snapshot(
        _realtime_values(completed),
        _realtime_values(unfinished, track_final=False),
        is_final=False,
    )

    plan = plan_final_reuse(
        checkpoint,
        (completed, unfinished),
        backend_id="mossformer2_ss_16k",
    )

    assert plan.reused_track_keys == (("session", 0, 1),)
    assert plan.missing_sources == (unfinished,)
    assert plan.rejected == ((('session', 0, 2), "track_not_final"),)


def test_runtime_exact_fast_path_reuses_tail_flushed_l6_without_offline_models():
    source = _source()
    snapshot = _snapshot(_realtime_values(source))
    calls: list[str] = []

    class Pipeline:
        def process_l4_sealed(self, *_args, **_kwargs):
            calls.append("l4")
            raise AssertionError("exact reuse must not run offline L4")

        def process_l5_sealed(self, *_args, **_kwargs):
            calls.append("l5")
            raise AssertionError("exact reuse must not run offline L5")

    class L6:
        def process(self, results):
            calls.append("l6")
            raise AssertionError("exact reuse must not run offline L6")

    runtime = SimpleNamespace(
        offline_l4_sources=(source,),
        config=SimpleNamespace(
            layer4=SimpleNamespace(default_backend="mossformer2_ss_16k"),
            layer6=SimpleNamespace(enabled=True),
        ),
        realtime_postprocessing=SimpleNamespace(final_snapshot=snapshot),
        build_offline_l4_pipeline=lambda _selected: Pipeline(),
        build_offline_l6_pipeline=lambda: L6(),
        _realtime_chunk_seconds=4,
        _campplus_cached_embedder=None,
    )

    outcome = ApplicationRuntime.reconcile_final_layer456(runtime)

    assert calls == []
    assert outcome.exact_fast_path
    assert outcome.recomputed_track_keys == ()
    assert outcome.diagnostics["additional_mf2_track_count"] == 0
    assert outcome.diagnostics["canonical_l6_reused"] is True
    assert outcome.l6_result.metadata["canonical"] is True
    assert outcome.l6_result.metadata["realtime_provisional"] is False
    assert outcome.l6_result.metadata["canonical_l6_reused"] is True


@pytest.mark.parametrize(
    "invalid_l6",
    ("wrong_session", "not_streaming_final", "short_watermark", "long_watermark"),
)
def test_runtime_recomputes_l6_when_streaming_final_identity_is_not_exact(
    invalid_l6,
):
    source = _source()
    snapshot = _snapshot(_realtime_values(source))
    l6 = snapshot.l6_result
    if invalid_l6 == "wrong_session":
        l6 = replace(l6, session_id="another-session")
    else:
        metadata = dict(l6.metadata)
        if invalid_l6 == "not_streaming_final":
            metadata["streaming_final"] = False
        elif invalid_l6 == "short_watermark":
            metadata["realtime_l6_valid_through_sample_48k"] -= 960
        else:
            metadata["realtime_l6_valid_through_sample_48k"] += 960
        l6 = replace(l6, metadata=metadata)
    snapshot = replace(snapshot, l6_result=l6)
    calls: list[str] = []

    class Pipeline:
        def process_l4_sealed(self, *_args, **_kwargs):
            raise AssertionError("exact L4 reuse must remain active")

        def process_l5_sealed(self, *_args, **_kwargs):
            raise AssertionError("exact L5 reuse must remain active")

    class L6:
        def process(self, results):
            calls.append("l6")
            assert len(results) == 1
            return "l6-final"

    runtime = SimpleNamespace(
        offline_l4_sources=(source,),
        config=SimpleNamespace(
            layer4=SimpleNamespace(default_backend="mossformer2_ss_16k"),
            layer6=SimpleNamespace(enabled=True),
        ),
        realtime_postprocessing=SimpleNamespace(final_snapshot=snapshot),
        build_offline_l4_pipeline=lambda _selected: Pipeline(),
        build_offline_l6_pipeline=lambda: L6(),
        _realtime_chunk_seconds=4,
        _campplus_cached_embedder=None,
    )

    outcome = ApplicationRuntime.reconcile_final_layer456(runtime)

    assert calls == ["l6"]
    assert outcome.l6_result == "l6-final"
    assert outcome.diagnostics["canonical_l6_reused"] is False


def test_runtime_recomputes_l6_when_realtime_snapshot_contains_an_extra_track():
    sealed = _source(track_id=1, seconds=4)
    filtered_out = _source(track_id=2, seconds=3)
    snapshot = _snapshot(
        _realtime_values(sealed),
        _realtime_values(filtered_out),
    )
    calls: list[str] = []

    class Pipeline:
        def process_l4_sealed(self, *_args, **_kwargs):
            raise AssertionError("sealed track still has exact realtime L4")

        def process_l5_sealed(self, *_args, **_kwargs):
            raise AssertionError("sealed track still has exact realtime L5")

    class L6:
        def process(self, results):
            calls.append("l6")
            assert tuple(item.source.track_id for item in results) == (1,)
            return "l6-filtered-final"

    runtime = SimpleNamespace(
        offline_l4_sources=(sealed,),
        config=SimpleNamespace(
            layer4=SimpleNamespace(default_backend="mossformer2_ss_16k"),
            layer6=SimpleNamespace(enabled=True),
        ),
        realtime_postprocessing=SimpleNamespace(final_snapshot=snapshot),
        build_offline_l4_pipeline=lambda _selected: Pipeline(),
        build_offline_l6_pipeline=lambda: L6(),
        _realtime_chunk_seconds=4,
        _campplus_cached_embedder=None,
    )

    outcome = ApplicationRuntime.reconcile_final_layer456(runtime)

    assert outcome.exact_fast_path
    assert calls == ["l6"]
    assert outcome.l6_result == "l6-filtered-final"
    assert outcome.diagnostics["canonical_l6_reused"] is False


def test_runtime_uses_completed_track_checkpoint_after_global_abort():
    source = _source()
    checkpoint = _snapshot(
        _realtime_values(source, track_final=True),
        is_final=False,
    )
    calls: list[str] = []

    class Pipeline:
        def process_l4_sealed(self, *_args, **_kwargs):
            calls.append("l4")
            raise AssertionError("completed checkpoint must avoid offline L4")

        def process_l5_sealed(self, *_args, **_kwargs):
            calls.append("l5")
            raise AssertionError("completed checkpoint must avoid offline L5")

    class L6:
        def process(self, results):
            calls.append("l6")
            assert len(results) == 1
            return "l6-final"

    runtime = SimpleNamespace(
        offline_l4_sources=(source,),
        config=SimpleNamespace(
            layer4=SimpleNamespace(default_backend="mossformer2_ss_16k"),
            layer6=SimpleNamespace(enabled=True),
        ),
        realtime_postprocessing=SimpleNamespace(
            reuse_snapshot=checkpoint,
            final_snapshot=None,
        ),
        build_offline_l4_pipeline=lambda _selected: Pipeline(),
        build_offline_l6_pipeline=lambda: L6(),
        _realtime_chunk_seconds=4,
        _campplus_cached_embedder=None,
    )

    outcome = ApplicationRuntime.reconcile_final_layer456(runtime)

    assert calls == ["l6"]
    assert outcome.exact_fast_path
    assert outcome.recomputed_track_keys == ()
    assert outcome.diagnostics["reuse_snapshot_is_global_final"] is False


def test_runtime_recomputes_only_the_rejected_track():
    good = _source(track_id=1)
    degraded = _source(track_id=2)
    snapshot = _snapshot(
        _realtime_values(good),
        _realtime_values(degraded, degraded=True),
    )
    fallback_l4, fallback_l5 = _realtime_values(degraded)
    calls: list[object] = []

    class Pipeline:
        def process_l4_sealed(self, sources, *, merge_candidates):
            calls.append(("l4", sources, merge_candidates))
            return (fallback_l4,)

        def process_l5_sealed(self, processed):
            calls.append(("l5", processed))
            return (fallback_l5,)

    class L6:
        def process(self, results):
            calls.append(("l6", tuple(item.source.track_id for item in results)))
            return "l6-final"

    runtime = SimpleNamespace(
        offline_l4_sources=(good, degraded),
        config=SimpleNamespace(
            layer4=SimpleNamespace(default_backend="mossformer2_ss_16k"),
            layer6=SimpleNamespace(enabled=True),
        ),
        realtime_postprocessing=SimpleNamespace(final_snapshot=snapshot),
        build_offline_l4_pipeline=lambda _selected: Pipeline(),
        build_offline_l6_pipeline=lambda: L6(),
        _realtime_chunk_seconds=4,
        _campplus_cached_embedder=None,
    )

    outcome = ApplicationRuntime.reconcile_final_layer456(runtime)

    assert calls[0] == ("l4", (degraded,), False)
    assert calls[1] == ("l5", (fallback_l4,))
    assert calls[2] == ("l6", (1, 2))
    assert outcome.reused_track_keys == (("session", 0, 1),)
    assert outcome.recomputed_track_keys == (("session", 0, 2),)
    assert not outcome.exact_fast_path


def test_track_load_diagnostics_exposes_fragment_overlap_pressure():
    first = _source(track_id=1, seconds=4)
    second = _source(track_id=2, seconds=4)
    values = track_load_diagnostics((first, second), chunk_samples_48k=4 * 48_000)

    assert values["sealed_track_count"] == 2
    assert values["source_audio_seconds"] == 8.0
    assert values["timeline_span_seconds"] == 4.0
    assert values["overlap_load_ratio"] == 2.0
