from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np

from app.final_layer456 import plan_final_reuse, track_load_diagnostics
from app.realtime_postprocessing import RealtimePostprocessingSnapshot
from app.runtime import ApplicationRuntime
from layer4_speech_separation import (
    Layer4LongAudioInput,
    Layer4OfflineResult,
    Layer4ProcessedAudio,
    SpeakerCountDecision,
)


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


def _realtime_values(source: Layer4LongAudioInput, *, degraded: bool = False):
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


def _snapshot(*values):
    l4 = tuple(value[0] for value in values)
    l5 = tuple(value[1] for value in values)
    return RealtimePostprocessingSnapshot(
        "session", 3, True,
        max(item.source.end_sample for item in l4),
        len(values), l4, l5, object(),
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


def test_runtime_exact_fast_path_never_calls_offline_l4_or_l5():
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
    assert outcome.exact_fast_path
    assert outcome.recomputed_track_keys == ()
    assert outcome.diagnostics["additional_mf2_track_count"] == 0


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
