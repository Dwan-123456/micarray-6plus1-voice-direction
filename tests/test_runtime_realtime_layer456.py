from __future__ import annotations

import hashlib
from types import SimpleNamespace
from threading import Event
from time import monotonic

import numpy as np
import pytest

import app.runtime as runtime_module
from app.runtime import ApplicationRuntime
from common.config import load_config
from layer4_speech_separation import Layer4LongAudioInput
from layer3_direction_signal import (
    L3_MODE_DS_BASELINE,
    L3_MODE_LOADED_MVDR,
    L3_MODE_OPTIMIZED,
)


CONFIG = "config/config.yaml"


class _Pipeline:
    running = False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def read(self, timeout=None):
        return None

    def take_health_events(self):
        return ()


class _Serial:
    def set_light(self, _enabled):
        return None


def _source(*, seconds=10):
    waveform = np.zeros(seconds * 48_000, dtype=np.float32)
    return Layer4LongAudioInput(
        "session:epoch0:track1:start0",
        hashlib.sha256(waveform.tobytes()).hexdigest(),
        "session",
        0,
        1,
        20.0,
        0,
        48_000,
        waveform,
        tuple((sample, 1) for sample in range(960, len(waveform) + 1, 960)),
    )


def _runtime(tmp_path):
    return ApplicationRuntime(
        load_config(CONFIG, environ={}),
        project_root=tmp_path,
        pipeline=_Pipeline(),
        serial_device=_Serial(),
    )


def test_realtime_chunk_seconds_can_change_before_formal_recording(tmp_path):
    runtime = _runtime(tmp_path)

    assert runtime.realtime_chunk_seconds == 4
    assert runtime.set_realtime_chunk_seconds(3) == 3
    assert runtime.realtime_chunk_seconds == 3
    assert runtime.set_realtime_chunk_seconds(15) == 15
    with pytest.raises(ValueError, match="integer"):
        runtime.set_realtime_chunk_seconds(True)
    with pytest.raises(ValueError, match="between 3 and 15"):
        runtime.set_realtime_chunk_seconds(2)

    streaming = runtime.config.layer4.streaming.__class__.model_validate({
        **runtime.config.layer4.streaming.model_dump(),
        "chunk_seconds": 4,
        "overlap_seconds": 3,
    })
    runtime.config = runtime.config.model_copy(update={
        "layer4": runtime.config.layer4.model_copy(update={"streaming": streaming}),
    })
    with pytest.raises(ValueError, match="greater than.*overlap"):
        runtime.set_realtime_chunk_seconds(3)

    runtime._ephemeral_live_capture = True
    runtime._ephemeral_recording_active = True
    with pytest.raises(RuntimeError, match="正式录音进行中"):
        runtime.set_realtime_chunk_seconds(10)
    runtime.close()


def test_realtime_chunk_seconds_can_change_after_stopped_generation(tmp_path):
    runtime = _runtime(tmp_path)
    service = runtime.realtime_postprocessing
    runtime.realtime_postprocessing = SimpleNamespace(
        active=False,
        status=SimpleNamespace(submitted_blocks=27),
    )
    runtime._realtime_generation_chunk_seconds = 4

    assert runtime.set_realtime_chunk_seconds(7) == 7
    assert runtime.realtime_chunk_seconds == 7
    assert runtime.realtime_applied_chunk_seconds == 4

    runtime.realtime_postprocessing = service
    runtime.close()


def test_realtime_chunk_seconds_rejects_live_generation_even_before_first_chunk(tmp_path):
    runtime = _runtime(tmp_path)
    runtime._thread = SimpleNamespace(is_alive=lambda: True)

    with pytest.raises(RuntimeError, match="仍在运行"):
        runtime.set_realtime_chunk_seconds(7)

    assert runtime.realtime_chunk_seconds == 4
    runtime._thread = None
    runtime.close()


def test_realtime_chunk_generation_binds_l4_and_l6_together(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    captured = []
    backend = object()
    embedder = object()
    scorer = object()
    monkeypatch.setattr(runtime, "_get_layer4_backend", lambda _selected: backend)
    monkeypatch.setattr(runtime, "_get_campplus_cached_embedder", lambda: embedder)
    monkeypatch.setattr(runtime, "_get_dnsmos_scorer", lambda: scorer)

    class Processor:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(runtime_module, "IncrementalLayer456Processor", Processor)
    runtime.set_realtime_chunk_seconds(7)
    assert runtime._bind_realtime_chunk_generation() == 7

    # A stopped run may be inspected while the next run is already configured.
    # Its final diagnostics/factory must keep the completed generation value.
    runtime.set_realtime_chunk_seconds(5)
    runtime._build_realtime_postprocessor()
    assert captured[-1]["chunk_samples_48k"] == 7 * 48_000
    assert captured[-1]["l6_interval_samples_48k"] == 7 * 48_000
    assert captured[-1]["spool_min_free_bytes"] == 5 * 1024**3

    assert runtime._bind_realtime_chunk_generation() == 5
    runtime._build_realtime_postprocessor()
    assert captured[-1]["chunk_samples_48k"] == 5 * 48_000
    assert captured[-1]["l6_interval_samples_48k"] == 5 * 48_000
    runtime.close()


def test_runtime_handoff_uses_configured_chunk_and_backfill_ready_gate(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.set_realtime_chunk_seconds(7)
    calls = []
    resolutions = []
    submitted = []
    source = _source()

    class Hub:
        def claim_streaming_chunks(self, **kwargs):
            calls.append(kwargs)
            return (source,)

        def resolve_streaming_chunk(self, value, *, accepted):
            resolutions.append((value, accepted))

    class Service:
        enabled = True
        available_slots = 4

        def submit(self, value, *, is_final_chunk=False):
            submitted.append((value, is_final_chunk))
            return True

    runtime.track_audio_stream = Hub()
    runtime.realtime_postprocessing = Service()
    runtime._confirmed_backfill_ready_ids.add(("session", 0, 1))

    assert runtime._offer_realtime_postprocessing_chunks() == 1
    assert calls == [{
        "chunk_samples": 7 * 48_000,
        "ready_track_keys": {("session", 0, 1)},
        "flush": False,
        "max_chunks": 4,
    }]
    assert submitted == [(source, False)]
    assert resolutions == [(source, True)]


def test_runtime_marks_only_short_flush_block_as_final_input(tmp_path):
    runtime = _runtime(tmp_path)
    tail = _source(seconds=3)
    submitted = []
    resolutions = []
    runtime.track_audio_stream = SimpleNamespace(
        claim_streaming_chunks=lambda **_kwargs: (tail,),
        resolve_streaming_chunk=lambda value, *, accepted: resolutions.append(
            (value, accepted)
        ),
    )
    runtime.realtime_postprocessing = SimpleNamespace(
        enabled=True,
        available_slots=1,
        submit=lambda value, *, is_final_chunk=False: (
            submitted.append((value, is_final_chunk)) or True
        ),
    )

    assert runtime._offer_realtime_postprocessing_chunks(flush=True) == 1
    assert submitted == [(tail, True)]
    assert resolutions == [(tail, True)]


def test_runtime_submits_tail_then_independent_track_finalization(tmp_path):
    runtime = _runtime(tmp_path)
    tail = _source(seconds=3)
    identity = ("session", 0, 1)
    submitted = []
    resolved_chunks = []
    resolved_finalizations = []

    class Hub:
        def claim_streaming_chunks(self, **_kwargs):
            return (tail,)

        def resolve_streaming_chunk(self, value, *, accepted):
            resolved_chunks.append((value, accepted))

        def claim_streaming_finalizations(self, **kwargs):
            assert kwargs == {
                "ready_track_keys": {identity},
                "flush": True,
                "max_tracks": 1,
            }
            return (identity,)

        def resolve_streaming_finalization(self, value, *, accepted):
            resolved_finalizations.append((value, accepted))

    class Service:
        enabled = True
        available_slots = 2

        def submit(self, value, *, is_final_chunk=False):
            submitted.append(("audio", value, is_final_chunk))
            return True

        def submit_track_final(self, value):
            submitted.append(("final", value))
            return True

    runtime.track_audio_stream = Hub()
    runtime.realtime_postprocessing = Service()
    runtime._confirmed_backfill_ready_ids.add(identity)

    assert runtime._offer_realtime_postprocessing_chunks(
        flush=True,
        max_chunks=2,
    ) == 2
    assert submitted == [
        ("audio", tail, False),
        ("final", identity),
    ]
    assert resolved_chunks == [(tail, True)]
    assert resolved_finalizations == [(identity, True)]


def test_runtime_releases_rejected_claim_without_offering_later_chunks(tmp_path):
    runtime = _runtime(tmp_path)
    first = _source()
    second = Layer4LongAudioInput(
        asset_id="session:epoch0:track1:start480000",
        sha256=first.sha256,
        session_id="session",
        stream_epoch=0,
        track_id=1,
        theta_deg=first.theta_deg,
        start_sample=480_000,
        sample_rate=48_000,
        waveform=first.waveform,
        l2_direction_counts=tuple(
            (sample + 480_000, count) for sample, count in first.l2_direction_counts
        ),
    )
    resolved = []
    submitted = []
    runtime.track_audio_stream = SimpleNamespace(
        claim_streaming_chunks=lambda **_kwargs: (first, second),
        resolve_streaming_chunk=lambda value, *, accepted: resolved.append(
            (value.start_sample, accepted)
        ),
    )
    runtime.realtime_postprocessing = SimpleNamespace(
        enabled=True,
        available_slots=2,
        submit=lambda value, *, is_final_chunk=False: (
            submitted.append(value.start_sample) or False
        ),
    )

    assert runtime._offer_realtime_postprocessing_chunks() == 0
    assert submitted == [0]
    assert resolved == [(0, False), (480_000, False)]


def test_runtime_flush_intersects_backfill_ready_ids_with_retained_tracks(tmp_path):
    runtime = _runtime(tmp_path)
    calls = []
    runtime._confirmed_backfill_ready_ids.update({
        ("session", 0, 1),
        ("session", 0, 2),
    })
    runtime.track_audio_stream = SimpleNamespace(
        claim_streaming_chunks=lambda **kwargs: calls.append(kwargs) or (),
    )
    runtime.realtime_postprocessing = SimpleNamespace(
        enabled=True,
        available_slots=1,
    )

    assert runtime._offer_realtime_postprocessing_chunks(
        flush=True,
        allowed_track_keys={("session", 0, 2)},
    ) == 0
    assert calls[0]["ready_track_keys"] == {("session", 0, 2)}


def test_running_mode_switch_is_locked_only_after_first_streaming_admission(tmp_path):
    runtime = _runtime(tmp_path)
    runtime._thread = SimpleNamespace(is_alive=lambda: True)
    runtime.realtime_postprocessing = SimpleNamespace(
        status=SimpleNamespace(submitted_blocks=0),
    )

    assert runtime.set_l3_processing_mode(
        L3_MODE_LOADED_MVDR
    ) == L3_MODE_LOADED_MVDR
    runtime.realtime_postprocessing.status.submitted_blocks = 1
    with pytest.raises(RuntimeError, match="accepted its first chunk"):
        runtime.set_l3_processing_mode(L3_MODE_OPTIMIZED)
    assert runtime.l3_processing_mode == L3_MODE_LOADED_MVDR
    assert runtime.set_l3_processing_mode(
        L3_MODE_LOADED_MVDR
    ) == L3_MODE_LOADED_MVDR

    runtime._thread = None
    assert runtime.set_l3_processing_mode(L3_MODE_DS_BASELINE) == L3_MODE_DS_BASELINE


def test_full_backfill_queue_never_blocks_l2_and_releases_the_id(tmp_path):
    runtime = _runtime(tmp_path)
    key = ("session", 0, 7)
    runtime._confirmed_backfill_history.append(SimpleNamespace(
        session_id="session",
        stream_epoch=0,
        decision_sample=1_920,
    ))
    while not runtime._confirmed_backfill_work.full():
        runtime._confirmed_backfill_work.put_nowait(object())
    item = SimpleNamespace(key=SimpleNamespace(
        session_id="session",
        stream_epoch=0,
        decision_sample=3_840,
    ))
    track = SimpleNamespace(
        track_state="confirmed",
        track_id=7,
        first_seen_sample=2_880,
    )
    runtime._realtime_chunk_signal.clear()

    began = monotonic()
    runtime._schedule_confirmed_backfill(
        item,
        (track,),
        processing_mode=L3_MODE_OPTIMIZED,
        l2_direction_count=1,
    )

    assert monotonic() - began < 0.1
    assert key in runtime._confirmed_backfill_ready_ids
    assert runtime._realtime_chunk_signal.is_set()
    assert "queue full" in str(runtime.dev_audio_tracking_error)


@pytest.mark.parametrize(
    ("stage", "failure"),
    (
        ("history", "history lookup failed"),
        ("insert", "Hub insert failed"),
        ("prepend", "UI prepend failed"),
    ),
)
def test_confirmed_backfill_failure_degrades_and_releases_realtime_gate(
    tmp_path, stage, failure,
):
    runtime = _runtime(tmp_path)
    key = ("session", 0, 7)
    track = SimpleNamespace(session_id=key[0], stream_epoch=key[1], track_id=key[2])
    work = runtime_module._ConfirmedBackfillWork(
        track=track,
        windows=(),
        processing_mode=L3_MODE_OPTIMIZED,
        l2_direction_count=1,
    )

    class Hub:
        @staticmethod
        def missing_backfill_decisions(**_kwargs):
            if stage == "history":
                raise RuntimeError(failure)
            return ()

        @staticmethod
        def missing_backfill_windows(audio_windows):
            assert audio_windows == ()
            return (SimpleNamespace(processing_mode=L3_MODE_OPTIMIZED),)

        @staticmethod
        def insert_backfill(_windows, *, l2_direction_count):
            assert l2_direction_count == 1
            if stage == "insert":
                raise RuntimeError(failure)
            return (object(),)

    class Tracker:
        @staticmethod
        def prepend_backfill_hops(*_args, **_kwargs):
            raise RuntimeError(failure)

    runtime.track_audio_stream = Hub()
    runtime.dev_audio_tracker = Tracker() if stage == "prepend" else None
    runtime.realtime_postprocessing = SimpleNamespace(enabled=True)
    runtime._realtime_chunk_signal.clear()
    runtime._confirmed_backfill_work.put_nowait(work)
    runtime._confirmed_backfill_work.put_nowait(runtime_module._PIPELINE_EOS)

    runtime._run_confirmed_backfill()

    assert key in runtime._confirmed_backfill_ready_ids
    assert runtime._realtime_chunk_signal.is_set()
    assert "backfill degraded" in str(runtime.dev_audio_tracking_error)
    assert failure in str(runtime.dev_audio_tracking_error)
    assert "zero-filled history gaps" in str(runtime.dev_audio_tracking_error)


def test_runtime_active_includes_chunk_and_model_sidecars(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.realtime_postprocessing = SimpleNamespace(active=True)

    assert runtime.active

    runtime.realtime_postprocessing.active = False
    runtime._realtime_chunk_thread = SimpleNamespace(is_alive=lambda: True)

    assert runtime.active

    runtime._realtime_chunk_thread = SimpleNamespace(is_alive=lambda: False)
    assert not runtime.active


def test_runtime_processing_status_exposes_sidecar_without_changing_window_queue_contract(tmp_path):
    runtime = _runtime(tmp_path)
    status = runtime.processing_status

    assert set(status["queue_depths"]) == {
        "l2", "l3", "l3_prepared", "l3_host", "l5", "completion",
    }
    assert status["layer456_stream"]["state"] in {"idle", "disabled"}
    assert status["layer456_stream"]["queued_blocks"] == 0
    assert status["stage_alive"]["layer456_stream"] is False
    assert status["realtime_chunk_seconds"] == {
        "configured_default": 4,
        "next_generation": 4,
        "active_generation": None,
    }
    resources = status["layer456_resources"]
    assert resources["resident_memory_budget_bytes"] == 128 * 1024 * 1024
    assert resources["embedding_cache_max_segments"] == 600
    assert resources["embedding_cache_segments"] == 0
    assert resources["spool_min_free_bytes"] == 5 * 1024**3
    assert resources["spool_free_bytes"] is None or resources["spool_free_bytes"] >= 0
    assert resources["spool_low_space"] in {True, False, None}
    assert runtime.track_audio_stream.spool_min_free_bytes == 5 * 1024**3


def test_thirty_minute_equivalent_voiceprint_cache_remains_bounded(tmp_path):
    runtime = _runtime(tmp_path)

    class Embedder:
        @staticmethod
        def embed(waveform):
            return np.asarray((float(waveform[0]),), dtype=np.float32)

    runtime._campplus_embedder = Embedder()
    cached = runtime._get_campplus_cached_embedder()
    for segment_index in range(30 * 60 // 2):
        cached.embed(np.asarray((segment_index,), dtype=np.float32))

    assert cached.cached_segments == runtime.config.layer6.embedding_cache_max_segments
    assert cached.cached_segments == 600
    runtime.close()


def test_close_releases_layer456_snapshots_queues_models_and_countnet(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    events = []

    class Owned:
        def __init__(self, name):
            self.name = name

        def close(self):
            events.append(f"close:{self.name}")

    class Cached(Owned):
        cached_segments = 3

        def clear(self):
            events.append("clear:cache")

    backend = Owned("l4")
    cache = Cached("cache")
    campplus = Owned("campplus")
    scorer = Owned("dnsmos")
    runtime._layer4_backends["test"] = backend
    runtime._campplus_cached_embedder = cache
    runtime._campplus_embedder = campplus
    runtime._dnsmos_scorer = scorer
    runtime.realtime_postprocessing._final_snapshot = object()
    runtime.realtime_postprocessing._reuse_snapshot = object()
    runtime.latest_realtime_postprocessing.put_nowait(object())
    runtime.latest_l1.put_nowait(object())
    runtime.latest_dev_ui.put_nowait(object())
    runtime._completion_backlog.append(object())
    monkeypatch.setattr(runtime.speaker_counter, "close", lambda timeout: events.append("countnet"))
    monkeypatch.setattr(runtime_module.torch.cuda, "empty_cache", lambda: events.append("cuda"))

    runtime.close()

    assert runtime.realtime_postprocessing._final_snapshot is None
    assert runtime.realtime_postprocessing._reuse_snapshot is None
    assert runtime.latest_realtime_postprocessing.empty()
    assert runtime.latest_l1.empty()
    assert runtime.latest_dev_ui.empty()
    assert not runtime._completion_backlog
    assert runtime._layer4_backends == {}
    assert runtime._campplus_cached_embedder is None
    assert runtime._campplus_embedder is None
    assert runtime._dnsmos_scorer is None
    assert runtime.realtime_applied_chunk_seconds is None
    assert events.count("clear:cache") == 1
    assert set(events) >= {
        "countnet", "close:l4", "close:cache", "close:campplus", "close:dnsmos", "cuda",
    }


def test_close_joins_live_chunk_model_and_countnet_workers(tmp_path):
    runtime = _runtime(tmp_path)
    finalized = Event()

    class Processor:
        @staticmethod
        def finalize():
            finalized.set()
            return None

        @staticmethod
        def abort():
            return None

    runtime.realtime_postprocessing._factory = Processor
    runtime._bind_realtime_chunk_generation()
    runtime.realtime_postprocessing.start()
    runtime._start_realtime_chunk_producer()
    runtime.speaker_counter.set_enabled(True)
    model_worker = runtime.realtime_postprocessing._thread
    chunk_worker = runtime._realtime_chunk_thread
    countnet_worker = runtime.speaker_counter._thread
    assert model_worker is not None and model_worker.is_alive()
    assert chunk_worker is not None and chunk_worker.is_alive()
    assert countnet_worker is not None and countnet_worker.is_alive()

    runtime.close()

    assert finalized.is_set()
    assert not model_worker.is_alive()
    assert not chunk_worker.is_alive()
    assert not countnet_worker.is_alive()
    assert runtime._realtime_chunk_thread is None
    assert not runtime.realtime_postprocessing.active


def test_l3_handoff_only_signals_the_async_chunk_owner(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.realtime_postprocessing = SimpleNamespace(enabled=True)
    runtime._realtime_chunk_signal.clear()

    runtime._signal_realtime_postprocessing_chunks()

    assert runtime._realtime_chunk_signal.is_set()


def test_async_chunk_owner_uses_one_queue_bounded_claim_round(tmp_path):
    runtime = _runtime(tmp_path)
    source = _source()
    resolved = Event()
    claims = []

    class Hub:
        def claim_streaming_chunks(self, **kwargs):
            claims.append(kwargs)
            return (source,)

        def resolve_streaming_chunk(self, _value, *, accepted):
            assert accepted
            resolved.set()

    runtime.track_audio_stream = Hub()
    runtime.realtime_postprocessing = SimpleNamespace(
        enabled=True,
        available_slots=2,
        status=SimpleNamespace(error=None),
        active=True,
        submit=lambda _value, *, is_final_chunk=False: True,
    )
    runtime._start_realtime_chunk_producer()
    runtime._signal_realtime_postprocessing_chunks()
    assert resolved.wait(1.0)
    assert runtime._stop_realtime_chunk_producer(timeout=1.0)
    assert len(claims) == 1
    assert claims[0]["max_chunks"] == 2


def test_async_chunk_owner_retries_when_capacity_opens_without_another_l3_signal(
    tmp_path,
):
    runtime = _runtime(tmp_path)
    source = _source()
    capacity_was_checked = Event()
    claimed = Event()
    resolutions = []

    class Hub:
        def claim_streaming_chunks(self, **_kwargs):
            if claimed.is_set():
                return ()
            claimed.set()
            return (source,)

        def resolve_streaming_chunk(self, value, *, accepted):
            resolutions.append((value.start_sample, accepted))

    class Service:
        enabled = True
        active = True
        status = SimpleNamespace(error=None)

        def __init__(self):
            self.slots = 0

        @property
        def available_slots(self):
            capacity_was_checked.set()
            return self.slots

        @staticmethod
        def submit(_value, *, is_final_chunk=False):
            return True

    service = Service()
    runtime.track_audio_stream = Hub()
    runtime.realtime_postprocessing = service
    runtime._start_realtime_chunk_producer()
    runtime._signal_realtime_postprocessing_chunks()
    assert capacity_was_checked.wait(1.0)

    # No second L3 signal is sent: opening capacity alone must release the
    # complete chunk that was already waiting in the Hub.
    service.slots = 1
    assert claimed.wait(1.0)
    assert runtime._stop_realtime_chunk_producer(timeout=1.0)
    assert resolutions == [(source.start_sample, True)]


def test_stop_uses_finite_sidecar_timeout_before_sealing(tmp_path):
    runtime = _runtime(tmp_path)
    timeouts = []
    sealed = []
    runtime._flush_realtime_chunk_producer = lambda *, timeout, allowed_track_keys=None: (
        timeouts.append(("flush", timeout)) or True
    )
    runtime._stop_realtime_chunk_producer = lambda *, timeout: (
        timeouts.append(("producer", timeout)) or True
    )
    runtime.track_audio_stream = SimpleNamespace(
        seal=lambda *, allowed_track_keys=None: sealed.append(allowed_track_keys),
        sealed_tracks=(),
    )

    class Service:
        enabled = True
        active = False
        status = SimpleNamespace(error=None)

        def finish(self, *, timeout):
            timeouts.append(("model", timeout))
            return True

    runtime.realtime_postprocessing = Service()

    runtime.stop(drain_timeout_seconds=None)

    configured = float(runtime.config.runtime.graceful_shutdown_timeout_seconds)
    assert timeouts == [
        ("flush", configured),
        ("producer", configured),
        ("model", configured),
    ]
    assert sealed == [None]
    assert runtime.offline_l4_sealed
    assert runtime.offline_l4_sources == ()


def test_stop_withholds_canonical_seal_while_sidecar_is_still_active(tmp_path):
    runtime = _runtime(tmp_path)
    sealed = []
    flush_succeeds = False
    runtime._flush_realtime_chunk_producer = lambda **_kwargs: flush_succeeds
    runtime._stop_realtime_chunk_producer = lambda *, timeout: True
    runtime.track_audio_stream = SimpleNamespace(
        seal=lambda *, allowed_track_keys=None: sealed.append(allowed_track_keys),
        sealed_tracks=(),
    )
    class Service:
        enabled = True
        active = True
        status = SimpleNamespace(error=None)

        @staticmethod
        def abort(*, timeout):
            return False

        @staticmethod
        def finish(*, timeout):
            return True

    runtime.realtime_postprocessing = Service()

    runtime.stop(drain_timeout_seconds=0.1)

    assert sealed == []
    assert "canonical offline sealing was withheld" in str(runtime.last_error)
    with pytest.raises(RuntimeError, match="complete processing drain"):
        _ = runtime.offline_l4_sources

    runtime.realtime_postprocessing.active = False
    with pytest.raises(RuntimeError, match="not sealed"):
        _ = runtime.offline_l4_sources

    flush_succeeds = True
    runtime.stop(drain_timeout_seconds=0.1)

    assert sealed == [None]
    assert runtime.offline_l4_sealed
    assert runtime.offline_l4_sources == ()
    assert runtime.last_error is None
