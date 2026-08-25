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


def test_runtime_handoff_uses_configured_chunk_and_backfill_ready_gate(tmp_path):
    runtime = _runtime(tmp_path)
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
        "chunk_samples": 10 * 48_000,
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
