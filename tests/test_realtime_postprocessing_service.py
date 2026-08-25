from __future__ import annotations

import hashlib
from dataclasses import replace
from threading import Event
from time import monotonic

import numpy as np

from app.realtime_postprocessing import (
    RealtimePostprocessingService,
    RealtimePostprocessingSnapshot,
)
from layer4_speech_separation import Layer4LongAudioInput


def _source(index: int = 0) -> Layer4LongAudioInput:
    start = index * 480_000
    waveform = np.full(480_000, 0.01, dtype=np.float32)
    return Layer4LongAudioInput(
        asset_id=f"session:epoch0:track1:start{start}",
        sha256=hashlib.sha256(waveform.tobytes()).hexdigest(),
        session_id="session",
        stream_epoch=0,
        track_id=1,
        theta_deg=30.0,
        start_sample=start,
        sample_rate=48_000,
        waveform=waveform,
        l2_direction_counts=tuple(
            (sample, 1) for sample in range(start + 960, start + len(waveform) + 1, 960)
        ),
    )


class _Processor:
    def __init__(self) -> None:
        self.count = 0
        self.track_final_count = 0
        self.last_source = None

    def _snapshot(self, source, *, final=False):
        return RealtimePostprocessingSnapshot(
            "session", self.count + int(final), final, source.end_sample,
            self.count, (), (), None,
        )

    def push(self, source, *, is_final_chunk=False):
        self.count += 1
        self.last_source = source
        return self._snapshot(source)

    def finalize_track(self, identity):
        assert self.last_source is not None
        assert identity == ("session", 0, 1)
        self.track_final_count += 1
        value = self._snapshot(self.last_source)
        return replace(value, revision=value.revision + self.track_final_count)

    def finalize(self):
        return RealtimePostprocessingSnapshot(
            "session", self.count + 1, True, self.count * 480_000,
            self.count, (), (), None,
        )


def test_realtime_service_preloads_models_before_the_first_chunk() -> None:
    loaded = Event()

    def factory():
        loaded.set()
        return _Processor()

    service = RealtimePostprocessingService(factory, queue_chunks=1)
    service.start()

    assert loaded.wait(1.0)
    deadline = monotonic() + 1.0
    while service.status.state == "loading" and monotonic() < deadline:
        Event().wait(0.01)
    assert service.status.state == "waiting"
    assert service.status.model_load_seconds >= 0.0
    assert service.abort(timeout=2.0)


def test_realtime_service_drains_chunks_and_publishes_replaceable_final_snapshot():
    service = RealtimePostprocessingService(_Processor, queue_chunks=2)
    service.start()
    assert service.submit(_source(0))
    assert service.submit(_source(1))
    assert service.finish(timeout=2.0)
    snapshot = service.latest.get_nowait()
    assert snapshot.is_final
    assert service.final_snapshot is snapshot
    assert snapshot.processed_blocks == 2
    assert snapshot.valid_through_sample_48k == 960_000
    assert service.status.state == "final"
    assert service.status.submitted_blocks == 2


def test_realtime_service_retains_per_track_checkpoint_before_global_finish():
    service = RealtimePostprocessingService(_Processor, queue_chunks=2)
    service.start()
    assert service.submit(_source(0))
    assert service.submit_track_final(("session", 0, 1))

    deadline = monotonic() + 2.0
    while service.status.latest_revision < 2 and monotonic() < deadline:
        Event().wait(0.01)

    checkpoint = service.reuse_snapshot
    assert checkpoint is not None
    assert not checkpoint.is_final
    assert service.final_snapshot is None
    assert service.status.submitted_blocks == 2
    assert service.status.processed_blocks == 2
    assert service.abort(timeout=2.0)
    assert service.reuse_snapshot is checkpoint


def test_realtime_service_queue_overflow_is_explicit_and_never_blocks_caller():
    release = Event()

    class Slow(_Processor):
        def push(self, source, *, is_final_chunk=False):
            release.wait(2.0)
            return super().push(source, is_final_chunk=is_final_chunk)

    service = RealtimePostprocessingService(Slow, queue_chunks=1)
    service.start()
    assert service.submit(_source(0))
    # The worker may already own block 0, but at most one additional block can
    # wait. Continue until the bounded mailbox reports its explicit overflow.
    outcomes = [service.submit(_source(index)) for index in range(1, 5)]
    assert False in outcomes
    assert service.status.error == "realtime_layer456_queue_overflow"
    assert service.final_snapshot is None
    assert service.status.dropped_blocks >= 1
    release.set()
    service.abort(timeout=2.0)


def test_overflowed_prefix_can_never_publish_a_final_snapshot():
    started = Event()
    release = Event()

    class Slow(_Processor):
        def push(self, source, *, is_final_chunk=False):
            started.set()
            release.wait(2.0)
            return super().push(source, is_final_chunk=is_final_chunk)

    service = RealtimePostprocessingService(Slow, queue_chunks=1)
    service.start()
    assert service.submit(_source(0))
    assert started.wait(1.0)
    assert service.submit(_source(1))
    assert not service.submit(_source(2))
    release.set()

    assert service.finish(timeout=2.0)
    snapshots = []
    while not service.latest.empty():
        snapshots.append(service.latest.get_nowait())
    assert snapshots
    assert all(not item.is_final for item in snapshots)
    assert service.status.state == "failed"
    assert service.status.error == "realtime_layer456_queue_overflow"


def test_finish_timeout_is_bounded_and_suppresses_late_final_snapshot():
    started = Event()
    release = Event()

    class Stuck(_Processor):
        def push(self, source, *, is_final_chunk=False):
            started.set()
            release.wait(2.0)
            return super().push(source, is_final_chunk=is_final_chunk)

    service = RealtimePostprocessingService(Stuck, queue_chunks=1)
    service.start()
    assert service.submit(_source(0))
    assert started.wait(1.0)
    assert service.submit(_source(1))
    began = monotonic()
    assert not service.finish(timeout=0.05)
    assert monotonic() - began < 0.5
    assert service.status.error == "realtime_layer456_finish_timeout"

    release.set()
    assert service._finished.wait(2.0)
    assert not service.active
    assert service.status.dropped_blocks >= 1
    while not service.latest.empty():
        assert not service.latest.get_nowait().is_final
    assert service.final_snapshot is None


def test_snapshot_validation_rejects_unaligned_watermark():
    snapshot = RealtimePostprocessingSnapshot(
        "session", 1, False, 960, 0, (), (), None,
    )
    try:
        replace(snapshot, valid_through_sample_48k=961)
    except ValueError as exc:
        assert "20 ms" in str(exc)
    else:
        raise AssertionError("unaligned realtime watermark was accepted")
