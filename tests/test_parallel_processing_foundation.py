from __future__ import annotations

from dataclasses import dataclass
import threading

import numpy as np
import pytest

from app.compute_cache import (
    ArtifactTooLargeError,
    CachePartitionLimits,
    ComputeCache,
    DuplicateArtifactError,
    GpuArtifactError,
    RetiredWindowError,
    WindowArtifactStore,
)
from app.processing_contracts import (
    L2StageResult,
    L3StageResult,
    L4StageResult,
    ProcessingConfigSnapshot,
    StageState,
    WindowKey,
    WindowWorkItem,
)
from app.result_joiner import (
    DuplicateStageResultError,
    JoinerCapacityError,
    ResultDeliveryError,
    ResultJoiner,
    TimelineGapError,
)
from common.data_types import DecisionWindow


def _window(index: int, *, session_id: str = "parallel", epoch: int = 0) -> DecisionWindow:
    start = index * 960
    end = start + 7_680
    return DecisionWindow(
        session_id=session_id,
        stream_epoch=epoch,
        window_id=index,
        decision_sample=end,
        doa_start_sample=end - 1_920,
        doa_end_sample=end,
        context_start_sample=start,
        context_end_sample=end,
        sample_rate=48_000,
        samples=np.zeros((7_680, 8), dtype=np.float32),
        source_sequence_ids=tuple(range(index, index + 8)),
    )


def _work(index: int, *, session_id: str = "parallel", epoch: int = 0) -> WindowWorkItem:
    window = _window(index, session_id=session_id, epoch=epoch)
    snapshot = ProcessingConfigSnapshot(
        revision=3,
        config_hash="abc123",
        geometry_version="geometry-v1",
        audio_mode="raw",
        values={"gate": {"threshold": 0.6}, "models": ["nv"]},
    )
    return WindowWorkItem(WindowKey.from_window(window), window, snapshot, accepted_monotonic_ns=100 + index)


@dataclass(frozen=True, slots=True)
class _Payload:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    value: str

    @classmethod
    def for_key(cls, key: WindowKey, value: str) -> _Payload:
        return cls(key.session_id, key.stream_epoch, key.window_id, key.decision_sample, value)


def _stage_results(work: WindowWorkItem):
    key = work.key
    return (
        L2StageResult.completed(key, _Payload.for_key(key, "l2"), finished_monotonic_ns=101),
        L3StageResult.completed(key, _Payload.for_key(key, "l3"), finished_monotonic_ns=102),
        L4StageResult.completed(key, _Payload.for_key(key, "l4"), finished_monotonic_ns=103),
    )


def test_window_work_item_has_one_exact_key_and_deeply_immutable_config():
    work = _work(0)

    assert work.key == WindowKey("parallel", 0, 0, 7_680)
    assert work.config.values["models"] == ("nv",)
    with pytest.raises(TypeError):
        work.config.values["gate"]["threshold"] = 0.2
    with pytest.raises(ValueError, match="exactly match"):
        WindowWorkItem(WindowKey("parallel", 0, 1, 7_680), work.window, work.config)


def test_stage_results_reject_wrong_identity_and_define_terminal_state():
    work = _work(0)
    wrong = _Payload("other", 0, 0, 7_680, "l2")

    with pytest.raises(ValueError, match="identity"):
        L2StageResult.completed(work.key, wrong, finished_monotonic_ns=1)
    skipped = L3StageResult.terminal(
        work.key,
        StageState.SKIPPED,
        "gate_closed",
        finished_monotonic_ns=2,
    )
    assert skipped.is_terminal
    assert skipped.output is None


def test_window_artifact_cache_is_single_publish_and_evicts_complete_oldest_window():
    events = []
    store = WindowArtifactStore(
        "window",
        CachePartitionLimits(max_windows=2, max_bytes=64 * 1024, max_artifacts_per_window=2),
        on_evict=events.append,
    )
    keys = [_work(index).key for index in range(3)]

    store.publish(keys[0], "spectrum", np.zeros(128, dtype=np.float32))
    with pytest.raises(DuplicateArtifactError):
        store.publish(keys[0], "spectrum", np.zeros(128, dtype=np.float32))
    store.publish(keys[0], "noise", b"noise")
    store.publish(keys[1], "spectrum", np.ones(128, dtype=np.float32))
    result = store.publish(keys[2], "spectrum", np.ones(128, dtype=np.float32))

    assert {event.artifact_name for event in result.evicted} == {"spectrum", "noise"}
    assert result.evicted == tuple(events)
    assert store.get(keys[0], "spectrum") is None
    assert store.snapshot().windows == 2
    assert store.snapshot().current_bytes <= store.snapshot().max_bytes
    with pytest.raises(RetiredWindowError):
        store.publish(keys[0], "new", b"cannot-resurrect")


def test_cache_has_byte_and_gpu_hard_limits_and_partition_total_limit():
    store = WindowArtifactStore("small", CachePartitionLimits(max_windows=2, max_bytes=256))
    with pytest.raises(ArtifactTooLargeError):
        store.publish(_work(0).key, "large", np.zeros(1024, dtype=np.float32))

    class _CudaValue:
        device = "cuda:0"

    with pytest.raises(GpuArtifactError):
        store.publish(_work(0).key, "gpu", _CudaValue())

    with pytest.raises(ValueError, match="sum of partition"):
        ComputeCache(
            {"l2": CachePartitionLimits(2, 1024), "l3": CachePartitionLimits(2, 1024)},
            max_total_bytes=1024,
        )
    cache = ComputeCache(
        {"l2": CachePartitionLimits(2, 1024), "l3": CachePartitionLimits(2, 2048)},
        max_total_bytes=4096,
    )
    assert cache.current_bytes == 0


def test_joiner_accepts_out_of_order_stage_completion_but_commits_in_sample_order():
    first, second = _work(0), _work(1)
    joiner = ResultJoiner(max_pending_windows=4, max_pending_bytes=16 * 1024 * 1024)
    joiner.register(first)
    joiner.register(second)

    for result in reversed(_stage_results(second)):
        joiner.submit(result)
    assert joiner.drain_ready() == ()

    l2, l3, l4 = _stage_results(first)
    joiner.submit(l4)
    joiner.submit(l2)
    joiner.submit(l3)
    joined = joiner.drain_ready()

    assert tuple(item.key for item in joined) == (first.key, second.key)
    assert all(item.state is StageState.COMPLETED for item in joined)
    assert joiner.snapshot().committed_through == (("parallel", 0, second.key.decision_sample),)


def test_joiner_never_silently_crosses_gap_and_can_record_an_explicit_gap():
    first, third = _work(0), _work(2)
    joiner = ResultJoiner(max_pending_bytes=16 * 1024 * 1024)
    joiner.register(first)

    with pytest.raises(TimelineGapError, match="unreported"):
        joiner.register(third)
    joiner.register(third, preceding_gap_reason="runtime_overflow_before_registration")
    gaps = joiner.drain_gaps()
    assert len(gaps) == 1
    assert gaps[0].reason == "runtime_overflow_before_registration"


def test_joiner_skips_downstream_explicitly_and_rejects_duplicate_stage_publication():
    work = _work(0)
    joiner = ResultJoiner(max_pending_bytes=8 * 1024 * 1024)
    joiner.register(work)
    l2, _, _ = _stage_results(work)
    joiner.submit_l2(l2)
    with pytest.raises(DuplicateStageResultError):
        joiner.submit_l2(l2)
    joiner.skip_missing_downstream(work.key, "gate_closed")

    joined = joiner.drain_ready()
    assert len(joined) == 1
    assert joined[0].l3.state is joined[0].l4.state is StageState.SKIPPED
    assert joined[0].state is StageState.COMPLETED


def test_joiner_capacity_is_hard_and_never_evicts_pending_windows_silently():
    joiner = ResultJoiner(max_pending_windows=1, max_pending_bytes=8 * 1024 * 1024)
    joiner.register(_work(0))
    with pytest.raises(JoinerCapacityError):
        joiner.register(_work(1))
    assert joiner.pending_keys() == (_work(0).key,)


def test_joiner_callback_runs_outside_lock_and_failed_delivery_is_retryable():
    callback_observed_unlocked = threading.Event()
    callback_threads: list[threading.Thread] = []
    joiner: ResultJoiner

    def callback(_joined) -> None:
        worker = threading.Thread(
            target=lambda: (joiner.snapshot(), callback_observed_unlocked.set()),
            daemon=True,
        )
        callback_threads.append(worker)
        worker.start()
        if not callback_observed_unlocked.wait(1.0):
            raise RuntimeError("joiner callback still owns the timeline lock")
        raise RuntimeError("injected downstream delivery failure")

    joiner = ResultJoiner(
        max_pending_windows=2,
        max_pending_bytes=8 * 1024 * 1024,
        on_joined=callback,
    )
    work = _work(0)
    joiner.register(work)
    l2, l3, l4 = _stage_results(work)
    joiner.submit_l2(l2)
    joiner.submit_l3(l3)
    with pytest.raises(ResultDeliveryError, match="callback failed"):
        joiner.submit_l4(l4)

    for worker in callback_threads:
        worker.join(timeout=1.0)
    assert callback_observed_unlocked.is_set()
    retriable = joiner.drain_ready()
    assert len(retriable) == 1 and retriable[0].key == work.key
    assert joiner.snapshot().pending_windows == 0


def test_completed_epoch_history_is_pruned_across_one_thousand_restarts():
    joiner = ResultJoiner(max_pending_windows=2, max_pending_bytes=8 * 1024 * 1024)
    cache = ComputeCache(
        {"l2": CachePartitionLimits(max_windows=2, max_bytes=64 * 1024)},
        max_total_bytes=64 * 1024,
    )

    for epoch in range(1_000):
        work = _work(epoch, epoch=epoch)
        joiner.register(work)
        cache.publish("l2", work.key, "marker", b"done")
        for result in _stage_results(work):
            joiner.submit(result)
        assert joiner.drain_ready()[0].key == work.key
        cache.evict_window(work.key)
        closed = joiner.prune_completed_streams("parallel", before_epoch=epoch)
        cache.prune_stream_history(closed)

    assert len(joiner.snapshot().committed_through) == 1
    assert cache.snapshots()["l2"].retired_streams == 1


def test_cache_history_prune_retries_until_old_window_artifacts_are_gone():
    cache = ComputeCache(
        {"l2": CachePartitionLimits(max_windows=2, max_bytes=64 * 1024)},
        max_total_bytes=64 * 1024,
    )
    old = _work(0, epoch=0).key
    cache.publish("l2", old, "marker", b"still-live")

    assert cache.prune_stream_history((old.stream_key,)) == ()
    cache.evict_window(old)
    assert cache.prune_stream_history((old.stream_key,)) == (old.stream_key,)
    assert cache.snapshots()["l2"].retired_streams == 0
