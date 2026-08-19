from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest

from app.runtime import ApplicationRuntime
from app.processing_contracts import WindowKey
from common.config import load_config
from common.data_types import (
    CandidateDirection,
    DecisionWindow,
    EnhancedAudio,
    PipelineStatus,
    SpatialResponse,
)
from data_management import DecisionRecord, ResultWatermark
from gui.dev_test_ui.contracts import DevUiFrame, L1MeterSnapshot
from layer2_source_detection import (
    CandidateSearchDiagnostics,
    Layer2ExecutionState,
    Layer2PipelineResult,
    ProbabilityGateDecision,
    ProbabilityGateState,
)
from layer3_direction_signal import Layer3Output
from layer4_voice_classifier import Layer4Result, ModelPrediction, VoiceDetection


CONFIG = Path(__file__).parents[1] / "config/config.yaml"


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for staged runtime")
        time.sleep(0.002)


class _IdlePipeline:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self._wake = queue.Queue[object]()

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1
        self._wake.put(object())

    def read(self, timeout: float | None = None) -> None:
        try:
            self._wake.get(timeout=timeout)
        except queue.Empty:
            pass
        return None

    @staticmethod
    def take_health_events() -> tuple[()]:
        return ()


class _StubSerial:
    @staticmethod
    def write(packet: bytes) -> int:
        return len(packet)


class _MemoryRecordingStore:
    def __init__(self) -> None:
        self.records: list[DecisionRecord] = []
        self.watermarks: list[ResultWatermark] = []
        self.stop_reasons: list[str] = []
        self._lock = threading.Lock()

    def start_session(self, session_id: str, metadata: object) -> None:
        del session_id, metadata

    @staticmethod
    def set_recording_mode(mode: str) -> None:
        del mode

    @staticmethod
    def append_audio(block: object) -> None:
        del block

    def append_result(self, record: DecisionRecord) -> None:
        with self._lock:
            self.records.append(record)

    def advance_result_watermark(self, watermark: ResultWatermark) -> None:
        with self._lock:
            self.watermarks.append(watermark)

    def stop_session(self, reason: str) -> None:
        with self._lock:
            self.stop_reasons.append(reason)

    @staticmethod
    def start_recording() -> None:
        return None

    @staticmethod
    def pause_recording() -> None:
        return None

    def record_snapshot(self) -> tuple[DecisionRecord, ...]:
        with self._lock:
            return tuple(self.records)

    def watermark_snapshot(self) -> tuple[ResultWatermark, ...]:
        with self._lock:
            return tuple(self.watermarks)


@dataclass(frozen=True)
class _Interval:
    stage: str
    window_id: int
    started: float
    finished: float
    thread_name: str


class _StageProbe:
    def __init__(self) -> None:
        self._intervals: list[_Interval] = []
        self._lock = threading.Lock()

    def add(self, stage: str, window_id: int, started: float, finished: float) -> None:
        with self._lock:
            self._intervals.append(
                _Interval(stage, window_id, started, finished, threading.current_thread().name)
            )

    def get(self, stage: str, window_id: int) -> _Interval:
        with self._lock:
            return next(
                item
                for item in self._intervals
                if item.stage == stage and item.window_id == window_id
            )

    def count(self, stage: str) -> int:
        with self._lock:
            return sum(item.stage == stage for item in self._intervals)


class _StubUiAggregator:
    """The staged tests admit formal windows directly, without an L1 UI status frame."""

    @staticmethod
    def update_srp(*args: object, **kwargs: object) -> None:
        del args, kwargs

    @staticmethod
    def update_l3(*args: object, **kwargs: object) -> None:
        del args, kwargs

    @staticmethod
    def update_l4(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return SimpleNamespace(stage="l4")


class _FailingUiAggregator:
    @staticmethod
    def update_srp(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("injected UI projection failure")


def _window(
    window_id: int,
    *,
    session_id: str = "parallel-session",
    stream_epoch: int = 0,
) -> DecisionWindow:
    decision_sample = 15_360 + window_id * 960
    return DecisionWindow(
        session_id,
        stream_epoch,
        window_id,
        decision_sample,
        decision_sample - 1_920,
        decision_sample,
        decision_sample - 15_360,
        decision_sample,
        48_000,
        np.zeros((15_360, 8), dtype=np.float32),
        tuple(range(window_id, window_id + 16)),
    )


def _open_l2_output(window: DecisionWindow) -> Layer2PipelineResult:
    gate = ProbabilityGateDecision(
        window.session_id,
        window.stream_epoch,
        window.window_id,
        window.decision_sample,
        "mean_2x20ms_v1",
        ProbabilityGateState.OPEN,
        0.9,
        0.9,
        0.9,
        0.6,
        0,
        True,
        "threshold_passed",
    )
    response = SpatialResponse(
        window.session_id,
        window.stream_epoch,
        window.window_id,
        window.decision_sample,
        window.doa_start_sample,
        window.doa_end_sample,
        np.arange(360, dtype=np.float32),
        np.zeros(360, dtype=np.float32),
        np.zeros(360, dtype=np.float32),
    )
    candidate = CandidateDirection(
        window.session_id,
        window.stream_epoch,
        window.window_id,
        window.decision_sample,
        window.doa_start_sample,
        window.doa_end_sample,
        float((window.window_id * 10) % 360),
        1.0,
        0.9,
    )
    diagnostics = CandidateSearchDiagnostics(
        "single_pass", "parallel_runtime_test_v1", 0, 1, "single_pass", 1.0
    )
    return Layer2PipelineResult(
        Layer2ExecutionState.PROCESSED,
        gate,
        response,
        (candidate,),
        diagnostics,
    )


def _blocked_l2_output(window: DecisionWindow) -> Layer2PipelineResult:
    gate = ProbabilityGateDecision(
        window.session_id,
        window.stream_epoch,
        window.window_id,
        window.decision_sample,
        "mean_2x20ms_v1",
        ProbabilityGateState.CLOSED,
        0.1,
        0.1,
        0.1,
        0.6,
        0,
        False,
        "below_threshold",
    )
    return Layer2PipelineResult(Layer2ExecutionState.BLOCKED, gate, None, (), None)


def _l3_output(window: DecisionWindow, candidates: tuple[CandidateDirection, ...]) -> Layer3Output:
    return Layer3Output(
        tuple(
            EnhancedAudio(
                window.session_id,
                window.stream_epoch,
                window.window_id,
                window.decision_sample,
                window.context_start_sample,
                window.context_end_sample,
                candidate.theta_deg,
                48_000,
                "parallel_runtime_test",
                None,
                (),
                np.zeros(15_360, dtype=np.float32),
            )
            for candidate in candidates
        )
    )


def _l4_output(inputs: tuple[object, ...]) -> Layer4Result:
    detections = tuple(
        VoiceDetection(
            item.session_id,
            item.stream_epoch,
            item.window_id,
            item.decision_sample,
            item.theta_deg,
            0.8,
            True,
            "stub-primary",
        )
        for item in inputs
    )
    return Layer4Result(
        detections,
        (
            ModelPrediction(
                "stub-primary",
                np.full(len(inputs), 0.8, dtype=np.float32),
                0.01,
                {},
            ),
        ),
        "stub-primary",
        0.5,
    )


class _StubL2:
    def __init__(
        self,
        probe: _StageProbe,
        *,
        delay: float = 0.0,
        blocked: bool = False,
        fail_window: int | None = None,
        first_started: threading.Event | None = None,
        first_release: threading.Event | None = None,
    ) -> None:
        self.probe = probe
        self.delay = delay
        self.blocked = blocked
        self.fail_window = fail_window
        self.first_started = first_started
        self.first_release = first_release
        self.last_kalman_error = None
        self.last_id_tracking_error = None
        self.id_tracker = SimpleNamespace(active_track_count=0)
        self.voice_feedback: list[tuple[str, int, int, float]] = []

    @staticmethod
    def reset() -> None:
        return None

    def process(self, window: DecisionWindow, *args: object, **kwargs: object) -> Layer2PipelineResult:
        del args, kwargs
        started = time.monotonic()
        if window.window_id == 0 and self.first_started is not None:
            self.first_started.set()
        if window.window_id == 0 and self.first_release is not None:
            if not self.first_release.wait(5.0):
                raise TimeoutError("test did not release blocked L2")
        if self.delay:
            time.sleep(self.delay)
        try:
            if window.window_id == self.fail_window:
                raise RuntimeError(f"injected L2 failure for window {window.window_id}")
            return _blocked_l2_output(window) if self.blocked else _open_l2_output(window)
        finally:
            self.probe.add("l2", window.window_id, started, time.monotonic())

    def submit_voice_feedback(
        self, session_id: str, stream_epoch: int, decision_sample: int, theta_deg: float
    ) -> bool:
        self.voice_feedback.append((session_id, stream_epoch, decision_sample, theta_deg))
        return True


class _StubL3:
    def __init__(
        self,
        probe: _StageProbe,
        *,
        delay: float = 0.0,
        fail_window: int | None = None,
    ) -> None:
        self.probe = probe
        self.delay = delay
        self.fail_window = fail_window
        self._started: dict[int, float] = {}

    @staticmethod
    def clear_cache() -> None:
        return None

    def prepare(self, window: DecisionWindow, *, mode: str) -> DecisionWindow:
        del mode
        self._started[window.window_id] = time.monotonic()
        if self.delay:
            time.sleep(self.delay / 2)
        return window

    def process_prepared(
        self,
        window: DecisionWindow,
        candidates: tuple[CandidateDirection, ...],
        geometry: object,
    ) -> Layer3Output:
        del geometry
        if self.delay:
            time.sleep(self.delay / 2)
        try:
            if window.window_id == self.fail_window:
                raise RuntimeError(f"injected L3 failure for window {window.window_id}")
            return _l3_output(window, candidates)
        finally:
            self.probe.add(
                "l3", window.window_id, self._started.pop(window.window_id), time.monotonic()
            )


class _FirstBlockingL3(_StubL3):
    def __init__(
        self,
        probe: _StageProbe,
        *,
        started: threading.Event,
        release: threading.Event,
    ) -> None:
        super().__init__(probe)
        self.first_started = started
        self.first_release = release

    def prepare(self, window: DecisionWindow, *, mode: str) -> DecisionWindow:
        if window.window_id == 0:
            self.first_started.set()
            if not self.first_release.wait(8.0):
                raise TimeoutError("test did not release first L3 window")
        return super().prepare(window, mode=mode)


class _StubL4:
    threshold = 0.5

    def __init__(
        self,
        probe: _StageProbe,
        *,
        delay: float = 0.0,
        fail_window: int | None = None,
    ) -> None:
        self.probe = probe
        self.delay = delay
        self.fail_window = fail_window

    def process(self, inputs: tuple[object, ...]) -> Layer4Result:
        window_id = inputs[0].window_id
        started = time.monotonic()
        if self.delay:
            time.sleep(self.delay)
        try:
            if window_id == self.fail_window:
                raise RuntimeError(f"injected L4 failure for window {window_id}")
            return _l4_output(inputs)
        finally:
            self.probe.add("l4", window_id, started, time.monotonic())


def _config(
    *,
    l2_queue_windows: int = 4,
    l3_queue_windows: int = 3,
    l4_queue_windows: int = 3,
    completion_queue_windows: int = 8,
    max_inflight_windows: int = 16,
    compute_cache_max_bytes: int = 8_388_608,
    graceful_shutdown_timeout_seconds: float = 5.0,
):
    base = load_config(CONFIG, environ={})
    runtime = base.runtime.model_copy(
        update={
            "preferred_device": "cpu",
            "l2_queue_windows": l2_queue_windows,
            "l3_queue_windows": l3_queue_windows,
            "l4_queue_windows": l4_queue_windows,
            "completion_queue_windows": completion_queue_windows,
            "max_inflight_windows": max_inflight_windows,
            "compute_cache_max_bytes": compute_cache_max_bytes,
            "graceful_shutdown_timeout_seconds": graceful_shutdown_timeout_seconds,
        }
    )
    return base.model_copy(update={"runtime": runtime})


def _runtime(
    tmp_path: Path,
    *,
    config=None,
    l2: _StubL2 | None = None,
    l3: _StubL3 | None = None,
    l4: _StubL4 | None = None,
) -> tuple[ApplicationRuntime, _MemoryRecordingStore, _StageProbe]:
    probe = _StageProbe()
    store = _MemoryRecordingStore()
    l4 = l4 or _StubL4(probe)
    runtime = ApplicationRuntime(
        config or _config(),
        project_root=tmp_path,
        pipeline=_IdlePipeline(),
        serial_device=_StubSerial(),
        recording_store=store,
        layer4_engine=l4,
        source_probability_provider=lambda window: (),
    )
    runtime._layer2 = l2 or _StubL2(probe)
    runtime._layer3 = l3 or _StubL3(probe)
    return runtime, store, probe


def _start_with_stubs(
    tmp_path: Path,
    *,
    config=None,
    l2_factory: Callable[[_StageProbe], _StubL2] | None = None,
    l3_factory: Callable[[_StageProbe], _StubL3] | None = None,
    l4_factory: Callable[[_StageProbe], _StubL4] | None = None,
) -> tuple[ApplicationRuntime, _MemoryRecordingStore, _StageProbe]:
    probe = _StageProbe()
    l2 = (l2_factory or (lambda value: _StubL2(value)))(probe)
    l3 = (l3_factory or (lambda value: _StubL3(value)))(probe)
    l4 = (l4_factory or (lambda value: _StubL4(value)))(probe)
    runtime, store, _ = _runtime(tmp_path, config=config, l2=l2, l3=l3, l4=l4)
    runtime.start()
    runtime._ui_aggregator = _StubUiAggregator()
    return runtime, store, probe


def test_completed_l4_has_independent_bounded_latest_only_ui_mailbox(tmp_path: Path) -> None:
    runtime, _store, _probe = _start_with_stubs(tmp_path)
    try:
        runtime._admit_window(_window(0))
        _wait_until(lambda: runtime.processing_status["l4_actual_completed"] >= 1)
        _wait_until(lambda: runtime.latest_l4_dev_ui.qsize() == 1)
        first = runtime.latest_l4_dev_ui.get_nowait()
        assert isinstance(first, DevUiFrame)
        assert first.l4_result is not None
        assert first.spatial_response is not None
        assert first.spatial_response.window_id == first.l4_result.detections[0].window_id == 0

        runtime._admit_window(_window(1))
        runtime._admit_window(_window(2))
        _wait_until(lambda: runtime.processing_status["l4_actual_completed"] >= 3)
        assert runtime.latest_l4_dev_ui.maxsize == 1
        assert runtime.latest_l4_dev_ui.qsize() == 1
        latest = runtime.latest_l4_dev_ui.get_nowait()
        assert latest.spatial_response.window_id == latest.l4_result.detections[0].window_id == 2
    finally:
        runtime.stop()


def test_late_ordered_commit_from_old_epoch_cannot_update_new_epoch_ui(tmp_path: Path) -> None:
    runtime, _store, _probe = _runtime(tmp_path)
    meter = L1MeterSnapshot(
        "parallel-session", 1, 960, 0,
        np.full(8, -40.0, np.float32),
        np.full(8, -20.0, np.float32),
        np.zeros(8, np.bool_), "unknown", "idle",
    )
    status = PipelineStatus(
        "warming_up", "parallel-session", 1, 960, 15_360, "Warming"
    )
    runtime._ui_aggregator.update_l1(meter, status)
    joined = SimpleNamespace(work_item=SimpleNamespace(window=_window(0, stream_epoch=0)))

    runtime._publish_joined_ui(
        joined=joined,
        values={},
        l2_output=None,
        l3_output=None,
        l4_result=None,
        response=None,
        candidates=(),
        search_diagnostics=None,
        gate_decision=None,
        previews=(),
        tracked_audio=(),
        compute_ms=0.0,
        latency_ms=0.0,
        stage_timings={},
    )

    assert runtime._ui_aggregator.current_stream == ("parallel-session", 1)
    assert runtime._ui_aggregator.frame().spatial_response is None
    assert runtime.latest_dev_ui.empty()


def test_completed_l4_voice_angle_is_sent_back_to_l2_without_private_id(tmp_path: Path) -> None:
    base = _config()
    tracking = base.layer2.direction_id_tracking.model_copy(update={"enabled": True})
    layer2 = base.layer2.model_copy(update={"direction_id_tracking": tracking})
    config = base.model_copy(update={"layer2": layer2})
    runtime, _store, _probe = _start_with_stubs(tmp_path, config=config)
    try:
        runtime._admit_window(_window(0))
        _wait_until(lambda: runtime.processing_status["l4_actual_completed"] >= 1)
        feedback = runtime._layer2.voice_feedback
        assert len(feedback) == 1
        session_id, stream_epoch, decision_sample, theta_deg = feedback[0]
        assert (session_id, stream_epoch, decision_sample) == (
            _window(0).session_id, _window(0).stream_epoch, _window(0).decision_sample
        )
        assert theta_deg == 0.0
    finally:
        runtime.stop()


def test_stages_overlap_across_windows_but_preserve_same_window_dependencies_and_ordered_commit(
    tmp_path: Path,
) -> None:
    runtime, store, probe = _start_with_stubs(
        tmp_path,
        l2_factory=lambda value: _StubL2(value, delay=0.03),
        l3_factory=lambda value: _StubL3(value, delay=0.06),
        l4_factory=lambda value: _StubL4(value, delay=0.08),
    )
    try:
        for window_id in range(4):
            runtime._admit_window(_window(window_id))
        _wait_until(lambda: len(store.record_snapshot()) == 4)

        for window_id in range(4):
            l2 = probe.get("l2", window_id)
            l3 = probe.get("l3", window_id)
            l4 = probe.get("l4", window_id)
            assert l2.finished <= l3.started
            assert l3.finished <= l4.started
            assert l2.thread_name.endswith("-l2")
            assert l3.thread_name.endswith("-l3")
            assert l4.thread_name.endswith("-l4")

        assert probe.get("l2", 1).started < probe.get("l3", 0).finished
        assert probe.get("l3", 1).started < probe.get("l4", 0).finished
        assert [item.window_id for item in store.record_snapshot()] == [0, 1, 2, 3]
        assert [item.sample for item in store.watermark_snapshot()] == [
            15_360,
            16_320,
            17_280,
            18_240,
        ]
        assert runtime.compute_cache_bytes == 0
    finally:
        runtime.stop()


def test_gate_skip_is_formal_and_never_runs_l3_or_l4(tmp_path: Path) -> None:
    runtime, store, probe = _start_with_stubs(
        tmp_path,
        l2_factory=lambda value: _StubL2(value, blocked=True),
    )
    try:
        runtime._admit_window(_window(0))
        _wait_until(lambda: len(store.record_snapshot()) == 1)
        record = store.record_snapshot()[0]
        assert record.stage_statuses == {
            "l2": "completed",
            "l3": "skipped",
            "l4": "skipped",
        }
        assert record.status == "ok"
        assert record.gate_decision is not None
        assert record.gate_decision["state"] == "closed"
        assert probe.count("l3") == 0
        assert probe.count("l4") == 0
    finally:
        runtime.stop()


def test_open_l2_with_no_candidates_skips_l3_prepare_and_finishes_empty_l4(
    tmp_path: Path,
) -> None:
    probe = _StageProbe()

    class EmptyCandidateL2(_StubL2):
        def process(self, window: DecisionWindow, *args: object, **kwargs: object):
            output = super().process(window, *args, **kwargs)
            return Layer2PipelineResult(
                output.state,
                output.gate_decision,
                output.spatial_response,
                (),
                output.search_diagnostics,
            )

    class PrepareForbiddenL3(_StubL3):
        prepare_calls = 0
        process_calls = 0

        def prepare(self, window: DecisionWindow, *, mode: str):
            del window, mode
            self.prepare_calls += 1
            raise AssertionError("empty candidates must not prepare L3")

        def process_prepared(self, *args: object, **kwargs: object):
            del args, kwargs
            self.process_calls += 1
            raise AssertionError("empty candidates must not process prepared L3")

    class EmptyBatchL4:
        threshold = 0.5

        def __init__(self) -> None:
            self.calls = 0

        def process(self, inputs: tuple[object, ...]) -> Layer4Result:
            self.calls += 1
            assert inputs == ()
            return _l4_output(inputs)

    layer3 = PrepareForbiddenL3(probe)
    layer4 = EmptyBatchL4()
    runtime, store, _ = _runtime(
        tmp_path,
        l2=EmptyCandidateL2(probe),
        l3=layer3,
        l4=layer4,
    )
    runtime.start()
    runtime._ui_aggregator = _StubUiAggregator()
    try:
        runtime._admit_window(_window(0))
        _wait_until(lambda: len(store.record_snapshot()) == 1)
        record = store.record_snapshot()[0]
        assert layer3.prepare_calls == 0
        assert layer3.process_calls == 0
        assert layer4.calls == 1
        assert record.stage_statuses == {
            "l2": "completed",
            "l3": "completed",
            "l4": "completed",
        }
        assert record.enhanced_audio == ()
        assert record.voice_direction_count == 0
    finally:
        runtime.stop()


@pytest.mark.parametrize("failed_stage", ["l2", "l3", "l4"])
def test_stage_failure_has_explicit_terminal_state_and_later_window_continues(
    tmp_path: Path,
    failed_stage: str,
) -> None:
    runtime, store, _ = _start_with_stubs(
        tmp_path,
        l2_factory=lambda value: _StubL2(
            value, fail_window=1 if failed_stage == "l2" else None
        ),
        l3_factory=lambda value: _StubL3(
            value, fail_window=1 if failed_stage == "l3" else None
        ),
        l4_factory=lambda value: _StubL4(
            value, fail_window=1 if failed_stage == "l4" else None
        ),
    )
    try:
        for window_id in range(3):
            runtime._admit_window(_window(window_id))
        _wait_until(lambda: len(store.record_snapshot()) == 3)
        records = store.record_snapshot()
        assert [item.window_id for item in records] == [0, 1, 2]
        assert records[0].status == "ok"
        assert records[2].status == "ok"
        assert records[1].stage_statuses[failed_stage] == "failed"
        assert records[1].status == "error"
        if failed_stage == "l2":
            assert records[1].stage_statuses["l3"] == "skipped"
            assert records[1].stage_statuses["l4"] == "skipped"
        if failed_stage == "l3":
            assert records[1].stage_statuses["l4"] == "skipped"
        assert runtime.processing_status["completed_counts"]["commit"] == 3
        assert runtime.processing_status["error_counts"][failed_stage] == 1
    finally:
        runtime.stop()


def test_l2_admission_overflow_explicitly_drops_oldest_waiting_window(tmp_path: Path) -> None:
    first_started = threading.Event()
    first_release = threading.Event()
    config = _config(
        l2_queue_windows=1,
        l3_queue_windows=1,
        l4_queue_windows=1,
        completion_queue_windows=3,
        max_inflight_windows=6,
    )
    runtime, store, _ = _start_with_stubs(
        tmp_path,
        config=config,
        l2_factory=lambda value: _StubL2(
            value, first_started=first_started, first_release=first_release
        ),
    )
    try:
        runtime._admit_window(_window(0))
        assert first_started.wait(2.0)
        runtime._admit_window(_window(1))
        runtime._admit_window(_window(2))
        first_release.set()
        _wait_until(lambda: len(store.record_snapshot()) == 3)
        records = store.record_snapshot()
        assert [item.window_id for item in records] == [0, 1, 2]
        assert records[1].stage_statuses == {
            "l2": "dropped",
            "l3": "dropped",
            "l4": "dropped",
        }
        assert records[1].terminal_reason == "l2_admission_queue_overflow"
        assert runtime.processing_drops >= 1
        dropped = store.watermark_snapshot()[1].dropped_windows
        assert dropped[0]["window_id"] == 1
        assert dropped[0]["reason"] == "l2_admission_queue_overflow"
    finally:
        first_release.set()
        runtime.stop()


def test_graceful_stop_drains_every_admitted_window(tmp_path: Path) -> None:
    runtime, store, _ = _start_with_stubs(
        tmp_path,
        l2_factory=lambda value: _StubL2(value, delay=0.015),
        l3_factory=lambda value: _StubL3(value, delay=0.025),
        l4_factory=lambda value: _StubL4(value, delay=0.025),
    )
    for window_id in range(4):
        runtime._admit_window(_window(window_id))
    runtime.stop()
    assert [item.window_id for item in store.record_snapshot()] == [0, 1, 2, 3]
    assert [item.sample for item in store.watermark_snapshot()] == [
        15_360,
        16_320,
        17_280,
        18_240,
    ]
    assert runtime.processing_running is False
    assert runtime.processing_queue_depths == {"l2": 0, "l3": 0, "l4": 0, "completion": 0}
    assert store.stop_reasons == ["normal"]


def test_processing_status_exposes_bounded_queues_and_cache_never_exceeds_hard_limit(
    tmp_path: Path,
) -> None:
    config = _config(compute_cache_max_bytes=8_388_608)
    runtime, _, _ = _runtime(tmp_path, config=config)
    payload = np.zeros(196_608, dtype=np.float32)
    for window_id in range(40):
        runtime._cache_publish(
            "l2",
            # Cache keys intentionally use a separate stream from formal runtime work.
            key=WindowKey("cache-stress", 0, window_id, 15_360 + window_id * 960),
            name="stress",
            value=payload,
        )
        status = runtime.processing_status
        assert status["cache_bytes"] <= status["cache_max_bytes"]
        assert status["inflight_windows"] == 0

    status = runtime.processing_status
    assert status["queue_capacities"] == {
        "l2": config.runtime.l2_queue_windows,
        "l3": config.runtime.l3_queue_windows,
        "l4": config.runtime.l4_queue_windows,
        "completion": config.runtime.completion_queue_windows,
    }
    snapshots = runtime._compute_cache.snapshots()
    assert snapshots["l2"].current_bytes <= snapshots["l2"].max_bytes
    assert snapshots["l2"].windows <= snapshots["l2"].max_windows
    assert status["cache_bytes"] == runtime.compute_cache_bytes


def test_ui_projection_failure_never_stops_formal_commit_or_later_windows(tmp_path: Path) -> None:
    runtime, store, _ = _start_with_stubs(tmp_path)
    runtime._ui_aggregator = _FailingUiAggregator()
    try:
        for window_id in range(3):
            runtime._admit_window(_window(window_id))
        _wait_until(lambda: len(store.record_snapshot()) == 3)
        assert [item.window_id for item in store.record_snapshot()] == [0, 1, 2]
        assert [item.sample for item in store.watermark_snapshot()] == [15_360, 16_320, 17_280]
        assert runtime.dev_ui_error == "injected UI projection failure"
        assert runtime.processing_running is True
    finally:
        runtime.stop()


def test_sustained_slow_l3_uses_latest_wins_without_stopping_l1(tmp_path: Path) -> None:
    """Regression for the real 245 ms L3 / 20 ms admission failure."""

    config = _config(
        l2_queue_windows=2,
        l3_queue_windows=1,
        l4_queue_windows=1,
        completion_queue_windows=4,
        max_inflight_windows=7,
        graceful_shutdown_timeout_seconds=5.0,
    )
    runtime, store, probe = _start_with_stubs(
        tmp_path,
        config=config,
        l3_factory=lambda value: _StubL3(value, delay=0.245),
    )
    admitted: list[bool] = []
    admission_latencies: list[float] = []
    try:
        for window_id in range(120):
            started = time.monotonic()
            admitted.append(runtime._admit_window(_window(window_id)))
            admission_latencies.append(time.monotonic() - started)
            assert admission_latencies[-1] < 0.05
            time.sleep(0.02)

        _wait_until(lambda: len(store.record_snapshot()) == 120, timeout=8.0)
        assert runtime.last_error is None
        assert runtime.processing_running is True
        assert runtime.processing_status["inflight_windows"] <= 7
        assert runtime.processing_status["cache_bytes"] <= runtime.processing_status["cache_max_bytes"]
    finally:
        runtime.stop()

    records = store.record_snapshot()
    assert len(records) == 120
    assert [item.window_id for item in records] == list(range(120))
    assert any(item.stage_statuses["l3"] == "dropped" for item in records)
    assert any(item.stage_statuses["l3"] == "completed" for item in records)
    assert any(item.stage_statuses["l2"] == "dropped" for item in records)
    assert any(admitted) and not all(admitted)
    assert max(admission_latencies) < 0.05
    assert runtime.processing_status["admission_rejections"] >= 1
    assert len(store.watermark_snapshot()) == 120
    assert any(item.dropped_windows for item in store.watermark_snapshot())
    assert runtime.processing_running is False


class _BlockingCommitStore(_MemoryRecordingStore):
    def __init__(self) -> None:
        super().__init__()
        self.commit_entered = threading.Event()
        self.commit_release = threading.Event()

    def append_result(self, record: DecisionRecord) -> None:
        self.commit_entered.set()
        if not self.commit_release.wait(5.0):
            raise TimeoutError("test did not release blocked commit")
        super().append_result(record)


def test_completion_saturation_rejects_admission_quickly_and_stays_bounded(
    tmp_path: Path,
) -> None:
    probe = _StageProbe()
    store = _BlockingCommitStore()
    config = _config(
        l2_queue_windows=1,
        l3_queue_windows=1,
        l4_queue_windows=1,
        completion_queue_windows=1,
        max_inflight_windows=6,
    )
    runtime = ApplicationRuntime(
        config,
        project_root=tmp_path,
        pipeline=_IdlePipeline(),
        serial_device=_StubSerial(),
        recording_store=store,
        layer4_engine=_StubL4(probe),
        source_probability_provider=lambda window: (),
    )
    runtime._layer2 = _StubL2(probe)
    runtime._layer3 = _StubL3(probe)
    runtime.start()
    runtime._ui_aggregator = _StubUiAggregator()
    try:
        runtime._admit_window(_window(0))
        assert store.commit_entered.wait(2.0)
        next_id = 1
        while not runtime.processing_status["completion_congested"] and next_id < 20:
            runtime._admit_window(_window(next_id))
            next_id += 1
            time.sleep(0.005)
        _wait_until(lambda: runtime.processing_status["completion_congested"])

        started = time.monotonic()
        accepted = runtime._admit_window(_window(next_id))
        elapsed = time.monotonic() - started
        assert accepted is False
        assert elapsed < 0.05
        status = runtime.processing_status
        assert status["admission_rejections"] >= 1
        assert status["last_admission_rejection"][0] == next_id
        assert status["completion_backlog_depth"] <= status["completion_backlog_capacity"]
        assert status["inflight_windows"] <= config.runtime.max_inflight_windows
    finally:
        store.commit_release.set()
        runtime.stop()


def test_cross_epoch_reorder_pending_is_hard_bounded_while_old_head_is_blocked(
    tmp_path: Path,
) -> None:
    first_started = threading.Event()
    first_release = threading.Event()
    config = _config(
        l2_queue_windows=4,
        l3_queue_windows=1,
        l4_queue_windows=1,
        completion_queue_windows=1,
        max_inflight_windows=9,
        graceful_shutdown_timeout_seconds=8.0,
    )
    runtime, store, _ = _start_with_stubs(
        tmp_path,
        config=config,
        l3_factory=lambda value: _FirstBlockingL3(
            value, started=first_started, release=first_release
        ),
    )
    max_commit_pending = 0
    try:
        assert runtime._admit_window(_window(0, stream_epoch=0))
        assert first_started.wait(2.0)
        for window_id in range(1, 121):
            started = time.monotonic()
            runtime._admit_window(_window(window_id, stream_epoch=1))
            assert time.monotonic() - started < 0.05
            status = runtime.processing_status
            max_commit_pending = max(max_commit_pending, status["commit_pending_count"])
            assert status["commit_pending_count"] <= status["commit_pending_capacity"]
            assert status["inflight_windows"] <= config.runtime.max_inflight_windows
            time.sleep(0.002)

        _wait_until(lambda: runtime.processing_status["commit_reorder_congested"])
        assert runtime.processing_status["admission_rejections"] > 0
        assert runtime.last_error is None
    finally:
        first_release.set()
        runtime.stop()

    assert max_commit_pending <= runtime.processing_status["commit_pending_capacity"]
    assert [item.window_id for item in store.record_snapshot()] == list(range(121))
    assert len(store.watermark_snapshot()) == 121
    assert runtime._result_joiner.drain_gaps() == ()
    assert runtime.processing_running is False


def test_partial_thread_start_failure_wakes_and_joins_started_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, store, _ = _runtime(tmp_path)
    original_start = threading.Thread.start

    def fail_l3_start(thread: threading.Thread) -> None:
        if thread.name == "application-runtime-l3":
            raise RuntimeError("injected l3 thread start failure")
        original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_l3_start)
    with pytest.raises(RuntimeError, match="injected l3 thread start failure"):
        runtime.start()

    assert not any(
        thread.is_alive() and thread.name.startswith("application-runtime-")
        for thread in threading.enumerate()
    )
    assert runtime.processing_running is False
    assert runtime.active is False
    assert store.stop_reasons == ["runtime_start_failed"]


def test_recording_mode_start_failure_rolls_back_session(tmp_path: Path) -> None:
    class _ModeFailingStore(_MemoryRecordingStore):
        @staticmethod
        def set_recording_mode(mode: str) -> None:
            del mode
            raise RuntimeError("injected recording mode failure")

    probe = _StageProbe()
    store = _ModeFailingStore()
    runtime = ApplicationRuntime(
        _config(),
        project_root=tmp_path,
        pipeline=_IdlePipeline(),
        serial_device=_StubSerial(),
        recording_store=store,
        layer4_engine=_StubL4(probe),
        source_probability_provider=lambda window: (),
    )
    runtime._layer2 = _StubL2(probe)
    runtime._layer3 = _StubL3(probe)

    with pytest.raises(RuntimeError, match="injected recording mode failure"):
        runtime.start()
    assert store.stop_reasons == ["runtime_start_failed"]
    assert runtime.active is False


def test_admission_audit_ranges_have_a_true_hard_bound(tmp_path: Path) -> None:
    runtime, _, _ = _runtime(tmp_path)

    for index in range(300):
        runtime._record_admission_rejection(
            SimpleNamespace(
                key=WindowKey(
                    "bounded-audit",
                    0,
                    index * 2,
                    15_360 + index * 1_920,
                )
            ),
            "injected_non_contiguous_rejection",
        )

    assert len(runtime._rejected_admission_ranges) == 255
    assert runtime.processing_status["admission_audit_overflows"] == 45
    assert runtime._processing_abort.is_set()
    assert "audit capacity exhausted" in str(runtime.processing_error)


def test_new_epoch_prunes_obsolete_preceding_gap_reason(tmp_path: Path) -> None:
    runtime, _, _ = _runtime(tmp_path)
    old = SimpleNamespace(key=WindowKey("epoch-prune", 0, 0, 15_360))
    runtime._record_admission_rejection(old, "old_epoch_drop")

    runtime._observe_stream_epoch(WindowKey("epoch-prune", 1, 1, 15_360))

    assert ("epoch-prune", 0) not in runtime._preceding_gap_reasons
    assert runtime._latest_stream_epoch_by_session == {"epoch-prune": 1}


def test_direction_stage_contract_requires_exact_ordered_outputs() -> None:
    candidates = (SimpleNamespace(theta_deg=5.0), SimpleNamespace(theta_deg=50.0))

    ApplicationRuntime._validate_direction_outputs(
        "test", candidates, (SimpleNamespace(theta_deg=5.0), SimpleNamespace(theta_deg=50.0))
    )
    with pytest.raises(RuntimeError, match="expected 2 ordered outputs"):
        ApplicationRuntime._validate_direction_outputs(
            "test", candidates, (SimpleNamespace(theta_deg=5.0),)
        )
    with pytest.raises(RuntimeError, match="does not match candidate"):
        ApplicationRuntime._validate_direction_outputs(
            "test", candidates, (SimpleNamespace(theta_deg=50.0), SimpleNamespace(theta_deg=5.0))
        )


def test_runtime_retains_recording_session_ownership_when_finalize_fails(
    tmp_path: Path,
) -> None:
    class _RetryableStopStore(_MemoryRecordingStore):
        fail_stop = True

        def stop_session(self, reason: str) -> None:
            if self.fail_stop:
                raise RuntimeError("writer still active")
            super().stop_session(reason)

    probe = _StageProbe()
    store = _RetryableStopStore()
    runtime = ApplicationRuntime(
        _config(),
        project_root=tmp_path,
        pipeline=_IdlePipeline(),
        serial_device=_StubSerial(),
        recording_store=store,
        layer4_engine=_StubL4(probe),
        source_probability_provider=lambda window: (),
    )
    runtime._recording_session_started = True

    runtime.stop()
    assert runtime._recording_session_started is True
    assert "writer still active" in str(runtime.last_error)

    store.fail_stop = False
    runtime.stop()
    assert runtime._recording_session_started is False
    assert store.stop_reasons == ["runtime_error"]


def test_stuck_input_worker_cannot_finalize_recording_until_retry(
    tmp_path: Path,
) -> None:
    class _StuckInputPipeline:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()

        @staticmethod
        def start() -> None:
            return None

        @staticmethod
        def stop() -> None:
            # Deliberately does not wake read(); this models a failed or stuck
            # device backend during shutdown.
            return None

        def read(self, timeout: float | None = None) -> None:
            del timeout
            self.entered.set()
            self.release.wait(5.0)
            return None

        @staticmethod
        def take_health_events() -> tuple[()]:
            return ()

    pipeline = _StuckInputPipeline()
    probe = _StageProbe()
    store = _MemoryRecordingStore()
    runtime = ApplicationRuntime(
        _config(graceful_shutdown_timeout_seconds=0.1),
        project_root=tmp_path,
        pipeline=pipeline,
        serial_device=_StubSerial(),
        recording_store=store,
        layer4_engine=_StubL4(probe),
        source_probability_provider=lambda window: (),
    )
    runtime._layer2 = _StubL2(probe)
    runtime._layer3 = _StubL3(probe)
    runtime.start()
    assert pipeline.entered.wait(1.0)

    runtime.stop()
    assert runtime._thread is not None and runtime._thread.is_alive()
    assert runtime._recording_session_started is True
    assert store.stop_reasons == []
    assert "input worker did not stop" in str(runtime.last_error)
    with pytest.raises(RuntimeError, match="input worker is still alive"):
        runtime.close()

    pipeline.release.set()
    _wait_until(lambda: runtime._thread is not None and not runtime._thread.is_alive())
    runtime.stop()
    assert runtime._thread is None
    assert runtime._recording_session_started is False
    assert store.stop_reasons == ["runtime_error"]
