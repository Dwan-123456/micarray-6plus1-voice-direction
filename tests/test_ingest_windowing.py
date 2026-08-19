from __future__ import annotations

import numpy as np
import pytest

from common.data_types import IngestedAudioBlock
from ingest import IngestCoordinator
from layer1_input.interface import DecodedAudio, InputHealthEvent
from layer1_input.calibration import ChannelCalibrator
from layer1_input.configuration import CalibrationConfig
from layer1_input.continuity import CalibrationContinuityGuard, continuity_decision
from layer1_input.capture import AudioCapture
from windowing import WindowAssembler


def frame(sequence: int, start: int, size: int, *, rate: int = 48_000) -> DecodedAudio:
    physical = np.repeat(np.arange(start, start + size, dtype=np.float32)[:, None], 7, axis=1)
    values = np.column_stack((physical, np.zeros(size, np.float32)))
    native = np.zeros((size, 8), np.float32)
    return DecodedAudio(values, rate, sequence, start / 48_000, native_samples=native)


def run_chunks(sizes: list[int]):
    coordinator = IngestCoordinator(session_id="test")
    assembler = WindowAssembler()
    output, start = [], 0
    for sequence, size in enumerate(sizes):
        output.extend(assembler.add(coordinator.ingest(frame(sequence, start, size))))
        start += size
    return output


def test_first_window_and_hops_are_exact_and_readonly():
    windows = run_chunks([960] * 18)
    assert [window.decision_sample for window in windows] == [15_360, 16_320, 17_280]
    first = windows[0]
    assert first.samples.shape == (15_360, 8)
    assert np.array_equal(first.samples[:, 0], np.arange(15_360, dtype=np.float32))
    assert np.count_nonzero(first.samples[:, 7]) == 0
    assert (first.doa_start_sample, first.context_start_sample) == (13_440, 0)
    assert not first.samples.flags.writeable
    with pytest.raises(ValueError):
        first.samples.setflags(write=True)


def test_arbitrary_chunking_produces_identical_windows():
    regular = run_chunks([960] * 20)
    irregular = run_chunks([137, 2000, 1, 4022, 7000, 6040])
    assert len(regular) == len(irregular)
    for left, right in zip(regular, irregular, strict=True):
        assert left.decision_sample == right.decision_sample
        assert np.array_equal(left.samples, right.samples)


def test_health_event_and_sequence_gap_increment_epoch_only_once():
    coordinator = IngestCoordinator(session_id="test")
    first = coordinator.ingest(frame(0, 0, 960))
    event = InputHealthEvent(0, 0.02, "handoff_drop", 0, 2, 960, "queue full")
    second = coordinator.ingest(frame(2, 1920, 960), (event,))
    assert (first.stream_epoch, second.stream_epoch, second.start_sample) == (0, 1, 0)
    assert len(coordinator.discontinuities) == 1


def test_sequence_or_timestamp_break_restarts_epoch_with_current_block():
    coordinator = IngestCoordinator(session_id="test")
    coordinator.ingest(frame(0, 0, 960))
    broken = frame(3, 960, 960)
    block = coordinator.ingest(broken)
    assert (block.stream_epoch, block.start_sample, block.end_sample) == (1, 0, 960)


def test_ingested_block_owns_immutable_copy():
    source = np.zeros((10, 8), np.float32)
    block = IngestedAudioBlock("s", 0, 0, 10, 48_000, 0, 0.0, source)
    source[:] = 1
    assert np.count_nonzero(block.samples) == 0
    with pytest.raises(ValueError):
        block.samples.setflags(write=True)


def test_epoch_change_clears_window_history_and_keeps_window_ids_monotonic():
    coordinator, assembler = IngestCoordinator(session_id="test"), WindowAssembler()
    windows = []
    for sequence in range(16):
        windows.extend(assembler.add(coordinator.ingest(frame(sequence, sequence * 960, 960))))
    event = InputHealthEvent(0, 1.0, "device_restart", 15, 20, None, "restart")
    for index in range(16):
        start = index * 960
        events = (event,) if index == 0 else ()
        windows.extend(assembler.add(coordinator.ingest(frame(20 + index, start, 960), events)))
    assert [(item.window_id, item.stream_epoch, item.decision_sample) for item in windows] == [
        (0, 0, 15_360),
        (1, 1, 15_360),
    ]


def test_guard_and_coordinator_share_continuity_decision():
    previous = frame(0, 0, 4)
    current = frame(2, 4, 4)
    assert continuity_decision(previous, current).reason == "sequence_gap"
    calibrator = ChannelCalibrator(
        CalibrationConfig(
            gains=(1.0,) * 7,
            polarity=(1,) * 7,
            delay_samples=(1, 0, 0, 0, 0, 0, 0),
        )
    )
    guard = CalibrationContinuityGuard(calibrator)
    guard.start()
    guard.process(previous)
    output = guard.process(current)
    assert output.samples[0, 0] == 0  # history was reset before current block


def test_numbered_capture_handoff_is_bounded_and_reports_drop():
    capture = AudioCapture("test", "test", 48_000, 8, 2)
    receiver = capture.subscribe_numbered(maxsize=1)
    capture._stream_origin_monotonic = 10.0
    capture._callback(np.zeros((2, 8), dtype=np.int16), 2, None, None)
    capture._callback(np.ones((2, 8), dtype=np.int16), 2, None, None)
    visible = receiver.get_nowait()
    events = capture.take_health_events()
    assert visible.sequence_id == 1
    assert len(events) == 1 and events[0].kind == "handoff_drop"
    assert (events[0].last_sequence_id_before_gap, events[0].first_sequence_id_after_gap) == (None, 1)


def test_numbered_capture_coalesces_one_contiguous_overflow_burst():
    capture = AudioCapture("test", "test", 48_000, 8, 2)
    receiver = capture.subscribe_numbered(maxsize=1)
    capture._stream_origin_monotonic = 10.0
    capture._callback(np.zeros((2, 8), dtype=np.int16), 2, None, None)
    capture._callback(np.ones((2, 8), dtype=np.int16), 2, None, None)
    capture._callback(np.full((2, 8), 2, dtype=np.int16), 2, None, None)

    visible = receiver.get_nowait()
    events = capture.take_health_events()
    status = capture.status()

    assert visible.sequence_id == 2
    assert len(events) == 1
    assert events[0].kind == "handoff_drop"
    assert events[0].last_sequence_id_before_gap is None
    assert events[0].first_sequence_id_after_gap == 2
    assert events[0].lost_sample_count == 4
    assert status["handoff_drop_count"] == 2
    assert status["handoff_queue_capacity"] == 1
    assert status["handoff_queue_high_water"] == 1


def test_capture_callback_defers_rms_work_to_status(monkeypatch):
    capture = AudioCapture("test", "test", 48_000, 8, 2)
    samples = np.full((2, 8), 16_384, dtype=np.int16)

    with monkeypatch.context() as context:
        context.setattr(np, "sqrt", lambda _value: (_ for _ in ()).throw(
            AssertionError("RMS must not run in the PortAudio callback")
        ))
        capture._callback(samples, 2, None, None)

    assert capture.status()["rms_levels"] == pytest.approx([0.5] * 8)
