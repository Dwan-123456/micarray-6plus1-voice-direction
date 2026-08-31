from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from common.data_types import CalibrationMetadata, IngestedAudioBlock
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
    windows = run_chunks([960] * 10)
    assert [window.decision_sample for window in windows] == [7_680, 8_640, 9_600]
    first = windows[0]
    assert first.samples.shape == (7_680, 8)
    assert np.array_equal(first.samples[:, 0], np.arange(7_680, dtype=np.float32))
    assert np.count_nonzero(first.samples[:, 7]) == 0
    assert (first.doa_start_sample, first.context_start_sample) == (5_760, 0)
    assert not first.samples.flags.writeable
    with pytest.raises(ValueError):
        first.samples.setflags(write=True)


def test_arbitrary_chunking_produces_identical_windows():
    regular = run_chunks([960] * 12)
    irregular = run_chunks([137, 2000, 1, 4022, 5360])
    assert len(regular) == len(irregular)
    for left, right in zip(regular, irregular, strict=True):
        assert left.decision_sample == right.decision_sample
        assert np.array_equal(left.samples, right.samples)


def test_window_exposes_only_calibrated_physical_history_for_music():
    samples = np.zeros((7_680, 8), np.float32)
    samples[:, :7] = np.arange(7, dtype=np.float32)[None, :]
    samples[:, 7] = 10_000.0
    config = CalibrationConfig((2.0,) * 7, (1,) * 7, (0,) * 7)
    calibrated = ChannelCalibrator(config).process(DecodedAudio(samples, 48_000, 0, 0.0))
    block = IngestCoordinator(session_id="music-input").ingest(calibrated)
    window = WindowAssembler().add(block)[0]

    assert window.samples.shape == (7_680, 8)
    assert window.physical_samples.shape == (7_680, 7)
    assert np.all(window.hardware_mix == 10_000.0)
    assert not np.any(window.physical_samples == 10_000.0)
    assert window.available_history_samples == 7_680
    history = window.physical_history(160)
    assert history.shape == (7_680, 7)
    assert window.physical_history_start_sample(160) == window.decision_sample - 7_680
    assert not history.flags.writeable
    for unavailable_ms in (200, 240, 320):
        with pytest.raises(ValueError, match="exceeds"):
            window.physical_history(unavailable_ms)
    with pytest.raises(ValueError, match="160/200/240/320"):
        window.physical_history(220)


def test_rolling_window_contract_uses_session_epoch_and_absolute_sample():
    first, second = run_chunks([960] * 9)
    assert second.is_contiguous_successor_of(first)
    assert first.rolling_state_key == ("test", 0, 7_680)
    assert second.rolling_update_start_sample == 7_680
    assert np.array_equal(second.physical_samples[-960:, 0], np.arange(7_680, 8_640))
    assert not hasattr(second, "track_id")


def test_ten_minute_contiguous_stream_never_invents_a_discontinuity():
    """30,000 20 ms blocks model ten minutes without retaining old windows."""

    coordinator = IngestCoordinator(session_id="ten-minute-contiguous")
    assembler = WindowAssembler()
    samples = np.zeros((960, 8), dtype=np.float32)
    final_window = None

    for sequence in range(30_000):
        decoded = DecodedAudio(
            samples,
            48_000,
            sequence,
            sequence * 0.02,
        )
        block = coordinator.ingest(decoded)
        assert block.stream_epoch == 0
        windows = assembler.add(block)
        assert len(windows) == (0 if sequence < 7 else 1)
        if windows:
            final_window = windows[0]
            assert final_window.stream_epoch == 0

    assert coordinator.stream_epoch == 0
    assert coordinator.discontinuities == []
    assert final_window is not None
    assert (final_window.window_id, final_window.decision_sample) == (29_992, 28_800_000)
    assert assembler.status.stream_epoch == 0


def test_health_event_and_sequence_gap_increment_epoch_only_once():
    coordinator = IngestCoordinator(session_id="test")
    first = coordinator.ingest(frame(0, 0, 960))
    event = InputHealthEvent(0, 0.02, "handoff_drop", 0, 2, 960, "queue full")
    second = coordinator.ingest(frame(2, 1920, 960), (event,))
    assert (first.stream_epoch, second.stream_epoch, second.start_sample) == (0, 1, 0)
    assert len(coordinator.discontinuities) == 1


def test_long_run_health_diagnostics_are_bounded():
    coordinator = IngestCoordinator(session_id="bounded-health")
    for event_id in range(1_100):
        coordinator.publish_health_event(
            InputHealthEvent(
                event_id, float(event_id), "handoff_drop", None, None, 960, "drop"
            )
        )
    for index in range(300):
        coordinator._reset(f"test_{index}")

    assert len(coordinator._seen_event_ids) == 1_024
    assert len(coordinator._seen_event_order) == 1_024
    assert len(coordinator.discontinuities) == 256
    assert coordinator.discontinuities[0].reason == "test_44"


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
    for sequence in range(8):
        windows.extend(assembler.add(coordinator.ingest(frame(sequence, sequence * 960, 960))))
    event = InputHealthEvent(0, 1.0, "device_restart", 15, 20, None, "restart")
    for index in range(8):
        start = index * 960
        events = (event,) if index == 0 else ()
        windows.extend(assembler.add(coordinator.ingest(frame(20 + index, start, 960), events)))
    assert [(item.window_id, item.stream_epoch, item.decision_sample) for item in windows] == [
        (0, 0, 7_680),
        (1, 1, 7_680),
    ]


def test_calibration_version_hash_change_restarts_epoch_and_window_history():
    coordinator = IngestCoordinator(session_id="calibration-boundary")
    first_config = CalibrationConfig((1.0,) * 7, (1,) * 7, (0,) * 7)
    second_config = CalibrationConfig((1.1,) * 7, (1,) * 7, (0,) * 7)
    raw_a = DecodedAudio(np.zeros((960, 8), np.float32), 48_000, 0, 0.0)
    raw_b = DecodedAudio(np.zeros((960, 8), np.float32), 48_000, 1, 0.02)
    first = coordinator.ingest(ChannelCalibrator(first_config).process(raw_a))
    second = coordinator.ingest(ChannelCalibrator(second_config).process(raw_b))
    assert first.calibration.calibration_hash != second.calibration.calibration_hash
    assert (first.stream_epoch, second.stream_epoch, second.start_sample) == (0, 1, 0)
    assert coordinator.discontinuities[-1].reason == "calibration_change"


def test_window_rejects_calibration_change_inside_one_epoch():
    assembler = WindowAssembler()
    first = IngestedAudioBlock("s", 0, 0, 960, 48_000, 0, 0.0, np.zeros((960, 8), np.float32))
    other = CalibrationMetadata(
        "verified", "other-v1", "a" * 64, "gain_polarity_integer_delay_v1", "b" * 64
    )
    assembler.add(first)
    with pytest.raises(ValueError, match="calibration boundary"):
        assembler.add(replace(first, start_sample=960, end_sample=1920, sequence_id=1, calibration=other))


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


def test_capture_recent_audio_is_bounded_to_one_second():
    capture = AudioCapture("test", "test", 48_000, 8, 960)

    assert capture._recent.maxlen == 50


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
