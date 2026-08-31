from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.runtime import ApplicationRuntime
from common.config import ProjectConfig, load_config
from common.data_types import CalibrationMetadata, DecisionWindow, IngestedAudioBlock
from common.geometry import MIC_POSITIONS_M, physical_6plus1_geometry
from layer2_source_detection import ProbabilityGateDecision, ProbabilityGateState
from source_counting import (
    IncrementalGccPhatSourceCounter,
    SourceCounterConfig,
    SourceCountSnapshot,
)


CONFIG = Path(__file__).parents[1] / "config" / "config.yaml"


def _directional_noise(
    directions: tuple[float, ...],
    *,
    levels: tuple[float, ...] | None = None,
    seed: int = 7,
    samples: int = 18_000,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frequencies = np.fft.rfftfreq(samples, 1.0 / 48_000.0)
    band = (frequencies >= 2_000.0) & (frequencies <= 4_000.0)
    output = np.zeros((samples, 8), dtype=np.float64)
    gains = levels or tuple(1.0 for _ in directions)
    for theta, gain in zip(directions, gains, strict=True):
        source = np.zeros(frequencies.size, dtype=np.complex128)
        source[band] = gain * (
            rng.normal(size=band.sum()) + 1j * rng.normal(size=band.sum())
        )
        unit = np.asarray((np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))))
        delays = -(MIC_POSITIONS_M @ unit) / 343.0
        for microphone in range(7):
            output[:, microphone] += np.fft.irfft(
                source * np.exp(-2j * np.pi * frequencies * delays[microphone]),
                n=samples,
            )
    peak = float(np.max(np.abs(output[:, :7])))
    if peak:
        output[:, :7] *= 0.60 / peak
    return np.asarray(output, dtype=np.float32)


def _window(audio: np.ndarray, index: int, *, session: str = "count") -> DecisionWindow:
    start = index * 960
    decision = 7_680 + start
    return DecisionWindow(
        session,
        0,
        index,
        decision,
        decision - 1_920,
        decision,
        decision - 7_680,
        decision,
        48_000,
        np.ascontiguousarray(audio[start : start + 7_680]),
        (index,),
        (),
    )


def _counter() -> IncrementalGccPhatSourceCounter:
    config = SourceCounterConfig.from_project(load_config(CONFIG))
    return IncrementalGccPhatSourceCounter(config)


@pytest.mark.parametrize(
    ("audio", "expected"),
    (
        (np.zeros((18_000, 8), dtype=np.float32), 0),
        (_directional_noise((70.0,)), 1),
        (_directional_noise((30.0, 210.0), levels=(1.0, 0.45)), 2),
    ),
)
def test_incremental_gcc_phat_counts_zero_one_or_two_prominent_sources(
    audio: np.ndarray,
    expected: int,
) -> None:
    counter = _counter()
    first = counter.process(_window(audio, 0), physical_6plus1_geometry())
    second = counter.process(_window(audio, 1), physical_6plus1_geometry())

    assert first.source_count is None
    assert second.source_count == expected
    assert counter.last_raw_count == expected


def test_uncorrelated_high_level_background_is_not_a_directional_source() -> None:
    rng = np.random.default_rng(91)
    audio = np.zeros((18_000, 8), dtype=np.float32)
    audio[:, :7] = 0.15 * rng.normal(size=(18_000, 7))
    counter = _counter()

    counter.process(_window(audio, 0), physical_6plus1_geometry())
    result = counter.process(_window(audio, 1), physical_6plus1_geometry())

    assert result.source_count == 0
    assert counter.last_first_peak < counter.config.first_peak_threshold


def test_nonoverlapping_directions_are_not_counted_as_simultaneous_sources() -> None:
    first = _directional_noise((30.0,), seed=3)
    second = _directional_noise((210.0,), seed=5)
    audio = first.copy()
    audio[4_800:] = second[4_800:]
    counter = _counter()

    counter.process(_window(audio, 0), physical_6plus1_geometry())
    result = counter.process(_window(audio, 1), physical_6plus1_geometry())

    assert result.source_count == 1
    assert counter.last_raw_count == 1
    assert counter.last_coactive_frames < counter.config.coactivity_required_frames


@pytest.mark.parametrize(
    "audio",
    (
        _directional_noise((20.0, 50.0), levels=(1.0, 0.45)),
        _directional_noise((0.0, 30.0), levels=(1.0, 0.70), seed=4),
    ),
)
def test_sources_closer_than_angular_resolution_are_counted_as_one(
    audio: np.ndarray,
) -> None:
    counter = _counter()

    counter.process(_window(audio, 0), physical_6plus1_geometry())
    result = counter.process(_window(audio, 1), physical_6plus1_geometry())

    assert result.source_count == 1
    assert counter.last_raw_count == 1


def test_normal_updates_transform_only_two_new_stft_frames() -> None:
    audio = _directional_noise((45.0,), samples=20_000)
    counter = _counter()

    counter.process(_window(audio, 0), physical_6plus1_geometry())
    assert (counter.last_update_kind, counter.last_added_frames, counter.last_removed_frames) == (
        "rebuilt",
        15,
        0,
    )

    counter.process(_window(audio, 1), physical_6plus1_geometry())
    assert (counter.last_update_kind, counter.last_added_frames, counter.last_removed_frames) == (
        "advanced",
        2,
        2,
    )

    # A skipped L2 window is caught up from only the six missing new frames.
    counter.process(_window(audio, 4), physical_6plus1_geometry())
    assert (counter.last_update_kind, counter.last_added_frames, counter.last_removed_frames) == (
        "advanced",
        6,
        6,
    )


@pytest.mark.parametrize("latest_index", range(1, 8))
def test_incremental_state_matches_rebuild_for_latest_context(latest_index: int) -> None:
    audio = _directional_noise((25.0, 205.0), levels=(1.0, 0.55), samples=20_000)
    rolling = _counter()
    rebuilt = _counter()

    rolling.process(_window(audio, 0), physical_6plus1_geometry())
    rolling.process(_window(audio, latest_index), physical_6plus1_geometry())
    rebuilt.process(_window(audio, latest_index), physical_6plus1_geometry())

    np.testing.assert_allclose(rolling._cross_sum, rebuilt._cross_sum, rtol=1e-12, atol=1e-12)
    assert rolling._power_sum == pytest.approx(rebuilt._power_sum, rel=1e-12, abs=1e-12)
    assert rolling.last_raw_count == rebuilt.last_raw_count
    assert (rolling.last_added_frames, rolling.last_removed_frames) == (
        latest_index * 2,
        latest_index * 2,
    )


def test_v14_config_without_source_counting_defaults_to_enabled_counting() -> None:
    payload = load_config(CONFIG).model_dump()
    payload.pop("source_counting")

    legacy = ProjectConfig.model_validate(payload)

    assert legacy.source_counting.enabled
    assert not legacy.source_counting.music_order_from_source_count
    assert SourceCounterConfig.from_project(legacy).enabled


def test_runtime_publishes_each_window_only_to_the_single_l2_worker() -> None:
    runtime = ApplicationRuntime(load_config(CONFIG), project_root=CONFIG.parent.parent)
    samples = np.zeros((7_680, 8), dtype=np.float32)
    block = IngestedAudioBlock(
        "fanout",
        0,
        0,
        len(samples),
        48_000,
        0,
        0.0,
        samples,
        calibration=CalibrationMetadata.unverified_identity(),
    )

    runtime._publish_block(block)

    latest = runtime.latest_windows.get_nowait()
    assert runtime._l2_windows.get_nowait().window is latest
    assert not hasattr(runtime, "_source_count_windows")


def _gate(window: DecisionWindow, *, opened: bool) -> ProbabilityGateDecision:
    value = 0.9 if opened else 0.1
    return ProbabilityGateDecision(
        window.session_id,
        window.stream_epoch,
        window.window_id,
        window.decision_sample,
        "test_gate",
        ProbabilityGateState.OPEN if opened else ProbabilityGateState.CLOSED,
        value,
        value,
        value,
        0.6,
        0,
        opened,
        "open" if opened else "closed",
    )


class _PlannedCounter:
    def __init__(self, values: list[int | None | Exception]) -> None:
        self.values = list(values)
        self.calls = 0
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def process(self, window: DecisionWindow, geometry: object) -> SourceCountSnapshot:
        del geometry
        self.calls += 1
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return SourceCountSnapshot(
            window.session_id,
            window.stream_epoch,
            window.window_id,
            window.decision_sample,
            value,
            time.monotonic(),
        )


def test_gate_and_controls_resolve_same_window_count_before_music() -> None:
    runtime = ApplicationRuntime(load_config(CONFIG), project_root=CONFIG.parent.parent)
    audio = _directional_noise((30.0,), samples=20_000)
    windows = tuple(_window(audio, index) for index in range(5))
    counter = _PlannedCounter([0, None, 0, 1, 2])
    runtime._source_counter = counter

    closed, order, reason, elapsed = runtime._prepare_source_count_plan(
        windows[0],
        _gate(windows[0], opened=False),
        enabled=True,
        follow_order=True,
        control_revision=0,
    )
    assert closed.source_count == 0
    assert (order, reason) == (None, "closed")
    assert elapsed >= 0.0
    assert counter.calls == 1

    expected = (
        (None, 1, None),
        (0, 1, None),
        (1, 1, None),
        (2, 2, None),
    )
    for window, values in zip(windows[1:], expected, strict=True):
        snapshot, order, reason, _elapsed = runtime._prepare_source_count_plan(
            window,
            _gate(window, opened=True),
            enabled=True,
            follow_order=True,
            control_revision=0,
        )
        assert (snapshot.source_count, order, reason) == values
    assert counter.calls == 5


def test_gate_closed_keeps_source_count_state_advancing() -> None:
    runtime = ApplicationRuntime(load_config(CONFIG), project_root=CONFIG.parent.parent)
    audio = _directional_noise((30.0,), samples=20_000)
    first, second = _window(audio, 0), _window(audio, 1)
    counter = _PlannedCounter([None, 1])
    runtime._source_counter = counter

    first_result = runtime._prepare_source_count_plan(
        first,
        _gate(first, opened=False),
        enabled=True,
        follow_order=True,
        control_revision=0,
    )
    second_result = runtime._prepare_source_count_plan(
        second,
        _gate(second, opened=False),
        enabled=True,
        follow_order=True,
        control_revision=0,
    )

    assert first_result[0].source_count is None
    assert second_result[0].source_count == 1
    assert first_result[1:3] == second_result[1:3] == (None, "closed")
    assert counter.calls == 2
    assert counter.resets == 1


def test_disabled_source_count_stops_processing_and_uses_fixed_order_two() -> None:
    runtime = ApplicationRuntime(load_config(CONFIG), project_root=CONFIG.parent.parent)
    audio = _directional_noise((30.0,), samples=20_000)
    window = _window(audio, 0)
    counter = _PlannedCounter([1])
    runtime._source_counter = counter

    snapshot, order, reason, elapsed = runtime._prepare_source_count_plan(
        window,
        _gate(window, opened=True),
        enabled=False,
        follow_order=False,
        control_revision=1,
    )

    assert snapshot.source_count is None
    assert (order, reason, elapsed) == (2, None, 0.0)
    assert counter.calls == 0


def test_fixed_order_two_survives_count_warming_or_fault_without_main_error() -> None:
    runtime = ApplicationRuntime(load_config(CONFIG), project_root=CONFIG.parent.parent)
    audio = _directional_noise((45.0,), samples=20_000)
    first, second = _window(audio, 0), _window(audio, 1)
    runtime._source_counter = _PlannedCounter([None, RuntimeError("count-only")])
    runtime.last_error = "main-chain-intact"

    warming = runtime._prepare_source_count_plan(
        first,
        _gate(first, opened=True),
        enabled=True,
        follow_order=False,
        control_revision=0,
    )
    fault = runtime._prepare_source_count_plan(
        second,
        _gate(second, opened=True),
        enabled=True,
        follow_order=False,
        control_revision=0,
    )

    assert warming[1:3] == (2, None)
    assert fault[1:3] == (2, None)
    assert runtime.last_error == "main-chain-intact"
    assert "count-only" in (runtime.source_count_last_error or "")
    assert runtime.performance_snapshot["source_count_frames_per_second"] == 1
    assert runtime.performance_snapshot["source_count_faults_per_second"] == 1


def test_follow_order_maps_warming_or_count_fault_to_one() -> None:
    runtime = ApplicationRuntime(load_config(CONFIG), project_root=CONFIG.parent.parent)
    audio = _directional_noise((45.0,), samples=20_000)
    first, second = _window(audio, 0), _window(audio, 1)
    runtime._source_counter = _PlannedCounter([None, RuntimeError("count-only")])

    warming = runtime._prepare_source_count_plan(
        first,
        _gate(first, opened=True),
        enabled=True,
        follow_order=True,
        control_revision=0,
    )
    fault = runtime._prepare_source_count_plan(
        second,
        _gate(second, opened=True),
        enabled=True,
        follow_order=True,
        control_revision=0,
    )

    assert warming[1:3] == (1, None)
    assert fault[1:3] == (1, None)


def test_manual_source_count_and_music_follow_controls_are_atomic() -> None:
    runtime = ApplicationRuntime(load_config(CONFIG), project_root=CONFIG.parent.parent)
    assert runtime.source_counting_enabled
    assert not runtime.music_order_follows_source_count
    source_revision = runtime._source_count_control_revision

    assert runtime.set_music_order_follows_source_count(True)
    assert runtime.music_order_follows_source_count
    assert runtime.current_music_effective_order == 1
    assert runtime._source_count_control_revision == source_revision
    assert not runtime.set_source_counting_enabled(False)
    assert not runtime.source_counting_enabled
    assert not runtime.music_order_follows_source_count
    assert runtime.current_music_effective_order == 2
    with pytest.raises(ValueError, match="enable source counting"):
        runtime.set_music_order_follows_source_count(True)
    with pytest.raises(ValueError, match="fixed MUSIC order is 2"):
        runtime.set_music_effective_order_limit(1)


def test_music_follow_toggle_does_not_reset_continuous_source_count_state() -> None:
    runtime = ApplicationRuntime(load_config(CONFIG), project_root=CONFIG.parent.parent)
    audio = _directional_noise((30.0,), samples=20_000)
    first, second = _window(audio, 0), _window(audio, 1)
    counter = _PlannedCounter([None, 1])
    runtime._source_counter = counter

    first_result = runtime._prepare_source_count_plan(
        first,
        _gate(first, opened=True),
        enabled=True,
        follow_order=False,
        control_revision=runtime._source_count_control_revision,
    )
    runtime.set_music_order_follows_source_count(True)
    second_result = runtime._prepare_source_count_plan(
        second,
        _gate(second, opened=True),
        enabled=True,
        follow_order=True,
        control_revision=runtime._source_count_control_revision,
    )

    assert first_result[1] == 2
    assert second_result[0].source_count == 1
    assert second_result[1] == 1
    assert counter.calls == 2
    assert counter.resets == 1


def test_source_count_snapshot_rejects_non_contract_counts() -> None:
    for invalid in (3, True, 1.0):
        with pytest.raises(ValueError, match="0, 1, 2"):
            SourceCountSnapshot("session", 0, 0, 7_680, invalid, 1.0)
    for invalid_time in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="publication time"):
            SourceCountSnapshot("session", 0, 0, 7_680, 1, invalid_time)


def test_runtime_starts_and_stops_source_counting_inside_l2_worker() -> None:
    class ExhaustedPipeline:
        source = SimpleNamespace(exhausted=True)

        def start(self) -> None:
            pass

        def read(self, timeout: float):
            del timeout
            return None

        def take_health_events(self) -> tuple[()]:
            return ()

        def stop(self) -> None:
            pass

    class SerialStub:
        @staticmethod
        def write(packet: bytes) -> int:
            return len(packet)

        @staticmethod
        def stop() -> None:
            pass

    runtime = ApplicationRuntime(
        load_config(CONFIG),
        project_root=CONFIG.parent.parent,
        pipeline=ExhaustedPipeline(),
        serial_device=SerialStub(),
    )
    runtime.start()
    deadline = time.monotonic() + 2.0
    while not runtime._input_done.is_set() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert not runtime.running
    assert runtime.processing_running
    assert runtime.source_count_running
    runtime.stop()
    assert not runtime.processing_running
    assert not runtime.source_count_running


def test_stop_retains_timed_out_l2_worker_and_prevents_restart() -> None:
    runtime = ApplicationRuntime(load_config(CONFIG), project_root=CONFIG.parent.parent)
    runtime.pipeline = SimpleNamespace(stop=lambda: None)
    release = threading.Event()
    worker = threading.Thread(target=release.wait, daemon=True)
    runtime._l2_thread = worker
    worker.start()
    try:
        runtime.stop(timeout=0.001)

        assert runtime._l2_thread is worker
        assert runtime.processing_running
        assert "shutdown timeout" in (runtime.last_error or "")
        runtime.start()
        assert runtime._l2_thread is worker
    finally:
        release.set()
        worker.join(1.0)
        runtime.stop(timeout=0.1)

    assert runtime._l2_thread is None
