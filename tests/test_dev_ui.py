from __future__ import annotations

import time
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.runtime import ApplicationRuntime
from common.config import load_config
from common.data_types import (
    CandidateDirection,
    IngestedAudioBlock,
    PipelineStatus,
    SpatialResponse,
    TrackedDirection,
)
from common.geometry import MIC_POSITIONS_M
from gui.dev_test_ui.aggregator import DevUiAggregator, PerformanceTracker
from gui.dev_test_ui.contracts import L1MeterSnapshot, TrackedAudioSnapshot
from gui.dev_test_ui.settings import DevUiSettings
from layer1_input.interface import DecodedAudio
from layer1_input.sources import LiveSipeedSource, WavAudioSource
from layer2_source_detection.iterative import CandidateSearchDiagnostics
from layer2_source_detection.probability_gate import (
    ProbabilityGateDecision,
    ProbabilityGateState,
    SourceProbability20ms,
    SourceProbabilityState,
)
from layer4_voice_classifier import Layer4Result, ModelPrediction, VoiceDetection


CONFIG = Path(__file__).parents[1] / "config/config.yaml"


def _write_test_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(samples.shape[1])
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


class StubPipeline:
    def __init__(self, frames):
        self.frames = iter(frames)

    def start(self):
        pass

    def stop(self):
        pass

    def read(self, timeout=None):
        del timeout
        try:
            return next(self.frames)
        except StopIteration:
            time.sleep(0.001)
            return None

    def take_health_events(self):
        return ()


class StubSerial:
    def write(self, packet):
        return len(packet)


def _aggregator_meter(
    *, session_id: str = "aggregator-test", epoch: int = 0, end_sample: int = 15_360
) -> L1MeterSnapshot:
    return L1MeterSnapshot(
        session_id,
        epoch,
        end_sample,
        end_sample // 960,
        np.full(8, -40.0, np.float32),
        np.full(8, -20.0, np.float32),
        np.zeros(8, np.bool_),
        "unknown",
        "idle",
    )


def _open_l2_result(
    *, session_id: str = "aggregator-test", epoch: int = 0,
    window_id: int = 0, revision: int = 3,
):
    decision_sample = 15_360 + window_id * 960
    raw = np.zeros(360, np.float32)
    normalized = np.zeros(360, np.float32)
    raw[30], normalized[30] = 0.2, 0.8
    response = SpatialResponse(
        session_id, epoch, window_id, decision_sample,
        decision_sample - 1_920, decision_sample,
        np.arange(360, dtype=np.float32), raw, normalized,
    )
    candidate = CandidateDirection(
        session_id, epoch, window_id, decision_sample,
        decision_sample - 1_920, decision_sample, 30.0, 0.2, 0.8,
    )
    gate = ProbabilityGateDecision(
        session_id, epoch, window_id, decision_sample,
        "mean_2x20ms_v1", ProbabilityGateState.OPEN,
        0.70, 0.80, 0.75, 0.60, 2, True, "probability_at_or_above_threshold",
    )
    diagnostics = CandidateSearchDiagnostics(
        "single_pass", "srp_phat_single_pass_v1", revision,
        1, "single_pass", 1.0,
    )
    return response, (candidate,), gate, diagnostics


def _l4_result(
    *, session_id: str = "aggregator-test", epoch: int = 0, window_id: int = 0
) -> Layer4Result:
    decision_sample = 15_360 + window_id * 960
    probability = np.asarray([0.9], dtype=np.float32)
    return Layer4Result(
        (
            VoiceDetection(
                session_id, epoch, window_id, decision_sample,
                30.0, 0.9, True, "test-model",
            ),
        ),
        (ModelPrediction("test-model", probability, 1.0, {}),),
        "test-model",
        0.7,
    )


def test_operator_settings_round_trip_without_overwriting_each_other(tmp_path):
    settings = DevUiSettings(tmp_path)
    assert settings.load_direction_threshold(.35) == .35
    assert settings.load_iterative_peak_search_enabled() is False
    assert settings.load_direction_kalman_enabled() is False
    assert settings.load_direction_id_tracking_enabled() is False
    assert settings.load_gate_probability_threshold(0.60) == 0.60
    assert settings.load_l1_pre_denoise_enabled(False) is False

    settings.save_direction_threshold(.67)
    assert settings.save_iterative_peak_search_enabled(True) is True
    assert settings.load_direction_threshold(.35) == .67
    settings.save_direction_threshold(.42)
    settings.save_direction_kalman_enabled(True)
    settings.save_direction_id_tracking_enabled(True)
    assert settings.save_direction_kalman_q_scale(1.2) == 1.2
    assert settings.save_direction_kalman_r_scale(0.8) == 0.8
    assert settings.save_gate_probability_threshold(0.73) == 0.73
    assert settings.save_l1_pre_denoise_enabled(True) is True

    loaded = DevUiSettings(tmp_path)
    assert loaded.load_direction_threshold(.35) == .42
    assert loaded.load_iterative_peak_search_enabled() is True
    assert loaded.load_direction_kalman_enabled() is True
    assert loaded.load_direction_id_tracking_enabled() is True
    assert loaded.load_direction_kalman_q_scale(1.0) == 1.2
    assert loaded.load_direction_kalman_r_scale(1.0) == 0.8
    assert loaded.load_gate_probability_threshold(0.60) == 0.73
    assert loaded.load_l1_pre_denoise_enabled(False) is True

    payload = loaded.path.read_text(encoding="utf-8")
    assert '"layer2_direction_threshold": 0.42' in payload
    assert '"layer2_iterative_peak_search_enabled": true' in payload


def test_operator_settings_reject_invalid_values(tmp_path):
    settings = DevUiSettings(tmp_path)
    with pytest.raises(ValueError):
        settings.save_direction_kalman_q_scale(0.03)
    with pytest.raises(ValueError, match="Gate probability threshold"):
        settings.save_gate_probability_threshold(1.01)
    with pytest.raises(ValueError, match="must be bool"):
        settings.save_l1_pre_denoise_enabled(1)


def test_kalman_q_r_control_stages_with_buttons_and_applies_explicitly(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from gui.dev_test_ui.panels import KalmanNoiseScaleControl

    app = QApplication.instance() or QApplication([])
    control = KalmanNoiseScaleControl("Q倍率", 1.0)
    applied = []
    control.apply_requested.connect(applied.append)
    control.plus_button.click()
    assert control.staged_value == 1.1
    assert applied == []
    control.apply_button.click()
    assert applied == [1.1]
    control.commit(1.1, pending=True)
    assert "1.10" in control.value_label.text()
    control.deleteLater()
    app.processEvents()


def test_performance_tracker_resets_on_epoch_and_observes_rate():
    tracker = PerformanceTracker(sample_rate=48_000, required_samples=15_360, window_count=500, rate_seconds=5)
    samples = np.zeros((960, 8), np.float32)
    tracker.add_block(IngestedAudioBlock("s", 0, 0, 960, 48_000, 0, 0.0, samples), 10.0)
    tracker.add_block(IngestedAudioBlock("s", 0, 960, 1920, 48_000, 1, .02, samples), 11.0)
    tracker.add_timing("s", 0, 1, 4.0, 6.0, l2_ms=1.0, l3_ms=2.0, l4_ms=3.0)
    snap = tracker.snapshot(PipelineStatus("running", "s", 0, 15_360, 15_360, "Ready"))
    assert snap.observed_sample_rate_hz == 960
    assert snap.compute_time_ms_current == 4.0 and snap.latency_ms_current == 6.0
    assert snap.l2_time_ms_last_second_avg == 1.0
    assert snap.l3_time_ms_last_second_avg == 2.0
    assert snap.l4_time_ms_last_second_avg == 3.0
    assert snap.l2_refresh_hz_last_second == 1.0
    assert snap.l3_refresh_hz_last_second == 1.0
    assert snap.l4_refresh_hz_last_second == 1.0
    reset = tracker.snapshot(PipelineStatus("warming_up", "s", 1, 960, 15_360, "Warming"))
    assert reset.observed_sample_rate_hz is None and reset.compute_time_ms_current is None


def test_performance_tracker_averages_only_completed_stages_from_last_second(monkeypatch):
    import gui.dev_test_ui.aggregator as aggregator_module

    tracker = PerformanceTracker(sample_rate=48_000, required_samples=15_360, window_count=500, rate_seconds=5)
    tracker.add_timing(
        "s", 0, 1, 10.0, 10.0,
        l2_ms=1.0, l3_ms=4.0, l4_ms=None, completed_monotonic=10.0,
    )
    tracker.add_timing(
        "s", 0, 2, 20.0, 20.0,
        l2_ms=3.0, l3_ms=8.0, l4_ms=5.0, completed_monotonic=10.8,
    )
    monkeypatch.setattr(aggregator_module, "monotonic", lambda: 10.9)
    status = PipelineStatus("running", "s", 0, 15_360, 15_360, "Ready")
    snapshot = tracker.snapshot(status)
    assert snapshot.l2_time_ms_last_second_avg == 2.0
    assert snapshot.l3_time_ms_last_second_avg == 6.0
    assert snapshot.l4_time_ms_last_second_avg == 5.0
    assert snapshot.l2_refresh_hz_last_second == 2.0
    assert snapshot.l3_refresh_hz_last_second == 2.0
    assert snapshot.l4_refresh_hz_last_second == 1.0

    monkeypatch.setattr(aggregator_module, "monotonic", lambda: 11.1)
    snapshot = tracker.snapshot(status)
    assert snapshot.l2_time_ms_last_second_avg == 3.0
    assert snapshot.l3_time_ms_last_second_avg == 8.0
    assert snapshot.l4_time_ms_last_second_avg == 5.0
    assert snapshot.l2_refresh_hz_last_second == 1.0
    assert snapshot.l3_refresh_hz_last_second == 1.0
    assert snapshot.l4_refresh_hz_last_second == 1.0


def test_ui_aggregator_clears_old_and_ignores_late_l2_results_on_epoch_change():
    performance = PerformanceTracker(sample_rate=48_000, required_samples=15_360, window_count=10, rate_seconds=5)
    aggregator = DevUiAggregator(performance)

    def meter(epoch: int) -> L1MeterSnapshot:
        return L1MeterSnapshot(
            "epoch-test", epoch, 15_360, epoch, np.full(8, -40.0, np.float32),
            np.full(8, -20.0, np.float32), np.zeros(8, np.bool_), "unknown", "idle",
        )

    old_status = PipelineStatus("running", "epoch-test", 0, 15_360, 15_360, "Ready")
    aggregator.update_l1(meter(0), old_status)
    gate = ProbabilityGateDecision(
        "epoch-test", 0, 0, 15_360, "mean_2x20ms_v1",
        ProbabilityGateState.WARMING_UP, None, None, None, 0.60, 0, False,
        "upstream_probability_warming_up",
    )
    aggregator.update_srp(
        None, (), "BACKGROUND_ONLY", gate_decision=gate,
        gate_threshold=0.60, gate_config_revision=0,
        direction_threshold=0.35, iterative_peak_search_enabled=False,
        direction_kalman_enabled=False, direction_id_tracking_enabled=False,
        direction_kalman_q_scale=1.0, direction_kalman_r_scale=1.0,
        scan_config_revision=0,
    )

    new_status = PipelineStatus("warming_up", "epoch-test", 1, 960, 15_360, "Warming")
    current = aggregator.update_l1(meter(1), new_status)
    assert current.gate_decision is None
    assert "WARMING_UP" in current.missing_reasons["srp"]

    late = aggregator.update_srp(
        None, (), "BACKGROUND_ONLY", gate_decision=gate,
        gate_threshold=0.60, gate_config_revision=0,
        direction_threshold=0.35, iterative_peak_search_enabled=False,
        direction_kalman_enabled=False, direction_id_tracking_enabled=False,
        direction_kalman_q_scale=1.0, direction_kalman_r_scale=1.0,
        scan_config_revision=0,
    )
    assert late.pipeline_status.stream_epoch == 1
    assert late.gate_decision is None


def test_gate_unavailable_preserves_l3_listening_rows_across_epoch_recovery():
    performance = PerformanceTracker(
        sample_rate=48_000, required_samples=15_360, window_count=10, rate_seconds=5,
    )
    aggregator = DevUiAggregator(performance)
    status0 = PipelineStatus(
        "running", "aggregator-test", 0, 15_360, 15_360, "Ready",
    )
    aggregator.update_l1(_aggregator_meter(), status0)
    response, candidates, gate, diagnostics = _open_l2_result(window_id=0)
    aggregator.update_srp(
        response,
        candidates,
        gate_decision=gate,
        search_diagnostics=diagnostics,
        gate_threshold=0.60,
        gate_config_revision=2,
        direction_threshold=0.35,
        iterative_peak_search_enabled=False,
        direction_kalman_enabled=False,
        direction_id_tracking_enabled=False,
        direction_kalman_q_scale=1.0,
        direction_kalman_r_scale=1.0,
        scan_config_revision=3,
    )
    row0 = TrackedAudioSnapshot(
        "aggregator-test", 0, 7, "active", 30.0, 0.8, 9_600,
    )
    frame = aggregator.update_l3((), tracked_audio=(row0,))
    assert frame.tracked_audio == (row0,)

    # Gate unavailable replaces window-scoped SRP/L3 previews, but must not
    # clear the session-scoped listening cache rows.
    unavailable = aggregator.update_srp(None, (), "GATE UNAVAILABLE")
    assert unavailable.spatial_response is None
    assert unavailable.tracked_audio == (row0,)

    # Runtime re-projects the retained rows with the new epoch identity while
    # L2 warms up, keeping DevUiFrame identities internally consistent.
    status1 = PipelineStatus(
        "warming_up", "aggregator-test", 1, 960, 15_360, "Warming",
    )
    aggregator.update_l1(_aggregator_meter(epoch=1, end_sample=960), status1)
    row1 = TrackedAudioSnapshot(
        "aggregator-test", 1, 7, "ended", 30.0, 0.8, 9_600,
    )
    warming = aggregator.update_l3(
        (), "WARMING_UP: waiting for Layer 2", tracked_audio=(row1,),
    )
    assert warming.pipeline_status.stream_epoch == 1
    assert warming.tracked_audio == (row1,)


def test_ui_aggregator_ignores_late_l4_from_old_epoch_and_old_window_without_side_effects():
    performance = PerformanceTracker(
        sample_rate=48_000, required_samples=15_360, window_count=10, rate_seconds=5
    )
    aggregator = DevUiAggregator(performance)
    status = PipelineStatus("running", "aggregator-test", 0, 15_360, 15_360, "Ready")
    aggregator.update_l1(_aggregator_meter(), status)

    def publish_l2(response, candidates, gate, diagnostics):
        return aggregator.update_srp(
            response,
            candidates,
            gate_decision=gate,
            search_diagnostics=diagnostics,
            gate_threshold=0.60,
            gate_config_revision=2,
            direction_threshold=0.35,
            iterative_peak_search_enabled=False,
            direction_kalman_enabled=False,
            direction_id_tracking_enabled=False,
            direction_kalman_q_scale=1.0,
            direction_kalman_r_scale=1.0,
            scan_config_revision=3,
        )

    response0, candidates0, gate0, diagnostics0 = _open_l2_result(window_id=0)
    publish_l2(response0, candidates0, gate0, diagnostics0)
    accepted = aggregator.update_l4(_l4_result(window_id=0))
    assert accepted.l4_result is not None

    response1, candidates1, gate1, diagnostics1 = _open_l2_result(window_id=1)
    publish_l2(response1, candidates1, gate1, diagnostics1)
    late_window = aggregator.update_l4(_l4_result(window_id=0))
    assert late_window.spatial_response is response1
    assert late_window.l4_result is None

    epoch1 = PipelineStatus("warming_up", "aggregator-test", 1, 960, 15_360, "Warming")
    aggregator.update_l1(_aggregator_meter(epoch=1, end_sample=960), epoch1)
    late_epoch = aggregator.update_l4(_l4_result(epoch=0, window_id=1))
    assert late_epoch.pipeline_status.stream_epoch == 1
    assert late_epoch.spatial_response is None
    assert late_epoch.l4_result is None


def test_runtime_connects_probability_gate_to_ui_during_upstream_warmup(tmp_path):
    rng = np.random.default_rng(7)
    frames = [DecodedAudio(rng.normal(0, .1, (960, 8)).astype(np.float32), 48_000, index, index * .02) for index in range(16)]
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}), project_root=tmp_path, pipeline=StubPipeline(frames), serial_device=StubSerial()
    )
    runtime.start()
    deadline, frame = time.monotonic() + 3, None
    while time.monotonic() < deadline:
        if not runtime.latest_dev_ui.empty():
            candidate = runtime.latest_dev_ui.get_nowait()
            if candidate.gate_decision is not None:
                frame = candidate
                break
        time.sleep(.005)
    runtime.stop()
    assert frame is not None
    assert frame.gate_decision is not None
    assert frame.gate_decision.window_id == 0
    assert frame.gate_decision.state is ProbabilityGateState.WARMING_UP
    assert frame.gate_decision.probability_40ms is None
    assert frame.performance.latency_ms_current >= frame.performance.compute_time_ms_current >= 0


def test_gate_blocked_frame_clears_previous_polar_snapshot(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from gui.dev_test_ui.app import build_window

    config_path = tmp_path / "config" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    app, window = build_window(config_path)
    try:
        window.srp_polar._snapshot = object()
        window._last_rendered_window = ("old-session", 0, 7)

        blocked = SimpleNamespace(
            previews=(), gate_decision=None,
            l1=None, spatial_response=None,
            spatial_published_monotonic=None, candidates=(), search_diagnostics=None,
            gate_config_revision=None,
            missing_reasons={"srp": "UNAVAILABLE: upstream_probability_warming_up"},
            performance=None,
        )
        window._render_frame(blocked)

        assert window.srp_polar._snapshot is None
        assert window._last_rendered_window is None
    finally:
        window.close()
        app.processEvents()


def test_l4_panel_retains_completed_result_across_dropped_frames_until_stale(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from gui.dev_test_ui.app import build_window

    app, window = build_window(CONFIG)
    try:
        detection = SimpleNamespace(
            window_id=11,
            theta_deg=30.0,
            probability=0.9,
        )
        result = SimpleNamespace(
            detections=(detection,),
            primary_model_id="test-model",
            threshold=0.7,
        )
        completed = SimpleNamespace(l4_result=result, missing_reasons={})
        dropped = SimpleNamespace(
            l4_result=None,
            missing_reasons={"cnn": "L4 DROPPED: overload"},
        )

        # A dropped ordered commit before/after a completed immediate result
        # must not erase the useful result during the configured retention.
        window._update_l4_panel(dropped, completed, now=10.0)
        assert window.cnn_panel._result is result
        window._update_l4_panel(dropped, None, now=10.1)
        assert window.cnn_panel._result is result

        window._update_l4_panel(dropped, None, now=10.6)
        assert window.cnn_panel._result is None
        assert "STALE" in window.cnn_panel.summary.text()
    finally:
        window.close()
        app.processEvents()


def test_l4_panel_clears_old_epoch_cache_and_ignores_old_immediate_frame(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from gui.dev_test_ui.app import build_window

    app, window = build_window(CONFIG)
    try:
        result = SimpleNamespace(
            detections=(), primary_model_id="test-model", threshold=0.7,
        )
        old_status = SimpleNamespace(session_id="stream", stream_epoch=0)
        new_status = SimpleNamespace(session_id="stream", stream_epoch=1)
        completed_old = SimpleNamespace(
            l4_result=result, missing_reasons={}, pipeline_status=old_status,
        )
        warming_new = SimpleNamespace(
            l4_result=None,
            missing_reasons={"cnn": "WARMING_UP"},
            pipeline_status=new_status,
        )

        window._update_l4_panel(None, completed_old, now=10.0)
        assert window._last_l4_frame is completed_old
        window._update_l4_panel(warming_new, completed_old, now=10.1)

        assert window._last_l4_frame is None
        assert window.cnn_panel._result is None
        assert "WARMING" in window.cnn_panel.summary.text()
    finally:
        window.close()
        app.processEvents()


def test_gate_probability_slider_emits_runtime_value(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from gui.dev_test_ui.panels import GateProbabilityThresholdControl

    app = QApplication.instance() or QApplication([])
    control = GateProbabilityThresholdControl(0.60)
    emitted = []
    control.threshold_changed.connect(emitted.append)
    control.slider.setValue(73)
    assert emitted == [0.73]
    assert control.value == 0.73
    assert control.value_label.text() == "0.73"
    control.close()
    app.processEvents()


def test_gate_probability_threshold_restores_and_persists_from_test_ui(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from gui.dev_test_ui.app import build_window

    config_path = tmp_path / "config" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    settings = DevUiSettings(tmp_path)
    settings.save_gate_probability_threshold(0.73)

    app, window = build_window(config_path)
    try:
        assert window._runtime.gate_probability_threshold == 0.73
        window.gate_threshold.slider.setValue(68)
        app.processEvents()
        assert window._runtime.gate_probability_threshold == 0.68
        assert DevUiSettings(tmp_path).load_gate_probability_threshold() == 0.68
    finally:
        window.close()
        app.processEvents()


def test_l3_mode_button_switches_runtime_before_capture(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from gui.dev_test_ui.app import build_window

    config_path = tmp_path / "config" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    app, window = build_window(config_path)
    try:
        assert window._runtime.l3_processing_mode == "optimized"
        assert window.bf_panel.mode_switch.text() == "BF：优化算法"
        window.bf_panel.mode_switch.click()
        app.processEvents()
        assert window._runtime.l3_processing_mode == "ds_baseline"
        assert window.bf_panel.mode_switch.text() == "BF：DS基线"
        assert window.bf_panel.mode_switch.isEnabled()
        window.bf_panel.mode_switch.click()
        app.processEvents()
        assert window._runtime.l3_processing_mode == "constant_beamwidth_baseline"
        assert window.bf_panel.mode_switch.text() == "BF：恒定波束30°"
        window.bf_panel.mode_switch.click()
        app.processEvents()
        assert window._runtime.l3_processing_mode == "optimized"
    finally:
        window.close()
        app.processEvents()


def test_l1_l2_l3_outputs_render_in_test_ui(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from gui.dev_test_ui.app import build_window

    sample_rate, sample_count = 48_000, 18 * 960
    theta = np.deg2rad(45.0)
    direction = np.array((np.cos(theta), np.sin(theta)))
    delays = -(MIC_POSITIONS_M @ direction) / 343.0
    time_axis = np.arange(sample_count, dtype=np.float64) / sample_rate
    samples = np.zeros((sample_count, 7), dtype=np.float64)
    for frequency in np.arange(600.0, 3_901.0, 137.0):
        samples += np.cos(2.0 * np.pi * frequency * (time_axis[:, None] - delays[None, :]))
    samples = np.asarray(0.1 * samples / np.max(np.abs(samples)), dtype=np.float32)
    frames = []
    for start in range(0, sample_count, 960):
        physical = samples[start:start + 960]
        native = np.column_stack(
            (physical[:, :6], np.zeros(len(physical), np.float32), physical[:, 6])
        )
        logical = np.column_stack((physical, np.zeros(len(physical), np.float32)))
        frames.append(
            DecodedAudio(
                logical, sample_rate, start // 960, start / sample_rate,
                native_samples=native,
            )
        )
    runtime_config = load_config(CONFIG, environ={})

    def open_probabilities(window):
        return (
            SourceProbability20ms(
                window.session_id, window.stream_epoch, window.doa_start_sample,
                window.doa_start_sample + 960, 0.9, SourceProbabilityState.READY, "test",
            ),
            SourceProbability20ms(
                window.session_id, window.stream_epoch, window.doa_start_sample + 960,
                window.doa_end_sample, 0.9, SourceProbabilityState.READY, "test",
            ),
        )

    runtime = ApplicationRuntime(
        runtime_config, project_root=tmp_path,
        pipeline=StubPipeline(frames), serial_device=StubSerial(),
        source_probability_provider=open_probabilities,
    )
    # This branch tests the L3 public-ID boundary while the separate L2 branch
    # owns production tracking. Supply explicit authoritative IDs in this
    # integration fixture; L3 itself must never synthesize them.
    original_l2_process = runtime._layer2.process

    def process_with_public_ids(*args, **kwargs):
        result = original_l2_process(*args, **kwargs)
        directions = tuple(
            TrackedDirection(
                item.session_id, item.stream_epoch, item.window_id, item.decision_sample,
                item.doa_start_sample, item.doa_end_sample, index + 1, index + 1,
                item.theta_deg, item.theta_deg, item.raw_score, item.normalized_score,
                "confirmed", True, index == 0, item.doa_start_sample,
                item.decision_sample, 0, False,
            )
            for index, item in enumerate(result.candidates)
        )
        return replace(result, directions=directions, active_tracks=directions)

    runtime._layer2.process = process_with_public_ids
    runtime.start()
    deadline, frame = time.monotonic() + 8, None
    while time.monotonic() < deadline:
        while not runtime.latest_dev_ui.empty():
            candidate = runtime.latest_dev_ui.get_nowait()
            if candidate.previews:
                frame = candidate
        if frame is not None:
            break
        time.sleep(.01)
    runtime.stop()
    runtime.recording_store.close()

    assert frame is not None and frame.spatial_response is not None
    assert len(frame.previews) == len(frame.candidates) >= 1
    assert all(preview.window_id == frame.spatial_response.window_id for preview in frame.previews)
    assert all(preview.waveform.shape == (7_680,) for preview in frame.previews)
    assert all(not hasattr(preview, "spectrogram") for preview in frame.previews)

    app, window = build_window(CONFIG)
    try:
        window._render_frame(frame)
        app.processEvents()
        assert "P 0.90" in window.gate_readout.text()
        assert "Gate OPEN" in window.gate_readout.text()
        assert "L2内部平滑" in window.bf_panel.help.text()
        assert "Directional Audio Preview" in window.bf_panel.title()
    finally:
        window.close()
        app.processEvents()


def test_test_ui_uses_runtime_sidecar_tracker_only_for_listening_cache(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from gui.dev_test_ui.app import build_window

    app, window = build_window(CONFIG)
    try:
        runtime = window._runtime
        assert runtime.dev_audio_tracker is not None
        assert runtime.dev_audio_tracker.__class__.__name__ == "AudioIdTracker"
        assert isinstance(runtime.pipeline.source, LiveSipeedSource)
    finally:
        window.close()
        app.processEvents()


def test_test_ui_accepts_backend_injected_wav_and_keeps_default_ui_input_hidden(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from gui.dev_test_ui.app import build_window

    config_path = tmp_path / "config" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    wav_path = tmp_path / "recorded_test.wav"
    _write_test_wav(wav_path, np.zeros((18 * 960, 7), np.float32))

    app, window = build_window(config_path, input_wav=wav_path, auto_start=True)
    try:
        assert isinstance(window._runtime.pipeline.source, WavAudioSource)
        assert window._runtime.pipeline.source.realtime is True
        assert "模拟测试" in window.windowTitle()
        assert "recorded_test.wav" in window.windowTitle()
        assert not hasattr(window, "input_source_selector")
        deadline = time.monotonic() + 10
        while (not window._runtime.input_exhausted or window._runtime.active) and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert window._runtime.input_exhausted
        assert not window._runtime.active
        assert window._frame is not None
    finally:
        window.close()
        app.processEvents()


def test_complete_recording_mode_exposes_only_simulation_controls_and_name(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from data_management.manifests import sha256_file
    from gui.dev_test_ui.app import build_window
    from layer1_input.recording_replay import RecordingReplaySource
    import json

    root = tmp_path / "named-recording"
    root.mkdir()
    audio = root / "native_8ch.wav"
    _write_test_wav(audio, np.zeros((18 * 960, 8), np.float32))
    hotmaps = root / "hotmaps.jsonl"
    hotmaps.write_text(
        json.dumps({
            "sequence_id": 0,
            "timestamp": 0.0,
            "received_at": None,
            "playback_sample": 0,
            "matrix": np.zeros((16, 16), dtype=np.uint8).tolist(),
        }) + "\n",
        encoding="utf-8",
    )
    manifest = root / "recording_manifest.json"
    manifest.write_text(json.dumps({
        "display_name": "我命名的阵列录音",
        "assets": [
            {"kind": "native_8ch", "path": audio.name, "sha256": sha256_file(audio)},
            {"kind": "cdc_hotmaps", "path": hotmaps.name, "sha256": sha256_file(hotmaps)},
        ],
    }), encoding="utf-8")

    app, window = build_window(CONFIG, replay_recording=manifest, auto_start=False)
    try:
        assert isinstance(window._runtime.pipeline.source, RecordingReplaySource)
        assert "模拟输入模式" in window.windowTitle()
        assert "我命名的阵列录音" in window.windowTitle()
        assert window.replay_name.text().endswith("我命名的阵列录音")
        assert window.replay_start.text() == "开始/继续"
        assert window.replay_pause.text() == "暂停"
        assert window.replay_restart.text() == "从头重播"
        assert not window.start_button.isVisible()
        assert not window.stop_button.isVisible()
        window._frame = object()
        window._last_l4_frame = object()
        window._restart_replay()
        assert window._frame is None
        assert window._last_l4_frame is None
        assert "replay restarted" in window.srp_header.text()
    finally:
        window.close()
        app.processEvents()


def test_window_has_four_equal_grid_cells_and_fixed_performance_bar(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from gui.dev_test_ui.app import build_window

    app, window = build_window(CONFIG)
    try:
        assert not window.isFullScreen()
        quadrants = window.findChild(object, "quadrants")
        layout = quadrants.layout()
        assert layout.count() == 4
        assert layout.rowStretch(0) == layout.rowStretch(1) == 1
        assert layout.columnStretch(0) == layout.columnStretch(1) == 1
        assert window.performance_bar.height() == 56
        assert window.performance_bar.text() == (
            "上一秒性能 | L2 N/A / 0.0 Hz | L3 N/A / 0.0 Hz | L4 N/A / 0.0 Hz"
        )
        assert window.start_button.text() == "启动采集"
        assert window.stop_button.text() == "停止采集"
        assert window.start_button.isEnabled() and not window.stop_button.isEnabled()
        assert window.light_on.isEnabled() and window.light_off.isEnabled()
        assert window.srp_threshold.parentWidget() is not window.srp_polar
        right_layout = window.srp_threshold.parentWidget().layout()
        assert right_layout.indexOf(window.gate_threshold) == 0
        assert right_layout.indexOf(window.srp_iterative) == 1
        assert right_layout.indexOf(window.srp_id_tracking) == 2
        assert right_layout.indexOf(window.srp_kalman) == 3
        assert right_layout.indexOf(window.srp_kalman_q) == 4
        assert right_layout.indexOf(window.srp_kalman_r) == 5
        assert right_layout.indexOf(window.gate_readout) == 6
        assert right_layout.indexOf(window.srp_threshold) == 7
        assert window.srp_iterative.parentWidget() is not window.srp_polar
        decision = ProbabilityGateDecision(
            "ui-test", 0, 12, 26_880, "mean_2x20ms_v1", ProbabilityGateState.OPEN,
            0.55, 0.75, 0.65, 0.60, 4, True, "probability_at_or_above_threshold",
        )
        window.gate_readout.set_decision(decision)
        assert "P 0.65" in window.gate_readout.text()
        assert "Gate OPEN" in window.gate_readout.text()
        assert window.gate_readout.height() == 30
        window._enter_stopped_state()
        stopped_header = window.srp_header.text()
        time.sleep(.02)
        window._refresh()
        assert window.srp_header.text() == stopped_header
        assert "age —" in stopped_header and "STALE" not in stopped_header

        window.resize(1920, 1080)
        app.processEvents()
        layout = quadrants.layout()
        before = [layout.itemAtPosition(row, column).widget().geometry().getRect() for row in range(2) for column in range(2)]
        for index in range(20):
            window.l1_header.setText(f"RUNNING | sample {index:012d} | seq {index:08d} | age 000 ms")
            window.srp_header.setText(f"LIVE | window {index:08d} | sample {index * 960:012d} | age 099 ms")
            window.global_status.setText(f"RUNNING | input drop {index:06d} | processing drop {index:06d}")
            app.processEvents()
        after = [layout.itemAtPosition(row, column).widget().geometry().getRect() for row in range(2) for column in range(2)]
        assert after == before
        # A one-pixel difference is the intentional grid separator/odd-pixel remainder.
        assert abs(before[0][2] - before[1][2]) <= 1
        assert abs(before[0][3] - before[2][3]) <= 1
    finally:
        window.close()
        app.processEvents()


def test_l3_listening_panel_hides_tracks_shorter_than_two_seconds(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from gui.dev_test_ui.app import build_window

    app, window = build_window(CONFIG)
    try:
        short = TrackedAudioSnapshot("s", 0, 1, "ended", 10.0, 0.5, 95_999)
        accepted = TrackedAudioSnapshot("s", 0, 2, "active", 20.0, 0.8, 96_000)
        longer = TrackedAudioSnapshot("s", 0, 3, "active", 30.0, 0.9, 144_000)
        reference = TrackedAudioSnapshot("s", 0, 0, "active", 0.0, 0.0, 48_000)
        window.bf_panel.set_tracks((short, accepted, longer, reference))
        app.processEvents()

        assert set(window.bf_panel._track_rows) == {0, 2, 3}
        ordered = [
            window.bf_panel.track_layout.itemAt(index).widget().track_id
            for index in range(3)
        ]
        assert ordered == [0, 3, 2]
        assert "Center Mic" in window.bf_panel._track_rows[0].label.text()
        assert "≥2.0s" in window.bf_panel.help.text()

        window.bf_panel.set_tracks(())
        app.processEvents()
        assert set(window.bf_panel._track_rows) == {0, 2, 3}

        next_session = TrackedAudioSnapshot(
            "next", 0, 0, "active", 0.0, 0.0, 48_000,
        )
        window.bf_panel.set_tracks((next_session,))
        app.processEvents()
        assert set(window.bf_panel._track_rows) == {0}
        assert window.bf_panel._track_stream == ("next", 0)

        window.bf_panel.clear_tracks()
        app.processEvents()
        assert window.bf_panel._track_rows == {}
    finally:
        window.close()
        app.processEvents()


def test_srp_response_radius_maps_zero_near_center_and_one_near_rim(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from gui.dev_test_ui.srp_panel import SrpPolarPanel

    outer = 200.0
    assert SrpPolarPanel._response_radius(outer, 0.0) == pytest.approx(7.0)
    assert SrpPolarPanel._response_radius(outer, 1.0) == pytest.approx(193.0)
    assert SrpPolarPanel._response_radius(outer, -1.0) == pytest.approx(7.0)
    assert SrpPolarPanel._response_radius(outer, 2.0) == pytest.approx(193.0)


def test_srp_candidate_style_encodes_identity_and_current_observation(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from gui.dev_test_ui.srp_panel import SrpPolarPanel

    new_colour, new_size = SrpPolarPanel._candidate_style(
        7, is_prediction=False, is_formal=False, is_new=True
    )
    pending_colour, pending_size = SrpPolarPanel._candidate_style(
        7, is_prediction=False, is_formal=False, is_new=False
    )
    pending_prediction_colour, pending_prediction_size = SrpPolarPanel._candidate_style(
        7, is_prediction=True, is_formal=False, is_new=False
    )
    red_live, red_live_size = SrpPolarPanel._candidate_style(
        1, is_prediction=False, is_formal=True, formal_color_slot=0
    )
    red_predicted, red_predicted_size = SrpPolarPanel._candidate_style(
        1, is_prediction=True, is_formal=True, formal_color_slot=0
    )
    green_live, green_live_size = SrpPolarPanel._candidate_style(
        3, is_prediction=False, is_formal=True, formal_color_slot=1
    )
    amber_live, amber_live_size = SrpPolarPanel._candidate_style(
        5, is_prediction=False, is_formal=True, formal_color_slot=2
    )

    assert new_colour.name() == pending_colour.name() == pending_prediction_colour.name() == "#929daa"
    assert new_size < pending_size
    assert pending_prediction_size == new_size
    assert red_live.name() == red_predicted.name() == "#ff3b30"
    assert red_predicted_size == new_size < red_live_size
    assert green_live.name() == "#2ecc71"
    assert green_live_size == red_live_size
    assert amber_live.name() == "#ffb000"
    assert amber_live_size == red_live_size
