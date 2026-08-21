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
from common.data_types import CandidateDirection, IngestedAudioBlock, ModelOrderEstimate, PipelineStatus, SpatialResponse, TrackedDirection
from common.geometry import MIC_POSITIONS_M
from gui.dev_test_ui.aggregator import DevUiAggregator, PerformanceTracker
from gui.dev_test_ui.contracts import BeamformPreview, L1MeterSnapshot, TrackedAudioSnapshot
from gui.dev_test_ui.settings import DevUiSettings
from layer1_input.interface import DecodedAudio
from layer1_input.sources import LiveSipeedSource, WavAudioSource
from layer2_source_detection.music import MusicDiagnostics, MusicStateDiagnostic
from layer2_source_detection.probability_gate import (
    ProbabilityGateDecision,
    ProbabilityGateState,
    SourceProbability20ms,
    SourceProbabilityState,
)
from layer4_voice_classifier import Layer4Result, ModelPrediction, VoiceDetection
from track_audio_stream import TrackVoiceAnnotation


CONFIG = Path(__file__).parents[1] / "config/config.yaml"


def test_waveform_voice_background_uses_stored_probability_and_live_ui_threshold():
    from PySide6.QtWidgets import QApplication

    from gui.dev_test_ui.panels import AudioWaveformThumbnail

    app = QApplication.instance() or QApplication([])
    widget = AudioWaveformThumbnail()
    annotations = (
        TrackVoiceAnnotation("s", 0, 1, 960, 7, 0, 960, 0.8, True, "nv", 0.7),
        TrackVoiceAnnotation("s", 0, 2, 1_920, 7, 960, 1_920, 0.2, False, "nv", 0.7),
        None,
    )
    widget.set_voice_annotations(annotations)

    widget.set_voice_threshold(0.7)
    assert widget._voice_columns(3) == (True, False, False)
    widget.set_voice_threshold(0.9)
    assert widget._voice_columns(3) == (False, False, False)
    widget.close()
    app.processEvents()


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
    model_order = ModelOrderEstimate(1, 43, 23, 1.0, 0, "ready")
    diagnostics = MusicDiagnostics(
        "frequency_normalized_music", "frequency_normalized_music_mdl_v1", revision,
        model_order, MusicStateDiagnostic("advanced", decision_sample - 960, decision_sample, 0, 23, 2, 2, False, "sample_continuous"),
        43, "ready",
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
    assert settings.load_music_effective_order_limit(3) == 3
    assert settings.load_music_dpd_rank1_enabled() is False
    assert settings.load_music_noise_whitening_enabled() is False
    assert settings.load_direction_kalman_enabled() is False
    assert settings.load_direction_id_tracking_enabled() is True
    assert settings.load_gate_probability_threshold(0.60) == 0.60
    assert settings.load_l1_pre_denoise_enabled(False) is False
    assert settings.load_l4_input_gain_compensation_enabled(True) is True

    settings.save_direction_threshold(.67)
    settings.save_music_effective_order_limit(1)
    settings.save_music_dpd_rank1_enabled(True)
    settings.save_music_noise_whitening_enabled(True)
    assert settings.load_direction_threshold(.35) == .67
    assert settings.load_music_effective_order_limit(3) == 1
    settings.save_direction_threshold(.42)
    settings.save_direction_kalman_enabled(True)
    settings.save_direction_id_tracking_enabled(False)
    assert settings.save_direction_kalman_q_scale(1.2) == 1.2
    assert settings.save_direction_kalman_r_scale(0.8) == 0.8
    assert settings.save_gate_probability_threshold(0.73) == 0.73
    assert settings.save_l1_pre_denoise_enabled(True) is True
    assert settings.save_l4_input_gain_compensation_enabled(False) is False

    loaded = DevUiSettings(tmp_path)
    assert loaded.load_direction_threshold(.35) == .42
    assert loaded.load_music_effective_order_limit(3) == 1
    assert loaded.load_music_dpd_rank1_enabled() is True
    assert loaded.load_music_noise_whitening_enabled() is True
    assert loaded.load_direction_kalman_enabled() is True
    assert loaded.load_direction_id_tracking_enabled() is False
    assert loaded.load_direction_kalman_q_scale(1.0) == 1.2
    assert loaded.load_direction_kalman_r_scale(1.0) == 0.8
    assert loaded.load_gate_probability_threshold(0.60) == 0.73
    assert loaded.load_l1_pre_denoise_enabled(False) is True
    assert loaded.load_l4_input_gain_compensation_enabled(True) is False

    payload = loaded.path.read_text(encoding="utf-8")
    assert '"layer2_direction_threshold": 0.42' in payload
    assert '"layer2_music_effective_order_limit": 1' in payload
    assert '"layer2_music_dpd_rank1_enabled": true' in payload
    assert '"layer2_music_noise_whitening_enabled": true' in payload
    assert '"layer2_direction_id_tracking_enabled": false' in payload
    assert '"layer4_input_gain_compensation_enabled": false' in payload
    assert "layer2_iterative_peak_search_enabled" not in payload


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


def test_beamform_panel_replaces_single_window_playback_with_downstream_switch(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from gui.dev_test_ui.panels import BeamformPanel

    config = load_config(CONFIG, environ={})
    app = QApplication.instance() or QApplication([])
    panel = BeamformPanel(config)
    assert not hasattr(panel, "preview_play")
    assert not hasattr(panel, "preview_stop")
    assert panel.downstream_switch.text() == "L3/L4：运行中"
    states = []
    panel.downstream_processing_changed.connect(states.append)
    panel.downstream_switch.click()
    assert states == [False]
    assert panel.downstream_switch.text() == "L3/L4：已停止"
    assert not panel.mode_switch.isEnabled()
    panel.deleteLater()
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
    assert snap.processed_windows_last_second == 1
    assert snap.dropped_windows_last_second == 0
    assert snap.drop_rate_last_second == 0.0
    reset = tracker.snapshot(PipelineStatus("warming_up", "s", 1, 960, 15_360, "Warming"))
    assert reset.observed_sample_rate_hz is None and reset.compute_time_ms_current is None


def test_performance_tracker_reports_last_second_processed_dropped_and_rate(monkeypatch):
    import gui.dev_test_ui.aggregator as aggregator_module

    tracker = PerformanceTracker(
        sample_rate=48_000, required_samples=15_360, window_count=500, rate_seconds=5,
    )
    tracker.add_timing(
        "s", 0, 1, 2.0, 3.0,
        l2_ms=1.0, completed_monotonic=10.2, processed=True,
    )
    tracker.add_timing(
        "s", 0, 2, 2.0, 3.0,
        l2_ms=1.0, completed_monotonic=10.4, processed=False,
    )
    tracker.add_drop("s", 0, dropped_monotonic=10.5)
    monkeypatch.setattr(aggregator_module, "monotonic", lambda: 10.9)

    snapshot = tracker.snapshot(PipelineStatus("running", "s", 0, 15_360, 15_360, "Ready"))

    assert snapshot.processed_windows_last_second == 1
    assert snapshot.dropped_windows_last_second == 1
    assert snapshot.drop_rate_last_second == pytest.approx(0.5)


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
        direction_threshold=0.35, direction_kalman_enabled=False,
        direction_kalman_q_scale=1.0, direction_kalman_r_scale=1.0,
        scan_config_revision=0,
    )

    new_status = PipelineStatus(
        "warming_up", "epoch-test", 1, 960, 15_360,
        "Warming; epoch_reset:health_event:input_overflow",
    )
    current = aggregator.update_l1(meter(1), new_status)
    assert current.gate_decision is None
    assert "WARMING_UP" in current.missing_reasons["srp"]
    assert "epoch_reset:health_event:input_overflow" in current.missing_reasons["srp"]

    late = aggregator.update_srp(
        None, (), "BACKGROUND_ONLY", gate_decision=gate,
        gate_threshold=0.60, gate_config_revision=0,
        direction_threshold=0.35, direction_kalman_enabled=False,
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
        direction_kalman_enabled=False,
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


def test_prediction_only_coasting_window_can_publish_l3_and_l4_without_music_response():
    performance = PerformanceTracker(
        sample_rate=48_000, required_samples=15_360, window_count=10, rate_seconds=5,
    )
    aggregator = DevUiAggregator(performance)
    status = PipelineStatus(
        "running", "aggregator-test", 0, 15_360, 15_360, "Ready",
    )
    aggregator.update_l1(_aggregator_meter(), status)
    decision_sample = 16_320
    direction = TrackedDirection(
        session_id="aggregator-test",
        stream_epoch=0,
        window_id=1,
        decision_sample=decision_sample,
        doa_start_sample=decision_sample - 1_920,
        doa_end_sample=decision_sample,
        track_id=7,
        rank=1,
        measured_theta_deg=None,
        theta_deg=30.0,
        raw_score=0.2,
        normalized_score=0.8,
        track_state="coasting",
        is_observed=False,
        is_new_track=False,
        first_seen_sample=12_480,
        last_observed_sample=15_360,
        missed_samples=960,
        kalman_applied=False,
    )
    gate = ProbabilityGateDecision(
        "aggregator-test", 0, 1, decision_sample, "mean_2x20ms_v1",
        ProbabilityGateState.CLOSED, 0.1, 0.1, 0.1, 0.7, 2, False,
        "probability_below_threshold",
    )
    aggregator.update_srp(
        None,
        (),
        "UNAVAILABLE: probability_below_threshold",
        gate_decision=gate,
        gate_threshold=0.7,
        gate_config_revision=2,
        direction_threshold=0.35,
        direction_kalman_enabled=False,
        direction_kalman_q_scale=1.0,
        direction_kalman_r_scale=1.0,
        scan_config_revision=3,
        directions=(direction,),
        active_tracks=(direction,),
    )
    preview = BeamformPreview(
        "aggregator-test", 0, 1, decision_sample, 30.0,
        np.zeros(1_920, np.float32), "optimized", track_id=7,
    )
    frame = aggregator.update_l3((preview,))
    assert frame.spatial_response is None
    assert frame.directions == (direction,)
    assert frame.previews == (preview,)

    probability = np.asarray([0.9], dtype=np.float32)
    l4 = Layer4Result(
        (
            VoiceDetection(
                "aggregator-test", 0, 1, decision_sample,
                30.0, 0.9, True, "test-model", track_id=7,
            ),
        ),
        (ModelPrediction("test-model", probability, 1.0, {}),),
        "test-model",
        0.7,
    )
    assert aggregator.update_l4(l4).l4_result is l4


def test_l2_drop_retains_last_music_and_gate_as_stale_snapshot():
    performance = PerformanceTracker(
        sample_rate=48_000, required_samples=15_360, window_count=10, rate_seconds=5,
    )
    aggregator = DevUiAggregator(performance)
    status = PipelineStatus(
        "running", "aggregator-test", 0, 15_360, 15_360, "Ready",
    )
    aggregator.update_l1(_aggregator_meter(), status)
    response, candidates, gate, diagnostics = _open_l2_result(window_id=0)
    completed = aggregator.update_srp(
        response,
        candidates,
        gate_decision=gate,
        search_diagnostics=diagnostics,
        gate_threshold=0.60,
        gate_config_revision=2,
        direction_threshold=0.35,
        direction_kalman_enabled=False,
        direction_kalman_q_scale=1.0,
        direction_kalman_r_scale=1.0,
        scan_config_revision=3,
    )

    dropped = aggregator.report_l2_drop(
        "L2 DROPPED: l2_admission_queue_overflow"
    )

    assert dropped.spatial_response is completed.spatial_response
    assert dropped.candidates == completed.candidates
    assert dropped.gate_decision is completed.gate_decision
    assert dropped.search_diagnostics is completed.search_diagnostics
    assert (
        dropped.spatial_published_monotonic
        == completed.spatial_published_monotonic
    )
    assert dropped.missing_reasons["srp"] == (
        "L2 DROPPED: l2_admission_queue_overflow"
    )


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
            direction_kalman_enabled=False,
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
    # The UI mailbox intentionally keeps only the newest frame.  Depending on
    # scheduling, window 1 may replace window 0 before this test consumes it.
    assert frame.gate_decision.window_id in {0, 1}
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


def test_test_ui_sends_led_off_only_after_microphone_start_succeeds(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from gui.dev_test_ui.app import build_window

    calls = []
    monkeypatch.setattr(ApplicationRuntime, "start", lambda _self: calls.append("microphone"))
    monkeypatch.setattr(
        ApplicationRuntime,
        "set_light",
        lambda _self, enabled: calls.append(("light", enabled)),
    )
    app, window = build_window(CONFIG)
    try:
        window._start_capture()
        deadline = time.monotonic() + 2.0
        while window._pending_command is not None and time.monotonic() < deadline:
            app.processEvents()
            window._poll_command()
            time.sleep(0.005)
        assert calls == ["microphone", ("light", False)]
        assert window.light_label.text() == "状态: Off (startup default)"
    finally:
        window.close()
        app.processEvents()


def test_test_ui_does_not_send_light_command_when_microphone_start_fails(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from gui.dev_test_ui.app import build_window

    calls = []

    def fail_start(_self):
        calls.append("microphone")
        raise OSError("microphone unavailable")

    monkeypatch.setattr(ApplicationRuntime, "start", fail_start)
    monkeypatch.setattr(
        ApplicationRuntime,
        "set_light",
        lambda _self, enabled: calls.append(("light", enabled)),
    )
    app, window = build_window(CONFIG)
    try:
        window._start_capture()
        deadline = time.monotonic() + 2.0
        while window._pending_command is not None and time.monotonic() < deadline:
            app.processEvents()
            window._poll_command()
            time.sleep(0.005)
        assert calls == ["microphone"]
        assert "microphone unavailable" in window.statusBar().currentMessage()
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
        assert window._runtime.l3_processing_mode == "loaded_mvdr_baseline"
        assert window.bf_panel.mode_switch.text() == "BF：Loaded MVDR基线"
        window.bf_panel.mode_switch.click()
        app.processEvents()
        assert window._runtime.l3_processing_mode == "subband_robust_baseline"
        assert window.bf_panel.mode_switch.text() == "BF：五频段鲁棒对照"
        window.bf_panel.mode_switch.click()
        app.processEvents()
        assert window._runtime.l3_processing_mode == "optimized"
    finally:
        window.close()
        app.processEvents()


def test_gain_compensation_control_is_in_l3_header_and_uses_state_colors(
    monkeypatch, tmp_path,
):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from gui.dev_test_ui.app import build_window

    config_path = tmp_path / "config" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    DevUiSettings(tmp_path).save_l4_input_gain_compensation_enabled(False)

    app, window = build_window(config_path)
    try:
        control = window.bf_panel.gain_compensation
        assert control.size() == window.bf_panel.mode_switch.size()
        assert control.size() == window.bf_panel.downstream_switch.size()
        assert control.text() == "连续轨响度补偿"
        assert control.parent() is window.bf_panel
        assert not hasattr(window.cnn_panel, "gain_compensation")
        assert not control.isChecked()
        assert "#5b6570" in control.styleSheet()

        control.click()
        app.processEvents()
        assert control.isChecked()
        assert "#16794b" in control.styleSheet()
        assert window._runtime.l4_input_gain_compensation_enabled is True
        assert DevUiSettings(tmp_path).load_l4_input_gain_compensation_enabled(False) is True
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

    assert frame is not None and frame.spatial_response is not None, (
        runtime.last_error,
        runtime.processing_error,
        runtime.dev_ui_error,
        runtime.dev_audio_tracking_error,
        runtime._stage_errors,
    )
    assert len(frame.previews) == len(frame.candidates) >= 1
    assert all(preview.window_id == frame.spatial_response.window_id for preview in frame.previews)
    assert all(
        preview.waveform.shape == (runtime_config.downstream_audio_window.samples,)
        for preview in frame.previews
    )
    assert all(not hasattr(preview, "spectrogram") for preview in frame.previews)

    app, window = build_window(CONFIG)
    try:
        window._render_frame(frame)
        app.processEvents()
        assert "P 0.90" in window.gate_readout.text()
        assert "Gate OPEN" in window.gate_readout.text()
        assert "L2 confirmed权威ID" in window.bf_panel.help.text()
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
        window._refresh_total_duration_text()
        assert "总处理时长 | L2 " in window.performance_bar.text()
        total_duration_text = window.performance_bar.text().split("总处理时长", 1)[1]
        assert "L2 N/A" not in total_duration_text
        assert "L3 N/A" not in total_duration_text
        assert "L4 N/A" not in total_duration_text
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
        assert window._runtime.pipeline.hotmap_source is None
        assert window._runtime.pipeline.hotmap_required is False
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
    from PySide6.QtCore import Qt
    from gui.dev_test_ui.app import build_window

    app, window = build_window(CONFIG)
    try:
        assert not window.isFullScreen()
        assert window.windowState() & Qt.WindowState.WindowMaximized
        quadrants = window.findChild(object, "quadrants")
        layout = quadrants.layout()
        assert layout.count() == 4
        assert layout.rowStretch(0) == layout.rowStretch(1) == 1
        assert layout.columnStretch(0) == layout.columnStretch(1) == 1
        assert window.performance_bar.height() == 56
        assert window.performance_bar.text() == (
            "上一秒性能 | L2 N/A | L3 N/A | L4 N/A / 0.0 Hz | "
            "20ms窗口 0 | 丢窗 0 | 丢窗率 0.0%"
        )
        assert window.start_button.text() == "启动采集"
        assert window.stop_button.text() == "停止采集"
        assert window.start_button.isEnabled() and not window.stop_button.isEnabled()
        assert window.light_on.isEnabled() and window.light_off.isEnabled()
        assert not hasattr(window, "calibration_label")
        assert window.srp_threshold.parentWidget() is not window.srp_polar
        right_layout = window.srp_threshold.parentWidget().layout()
        assert right_layout.indexOf(window.gate_threshold) == 0
        assert not hasattr(window, "srp_iterative")
        assert window.srp_id_tracking.text() == "ID Tracking"
        expected_id_colour = "#16794b" if window.srp_id_tracking.isChecked() else "#5b6570"
        assert expected_id_colour in window.srp_id_tracking.styleSheet()
        assert not hasattr(window, "direction_table")
        processing_switches = right_layout.itemAt(1).layout()
        assert processing_switches is not None
        assert processing_switches.indexOf(window.srp_kalman) == 0
        assert processing_switches.indexOf(window.music_dpd_rank1) == 1
        assert processing_switches.indexOf(window.music_noise_whitening) == 2
        assert processing_switches.stretch(0) == 1
        assert processing_switches.stretch(1) == 1
        assert processing_switches.stretch(2) == 1
        assert window.srp_kalman.text() == "Kalman"
        assert window.music_dpd_rank1.text() == "DPD"
        assert window.music_noise_whitening.text() == "Whitening"
        for switch in (
            window.srp_kalman,
            window.music_dpd_rank1,
            window.music_noise_whitening,
        ):
            expected_colour = "#16794b" if switch.isChecked() else "#5b6570"
            assert expected_colour in switch.styleSheet()
        assert right_layout.indexOf(window.srp_kalman_q) == 2
        assert right_layout.indexOf(window.srp_kalman_r) == 3
        assert right_layout.indexOf(window.gate_readout) == 4
        assert right_layout.indexOf(window.music_status) == 5
        order_tracking_row = right_layout.itemAt(6).layout()
        assert order_tracking_row is not None
        assert order_tracking_row.indexOf(window.music_order_limit) == 0
        assert order_tracking_row.indexOf(window.srp_id_tracking) == 1
        assert window.music_order_limit.maximumWidth() == 185
        assert window.music_order_limit.combo.width() == 64
        assert right_layout.indexOf(window.srp_threshold) == 7
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
        rendered_after_stop = []
        original_render = window._render_frame
        window._render_frame = lambda *args, **kwargs: rendered_after_stop.append((args, kwargs))
        window._runtime.latest_dev_ui.put_nowait(object())
        window._refresh()
        window._render_frame = original_render
        assert rendered_after_stop == []
        assert window.srp_header.text() == stopped_header

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


def test_stop_command_is_not_reported_complete_while_runtime_remains_active(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from gui.dev_test_ui.app import build_window

    app, window = build_window(CONFIG)
    try:
        window._runtime._recording_session_started = True
        window._submit_command("停止采集", lambda: None)
        deadline = time.monotonic() + 1.0
        while window._pending_command is not None and time.monotonic() < deadline:
            window._poll_command()
            app.processEvents()
        assert "停止采集失败" in window.statusBar().currentMessage()
        assert window.stop_button.isEnabled()
        assert "STOPPED | capture closed" not in window.l1_header.text()
        window._runtime._recording_session_started = False
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
        assert window.bf_panel._track_rows[3].label.text() == "3  30.0°"
        assert "#ffb000" in window.bf_panel._track_rows[3].label.styleSheet().lower()
        assert window.bf_panel._track_rows[2].label.text() == "2  20.0°"
        assert "#2ecc71" in window.bf_panel._track_rows[2].label.styleSheet().lower()
        assert "≥2.0s" in window.bf_panel.help.text()

        window.bf_panel.set_track_playback_progress(3, 0.4)
        assert window.bf_panel._track_rows[3].waveform._playback_progress == pytest.approx(0.4)
        assert window.bf_panel._track_rows[0].waveform._playback_progress is None
        assert window.bf_panel._track_rows[2].waveform._playback_progress is None
        window.bf_panel.clear_track_playback_progress()
        assert all(
            row.waveform._playback_progress is None
            for row in window.bf_panel._track_rows.values()
        )

        window.bf_panel.set_tracks(())
        app.processEvents()
        assert set(window.bf_panel._track_rows) == {0, 2, 3}

        # A complete tracker snapshot (identified by Center Mic) removes an ID
        # whose cache was explicitly filtered, instead of retaining a stale,
        # unplayable waveform row.
        window.bf_panel.set_tracks((longer, reference))
        app.processEvents()
        assert set(window.bf_panel._track_rows) == {0, 3}
        assert set(window.bf_panel._track_snapshots) == {0, 3}

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


def test_music_response_radius_maps_zero_near_center_and_one_near_rim(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from gui.dev_test_ui.srp_panel import MusicPolarPanel

    outer = 200.0
    assert MusicPolarPanel._response_radius(outer, 0.0) == pytest.approx(7.0)
    assert MusicPolarPanel._response_radius(outer, 1.0) == pytest.approx(193.0)
    assert MusicPolarPanel._response_radius(outer, -1.0) == pytest.approx(7.0)
    assert MusicPolarPanel._response_radius(outer, 2.0) == pytest.approx(193.0)


def test_music_panel_and_table_use_authoritative_track_fields(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    import gui.dev_test_ui.srp_panel as music_panel
    from gui.dev_test_ui.srp_panel import DirectionTrackTable, MusicPanelSnapshot, MusicPolarPanel

    app = QApplication.instance() or QApplication([])
    del app
    response, _, _, diagnostics = _open_l2_result()
    response = SpatialResponse(
        response.session_id, response.stream_epoch, response.window_id,
        response.decision_sample, response.doa_start_sample, response.doa_end_sample,
        response.theta_degrees, response.raw_scores, response.normalized_scores,
        diagnostics.model_order, diagnostics.valid_frequency_bins,
        diagnostics.covariance_quality, diagnostics.algorithm_version,
    )
    track = TrackedDirection(
        "aggregator-test", 0, 0, 15_360, 13_440, 15_360,
        7, 1, 30.0, 31.0, 0.2, 0.8, "confirmed", True, True,
        14_400, 15_360, 0, True,
    )
    snapshot = MusicPanelSnapshot(response, (track,), (track,), time.monotonic(), {7: 0.91})
    panel = MusicPolarPanel()
    table = DirectionTrackTable()
    now = [100.0]
    monkeypatch.setattr(music_panel, "monotonic", lambda: now[0])
    first_item = table.item(0, 0)
    panel.set_snapshot(snapshot)
    table.set_snapshot(snapshot)
    assert panel._snapshot.active_tracks[0].track_id == 7
    assert table.item(0, 0).text() == "7"
    assert table.item(0, 1).text() == "30.0°"
    assert table.item(0, 2).text() == "31.0°"
    assert table.item(0, 4).text() == "confirmed"
    assert table.item(0, 7).text() == "—"
    assert table.rowCount() == 3
    assert not table.alternatingRowColors()
    assert table.item(0, 0) is first_item
    assert table.item(1, 0).text() == table.item(2, 0).text() == ""

    raw_peak = CandidateDirection(
        response.session_id, response.stream_epoch, response.window_id,
        response.decision_sample, response.doa_start_sample, response.doa_end_sample,
        125.0, 0.4, 0.7,
    )
    raw_snapshot = MusicPanelSnapshot(
        response, (), (), time.monotonic(), {}, None, (raw_peak,), False,
    )
    panel.set_snapshot(raw_snapshot)
    table.set_snapshot(raw_snapshot)
    assert panel._snapshot.raw_peaks == (raw_peak,)
    assert panel._snapshot.direction_id_tracking_enabled is False
    assert all(
        table.item(row, column).text() == ""
        for row in range(3)
        for column in range(table.columnCount())
    )

    now[0] = 100.5
    response2 = replace(response, window_id=1, decision_sample=16_320,
                        doa_start_sample=14_400, doa_end_sample=16_320)
    track2 = replace(track, window_id=1, decision_sample=16_320,
                     doa_start_sample=14_400, doa_end_sample=16_320,
                     first_seen_sample=14_400, last_observed_sample=16_320)
    table.set_snapshot(MusicPanelSnapshot(
        response2, (track2,), (track2,), now[0], {7: 0.40},
    ))
    assert table.item(0, 7).text() == "—"

    now[0] = 101.1
    response3 = replace(response2, window_id=2, decision_sample=17_280,
                        doa_start_sample=15_360, doa_end_sample=17_280)
    track3 = replace(track2, window_id=2, decision_sample=17_280,
                     doa_start_sample=15_360, doa_end_sample=17_280,
                     last_observed_sample=17_280)
    table.set_snapshot(MusicPanelSnapshot(
        response3, (track3,), (track3,), now[0], {7: 0.40},
    ))
    assert table.item(0, 7).text() == "0.910"

    now[0] = 102.2
    response4 = replace(response3, window_id=3, decision_sample=18_240,
                        doa_start_sample=16_320, doa_end_sample=18_240)
    track4 = replace(track3, window_id=3, decision_sample=18_240,
                     doa_start_sample=16_320, doa_end_sample=18_240,
                     last_observed_sample=18_240)
    table.set_snapshot(MusicPanelSnapshot(
        response4, (track4,), (track4,), now[0], {7: 0.25},
    ))
    assert table.item(0, 7).text() == "0.400"

    table.set_snapshot(None)
    assert table.rowCount() == 3
    assert table.item(0, 0) is first_item
    assert all(
        table.item(row, column).text() == ""
        for row in range(3)
        for column in range(table.columnCount())
    )
