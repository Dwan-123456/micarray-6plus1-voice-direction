from __future__ import annotations

import queue
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import json
import pytest

from app.runtime import ApplicationRuntime
from common.config import load_config
from common.data_types import DecisionWindow, IngestedAudioBlock, TrackedDirection
from layer1_input.interface import DecodedAudio
from layer3_direction_signal import (
    L3_MODE_CONSTANT_BEAMWIDTH,
    L3_MODE_DS_BASELINE,
    L3_MODE_OPTIMIZED,
)


CONFIG = Path(__file__).parents[1] / "config/config.yaml"


class StubPipeline:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.started = self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def read(self, timeout=None):
        del timeout
        try:
            return next(self.frames)
        except StopIteration:
            time.sleep(0.001)
            return None

    def take_health_events(self):
        return ()


class RestartablePipeline(StubPipeline):
    def __init__(self, frames):
        self.template = tuple(frames)
        super().__init__(())

    def start(self):
        self.started += 1
        self.frames = iter(self.template)


class PushPipeline:
    def __init__(self):
        self.frames = queue.Queue()
        self.started = self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def read(self, timeout=None):
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def take_health_events(self):
        return ()

    def push(self, frame):
        self.frames.put(frame)


class StubSerial:
    def __init__(self):
        self.packets = []

    def write(self, packet):
        self.packets.append(packet)
        return len(packet)


def test_runtime_direction_threshold_is_live_and_validated(tmp_path):
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}), project_root=tmp_path, pipeline=StubPipeline([]), serial_device=StubSerial()
    )
    assert runtime.direction_threshold == .35
    assert runtime.set_direction_threshold(.72) == .72
    assert runtime.direction_threshold == .72
    assert runtime.direction_scan_config.direction_threshold == .72
    with pytest.raises(ValueError):
        runtime.set_direction_threshold(1.01)


def test_runtime_iterative_switch_is_strict_and_revisioned(tmp_path):
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}), project_root=tmp_path, pipeline=StubPipeline([]), serial_device=StubSerial()
    )
    assert runtime.iterative_peak_search_enabled is False
    revision = runtime.direction_scan_config_revision
    assert runtime.set_iterative_peak_search_enabled(True) is True
    assert runtime.iterative_peak_search_enabled is True
    assert runtime.direction_scan_config_revision == revision + 1
    runtime.set_iterative_peak_search_enabled(True)
    assert runtime.direction_scan_config_revision == revision + 1
    for invalid in (1, 0, "true", None):
        with pytest.raises(ValueError):
            runtime.set_iterative_peak_search_enabled(invalid)


def test_runtime_kalman_and_id_switches_are_independent_and_revisioned(tmp_path):
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}), project_root=tmp_path,
        pipeline=StubPipeline([]), serial_device=StubSerial(),
    )
    assert runtime.direction_kalman_enabled is False
    assert runtime.direction_id_tracking_enabled is False
    revision = runtime.direction_scan_config_revision
    with pytest.raises(ValueError, match="ID tracking"):
        runtime.set_direction_kalman_enabled(True)
    runtime.set_direction_id_tracking_enabled(True)
    assert runtime.direction_scan_config_revision == revision + 1
    runtime.set_direction_kalman_enabled(True)
    assert runtime.direction_kalman_enabled is True
    assert runtime.direction_id_tracking_enabled is True
    assert runtime.direction_scan_config_revision == revision + 2
    runtime.set_direction_id_tracking_enabled(False)
    assert runtime.direction_id_tracking_enabled is False
    assert runtime.direction_kalman_enabled is False
    assert runtime.direction_scan_config_revision == revision + 3
    for setter in (
        runtime.set_direction_kalman_enabled,
        runtime.set_direction_id_tracking_enabled,
    ):
        with pytest.raises(ValueError):
            setter(1)


def test_runtime_kalman_q_r_scales_are_live_validated_and_revisioned(tmp_path):
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}), project_root=tmp_path,
        pipeline=StubPipeline([]), serial_device=StubSerial(),
    )
    assert runtime.direction_kalman_q_scale == 1.0
    assert runtime.direction_kalman_r_scale == 1.0
    revision = runtime.direction_scan_config_revision
    assert runtime.set_direction_kalman_q_scale(1.1) == 1.1
    assert runtime.set_direction_kalman_r_scale(0.9) == 0.9
    assert runtime.direction_scan_config_revision == revision + 2
    for invalid in (0.0, 10.02, 0.03, float("nan")):
        with pytest.raises(ValueError):
            runtime.set_direction_kalman_q_scale(invalid)


def test_runtime_probability_gate_threshold_is_validated_and_revisioned(tmp_path):
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}), project_root=tmp_path, pipeline=StubPipeline([]), serial_device=StubSerial()
    )
    assert runtime.gate_probability_threshold == 0.60
    revision = runtime.gate_config_revision
    assert runtime.set_gate_probability_threshold(0.45) == 0.45
    assert runtime.gate_config_revision == revision + 1
    runtime.set_gate_probability_threshold(0.45)
    assert runtime.gate_config_revision == revision + 1
    with pytest.raises(ValueError, match="probability threshold"):
        runtime.set_gate_probability_threshold(1.01)


def test_runtime_l1_pre_denoise_switch_is_live_and_strict(tmp_path):
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}), project_root=tmp_path,
        pipeline=StubPipeline([]), serial_device=StubSerial(),
    )
    assert runtime.l1_pre_denoise_enabled is False
    assert runtime.set_l1_pre_denoise_enabled(True) is True
    assert runtime.l1_pre_denoise_enabled is True
    assert runtime.set_l1_pre_denoise_enabled(False) is False
    with pytest.raises(ValueError, match="must be bool"):
        runtime.set_l1_pre_denoise_enabled(1)


def test_runtime_l1_pre_denoise_live_switch_never_duplicates_sample_ranges(tmp_path):
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}), project_root=tmp_path,
        pipeline=StubPipeline([]), serial_device=StubSerial(),
    )
    blocks = tuple(
        IngestedAudioBlock(
            "session", 0, index * 960, (index + 1) * 960, 48_000, index, index * 0.02,
            np.full((960, 8), index + 1, np.float32),
        )
        for index in range(4)
    )
    selected = list(runtime._select_pre_denoise(blocks[0]))
    runtime.set_l1_pre_denoise_enabled(True)
    selected.extend(runtime._select_pre_denoise(blocks[1]))
    selected.extend(runtime._select_pre_denoise(blocks[2]))
    runtime.set_l1_pre_denoise_enabled(False)
    selected.extend(runtime._select_pre_denoise(blocks[3]))
    selected.extend(runtime._flush_pre_denoise())
    assert [(item.start_sample, item.end_sample) for item in selected] == [
        (0, 960), (960, 1920), (1920, 2880), (2880, 3840),
    ]
    np.testing.assert_array_equal(selected[-1].samples, blocks[-1].samples)


def test_runtime_l3_mode_switch_is_available_before_and_during_capture(tmp_path):
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}),
        project_root=tmp_path,
        pipeline=PushPipeline(),
        serial_device=StubSerial(),
    )
    assert runtime.l3_processing_mode == L3_MODE_OPTIMIZED
    assert runtime.set_l3_processing_mode(L3_MODE_DS_BASELINE) == L3_MODE_DS_BASELINE
    assert runtime.set_l3_processing_mode(L3_MODE_CONSTANT_BEAMWIDTH) == L3_MODE_CONSTANT_BEAMWIDTH
    runtime.start()
    try:
        assert runtime.running
        assert runtime.set_l3_processing_mode(L3_MODE_OPTIMIZED) == L3_MODE_OPTIMIZED
        assert runtime.l3_processing_mode == L3_MODE_OPTIMIZED
        with pytest.raises(ValueError, match="unsupported L3 processing mode"):
            runtime.set_l3_processing_mode("mvdr")
    finally:
        runtime.stop()
        runtime.close()


def test_application_runtime_owns_single_chain_and_emits_window(tmp_path):
    frames = [DecodedAudio(np.zeros((960, 8), np.float32), 48_000, index, index * 0.02) for index in range(16)]
    pipeline = StubPipeline(frames)
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}),
        project_root=tmp_path,
        pipeline=pipeline,
        serial_device=StubSerial(),
    )
    runtime.start()
    deadline = time.monotonic() + 1
    while runtime.latest_windows.empty() and time.monotonic() < deadline:
        time.sleep(0.005)
    runtime.stop()
    window = runtime.latest_windows.get_nowait()
    assert window.decision_sample == 15_360
    assert pipeline.started == 1 and pipeline.stopped >= 1


def test_runtime_pre_denoise_replaces_audio_before_window_and_preserves_timeline(tmp_path):
    base = load_config(CONFIG, environ={})
    config = base.model_copy(update={
        "layer1_imcra": base.layer1_imcra.model_copy(update={"warmup_seconds": 0.02}),
        "layer1_pre_denoise": base.layer1_pre_denoise.model_copy(update={"enabled": True}),
    })
    rng = np.random.default_rng(82)
    frames = [
        DecodedAudio(rng.normal(0.0, 0.03, (960, 8)).astype(np.float32), 48_000, index, index * 0.02)
        for index in range(17)
    ]
    runtime = ApplicationRuntime(
        config, project_root=tmp_path, pipeline=StubPipeline(frames), serial_device=StubSerial(),
    )
    runtime.start()
    deadline = time.monotonic() + 2.0
    while runtime.latest_windows.empty() and time.monotonic() < deadline:
        time.sleep(0.005)
    window = runtime.latest_windows.get_nowait()
    latest_l1 = runtime.latest_l1.get_nowait()
    runtime.stop()
    runtime.close()
    assert window.context_start_sample == 0
    assert window.context_end_sample == 15_360
    assert window.samples.shape == (15_360, 8)
    assert len(window.imcra_hops) == 16
    assert latest_l1.pre_denoise_enabled is True
    assert np.isfinite(window.samples).all()


def test_runtime_light_control_uses_official_commands(tmp_path):
    serial = StubSerial()
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}),
        project_root=tmp_path,
        pipeline=StubPipeline([]),
        serial_device=serial,
    )
    runtime.set_light(True)
    assert runtime.light_state == "on"
    runtime.set_light(False)
    assert serial.packets == [b"E", b"e"]
    assert runtime.light_state == "off"


def test_runtime_light_control_reports_write_failure_while_stopped(tmp_path):
    class FailingSerial:
        def write(self, _packet):
            raise OSError("control port unavailable")

    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}),
        project_root=tmp_path,
        pipeline=StubPipeline([]),
        serial_device=FailingSerial(),
    )
    with pytest.raises(OSError, match="control port unavailable"):
        runtime.set_light(True)
    assert runtime.light_state == "error"


def test_runtime_can_stop_and_start_a_new_capture_session(tmp_path):
    frames = [DecodedAudio(np.zeros((960, 8), np.float32), 48_000, index, index * .02) for index in range(16)]
    pipeline = RestartablePipeline(frames)
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}), project_root=tmp_path, pipeline=pipeline, serial_device=StubSerial()
    )
    session_ids = []
    for _ in range(2):
        runtime.start()
        deadline = time.monotonic() + 2
        while runtime.latest_windows.empty() and time.monotonic() < deadline:
            time.sleep(.005)
        assert runtime.running
        session_ids.append(runtime.coordinator.session_id)
        runtime.stop()
        assert not runtime.running
    assert pipeline.started == 2
    assert session_ids[0] != session_ids[1]
    runtime.close()


def test_runtime_connects_l1_l2_formal_recording_and_ui_control(tmp_path):
    rng = np.random.default_rng(11)
    frames = [
        DecodedAudio(
            rng.normal(0, .05, (960, 8)).astype(np.float32),
            48_000,
            index,
            index * .02,
            native_samples=np.zeros((960, 8), np.float32),
        )
        for index in range(18)
    ]
    config = load_config(CONFIG, environ={})
    runtime = ApplicationRuntime(config, project_root=tmp_path, pipeline=StubPipeline(frames), serial_device=StubSerial())
    runtime.start()
    runtime.begin_runtime_recording()
    deadline = time.monotonic() + 3
    while runtime.latest_dev_ui.empty() and time.monotonic() < deadline:
        time.sleep(.005)
    while runtime.processing_error is None and runtime._performance._current_window is None and time.monotonic() < deadline:
        time.sleep(.005)
    runtime.pause_runtime_recording()
    runtime.stop()

    roots = list((tmp_path / "data/runtime_sessions").glob("*/*/*"))
    assert len(roots) == 1
    manifest = json.loads((roots[0] / "session_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["algorithm_versions"]["layer2_direction_kalman"] == "damped_circular_kalman_v2"
    assert manifest["algorithm_versions"]["layer2_direction_id_tracking"] == "confidence_id_tracker_v2"
    assert manifest["recorded_intervals"]
    assert manifest["chunks"][0]["result_count"] >= 1
    results_asset = next(x for x in manifest["chunks"][0]["assets"] if x["kind"] == "results")
    result_rows = [json.loads(line) for line in (roots[0] / results_asset["path"]).read_text(encoding="utf-8").splitlines()]
    assert result_rows[1]["window_id"] == 0
    assert result_rows[1]["session_id"] == manifest["session_id"]
    assert "l2_pipeline_state=blocked" in result_rows[1]["diagnostics"]
    assert "l2_gate_backend=mean_2x20ms_v1" in result_rows[1]["diagnostics"]
    assert "l2_gate_state=warming_up" in result_rows[1]["diagnostics"]


def _listening_window_and_candidate():
    window = DecisionWindow(
        "s", 0, 0, 15_360, 13_440, 15_360, 0, 15_360, 48_000,
        np.zeros((15_360, 8), np.float32), (0,),
    )
    candidate = TrackedDirection(
        "s", 0, 0, 15_360, 13_440, 15_360, 7, 1,
        30.0, 30.0, 1.0, 0.8, "confirmed", True, False,
        0, 15_360, 0, True,
    )
    return window, candidate


class _CapturingLayer3:
    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def process(self, window, candidates, geometry, *, mode):
        self.calls.append((tuple(candidates), mode))
        if self.fail:
            raise RuntimeError("formal L3 failure")
        outputs = []
        for candidate in candidates:
            outputs.append(SimpleNamespace(
                session_id=window.session_id, stream_epoch=window.stream_epoch,
                window_id=window.window_id, decision_sample=window.decision_sample,
                track_id=candidate.track_id, rank=candidate.rank,
                theta_deg=candidate.theta_deg, sample_rate=48_000,
                algorithm=("ds_baseline" if mode == L3_MODE_DS_BASELINE else "imcra_spatial_separation"),
                fallback_reason=None, diagnostics=(),
                enhanced_audio=np.zeros(15_360, np.float32),
            ))
        return SimpleNamespace(enhanced_audio=tuple(outputs))

def _runtime_with_layer3(layer3):
    runtime = ApplicationRuntime.__new__(ApplicationRuntime)
    runtime._layer3 = layer3
    runtime._geometry = object()
    runtime._l3_mode_lock = threading.Lock()
    runtime._l3_processing_mode = L3_MODE_OPTIMIZED
    return runtime


def test_runtime_passes_only_formal_smoothed_candidates_to_l3_once():
    window, candidate = _listening_window_and_candidate()
    layer3 = _CapturingLayer3()
    runtime = _runtime_with_layer3(layer3)

    formal, l4_inputs = runtime._process_l3(window, (candidate,), (17,))

    assert len(layer3.calls) == 1
    assert tuple(item.theta_deg for item in layer3.calls[0][0]) == (30.0,)
    assert layer3.calls[0][1] == L3_MODE_OPTIMIZED
    assert len(formal) == 1 and formal[0].theta_deg == 30.0
    assert len(l4_inputs) == 1 and l4_inputs[0].waveform.shape == (15_360,)
    assert l4_inputs[0].track_id == 17
    assert l4_inputs[0].array_source_probabilities_20ms == (None,) * 16


def test_runtime_aligns_all_sixteen_context_imcra_probabilities_to_l4_audio():
    window, candidate = _listening_window_and_candidate()
    hops = tuple(
        SimpleNamespace(
            session_id=window.session_id,
            stream_epoch=window.stream_epoch,
            start_sample=index * 960,
            end_sample=(index + 1) * 960,
            state="ready",
            array_source_probability_20ms=index / 15.0,
        )
        for index in range(16)
    )
    window = replace(window, imcra_hops=hops)
    runtime = _runtime_with_layer3(_CapturingLayer3())

    _, l4_inputs = runtime._process_l3(window, (candidate,), (17,))

    assert l4_inputs[0].array_source_probabilities_20ms == pytest.approx(
        tuple(index / 15.0 for index in range(16))
    )


def test_runtime_does_not_hide_a_formal_l3_failure():
    window, candidate = _listening_window_and_candidate()
    runtime = _runtime_with_layer3(_CapturingLayer3(fail=True))
    with pytest.raises(RuntimeError, match="formal L3 failure"):
        runtime._process_l3(window, (candidate,), (17,))


def test_runtime_passes_selected_ds_baseline_mode_to_l3():
    window, candidate = _listening_window_and_candidate()
    layer3 = _CapturingLayer3()
    runtime = _runtime_with_layer3(layer3)
    runtime.set_l3_processing_mode(L3_MODE_DS_BASELINE)

    previews, _l4_inputs = runtime._process_l3(window, (candidate,), (17,))

    assert layer3.calls[0][1] == L3_MODE_DS_BASELINE
    assert previews[0].runtime_backend == "ds_baseline"


def test_runtime_mode_switch_never_changes_authoritative_l2_id():
    window, direction = _listening_window_and_candidate()
    layer3 = _CapturingLayer3()
    runtime = _runtime_with_layer3(layer3)

    for mode in (
        L3_MODE_OPTIMIZED,
        L3_MODE_DS_BASELINE,
        L3_MODE_CONSTANT_BEAMWIDTH,
    ):
        runtime.set_l3_processing_mode(mode)
        runtime._process_l3(window, (direction,))

    assert [call[0][0].track_id for call in layer3.calls] == [7, 7, 7]
    assert [call[0][0].rank for call in layer3.calls] == [1, 1, 1]
