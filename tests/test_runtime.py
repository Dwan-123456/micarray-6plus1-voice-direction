from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import json
import pytest
import torch

from app.runtime import ApplicationRuntime
from common.config import load_config
from common.data_types import (
    DecisionWindow,
    IngestedAudioBlock,
    TrackedDirection,
)
from layer1_input.interface import DecodedAudio
from layer3_direction_signal import (
    L3_MODE_DS_BASELINE,
    L3_MODE_LOADED_MVDR,
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


def test_runtime_uses_independent_l3_l4_l5_devices(tmp_path):
    config = load_config(CONFIG, environ={})
    runtime = ApplicationRuntime(
        config, project_root=tmp_path,
        pipeline=StubPipeline([]), serial_device=StubSerial(),
    )

    assert runtime.l3_device == "cpu"
    assert runtime.l4_device == ("cuda" if torch.cuda.is_available() else "cpu")
    assert runtime.l5_device == "cpu"
    assert runtime.processing_device == runtime.l3_device
    assert runtime.processing_status["devices"] == {
        "l1": "cpu", "l2": "cpu", "l3": runtime.l3_device,
        "l4": runtime.l4_device, "l5": "cpu",
    }
    runtime.close()


def test_legacy_runtime_device_falls_back_for_each_layer():
    config = load_config(CONFIG, environ={})
    legacy_runtime = config.runtime.model_copy(update={
        "preferred_device": "CPU",
        "l3_device": None,
        "l4_device": None,
        "l5_device": None,
    })
    legacy = config.model_copy(update={"runtime": legacy_runtime})

    assert ApplicationRuntime._resolve_layer_device(legacy, "l3_device") == "cpu"
    assert ApplicationRuntime._resolve_layer_device(legacy, "l4_device") == "cpu"
    assert ApplicationRuntime._resolve_layer_device(legacy, "l5_device") == "cpu"


def test_l4_cuda_policy_honors_cpu_fallback(monkeypatch):
    config = load_config(CONFIG, environ={})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert ApplicationRuntime._resolve_layer_device(config, "l4_device") == "cpu"


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


def test_offline_l4_builder_uses_its_independent_device(tmp_path, monkeypatch):
    captured = {}
    backend_builds = []
    quality_scorer = SimpleNamespace(score=lambda _waveform: (4.0, 4.0, 4.0))
    quality_artifacts = []
    campplus_builds = []

    def build_backend(artifact, *, device):
        captured.update(artifact=artifact, device=device)
        value = SimpleNamespace(backend_id="mossformer2_ss_16k")
        backend_builds.append(value)
        return value

    monkeypatch.setattr("app.runtime.MossFormer2Backend", build_backend)
    monkeypatch.setattr(
        "app.runtime.DnsMosScorer",
        lambda artifact: quality_artifacts.append(artifact) or quality_scorer,
    )
    monkeypatch.setattr(
        "app.runtime.CampPlusEmbedder",
        lambda _artifact: campplus_builds.append(object()) or campplus_builds[-1],
    )
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}), project_root=tmp_path,
        pipeline=StubPipeline([]), serial_device=StubSerial(),
    )

    pipeline = runtime.build_offline_l4_pipeline("mossformer2_ss_16k")

    assert captured["device"] == runtime.l4_device
    assert pipeline.backends["mossformer2_ss_16k"].backend_id == "mossformer2_ss_16k"
    assert pipeline.quality_scorer is quality_scorer
    second_pipeline = runtime.build_offline_l4_pipeline()
    first_l6 = runtime.build_offline_l6_pipeline()
    second_l6 = runtime.build_offline_l6_pipeline()
    assert second_pipeline.quality_scorer is quality_scorer
    assert second_pipeline.backends["mossformer2_ss_16k"] is pipeline.backends[
        "mossformer2_ss_16k"
    ]
    assert first_l6.config is runtime.config.layer6
    assert second_l6.embedder is first_l6.embedder
    assert len(backend_builds) == 1
    assert len(campplus_builds) == 1
    assert len(quality_artifacts) == 1
    runtime.close()


def test_offline_l4_releases_drained_l3_cuda_cache(tmp_path, monkeypatch):
    events = []

    class Stream:
        def synchronize(self):
            events.append("synchronize")

    def build_backend(artifact, *, device):
        del artifact
        return SimpleNamespace(backend_id="mossformer2_ss_16k", device=device)

    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}), project_root=tmp_path,
        pipeline=StubPipeline([]), serial_device=StubSerial(),
    )
    runtime.l3_device = runtime.l4_device = "cuda"
    runtime._l3_cuda_stream = Stream()
    monkeypatch.setattr(runtime._layer3, "clear_cache", lambda: events.append("clear"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty"))
    monkeypatch.setattr("app.runtime.MossFormer2Backend", build_backend)
    monkeypatch.setattr(
        "app.runtime.DnsMosScorer",
        lambda _artifact: SimpleNamespace(score=lambda _waveform: (4.0, 4.0, 4.0)),
    )

    pipeline = runtime.build_offline_l4_pipeline("mossformer2_ss_16k")

    assert events == ["synchronize", "clear", "empty"]
    assert pipeline.backends["mossformer2_ss_16k"].device == "cuda"
    runtime.close()


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


def test_first_confirmed_id_queues_only_pre_birth_one_second_backfill(tmp_path):
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}), project_root=tmp_path,
        pipeline=StubPipeline([]), serial_device=StubSerial(),
    )
    windows = tuple(
        DecisionWindow(
            "backfill-session", 0, index, decision,
            decision - 1_920, decision, decision - 7_680, decision,
            48_000, np.zeros((7_680, 8), np.float32), (index,),
        )
        for index, decision in enumerate((7_680, 8_640, 9_600, 10_560))
    )
    for window in windows:
        runtime._remember_confirmed_backfill_window(window)
    track = TrackedDirection(
        "backfill-session", 0, 3, 10_560, 8_640, 10_560,
        7, 1, 32.0, 30.0, 1.0, 0.9, "confirmed", True, False,
        9_600, 10_560, 0, True,
    )
    item = SimpleNamespace(key=SimpleNamespace(
        session_id="backfill-session", stream_epoch=0, decision_sample=10_560,
    ))

    runtime._schedule_confirmed_backfill(
        item, (track,), processing_mode=L3_MODE_OPTIMIZED,
        l2_direction_count=1,
    )
    queued = runtime._confirmed_backfill_work.get_nowait()
    runtime._schedule_confirmed_backfill(
        item, (track,), processing_mode=L3_MODE_OPTIMIZED,
        l2_direction_count=1,
    )

    assert tuple(window.decision_sample for window in queued.windows) == (7_680, 8_640)
    assert queued.track is track
    assert runtime._confirmed_backfill_work.empty()
    layer3 = _CapturingLayer3()
    runtime._layer3_backfill = layer3
    runtime.downstream_window_spec = SimpleNamespace(samples=7_680, decision_hops=8)
    runtime.track_audio_stream.observe_l2(
        identity=("backfill-session", 0, 3, 10_560),
        active_tracks=(track,), processing_mode=L3_MODE_OPTIMIZED,
        l2_direction_count=1,
    )
    runtime._process_confirmed_backfill(queued)
    sealed = runtime.track_audio_stream.seal()[0]

    assert tuple(
        candidate.theta_deg
        for candidates, _mode in layer3.calls
        for candidate in candidates
    ) == (30.0, 30.0)
    assert sealed.start_sample == 5_760
    runtime.close()


def test_processing_status_exposes_input_discontinuity_reason(tmp_path):
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}), project_root=tmp_path,
        pipeline=StubPipeline([]), serial_device=StubSerial(),
    )
    first = DecodedAudio(np.zeros((960, 8), np.float32), 48_000, 0, 0.0)
    after_gap = DecodedAudio(np.zeros((960, 8), np.float32), 48_000, 2, 0.04)
    runtime.coordinator.ingest(first)
    runtime.coordinator.ingest(after_gap)

    health = runtime.processing_status["input_health"]

    assert health["stream_epoch"] == 1
    assert health["discontinuity_count"] == 1
    assert health["last_discontinuity"]["reason"] == "sequence_gap"
    assert health["input_overflow_count"] == 0
    assert health["handoff_drop_count"] == 0
    runtime.close()


def test_runtime_processing_snapshot_freezes_music_and_imm_jpda_lifecycle(tmp_path):
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}), project_root=tmp_path,
        pipeline=StubPipeline([]), serial_device=StubSerial(),
    )
    snapshot = runtime._capture_processing_config()
    values = snapshot.values
    assert "iterative_peak_search_enabled" not in values
    assert values["direction_id_tracking_enabled"] is True
    assert values["music_history_ms"] in {160, 240, 320}
    assert values["music_stft"] == {
        "n_fft": 1024, "win_length": 960, "hop_length": 480, "window": "hann_periodic",
    }
    assert values["music_frequency_band_hz"] == (2_000.0, 4_000.0)
    assert values["music_order"] == {
        "source": "test_ui_manual",
        "value": 2,
        "min_valid_frequency_bins": 12,
    }
    assert values["association_lifecycle"]["coasting_ttl_ms"] > 0
    assert values["association_config_revision"] == 0
    assert values["kalman_config_revision"] == 0

    assert values["direction_kalman_enabled"] is True
    assert runtime.config.layer2.direction_id_tracking.backend == "circular_imm_jpda_v1"


def test_runtime_has_only_id_tracking_as_public_tracking_switch(tmp_path):
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}), project_root=tmp_path,
        pipeline=StubPipeline([]), serial_device=StubSerial(),
    )
    assert runtime.direction_kalman_enabled is True
    assert runtime.direction_id_tracking_enabled is True
    revision = runtime.direction_scan_config_revision
    assert runtime.set_direction_id_tracking_enabled(False) is False
    assert runtime.direction_id_tracking_enabled is False
    assert runtime.direction_scan_config_revision == revision + 1
    assert runtime.set_direction_id_tracking_enabled(False) is False
    assert runtime.direction_scan_config_revision == revision + 1
    assert runtime.set_direction_id_tracking_enabled(True) is True
    assert runtime.direction_id_tracking_enabled is True
    assert runtime.direction_scan_config_revision == revision + 2
    with pytest.raises(ValueError):
        runtime.set_direction_id_tracking_enabled(1)


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


def test_runtime_optional_music_filters_follow_config_and_are_revisioned(tmp_path):
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}), project_root=tmp_path,
        pipeline=StubPipeline([]), serial_device=StubSerial(),
    )
    assert runtime.music_dpd_rank1_enabled is False
    assert runtime.music_noise_whitening_enabled is False
    revision = runtime.direction_scan_config_revision
    assert runtime.set_music_dpd_rank1_enabled(True) is True
    assert runtime.set_music_noise_whitening_enabled(True) is True
    assert runtime.direction_scan_config_revision == revision + 2
    runtime.set_music_noise_whitening_enabled(True)
    assert runtime.direction_scan_config_revision == revision + 2
    assert runtime.direction_scan_config.dpd_rank1_enabled is True
    assert runtime.direction_scan_config.noise_whitening_enabled is True
    with pytest.raises(ValueError):
        runtime.set_music_dpd_rank1_enabled(1)
    with pytest.raises(ValueError):
        runtime.set_music_noise_whitening_enabled(1)


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
    assert runtime.set_l1_pre_denoise_enabled(False) is False
    assert runtime.l1_pre_denoise_enabled is False
    assert runtime.set_l1_pre_denoise_enabled(True) is True
    with pytest.raises(ValueError, match="must be bool"):
        runtime.set_l1_pre_denoise_enabled(1)


def test_runtime_l1_speaker_count_switch_is_live_and_strict(tmp_path):
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}), project_root=tmp_path,
        pipeline=StubPipeline([]), serial_device=StubSerial(),
    )
    assert runtime.l1_speaker_count_enabled is False
    assert runtime.set_l1_speaker_count_enabled(True) is True
    assert runtime.l1_speaker_count_enabled is True
    assert runtime.set_l1_speaker_count_enabled(False) is False
    with pytest.raises(ValueError, match="must be bool"):
        runtime.set_l1_speaker_count_enabled(1)


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


def test_runtime_marks_only_actual_imcra_outputs_for_center_preview(tmp_path):
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
    selected = list(runtime._select_pre_denoise_with_mode(blocks[0]))
    runtime.set_l1_pre_denoise_enabled(True)
    selected.extend(runtime._select_pre_denoise_with_mode(blocks[1]))
    selected.extend(runtime._select_pre_denoise_with_mode(blocks[2]))
    runtime.set_l1_pre_denoise_enabled(False)
    selected.extend(runtime._select_pre_denoise_with_mode(blocks[3]))
    selected.extend(runtime._flush_pre_denoise_with_mode())

    assert [(item.start_sample, denoised) for item, denoised in selected] == [
        (0, False),
        (960, True),
        (1920, False),
        (2880, False),
    ]


def test_runtime_caches_only_marked_imcra_center_selection(tmp_path, monkeypatch):
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}), project_root=tmp_path,
        pipeline=StubPipeline([]), serial_device=StubSerial(),
    )
    block = IngestedAudioBlock(
        "session", 0, 0, 960, 48_000, 0, 0.0, np.zeros((960, 8), np.float32),
    )
    cached, published = [], []
    runtime.dev_audio_tracker = SimpleNamespace(
        append_imcra_center_reference=lambda item, channel_index: cached.append(
            (item, channel_index)
        ),
    )
    monkeypatch.setattr(
        runtime, "_publish_l1_block",
        lambda item, received: published.append((item, received)),
    )

    runtime._publish_l1_selection(block, 1.0, denoised=False)
    runtime._publish_l1_selection(block, 2.0, denoised=True)

    assert cached == [(block, 6)]
    assert published == [(block, 1.0), (block, 2.0)]


def test_runtime_l3_mode_switch_is_available_before_and_during_capture(tmp_path):
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}),
        project_root=tmp_path,
        pipeline=PushPipeline(),
        serial_device=StubSerial(),
    )
    assert runtime.l3_processing_mode == L3_MODE_DS_BASELINE
    assert runtime.set_l3_processing_mode(L3_MODE_DS_BASELINE) == L3_MODE_DS_BASELINE
    assert runtime.set_l3_processing_mode(L3_MODE_LOADED_MVDR) == L3_MODE_LOADED_MVDR
    with pytest.raises(ValueError, match="unsupported L3 processing mode"):
        runtime.set_l3_processing_mode("subband_robust_baseline")
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
    frames = [DecodedAudio(np.zeros((960, 8), np.float32), 48_000, index, index * 0.02) for index in range(8)]
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
    assert window.decision_sample == 7_680
    assert pipeline.started == 1 and pipeline.stopped >= 1


def test_stop_waits_for_forced_commit_exit_and_reports_success(tmp_path):
    base = load_config(CONFIG, environ={})
    runtime_config = base.runtime.model_copy(update={
        "graceful_shutdown_timeout_seconds": 0.05,
    })
    runtime = ApplicationRuntime(
        base.model_copy(update={"runtime": runtime_config}),
        project_root=tmp_path,
        pipeline=StubPipeline([]),
        serial_device=StubSerial(),
    )
    commit = threading.Thread(target=lambda: time.sleep(0.08), daemon=True)
    runtime._processing_threads = {"commit": commit}
    runtime._processing_thread = commit
    commit.start()

    runtime.stop()

    assert not commit.is_alive()
    assert runtime.last_error is None
    assert runtime.active is False
    assert runtime.processing_queue_depths == {
        "l2": 0, "l3": 0, "l3_prepared": 0, "l3_host": 0,
        "l5": 0, "completion": 0,
    }


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
    # The runtime mailbox intentionally keeps only the newest window, so a
    # completed successor can replace window 0 before the test thread wakes.
    assert window.context_start_sample == window.window_id * 960
    assert window.context_end_sample == window.context_start_sample + 7_680
    assert window.samples.shape == (7_680, 8)
    assert len(window.imcra_hops) == 8
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
    frames = [DecodedAudio(np.zeros((960, 8), np.float32), 48_000, index, index * .02) for index in range(8)]
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


def test_live_test_ui_capture_keeps_imcra_but_resets_l2_before_temporary_l3(
    tmp_path, monkeypatch,
):
    runtime = ApplicationRuntime(
        load_config(CONFIG, environ={}),
        project_root=tmp_path,
        pipeline=StubPipeline([]),
        serial_device=StubSerial(),
        ephemeral_live_capture=True,
    )
    runtime.start()
    warmed_imcra = runtime.imcra
    warmed_l2 = runtime._layer2
    reset_calls = 0
    original_reset = warmed_l2.reset

    def recording_reset():
        nonlocal reset_calls
        reset_calls += 1
        original_reset()

    monkeypatch.setattr(warmed_l2, "reset", recording_reset)

    assert runtime.runtime_recording_mode == "temporary"
    assert runtime.runtime_recording_active is False
    assert runtime.downstream_processing_enabled is False
    assert runtime.set_downstream_processing_enabled(True) is False
    assert not (tmp_path / "data/runtime_sessions").exists()

    runtime.begin_runtime_recording()

    assert runtime.runtime_recording_active is True
    assert runtime.downstream_processing_enabled is True
    assert runtime.imcra is warmed_imcra
    assert runtime._layer2 is warmed_l2
    assert reset_calls == 1
    assert runtime._recording_session_started is False

    runtime.pause_runtime_recording()
    assert runtime.runtime_recording_active is False
    assert runtime.downstream_processing_enabled is False
    runtime.stop()
    runtime.close()
    assert not (tmp_path / "data/runtime_sessions").exists()


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
    assert manifest["algorithm_versions"]["layer2_direction_tracker"] == "circular_imm_jpda_v1"
    assert manifest["algorithm_versions"]["layer2_direction_id_tracking"] == "circular_imm_jpda_v1"
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
        "s", 0, 0, 7_680, 5_760, 7_680, 0, 7_680, 48_000,
        np.zeros((7_680, 8), np.float32), (0,),
    )
    candidate = TrackedDirection(
        "s", 0, 0, 7_680, 5_760, 7_680, 7, 1, 28.0, 30.0, 1.0, 0.8,
        "confirmed", True, False, 5_760, 7_680, 0, True,
    )
    return window, candidate


class _CapturingLayer3:
    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    @staticmethod
    def clear_cache():
        return None

    def process(self, window, candidates, geometry, *, mode):
        self.calls.append((tuple(candidates), mode))
        if self.fail:
            raise RuntimeError("formal L3 failure")
        outputs = []
        for candidate in candidates:
            algorithm = {
                L3_MODE_DS_BASELINE: "ds_baseline",
                L3_MODE_LOADED_MVDR: "loaded_mvdr_baseline",
            }.get(mode, "imcra_spatial_separation")
            outputs.append(SimpleNamespace(
                session_id=window.session_id, stream_epoch=window.stream_epoch,
                window_id=window.window_id, decision_sample=window.decision_sample,
                theta_deg=candidate.theta_deg, sample_rate=48_000,
                algorithm=algorithm,
                fallback_reason=None, diagnostics=(),
                enhanced_audio=np.zeros(7_680, np.float32),
                track_id=candidate.track_id,
            ))
        return SimpleNamespace(enhanced_audio=tuple(outputs))

def _runtime_with_layer3(layer3):
    runtime = ApplicationRuntime.__new__(ApplicationRuntime)
    runtime._layer3 = layer3
    runtime._geometry = object()
    runtime._thread = None
    runtime._realtime_mode_submission_lock = threading.RLock()
    runtime._l3_mode_lock = threading.Lock()
    runtime._l3_processing_mode = L3_MODE_OPTIMIZED
    return runtime


def test_runtime_passes_only_formal_smoothed_candidates_to_l3_once():
    window, candidate = _listening_window_and_candidate()
    layer3 = _CapturingLayer3()
    runtime = _runtime_with_layer3(layer3)

    formal = runtime._process_l3(window, (candidate,))

    assert len(layer3.calls) == 1
    assert tuple(item.theta_deg for item in layer3.calls[0][0]) == (30.0,)
    assert layer3.calls[0][1] == L3_MODE_OPTIMIZED
    assert len(formal) == 1 and formal[0].theta_deg == 30.0


def test_runtime_does_not_hide_a_formal_l3_failure():
    window, candidate = _listening_window_and_candidate()
    runtime = _runtime_with_layer3(_CapturingLayer3(fail=True))
    with pytest.raises(RuntimeError, match="formal L3 failure"):
        runtime._process_l3(window, (candidate,))


def test_runtime_passes_selected_ds_baseline_mode_to_l3():
    window, candidate = _listening_window_and_candidate()
    layer3 = _CapturingLayer3()
    runtime = _runtime_with_layer3(layer3)
    runtime.set_l3_processing_mode(L3_MODE_DS_BASELINE)

    previews = runtime._process_l3(window, (candidate,))

    assert layer3.calls[0][1] == L3_MODE_DS_BASELINE
    assert previews[0].runtime_backend == "ds_baseline"


def test_runtime_passes_selected_loaded_mvdr_baseline_mode_to_l3():
    window, candidate = _listening_window_and_candidate()
    layer3 = _CapturingLayer3()
    runtime = _runtime_with_layer3(layer3)
    runtime.set_l3_processing_mode(L3_MODE_LOADED_MVDR)

    previews = runtime._process_l3(window, (candidate,))

    assert layer3.calls[0][1] == L3_MODE_LOADED_MVDR
    assert previews[0].runtime_backend == "loaded_mvdr_baseline"


def test_runtime_mode_switch_never_changes_authoritative_l2_id():
    window, direction = _listening_window_and_candidate()
    layer3 = _CapturingLayer3()
    runtime = _runtime_with_layer3(layer3)

    for mode in (
        L3_MODE_OPTIMIZED,
        L3_MODE_DS_BASELINE,
        L3_MODE_LOADED_MVDR,
    ):
        runtime.set_l3_processing_mode(mode)
        runtime._process_l3(window, (direction,))

    assert [call[0][0].track_id for call in layer3.calls] == [7, 7, 7]
    assert [call[0][0].rank for call in layer3.calls] == [1, 1, 1]
