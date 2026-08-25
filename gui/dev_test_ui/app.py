from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import queue
import sys
import wave
from collections.abc import Mapping
from pathlib import Path
from time import monotonic, perf_counter
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from app.runtime import ApplicationRuntime
from common.config import load_config
from layer1_input.calibration import ChannelCalibrator
from layer1_input.configuration import CalibrationConfig
from layer1_input.pipeline import InputPipeline
from layer1_input.sources import WavAudioSource
from layer1_input.recording_replay import RecordingReplaySource


def _time(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f} ms"


def _runtime_processing_status(runtime: object) -> Mapping[str, Any] | None:
    """Read the public runtime diagnostic snapshot without coupling UI to queues."""

    try:
        status = getattr(runtime, "processing_status", None)
        status = status() if callable(status) else status
    except Exception:
        return None
    return status if isinstance(status, Mapping) else None


def _is_current_monotonic_l2_snapshot(
    snapshot: object,
    current_stream: tuple[str, int],
    previous_identity: tuple[str, int, int, int] | None,
) -> bool:
    identity = (
        str(getattr(snapshot, "session_id", "")),
        int(getattr(snapshot, "stream_epoch", -1)),
        int(getattr(snapshot, "window_id", -1)),
        int(getattr(snapshot, "decision_sample", -1)),
    )
    if identity[:2] != current_stream or min(identity[1:]) < 0:
        return False
    return not (
        previous_identity is not None
        and identity[:2] == previous_identity[:2]
        and identity[2:] <= previous_identity[2:]
    )


def _format_processing_pipeline_status(
    runtime: object, status: Mapping[str, Any] | None = None,
) -> str:
    status = _runtime_processing_status(runtime) if status is None else status
    if status is None:
        return "pipeline telemetry unavailable"

    depths = status.get("queue_depths", {})
    capacities = status.get("queue_capacities", {})
    alive = status.get("stage_alive", {})
    completed = status.get("completed_counts", {})
    errors = status.get("error_counts", {})

    def stage(label: str, queue_names: tuple[str, ...], worker_name: str) -> str:
        queue_name = next(
            (name for name in queue_names if isinstance(depths, Mapping) and name in depths),
            queue_names[0],
        )
        depth = int(depths.get(queue_name, 0)) if isinstance(depths, Mapping) else 0
        capacity = int(capacities.get(queue_name, 0)) if isinstance(capacities, Mapping) else 0
        running = bool(alive.get(worker_name, False)) if isinstance(alive, Mapping) else False
        done = int(completed.get(worker_name, 0)) if isinstance(completed, Mapping) else 0
        failed = int(errors.get(worker_name, 0)) if isinstance(errors, Mapping) else 0
        downstream_enabled = bool(status.get("downstream_processing_enabled", True))
        state = (
            "OFF"
            if worker_name in {"l3", "l5"} and not downstream_enabled
            else "RUN" if running else "STOP"
        )
        suffix = f" !{failed}" if failed else ""
        if worker_name == "l5":
            actual = max(0, int(status.get("l5_actual_completed", 0)))
            dropped = max(0, int(status.get("l5_dropped", 0)))
            actual_hz = max(0.0, float(status.get("l5_actual_hz", 0.0)))
            suffix += f" ok{actual} d{dropped} @{actual_hz:.0f}Hz"
        return f"{label} {depth}/{capacity} {state} #{done}{suffix}"

    segments = (
        stage("L2", ("l2",), "l2"),
        stage("L3", ("l3",), "l3"),
        stage("L5", ("l5",), "l5"),
        stage("JOIN", ("completion", "commit"), "commit"),
    )
    inflight = max(0, int(status.get("inflight_windows", 0)))
    cache_bytes = max(0, int(status.get("cache_bytes", 0)))
    cache_max_bytes = max(0, int(status.get("cache_max_bytes", 0)))
    cache_text = f"cache {cache_bytes / (1024 * 1024):.1f}/{cache_max_bytes / (1024 * 1024):.1f} MiB"
    return " | ".join((*segments, f"flight {inflight}", cache_text))


def _format_processing_pipeline_tooltip(
    runtime: object, status: Mapping[str, Any] | None = None,
) -> str:
    status = _runtime_processing_status(runtime) if status is None else status
    if status is None:
        return "当前Runtime未提供分层流水线诊断。"
    latest_errors = status.get("latest_errors", {})
    error_text = "无"
    if isinstance(latest_errors, Mapping):
        items = [f"{name}: {value}" for name, value in latest_errors.items() if value]
        if items:
            error_text = "；".join(items)
    return (
        "分层队列格式为 当前深度/容量；# 后为该阶段完成窗口数。\n"
        f"当前流水中窗口：{max(0, int(status.get('inflight_windows', 0)))}\n"
        f"入口丢窗累计：{max(0, int(status.get('processing_drops', 0)))}\n"
        f"L3设备缓存：{max(0, int(status.get('l3_device_cache_bytes', 0))) / (1024 * 1024):.1f} MiB，"
        f"Prepared {max(0, int(status.get('l3_prepared_cache_entries', 0)))}/"
        f"{max(0, int(status.get('l3_prepared_cache_limit', 0)))}\n"
        f"L5实际推理完成：{max(0, int(status.get('l5_actual_completed', 0)))}，"
        f"丢弃：{max(0, int(status.get('l5_dropped', 0)))}，"
        f"跳过：{max(0, int(status.get('l5_skipped', 0)))}，"
        f"最近1秒实际刷新率：{max(0.0, float(status.get('l5_actual_hz', 0.0))):.1f} Hz\n"
        f"L5显示邮箱：{max(0, int(status.get('l5_ui_mailbox_depth', 0)))}/"
        f"{max(0, int(status.get('l5_ui_mailbox_capacity', 0)))}，"
        f"latest-only覆盖：{max(0, int(status.get('l5_ui_mailbox_overwrites', 0)))}\n"
        f"最近阶段错误：{error_text}"
    )


def build_window(
    config_path: str | Path,
    *,
    input_wav: str | Path | None = None,
    replay_recording: str | Path | None = None,
    auto_start: bool = False,
):
    try:
        from PySide6.QtCore import QSignalBlocker, QTimer, Qt
        from PySide6.QtWidgets import (
            QApplication, QCheckBox, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
            QPushButton, QProgressBar, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
        )
    except ImportError as exc:
        raise RuntimeError("Development Test UI需要安装项目ui依赖") from exc

    from .panels import BeamformPanel, CnnPanel, Layer4AudioPanel, Layer6AudioPanel
    from .preview_player import PreviewPlayer
    from .audio_id_tracker import (
        AudioIdTracker,
        CENTER_IMCRA_TRACK_ID,
        CENTER_RAW_TRACK_ID,
    )
    from .offline_l4_store import OfflineLayer4UiStore
    from .offline_l6_store import OfflineLayer6UiStore
    from .panels import (
        GateProbabilityThresholdControl,
        DirectionIdTrackingControl,
        DoaBackendControl,
        MusicDpdRank1Control,
        MusicNoiseWhiteningControl,
        MusicOrderLimitControl,
        ProbabilityGateReadout,
        SrpThresholdControl,
    )
    from .settings import DevUiSettings
    from .srp_panel import MusicPanelSnapshot, MusicPolarPanel

    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    project_root = config_path.parent.parent
    if input_wav is not None and replay_recording is not None:
        raise ValueError("input_wav和replay_recording不能同时使用")
    replay_path = None if input_wav is None else Path(input_wav).resolve()
    recording_manifest = None if replay_recording is None else Path(replay_recording).resolve()
    replay_source = None
    pipeline = None
    if recording_manifest is not None:
        replay_source = RecordingReplaySource(
            recording_manifest,
            logical_channel_map=config.device.logical_channel_map,
            block_size=config.device.block_size_samples,
            autoplay=auto_start,
        )
        pipeline = InputPipeline(
            replay_source,
            ChannelCalibrator(CalibrationConfig.from_project(config)),
            timestamp_tolerance_ms=config.timing.timestamp_tolerance_ms,
        )
    elif replay_path is not None:
        with wave.open(str(replay_path), "rb") as source:
            channels = source.getnchannels()
            sample_rate = source.getframerate()
            sample_width = source.getsampwidth()
        if sample_rate != config.device.sample_rate or channels not in {7, 8} or sample_width != 2:
            raise ValueError("模拟测试音频必须是48 kHz、7或8通道、PCM16 WAV")
        channel_map = config.device.logical_channel_map if channels == 8 else None
        pipeline = InputPipeline(
            WavAudioSource(
                replay_path,
                block_size=config.device.block_size_samples,
                channel_map=channel_map,
                realtime=True,
            ),
            ChannelCalibrator(CalibrationConfig.from_project(config)),
            timestamp_tolerance_ms=config.timing.timestamp_tolerance_ms,
        )
    audio_id_tracker = AudioIdTracker(
        f"data/dev_test_ui/l3_audio_cache/current/run_{uuid4().hex}",
        project_root=project_root,
        downstream_window_samples=config.downstream_audio_window.samples,
        minimum_listening_track_seconds=config.dev_test_ui.minimum_listening_track_seconds,
    )
    live_microphone_mode = replay_source is None and replay_path is None
    runtime = ApplicationRuntime(
        config, project_root=project_root, pipeline=pipeline,
        dev_audio_tracker=audio_id_tracker,
        ephemeral_live_capture=live_microphone_mode,
    )
    ui_settings = DevUiSettings(config_path.parent.parent)
    persisted_l4_backend = ui_settings.load_layer4_backend(
        config.layer4.default_backend
    )
    persisted_threshold = ui_settings.load_direction_threshold(config.layer2.direction_threshold)
    runtime.set_direction_threshold(persisted_threshold)
    runtime.set_doa_backend(ui_settings.load_doa_backend(config.layer2.scanner_backend))
    runtime.set_music_effective_order_limit(ui_settings.load_music_effective_order_limit(
        config.layer2.effective_order_limit
    ))
    runtime.set_music_dpd_rank1_enabled(ui_settings.load_music_dpd_rank1_enabled(
        config.layer2.dpd_rank1_enabled
    ))
    runtime.set_music_noise_whitening_enabled(
        ui_settings.load_music_noise_whitening_enabled(
            config.layer2.noise_whitening_enabled
        )
    )
    runtime.set_direction_id_tracking_enabled(
        ui_settings.load_direction_id_tracking_enabled(True)
    )
    persisted_gate_threshold = ui_settings.load_gate_probability_threshold(
        config.layer2.probability_gate.threshold
    )
    runtime.set_gate_probability_threshold(persisted_gate_threshold)
    runtime.set_l1_pre_denoise_enabled(ui_settings.load_l1_pre_denoise_enabled(
        config.layer1_pre_denoise.enabled
    ))
    runtime.set_l1_speaker_count_enabled(ui_settings.load_l1_speaker_count_enabled(
        config.layer1_speaker_count.enabled
    ))
    runtime.set_l5_input_gain_compensation_enabled(
        ui_settings.load_l5_input_gain_compensation_enabled(
            config.layer5.input_gain_compensation.enabled
        )
    )

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self._runtime = runtime
            if replay_source is not None:
                self.setWindowTitle(f"Development Test UI — 模拟输入模式：{replay_source.display_name}")
            elif replay_path is not None:
                self.setWindowTitle(f"Development Test UI — 模拟测试：{replay_path.name}")
            else:
                self.setWindowTitle("6+1 Microphone Array — Development Test UI")
            self.setMinimumSize(1200, 700)
            self._frame = None
            self._l2_frame = None
            self._last_l2_identity = None
            self._last_l5_frame = None
            self._last_l5_seen = 0.0
            self._l5_is_stale = False
            self._selected = None
            self._last_l1_seen = 0.0
            self._record_started = None
            self._record_elapsed = 0.0
            self._commands = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dev-ui-command")
            self._pending_command: tuple[str, Future, object | None] | None = None
            self._closing = False
            self._last_rendered_window = None
            self._last_performance_refresh = 0.0
            self._last_runtime_state = "stopped"
            self._eof_stop_submitted = False
            self._audio_source_key = None
            self._offline_l4_pipeline = None
            self._l4_processed = ()
            self._l5_results = ()
            self._l6_result = None
            self._offline_stage_durations_seconds = {"l4": None, "l5": None, "l6": None}
            self._l4_store = OfflineLayer4UiStore()
            self._l6_store = OfflineLayer6UiStore()
            self._replay_previous_stream = None
            self._replay_reset_pending = False
            root = QWidget()
            outer = QVBoxLayout(root)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)
            self.global_status = QLabel(
                f"STOPPED | input drop 0 | processing drop 0 | L2 {runtime.doa_backend} + {runtime.processing_device.upper()} L3 | queue 0"
            )
            self.global_status.setFixedHeight(28)
            self.global_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            self.global_status.setStyleSheet("background:#17212b;color:#dce7f2;padding-left:8px")
            outer.addWidget(self.global_status)
            quadrants = QWidget()
            quadrants.setObjectName("quadrants")
            grid = QGridLayout(quadrants)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(1)
            for index in (0, 1):
                grid.setRowStretch(index, 1)
            for index in range(6):
                grid.setColumnStretch(index, 1)
            grid.addWidget(self._l1_panel(), 0, 0, 1, 3)
            grid.addWidget(self._doa_panel(), 0, 3, 1, 3)
            self.bf_panel = BeamformPanel(
                config, runtime.l5_input_gain_compensation_enabled
            )
            self.preview_player = PreviewPlayer(
                sample_rate=config.device.sample_rate, volume=config.dev_test_ui.preview_volume,
                loop_gap_ms=config.dev_test_ui.loop_gap_ms, autoplay=config.dev_test_ui.autoplay,
                peak_dbfs=config.dev_test_ui.preview_peak_dbfs,
                fade_ms=config.dev_test_ui.preview_fade_ms,
            )
            self.bf_panel.track_play_requested.connect(self._toggle_track_audio)
            self.bf_panel.track_stop_requested.connect(self._pause_track_audio)
            self.bf_panel.mode_change_requested.connect(self._change_l3_processing_mode)
            self.bf_panel.downstream_processing_changed.connect(
                self._set_downstream_processing
            )
            self.bf_panel.gain_compensation_changed.connect(
                self._set_l5_input_gain_compensation
            )
            self.bf_panel.send_requested.connect(self._send_l3_to_l4)
            self.bf_panel.set_processing_mode(runtime.l3_processing_mode)
            self.bf_panel.set_downstream_processing_enabled(
                runtime.downstream_processing_enabled
            )
            if live_microphone_mode:
                # The BF algorithm may be selected during L1/L2 warm-up even
                # though no L3 window is admitted yet.
                self.bf_panel.mode_switch.setEnabled(True)
            self.cnn_panel = CnnPanel(config.layer5.voice_probability_limit)
            self.l4_panel = Layer4AudioPanel(persisted_l4_backend)
            self.l6_panel = Layer6AudioPanel()
            self.l4_panel.track_play_requested.connect(self._toggle_l4_audio)
            self.l4_panel.track_stop_requested.connect(self._pause_track_audio)
            self.l4_panel.backend_changed.connect(ui_settings.save_layer4_backend)
            self.l6_panel.run_requested.connect(self._run_l6)
            self.l6_panel.track_play_requested.connect(self._toggle_l6_audio)
            self.l6_panel.track_stop_requested.connect(self._pause_track_audio)
            self.l4_panel.set_voice_threshold(config.layer5.voice_probability_limit)
            self.cnn_panel.threshold_changed.connect(
                self.l4_panel.set_voice_threshold
            )
            grid.addWidget(self.bf_panel, 1, 0, 1, 2)
            grid.addWidget(self.l4_panel, 1, 2, 1, 2)
            grid.addWidget(self.l6_panel, 1, 4, 1, 2)
            for item_index in range(grid.count()):
                quadrant = grid.itemAt(item_index).widget()
                quadrant.setMinimumSize(0, 0)
                quadrant.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            outer.addWidget(quadrants, 1)
            self.performance_bar = QLabel()
            self.performance_bar.setObjectName("performanceBar")
            self.performance_bar.setFixedHeight(config.dev_test_ui.performance_bar_height_px)
            self.performance_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.performance_bar.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
            self.performance_bar.setTextFormat(Qt.TextFormat.PlainText)
            self.performance_bar.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            outer.addWidget(self.performance_bar)
            self.setCentralWidget(root)
            self._set_performance(None)
            self._refresh_timer = QTimer(self)
            self._refresh_timer.setTimerType(Qt.TimerType.PreciseTimer)
            self._refresh_timer.timeout.connect(self._refresh)
            # Formal windows arrive at most every 20 ms (50 Hz). Polling the
            # same frame faster only repeats widget work and competes with the
            # CPU L1/L2 workers for the Python GIL.
            refresh_hz = max(
                config.dev_test_ui.l1_meter_refresh_hz,
                config.dev_test_ui.polar_refresh_hz,
                config.dev_test_ui.waveform_refresh_hz,
            )
            self._refresh_timer.start(max(1, round(1_000 / refresh_hz)))
            self._update_control_states()

        def _l1_panel(self):
            box = QGroupBox("L1 · Input / Lights / Scratch Recording")
            layout = QVBoxLayout(box)
            self.l1_header = QLabel("Stopped | session — | epoch 0 | sample 0 | age N/A")
            self.l1_header.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            layout.addWidget(self.l1_header)
            if replay_source is not None:
                replay_controls = QHBoxLayout()
                self.replay_name = QLabel(f"模拟输入模式 · {replay_source.display_name}")
                self.replay_start = QPushButton("开始/继续")
                self.replay_pause = QPushButton("暂停")
                self.replay_restart = QPushButton("从头重播")
                self.replay_status = QLabel("准备中")
                self.replay_start.clicked.connect(self._start_or_resume_replay)
                self.replay_pause.clicked.connect(self._pause_replay)
                self.replay_restart.clicked.connect(self._restart_replay)
                for widget in (
                    self.replay_name,
                    self.replay_start,
                    self.replay_pause,
                    self.replay_restart,
                    self.replay_status,
                ):
                    replay_controls.addWidget(widget)
                replay_controls.addStretch()
                layout.addLayout(replay_controls)
            controls = QHBoxLayout()
            self.start_button = QPushButton("启动采集")
            self.stop_button = QPushButton("停止采集")
            self.start_button.clicked.connect(self._start_capture)
            self.stop_button.clicked.connect(lambda: self._submit_command("停止采集", runtime.stop))
            self.light_on = QPushButton("灯光开")
            self.light_off = QPushButton("灯光关")
            self.light_on.clicked.connect(lambda: self._light_command(True))
            self.light_off.clicked.connect(lambda: self._light_command(False))
            self.light_label = QLabel("状态: Unknown")
            self.light_label.setMinimumWidth(160)
            self.light_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            for widget in (self.start_button, self.stop_button, self.light_on, self.light_off, self.light_label):
                controls.addWidget(widget)
                if replay_source is not None:
                    widget.setVisible(False)
            controls.addStretch()
            layout.addLayout(controls)
            recording = QHBoxLayout()
            self.record = QPushButton("录制")
            self.pause = QPushButton("暂停/继续")
            self.finish = QPushButton("结束")
            self.record.clicked.connect(self._begin_recording)
            self.pause.clicked.connect(self._pause_recording)
            self.finish.clicked.connect(self._finish_recording)
            self.recording_label = QLabel("状态: idle | 00:00.0 | 当前文件: —")
            self.recording_label.setMinimumWidth(420)
            self.recording_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            for widget in (self.record, self.pause, self.finish, self.recording_label):
                recording.addWidget(widget)
            recording.addStretch()
            layout.addLayout(recording)
            runtime_recording = QHBoxLayout()
            self.runtime_record = QPushButton("正式录音开始")
            self.runtime_pause = QPushButton("正式录音暂停")
            self.runtime_recording_label = QLabel("Runtime Recording: manual / paused")
            self.runtime_record.clicked.connect(self._begin_runtime_recording)
            self.runtime_pause.clicked.connect(self._pause_runtime_recording)
            for widget in (self.runtime_record, self.runtime_pause, self.runtime_recording_label):
                runtime_recording.addWidget(widget)
            runtime_recording.addStretch()
            layout.addLayout(runtime_recording)
            denoise = QHBoxLayout()
            self.pre_denoise_switch = QCheckBox("IMCRA预降噪")
            self.pre_denoise_switch.setChecked(runtime.l1_pre_denoise_enabled)
            self.pre_denoise_switch.setToolTip(
                "每个物理麦使用自己的IMCRA噪声PSD；40 ms窗、20 ms步长重叠重建"
            )
            self.pre_denoise_switch.toggled.connect(self._set_l1_pre_denoise)
            self.pre_denoise_label = QLabel("预降噪: OFF | 原始音频直通")
            denoise.addWidget(self.pre_denoise_switch)
            denoise.addWidget(self.pre_denoise_label)
            denoise.addStretch()
            layout.addLayout(denoise)
            countnet = QHBoxLayout()
            self.countnet_switch = QCheckBox("CountNet人数估计")
            self.countnet_switch.setChecked(runtime.l1_speaker_count_enabled)
            self.countnet_switch.setToolTip(
                "校准后Center Mic · 48→16 kHz · 5秒上下文 · 每100 ms异步推理"
            )
            self.countnet_switch.toggled.connect(self._set_l1_speaker_count)
            self.countnet_label = QLabel("Speech count: OFF")
            self.countnet_label.setStyleSheet("font-family:Consolas")
            countnet.addWidget(self.countnet_switch)
            countnet.addWidget(self.countnet_label)
            countnet.addStretch()
            layout.addLayout(countnet)
            self.imcra_label = QLabel("IMCRA: WAITING | noise MIC0— MIC1— MIC2— MIC3— MIC4— MIC5— Center—")
            self.imcra_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            self.imcra_label.setStyleSheet("font-family:Consolas")
            layout.addWidget(self.imcra_label)
            meters = QHBoxLayout()
            self.meter_bars, self.meter_labels = [], []
            for name in ("MIC0", "MIC1", "MIC2", "MIC3", "MIC4", "MIC5", "Center", "HardwareMix"):
                column = QVBoxLayout()
                bar = QProgressBar()
                bar.setOrientation(Qt.Orientation.Vertical)
                bar.setRange(-90, 0)
                bar.setValue(-90)
                bar.setTextVisible(False)
                label = QLabel(f"{name}\n-120.0 dB")
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setMinimumWidth(100)
                label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
                column.addWidget(bar, 1)
                column.addWidget(label)
                meters.addLayout(column, 1)
                self.meter_bars.append(bar)
                self.meter_labels.append((name, label))
            layout.addLayout(meters, 1)
            self._set_record_buttons("idle")
            return box

        def _doa_panel(self):
            box = QGroupBox("L2 · 可切换定位与追踪")
            layout = QVBoxLayout(box)
            self.srp_header = QLabel("UNAVAILABLE | session — | epoch 0 | window — | sample — | age N/A")
            self.srp_header.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            layout.addWidget(self.srp_header)
            splitter = QSplitter(Qt.Orientation.Horizontal)
            self.srp_polar = MusicPolarPanel(config.dev_test_ui.stale_after_ms)
            right = QWidget()
            right_layout = QVBoxLayout(right)
            right_layout.setContentsMargins(0, 0, 0, 0)
            right_layout.setSpacing(4)
            self.gate_threshold = GateProbabilityThresholdControl(
                runtime.gate_probability_threshold
            )
            self.doa_backend = DoaBackendControl(runtime.doa_backend)
            self.srp_threshold = SrpThresholdControl(runtime.direction_threshold)
            self.music_order_limit = MusicOrderLimitControl(runtime.music_effective_order_limit)
            self.music_dpd_rank1 = MusicDpdRank1Control(runtime.music_dpd_rank1_enabled)
            self.music_noise_whitening = MusicNoiseWhiteningControl(
                runtime.music_noise_whitening_enabled
            )
            music_backend_active = runtime.doa_backend == "frequency_normalized_music"
            self.music_dpd_rank1.setEnabled(music_backend_active)
            self.music_noise_whitening.setEnabled(music_backend_active)
            self.music_dpd_rank1.setToolTip(
                "仅使用通过直达声主导检验的频点，以rank-1 MUSIC逐频投票并执行圆周聚类；默认关闭。"
            )
            self.music_noise_whitening.setToolTip(
                "直接读取L1 IMCRA长期维护的完整7×7空间噪声协方差，同时白化观测协方差和steering。"
            )
            self.srp_id_tracking = DirectionIdTrackingControl(
                runtime.direction_id_tracking_enabled
            )
            self.srp_id_tracking.setToolTip(
                "开启时运行圆周IMM-JPDA、ID生命周期和2秒预测；关闭时仅显示DOA原始峰值。"
            )
            self.gate_readout = ProbabilityGateReadout()
            self.music_status = QLabel("MUSIC order=—  output=—  valid=—  status=UNAVAILABLE")
            self.music_status.setFixedHeight(30)
            self.music_status.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
            )
            self.music_status.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            self.music_status.setStyleSheet(
                "QLabel { background:#202a34; color:#9fb2c5; padding:0 8px; "
                "font-family:Consolas; font-weight:600; }"
            )
            right_layout.addWidget(self.doa_backend)
            right_layout.addWidget(self.gate_threshold)
            processing_switches = QHBoxLayout()
            processing_switches.setContentsMargins(0, 0, 0, 0)
            processing_switches.setSpacing(4)
            processing_switches.addWidget(self.srp_id_tracking, 1)
            processing_switches.addWidget(self.music_dpd_rank1, 1)
            processing_switches.addWidget(self.music_noise_whitening, 1)
            right_layout.addLayout(processing_switches)
            right_layout.addWidget(self.gate_readout)
            right_layout.addWidget(self.music_status)
            order_tracking_row = QHBoxLayout()
            order_tracking_row.setContentsMargins(0, 0, 0, 0)
            order_tracking_row.setSpacing(4)
            order_tracking_row.addWidget(self.music_order_limit, 0)
            right_layout.addLayout(order_tracking_row)
            right_layout.addWidget(self.srp_threshold)
            right_layout.addStretch(1)
            splitter.addWidget(self.srp_polar)
            splitter.addWidget(right)
            splitter.setSizes((700, 300))
            layout.addWidget(splitter, 1)
            self.srp_polar.candidate_selected.connect(self._select_candidate)
            self.doa_backend.backend_changed.connect(self._set_doa_backend)
            self.srp_threshold.threshold_changed.connect(self._set_srp_threshold)
            self.music_order_limit.order_changed.connect(self._set_music_order_limit)
            self.music_dpd_rank1.enabled_changed.connect(self._set_music_dpd_rank1)
            self.music_noise_whitening.enabled_changed.connect(self._set_music_noise_whitening)
            self.gate_threshold.threshold_changed.connect(self._set_gate_probability_threshold)
            self.srp_id_tracking.enabled_changed.connect(self._set_direction_id_tracking)
            return box

        def _set_doa_backend(self, backend: str):
            previous = runtime.doa_backend
            try:
                backend = ui_settings.save_doa_backend(backend)
                runtime.set_doa_backend(backend)
                self.doa_backend.set_backend(backend, pending=True)
                music = backend == "frequency_normalized_music"
                self.music_dpd_rank1.setEnabled(music)
                self.music_noise_whitening.setEnabled(music)
                self.statusBar().showMessage(
                    f"完整L2方案切换为 {'MUSIC' if music else 'GI-DOAEnet'} + Circular IMM-JPDA；下一窗口生效",
                    5000,
                )
            except Exception as exc:
                runtime.set_doa_backend(previous)
                with QSignalBlocker(self.doa_backend):
                    self.doa_backend.set_backend(previous)
                self.statusBar().showMessage(f"L2方案切换失败: {exc}", 8000)

        def _set_gate_probability_threshold(self, threshold: float):
            previous = runtime.gate_probability_threshold
            try:
                threshold = ui_settings.save_gate_probability_threshold(threshold)
                runtime.set_gate_probability_threshold(threshold)
                self.gate_threshold.set_value(threshold, pending=True)
                self.statusBar().showMessage(
                    f"L2 Gate probability threshold saved as {threshold:.2f}; next window applies",
                    3500,
                )
            except Exception as exc:
                runtime.set_gate_probability_threshold(previous)
                with QSignalBlocker(self.gate_threshold):
                    self.gate_threshold.set_value(previous)
                self.statusBar().showMessage(f"Failed to set L2 Gate threshold: {exc}", 8000)

        def _set_l1_pre_denoise(self, enabled: bool):
            previous = runtime.l1_pre_denoise_enabled
            try:
                enabled = ui_settings.save_l1_pre_denoise_enabled(bool(enabled))
                runtime.set_l1_pre_denoise_enabled(enabled)
                self.pre_denoise_label.setText(
                    "预降噪: ON | 等待下一完整20 ms替换" if enabled else "预降噪: OFF | 原始音频直通"
                )
                self.statusBar().showMessage(
                    f"L1 IMCRA预降噪{'开启' if enabled else '关闭'}；从下一完整音频块生效", 3500
                )
            except Exception as exc:
                runtime.set_l1_pre_denoise_enabled(previous)
                with QSignalBlocker(self.pre_denoise_switch):
                    self.pre_denoise_switch.setChecked(previous)
                self.statusBar().showMessage(f"L1预降噪切换失败: {exc}", 8000)

        def _set_l1_speaker_count(self, enabled: bool):
            previous = runtime.l1_speaker_count_enabled
            try:
                enabled = ui_settings.save_l1_speaker_count_enabled(bool(enabled))
                runtime.set_l1_speaker_count_enabled(enabled)
                self._set_text(
                    self.countnet_label,
                    "Speech count: WARMING_UP | 需要5秒连续Center音频"
                    if enabled else "Speech count: OFF",
                )
                self.statusBar().showMessage(
                    f"L1 CountNet人数估计{'开启' if enabled else '关闭'}", 3500
                )
            except Exception as exc:
                runtime.set_l1_speaker_count_enabled(previous)
                with QSignalBlocker(self.countnet_switch):
                    self.countnet_switch.setChecked(previous)
                self.statusBar().showMessage(f"L1 CountNet切换失败: {exc}", 8000)

        def _set_l5_input_gain_compensation(self, enabled: bool):
            previous = runtime.l5_input_gain_compensation_enabled
            try:
                enabled = ui_settings.save_l5_input_gain_compensation_enabled(bool(enabled))
                runtime.set_l5_input_gain_compensation_enabled(enabled)
                self.bf_panel.set_gain_compensation_enabled(enabled)
                self.statusBar().showMessage(
                    f"连续轨响度补偿{'开启' if enabled else '关闭'}；下一20 ms生效", 3500
                )
            except Exception as exc:
                runtime.set_l5_input_gain_compensation_enabled(previous)
                with QSignalBlocker(self.bf_panel.gain_compensation):
                    self.bf_panel.set_gain_compensation_enabled(previous)
                self.statusBar().showMessage(f"响度补偿切换失败: {exc}", 8000)

        def _set_srp_threshold(self, threshold: float):
            try:
                threshold = runtime.set_direction_threshold(threshold)
                ui_settings.save_direction_threshold(threshold)
                self.statusBar().showMessage(f"L2 candidate threshold已保存为 {threshold:.2f}", 2500)
                self.statusBar().showMessage(
                    f"L2 MUSIC candidate threshold saved as {threshold:.2f}; next window applies", 3500
                )
            except Exception as exc:
                self.statusBar().showMessage(f"保存L2 threshold失败: {exc}", 8000)

        def _set_music_order_limit(self, value: int):
            previous = runtime.music_effective_order_limit
            try:
                value = ui_settings.save_music_effective_order_limit(value)
                runtime.set_music_effective_order_limit(value)
                self.statusBar().showMessage(
                    f"MUSIC阶数已设为 {value}；下一窗口生效", 3500
                )
            except Exception as exc:
                runtime.set_music_effective_order_limit(previous)
                with QSignalBlocker(self.music_order_limit.combo):
                    self.music_order_limit.set_value(previous)
                self.statusBar().showMessage(f"设置MUSIC阶数上限失败: {exc}", 8000)

        def _set_music_dpd_rank1(self, enabled: bool):
            previous = runtime.music_dpd_rank1_enabled
            try:
                enabled = ui_settings.save_music_dpd_rank1_enabled(bool(enabled))
                runtime.set_music_dpd_rank1_enabled(enabled)
                self.music_dpd_rank1.set_enabled(enabled, pending=True)
                self.statusBar().showMessage(
                    f"DPD + rank-1 MUSIC {'已开启' if enabled else '已关闭'}；下一窗口生效", 3500
                )
            except Exception as exc:
                runtime.set_music_dpd_rank1_enabled(previous)
                with QSignalBlocker(self.music_dpd_rank1):
                    self.music_dpd_rank1.set_enabled(previous)
                self.statusBar().showMessage(f"DPD + rank-1 MUSIC切换失败: {exc}", 8000)

        def _set_music_noise_whitening(self, enabled: bool):
            previous = runtime.music_noise_whitening_enabled
            try:
                enabled = ui_settings.save_music_noise_whitening_enabled(bool(enabled))
                runtime.set_music_noise_whitening_enabled(enabled)
                self.music_noise_whitening.set_enabled(enabled, pending=True)
                self.statusBar().showMessage(
                    f"IMCRA噪声白化 {'已开启' if enabled else '已关闭'}；下一窗口生效", 3500
                )
            except Exception as exc:
                runtime.set_music_noise_whitening_enabled(previous)
                with QSignalBlocker(self.music_noise_whitening):
                    self.music_noise_whitening.set_enabled(previous)
                self.statusBar().showMessage(f"IMCRA噪声白化切换失败: {exc}", 8000)

        def _set_direction_id_tracking(self, enabled: bool):
            previous = runtime.direction_id_tracking_enabled
            try:
                enabled = ui_settings.save_direction_id_tracking_enabled(bool(enabled))
                runtime.set_direction_id_tracking_enabled(enabled)
                self.srp_id_tracking.set_enabled(enabled, pending=True)
                self.statusBar().showMessage(
                    "ID Tracking已开启；下一窗口开始建立权威ID"
                    if enabled else
                    "ID Tracking已关闭；仅显示MUSIC原始峰值，L3/L5将跳过",
                    3500,
                )
            except Exception as exc:
                runtime.set_direction_id_tracking_enabled(previous)
                with QSignalBlocker(self.srp_id_tracking):
                    self.srp_id_tracking.set_enabled(previous)
                self.statusBar().showMessage(f"ID Tracking切换失败: {exc}", 8000)

        def _submit_command(self, name, command, on_success=None):
            if self._pending_command is not None:
                self.statusBar().showMessage("上一条命令仍在执行，请稍候", 3000)
                return
            self._pending_command = (name, self._commands.submit(command), on_success)
            self.statusBar().showMessage(f"{name}…")
            self._update_control_states()

        def _poll_command(self):
            pending = self._pending_command
            if pending is None or not pending[1].done():
                return
            name, future, on_success = pending
            self._pending_command = None
            try:
                result = future.result()
                if on_success is not None:
                    on_success(result)
                if name in {"启动采集", "从头重播"}:
                    self._enter_starting_state()
                elif name in {"停止采集", "模拟输入已播放完成"}:
                    if runtime.active:
                        raise RuntimeError(
                            runtime.last_error or "Runtime仍有线程或录音会话未停止"
                        )
                    self._enter_stopped_state()
                self.statusBar().showMessage(f"{name}已完成", 3000)
            except Exception as exc:
                if name in {"灯光开", "灯光关"}:
                    self.light_label.setText("状态: Error")
                elif name == "L3发送到L4":
                    self.bf_panel.set_send_enabled(True)
                    self.l4_panel.set_processing(f"L4失败：{exc}")
                elif name == "运行L6":
                    self.l6_panel.set_error(str(exc))
                self.statusBar().showMessage(f"{name}失败: {exc}", 10000)
            self._update_control_states()

        @staticmethod
        def _set_text(widget, value):
            if widget.text() != value:
                widget.setText(value)

        def _enter_starting_state(self):
            self._frame = None
            self._l2_frame = None
            self._last_l2_identity = None
            self._last_l5_frame = None
            self._last_l5_seen = 0.0
            self._l5_is_stale = False
            self._last_rendered_window = None
            self._last_l1_seen = monotonic()
            self.srp_polar.set_snapshot(None, live=True)
            self.gate_readout.set_unavailable("WARMING")
            self.music_status.setText("MUSIC order=—  output=—  valid=—  status=WARMING")
            self.cnn_panel.set_unavailable("WARMING: waiting for completed L5 window")
            self.bf_panel.set_send_enabled(False)
            self.l4_panel.clear_tracks()
            self._l4_store.clear()
            self._offline_l4_pipeline = None
            self._l4_processed = ()
            self._l5_results = ()
            self._l6_result = None
            self._l6_store.clear()
            self.l6_panel.clear_tracks()
            self.l6_panel.set_run_enabled(False)
            self._offline_stage_durations_seconds = {"l4": None, "l5": None, "l6": None}
            self.srp_header.setText("WARMING | session — | epoch 0 | window — | sample — | age —")
            self.l1_header.setText("WARMING | waiting for the first audio block")

        def _enter_stopped_state(self):
            self._last_runtime_state = "stopped"
            self._last_l1_seen = 0.0
            self._last_l5_frame = None
            self._last_l5_seen = 0.0
            self._l5_is_stale = False
            self.srp_polar.set_live(False)
            self.gate_readout.set_unavailable("STOPPED")
            if self._l2_frame is None or self._l2_frame.spatial_response is None:
                self.music_status.setText("MUSIC order=—  output=—  valid=—  status=STOPPED")
            else:
                status = self.music_status.text()
                if status.endswith(" LIVE"):
                    status = status[:-5]
                elif status.endswith(" STALE"):
                    status = status[:-6]
                self.music_status.setText(f"{status} STOPPED")
            self.cnn_panel.set_unavailable("STOPPED")
            try:
                self.bf_panel.set_send_enabled(bool(runtime.offline_l4_sources))
            except RuntimeError:
                self.bf_panel.set_send_enabled(False)
            self.l1_header.setText("STOPPED | capture closed | age —")
            if self._l2_frame is not None and self._l2_frame.spatial_response is not None:
                response = self._l2_frame.spatial_response
                self.srp_header.setText(
                    f"STOPPED | session {response.session_id[:8]} | epoch {response.stream_epoch:03d} | "
                    f"window {response.window_id:08d} | sample {response.decision_sample:012d} | age —"
                )
            else:
                self.srp_header.setText("STOPPED | no completed SRP window | age —")

        def _light_command(self, enabled: bool):
            self.light_label.setText("状态: Pending")
            def completed(_result):
                state = "On" if enabled else "Off"
                self.light_label.setText(f"状态: {state} (commanded)")

            self._submit_command(
                "灯光开" if enabled else "灯光关",
                lambda: runtime.set_light(enabled),
                completed,
            )

        def _start_capture(self):
            # Release a possibly mapped cache snapshot before tracker.reset()
            # removes the previous Test UI session directory on Windows.
            self.preview_player.close()
            self._audio_source_key = None
            self.bf_panel.clear_tracks()
            self.light_label.setText("状态: Waiting for microphone")

            def start_and_default_light_off():
                # Do not touch CDC unless UAC capture has connected first.
                # A missing microphone therefore produces no light command
                # and no secondary light-control error.
                runtime.start()
                try:
                    runtime.set_light(False)
                except Exception:
                    # Startup LED-off is best effort. A connected microphone
                    # remains usable even when its optional CDC port is absent.
                    return False
                return True

            def completed(light_was_turned_off):
                if light_was_turned_off:
                    self.light_label.setText("状态: Off (startup default)")
                else:
                    self.light_label.setText("状态: Unknown")
                self.light_label.setToolTip("")

            self._submit_command("启动采集", start_and_default_light_off, completed)

        def _start_or_resume_replay(self):
            if replay_source is None:
                return
            replay_source.resume()
            runtime.set_pipeline_timing_paused("simulation_input_paused", False)
            if not runtime.active:
                self._start_capture()

        def _pause_replay(self):
            if replay_source is None:
                return
            replay_source.pause()
            runtime.set_pipeline_timing_paused("simulation_input_paused", True)

        def _clear_replay_results(self):
            current_l1 = None if self._frame is None else getattr(self._frame, "l1", None)
            if current_l1 is not None:
                self._replay_previous_stream = (
                    current_l1.session_id,
                    current_l1.stream_epoch,
                )
            else:
                self._replay_previous_stream = None
            self._replay_reset_pending = True
            for mailbox in (
                runtime.latest_dev_ui,
                getattr(runtime, "latest_l2_dev_ui", None),
                getattr(runtime, "latest_l5_dev_ui", None),
            ):
                if mailbox is None:
                    continue
                while True:
                    try:
                        mailbox.get_nowait()
                    except queue.Empty:
                        break
            self.preview_player.close()
            self._audio_source_key = None
            audio_id_tracker.reset()
            self.bf_panel.clear_tracks()
            self._frame = None
            self._l2_frame = None
            self._last_l2_identity = None
            self._last_l5_frame = None
            self._last_l5_seen = 0.0
            self._l5_is_stale = False
            self._selected = None
            self._last_rendered_window = None
            self.srp_polar.set_snapshot(None, live=True)
            self.gate_readout.set_unavailable("WARMING")
            self.music_status.setText("MUSIC order=—  output=—  valid=—  status=WARMING")
            self.cnn_panel.set_unavailable("WARMING: replay restarted")
            self.srp_header.setText("WARMING | replay restarted | waiting for new result")
            self.l1_header.setText("WARMING | replay restarted | waiting for first block")

        def _restart_replay(self):
            if replay_source is None:
                return
            self._clear_replay_results()
            # Every replay pass must submit its own EOF drain/stop.  Keeping
            # the previous pass' marker leaves Runtime active after the next
            # EOF, so TrackAudioStreamHub never seals and Send to L4 remains
            # disabled indefinitely.
            self._eof_stop_submitted = False
            runtime.reset_pipeline_total_durations()
            # Stop new L1 admission before the background command discards all
            # queued work. The one L3 item already executing is allowed to
            # return, but its late result is thrown away with the old graph.
            replay_source.pause()
            runtime.set_pipeline_timing_paused("simulation_input_paused", True)

            def discard_and_restart():
                if runtime.active:
                    runtime.discard_pending_for_replay()
                replay_source.replay()
                runtime.set_pipeline_timing_paused("simulation_input_paused", False)
                runtime.start()

            self._submit_command("从头重播", discard_and_restart)

        @staticmethod
        def _format_replay_time(seconds: float) -> str:
            value = max(0, int(seconds * 10))
            return f"{value // 600:02d}:{(value % 600) / 10:04.1f}"

        def _refresh_replay_controls(self):
            if replay_source is None:
                return
            status = replay_source.status()
            names = {
                "ready": "准备就绪",
                "playing": "正在播放",
                "paused": "已暂停",
                "ended": "播放完成",
                "stopped": "已停止",
                "error": "错误",
            }
            self.replay_status.setText(
                f"{names[status.state]} · {self._format_replay_time(status.current_seconds)} / "
                f"{self._format_replay_time(status.total_seconds)}"
            )
            if status.state == "ended":
                runtime.seal_pipeline_total_durations()
            busy = self._pending_command is not None
            self.replay_start.setEnabled(not busy and status.state in {"ready", "paused"})
            self.replay_pause.setEnabled(not busy and status.state == "playing")
            self.replay_restart.setEnabled(not busy and status.state not in {"ready", "error"})

        def _begin_recording(self):
            def started(_result):
                self._record_started, self._record_elapsed = monotonic(), 0.0

            self._submit_command("开始临时录音", runtime.begin_scratch, started)

        def _pause_recording(self):
            was_recording = runtime.scratch.state == "recording"

            def changed(_result):
                if was_recording and self._record_started is not None:
                    self._record_elapsed += monotonic() - self._record_started
                    self._record_started = None
                elif runtime.scratch.state == "recording":
                    self._record_started = monotonic()

            self._submit_command("暂停临时录音" if was_recording else "继续临时录音", runtime.pause_or_resume_scratch, changed)

        def _finish_recording(self):
            def finished(_result):
                if self._record_started is not None:
                    self._record_elapsed += monotonic() - self._record_started
                self._record_started = None

            self._submit_command("结束临时录音", runtime.finish_scratch, finished)

        def _begin_runtime_recording(self):
            if live_microphone_mode:
                # AudioIdTracker.reset() removes the Center preview files.
                # Release a possibly mapped playback snapshot first on Windows.
                self.preview_player.close()
                self._audio_source_key = None
                self.bf_panel.clear_tracks()

            def started(_result):
                self.bf_panel.set_downstream_processing_enabled(
                    runtime.downstream_processing_enabled
                )

            self._submit_command("正式录音开始", runtime.begin_runtime_recording, started)

        def _pause_runtime_recording(self):
            def paused(_result):
                self.bf_panel.set_downstream_processing_enabled(
                    runtime.downstream_processing_enabled
                )

            self._submit_command("正式录音暂停", runtime.pause_runtime_recording, paused)

        def _set_record_buttons(self, state: str):
            busy = self._pending_command is not None
            self.record.setEnabled(runtime.running and not busy and state != "finalizing")
            self.pause.setEnabled(runtime.running and not busy and state in {"recording", "paused"})
            self.finish.setEnabled(not busy and state in {"recording", "paused"})

        def _update_control_states(self):
            busy = self._pending_command is not None
            self.start_button.setEnabled(not runtime.active and not busy)
            self.stop_button.setEnabled(runtime.active and not busy)
            # SerialDevice.write() opens the CDC control port on demand, so
            # lighting is intentionally independent of audio capture state.
            self.light_on.setEnabled(not busy)
            self.light_off.setEnabled(not busy)
            self._set_record_buttons(runtime.scratch.state)
            self._refresh_replay_controls()

        def _select_candidate(self, theta: float, window_id: int):
            self._selected = (theta, window_id)
            self.bf_panel.select_preview(theta, window_id)
            self.cnn_panel.set_selection(theta, window_id)

        def _change_l3_processing_mode(self, mode: str):
            applied = runtime.set_l3_processing_mode(mode)
            # Release mapped playback files before the comparison cache is reset.
            self.preview_player.close()
            self._audio_source_key = None
            audio_id_tracker.seal_mode(None)
            self.bf_panel.reset_for_mode_change(applied)
            self._last_l5_frame = None
            self._last_l5_seen = 0.0
            self._l5_is_stale = False
            self.cnn_panel.set_unavailable("L3 MODE CHANGED: waiting for next window")
            label = {
                "optimized": "优化算法",
                "ds_baseline": "DS基线",
                "loaded_mvdr_baseline": "Loaded MVDR基线",
            }[applied]
            self.statusBar().showMessage(f"L3已切换为{label}；从下一个处理窗口生效", 5000)

        def _set_downstream_processing(self, enabled: bool):
            applied = runtime.set_downstream_processing_enabled(bool(enabled))
            runtime.set_pipeline_timing_paused(
                "downstream_disabled_by_test_ui",
                not applied,
                stages=("l3", "l5"),
            )
            self.bf_panel.set_downstream_processing_enabled(applied)
            if applied:
                self.bf_panel.set_unavailable(
                    "L3/L5已恢复；等待下一条L2结果。"
                )
                self.cnn_panel.set_unavailable("WARMING: waiting for completed L5 window")
                message = "L3/L5已恢复运行；L2后续窗口将重新进入下游"
            else:
                self.bf_panel.set_unavailable(
                    "L3/L5已由Test UI停止；L2继续独立运行。下方已有试听缓存仍可播放。"
                )
                self.cnn_panel.set_unavailable("STOPPED BY TEST UI; L2 remains active")
                message = "已切断L2→L3输入；L3与L5停止计算，L2继续运行"
            self.statusBar().showMessage(message, 5000)

        def _pause_track_audio(self):
            self.preview_player.pause()

        def _toggle_track_audio(self, track_id: int):
            if (
                self._audio_source_key is not None
                and self._audio_source_key[0] == "track"
                and int(self._audio_source_key[1]) == int(track_id)
            ):
                if not self.preview_player.play():
                    error = self.preview_player.take_error() or "unknown audio output error"
                    self.bf_panel.sync_track_playback_stopped()
                    self.statusBar().showMessage(f"试听失败：{error}", 5000)
                return
            cache_path = audio_id_tracker.audio_cache_path(track_id)
            if cache_path is None:
                if track_id == CENTER_RAW_TRACK_ID:
                    label = "Center Mic RAW"
                elif track_id == CENTER_IMCRA_TRACK_ID:
                    label = "Center Mic IMCRA"
                else:
                    label = f"ID-{track_id:03d}"
                self.statusBar().showMessage(f"{label} 暂无可播放缓存", 3000)
                self.bf_panel.sync_track_playback_stopped()
                return
            self.preview_player.stop()
            # Play the exact TrackAudioStreamHub waveform used by L5.  The
            # preview backend may still apply attenuation-only output safety,
            # but must never add a second listening-only loudness boost.
            self.preview_player.set_volume(1.0)
            self.preview_player.load_file(
                cache_path,
                delete_on_release=True,
            )
            self._audio_source_key = ("track", int(track_id), str(cache_path))
            if not self.preview_player.play():
                error = self.preview_player.take_error() or "unknown audio output error"
                self.bf_panel.sync_track_playback_stopped()
                self.statusBar().showMessage(f"试听失败：{error}", 5000)

        def _toggle_l4_audio(self, track_id: int):
            if (
                self._audio_source_key is not None
                and self._audio_source_key[0] == "l4_track"
                and int(self._audio_source_key[1]) == int(track_id)
            ):
                if not self.preview_player.play():
                    error = self.preview_player.take_error() or "unknown audio output error"
                    self.l4_panel.sync_track_playback_stopped()
                    self.statusBar().showMessage(f"L4试听失败：{error}", 5000)
                return
            cache_path = self._l4_store.audio_path(track_id)
            if cache_path is None:
                self.l4_panel.sync_track_playback_stopped()
                self.statusBar().showMessage(f"L4 ID-{track_id:03d}暂无可播放音频", 3000)
                return
            self.preview_player.stop()
            self.preview_player.set_volume(1.0)
            self.preview_player.load_wav_file(cache_path)
            self._audio_source_key = ("l4_track", int(track_id), str(cache_path))
            if not self.preview_player.play():
                error = self.preview_player.take_error() or "unknown audio output error"
                self.l4_panel.sync_track_playback_stopped()
                self.statusBar().showMessage(f"L4试听失败：{error}", 5000)

        def _toggle_l6_audio(self, speaker_id: int):
            if (
                self._audio_source_key is not None
                and self._audio_source_key[0] == "l6_speaker"
                and int(self._audio_source_key[1]) == int(speaker_id)
            ):
                if not self.preview_player.play():
                    self.l6_panel.sync_track_playback_stopped()
                return
            cache_path = self._l6_store.audio_path(speaker_id)
            if cache_path is None:
                self.l6_panel.sync_track_playback_stopped()
                self.statusBar().showMessage(f"L6 Speaker-{speaker_id}暂无可播放音频", 3000)
                return
            self.preview_player.stop()
            self.preview_player.set_volume(1.0)
            self.preview_player.load_wav_file(cache_path)
            self._audio_source_key = ("l6_speaker", int(speaker_id), str(cache_path))
            if not self.preview_player.play():
                error = self.preview_player.take_error() or "unknown audio output error"
                self.l6_panel.sync_track_playback_stopped()
                self.statusBar().showMessage(f"L6试听失败：{error}", 5000)

        def _send_l3_to_l4(self):
            self.preview_player.close()
            self._audio_source_key = None
            self.bf_panel.set_send_enabled(False)
            # A submission is a complete replacement, not an append.  Clear
            # both the visible rows and their backing preview files before the
            # selected offline L4 backend starts another pass over sealed L3.
            self.l4_panel.clear_tracks()
            self._l4_store.clear()
            self._offline_l4_pipeline = None
            self._l4_processed = ()
            self._l5_results = ()
            self._l6_result = None
            self._l6_store.clear()
            self.l6_panel.clear_tracks()
            self.l6_panel.set_run_enabled(False)
            self._offline_stage_durations_seconds = {"l4": None, "l5": None, "l6": None}
            self._refresh_total_duration_text()
            self.cnn_panel.set_unavailable("等待L4完成后自动处理")
            backend_id = self.l4_panel.backend_id
            backend_label = self.l4_panel.BACKEND_LABELS[backend_id]
            merge_candidates = self.l4_panel.merge_enabled
            merge_label = "合并" if merge_candidates else "保留双候选"
            self.l4_panel.set_processing(
                f"正在加载{backend_label}并处理全部L3长音频（{merge_label}）…"
            )

            def process_l4_and_l5():
                pipeline = runtime.build_offline_l4_pipeline(backend_id)
                l4_started = perf_counter()
                processed = tuple(
                    pipeline.process_l4_sealed(
                        runtime.offline_l4_sources,
                        merge_candidates=merge_candidates,
                    )
                )
                l4_elapsed = perf_counter() - l4_started
                l5_started = perf_counter()
                try:
                    l5_results = tuple(pipeline.process_l5_sealed(processed))
                except Exception as exc:
                    return (
                        pipeline, processed, (), exc, l4_elapsed,
                        perf_counter() - l5_started, merge_candidates,
                    )
                return (
                    pipeline, processed, l5_results, None, l4_elapsed,
                    perf_counter() - l5_started, merge_candidates,
                )

            def completed(value):
                (
                    pipeline, processed, l5_results, l5_error,
                    l4_elapsed, l5_elapsed, merged,
                ) = value
                self._offline_l4_pipeline = pipeline
                self._l4_processed = processed
                self._offline_stage_durations_seconds = {
                    "l4": l4_elapsed,
                    "l5": l5_elapsed,
                    "l6": None,
                }
                self._refresh_total_duration_text()
                self._l4_store.set_processed(self._l4_processed)
                if l5_error is not None:
                    self.l4_panel.set_tracks(
                        self._l4_store.snapshots(), unmerged=not merged,
                    )
                    self.l4_panel.set_l5_error(str(l5_error))
                    self.cnn_panel.set_unavailable(f"L5失败：{l5_error}")
                    self.bf_panel.set_send_enabled(True)
                    return
                self._l4_store.apply_l5(l5_results)
                self._l5_results = tuple(l5_results)
                self.l4_panel.set_tracks(
                    self._l4_store.snapshots(), l5_complete=True, unmerged=not merged,
                )
                detections = tuple(SimpleNamespace(
                    theta_deg=item.source.theta_deg,
                    probability=item.l5_probability,
                    window_id=item.source.end_sample // 960,
                ) for item in l5_results)
                self.cnn_panel.set_result(SimpleNamespace(
                    detections=detections,
                    primary_model_id=(
                        l5_results[0].l5_model_id if l5_results else "L5"
                    ),
                    threshold=pipeline.layer5.threshold,
                ))
                self.l6_panel.clear_tracks(
                    "L4已合并；关闭“合并”并重新发送后才能运行L6"
                    if merged else "L4双候选标注完成；可手动运行L6"
                )
                self.l6_panel.set_run_enabled(not merged and bool(l5_results))
                self.bf_panel.set_send_enabled(True)

            self._submit_command("L3发送到L4", process_l4_and_l5, completed)

        def _run_l6(self):
            if not self._l5_results:
                self.l6_panel.set_error("没有可用的L4双候选标注")
                return
            self.preview_player.close()
            self._audio_source_key = None
            self._l6_store.clear()
            self.l6_panel.clear_tracks()
            self.l6_panel.set_processing()

            def process_l6():
                pipeline = runtime.build_offline_l6_pipeline()
                started = perf_counter()
                return pipeline.process(self._l5_results), perf_counter() - started

            def completed(value):
                result, elapsed = value
                self._l6_result = result
                self._offline_stage_durations_seconds["l6"] = elapsed
                self._l6_store.set_result(result)
                self.l6_panel.set_tracks(self._l6_store.snapshots())
                self._refresh_total_duration_text()

            self._submit_command("运行L6", process_l6, completed)

        def _refresh(self):
            self._poll_command()
            self._refresh_replay_controls()
            self._refresh_total_duration_text()
            self.preview_player.validate_output()
            if not self.preview_player.playing:
                self.bf_panel.sync_track_playback_stopped()
            if self._audio_source_key is not None and self._audio_source_key[0] == "track":
                progress = self.preview_player.playback_progress
                if self.preview_player.playing or progress > 0.0:
                    self.bf_panel.set_track_playback_progress(
                        int(self._audio_source_key[1]), progress
                    )
                else:
                    self.bf_panel.clear_track_playback_progress()
            else:
                self.bf_panel.clear_track_playback_progress()
            if self._audio_source_key is not None and self._audio_source_key[0] == "l4_track":
                progress = self.preview_player.playback_progress
                if self.preview_player.playing or progress > 0.0:
                    self.l4_panel.set_track_playback_progress(
                        int(self._audio_source_key[1]), progress
                    )
                else:
                    self.l4_panel.clear_track_playback_progress()
            else:
                self.l4_panel.clear_track_playback_progress()
            if self._audio_source_key is not None and self._audio_source_key[0] == "l6_speaker":
                progress = self.preview_player.playback_progress
                if self.preview_player.playing or progress > 0.0:
                    self.l6_panel.set_track_playback_progress(
                        int(self._audio_source_key[1]), progress
                    )
                else:
                    self.l6_panel.clear_track_playback_progress()
            else:
                self.l6_panel.clear_track_playback_progress()
            playback_error = self.preview_player.take_error()
            if playback_error:
                self.statusBar().showMessage(f"试听输出：{playback_error}", 5000)
            replay_ended = (
                replay_source is not None
                and replay_source.status().state == "ended"
            )
            if (
                (runtime.input_exhausted or replay_ended)
                and runtime.active
                and self._pending_command is None
                and not self._eof_stop_submitted
            ):
                # Stage workers intentionally stay alive while Runtime is
                # active and only drain/exit after stop() sends EOS. Waiting
                # for processing_running to become false here deadlocks a
                # completed replay in RUNNING and prevents Hub sealing.
                self._eof_stop_submitted = True
                self._submit_command(
                    "模拟输入已播放完成",
                    lambda: runtime.stop(drain_timeout_seconds=None),
                )
            latest = None
            while True:
                try:
                    latest = runtime.latest_dev_ui.get_nowait()
                except queue.Empty:
                    break
            latest_l2 = None
            l2_mailbox = getattr(runtime, "latest_l2_dev_ui", None)
            if l2_mailbox is not None:
                while True:
                    try:
                        latest_l2 = l2_mailbox.get_nowait()
                    except queue.Empty:
                        break
            latest_l5 = None
            l5_mailbox = getattr(runtime, "latest_l5_dev_ui", None)
            if l5_mailbox is not None:
                while True:
                    try:
                        latest_l5 = l5_mailbox.get_nowait()
                    except queue.Empty:
                        break
            if latest is not None:
                if self._replay_reset_pending and self._replay_previous_stream is not None:
                    identity = None if latest.l1 is None else (latest.l1.session_id, latest.l1.stream_epoch)
                    if identity == self._replay_previous_stream:
                        latest = None
                    elif identity is not None:
                        self._replay_reset_pending = False
                elif self._replay_reset_pending and latest.l1 is not None:
                    self._replay_reset_pending = False
            # A completed stop may leave one diagnostic frame in the latest-
            # only mailboxes.  Keep the already-rendered final snapshot, but
            # never repaint that residual frame as LIVE after Runtime is idle.
            if not runtime.active:
                latest = None
                latest_l2 = None
                latest_l5 = None
            if latest_l2 is not None:
                identity = (
                    latest_l2.session_id,
                    latest_l2.stream_epoch,
                    latest_l2.window_id,
                    latest_l2.decision_sample,
                )
                current_stream = (
                    runtime.coordinator.session_id,
                    runtime.coordinator.stream_epoch,
                )
                previous = self._last_l2_identity
                if not _is_current_monotonic_l2_snapshot(
                    latest_l2, current_stream, previous
                ):
                    latest_l2 = None
                else:
                    self._l2_frame = latest_l2
                    self._last_l2_identity = identity
                    self._render_l2_frame(latest_l2)
            if latest is not None:
                self._frame = latest
                self._last_l1_seen = monotonic()
                # Render the ordered formal frame without allowing a terminal
                # DROPPED/SKIPPED frame to erase the independent latest
                # completed L5 result below.
                self._render_frame(latest, render_l2=False, render_l5=False)
            elif runtime.last_error:
                self._set_text(self.l1_header, f"ERROR | {runtime.last_error}")
            if latest_l5 is not None and self._frame is not None and self._frame.l1 is not None:
                immediate_identity = None if latest_l5.l1 is None else (
                    latest_l5.l1.session_id,
                    latest_l5.l1.stream_epoch,
                )
                current_identity = (self._frame.l1.session_id, self._frame.l1.stream_epoch)
                if immediate_identity != current_identity:
                    latest_l5 = None
            self._update_l5_panel(latest, latest_l5)
            if runtime.running:
                if self._frame is not None and monotonic() - self._last_l1_seen > config.dev_test_ui.stale_after_ms / 1000:
                    self._set_text(self.l1_header, "RUNNING | input data STALE (>500 ms)")
                if self.srp_polar._snapshot is not None and self.srp_polar._snapshot.age_ms > config.dev_test_ui.stale_after_ms:
                    self._set_text(self.srp_header, "RUNNING | SRP result STALE (>500 ms)")
            elif not runtime.active and self._pending_command is None and self._last_runtime_state != "stopped":
                self._enter_stopped_state()
            elapsed = self._record_elapsed + (
                monotonic() - self._record_started if self._record_started is not None else 0.0
            )
            if runtime.scratch.state in {"recording", "paused"}:
                self._set_text(self.recording_label,
                    f"状态: {runtime.scratch.state} | {int(elapsed // 60):02d}:{elapsed % 60:04.1f} | "
                    "当前文件: scratch/current"
                )
                self.recording_label.setToolTip(str(runtime.scratch.current_root))
            finalizing_replay = (
                self._pending_command is not None
                and self._pending_command[0] == "模拟输入已播放完成"
                and runtime.active
            )
            state = (
                "ERROR"
                if runtime.last_error
                else "FINALIZING"
                if finalizing_replay
                else "RUNNING"
                if runtime.running
                else "STOPPED"
            )
            formal_active = runtime.runtime_recording_active
            self._set_text(self.runtime_recording_label,
                f"Runtime Recording: {runtime.runtime_recording_mode} / "
                f"{'recording' if formal_active else 'paused'}"
            )
            self.bf_panel.set_downstream_processing_enabled(
                runtime.downstream_processing_enabled
            )
            if live_microphone_mode:
                self.bf_panel.downstream_switch.setEnabled(formal_active)
                self.bf_panel.mode_switch.setEnabled(True)
            self.runtime_record.setEnabled(runtime.running and not formal_active)
            self.runtime_pause.setEnabled(runtime.running and formal_active)
            if self._pending_command is not None:
                self.runtime_record.setEnabled(False)
                self.runtime_pause.setEnabled(False)
            self._update_control_states()
            status_snapshot = _runtime_processing_status(runtime)
            pipeline_status = _format_processing_pipeline_status(
                runtime, status_snapshot
            )
            self._set_text(
                self.global_status,
                f"{state:<7} | input drop {runtime.fanout.dropped_by_subscriber:06d} | "
                f"processing drop {runtime.processing_drops:06d} | L2 {runtime.doa_backend} | "
                f"{runtime.processing_device.upper()} L3 {runtime.l3_processing_mode} | "
                f"{pipeline_status} | "
                f"scratch queue {runtime.scratch.queued_blocks:03d}",
            )
            tooltip = _format_processing_pipeline_tooltip(runtime, status_snapshot)
            if self.global_status.toolTip() != tooltip:
                self.global_status.setToolTip(tooltip)
            self._last_runtime_state = state.casefold()
            if runtime.scratch_error or runtime.scratch.last_error:
                self._set_text(self.recording_label, f"状态: error | {runtime.scratch_error or runtime.scratch.last_error}")

        def _update_l5_panel(self, ordered_frame=None, immediate_frame=None, *, now=None):
            """Keep the newest valid L5 result until its configured stale limit.

            ``immediate_frame`` is a full, window-consistent DevUiFrame from
            the L5 worker.  Ordered audit frames may report a dropped window,
            but they are not allowed to clear that valid result immediately.
            """

            if runtime.active and not runtime.downstream_processing_enabled:
                self._last_l5_frame = None
                self._last_l5_seen = 0.0
                self._l5_is_stale = False
                self.cnn_panel.set_unavailable("STOPPED BY TEST UI; L2 remains active")
                return
            current = monotonic() if now is None else float(now)
            ordered_status = getattr(ordered_frame, "pipeline_status", None)
            ordered_stream = None if ordered_status is None else (
                ordered_status.session_id,
                ordered_status.stream_epoch,
            )
            cached_status = getattr(self._last_l5_frame, "pipeline_status", None)
            cached_stream = None if cached_status is None else (
                cached_status.session_id,
                cached_status.stream_epoch,
            )
            if (
                ordered_stream is not None
                and cached_stream is not None
                and ordered_stream != cached_stream
            ):
                self._last_l5_frame = None
                self._last_l5_seen = 0.0
                self._l5_is_stale = False
                self.cnn_panel.set_unavailable(
                    "WARMING: waiting for completed L5 window in the new stream epoch"
                )

            immediate_status = getattr(immediate_frame, "pipeline_status", None)
            immediate_stream = None if immediate_status is None else (
                immediate_status.session_id,
                immediate_status.stream_epoch,
            )
            if (
                ordered_stream is not None
                and immediate_stream is not None
                and immediate_stream != ordered_stream
            ):
                immediate_frame = None
            selected = None
            if immediate_frame is not None and getattr(immediate_frame, "l5_result", None) is not None:
                selected = immediate_frame
            elif ordered_frame is not None and getattr(ordered_frame, "l5_result", None) is not None:
                selected = ordered_frame
            if selected is not None:
                self._last_l5_frame = selected
                self._last_l5_seen = current
                self._l5_is_stale = False
                self.cnn_panel.set_result(selected.l5_result)
                return

            if self._last_l5_frame is None:
                if ordered_frame is not None:
                    self.cnn_panel.set_unavailable(
                        ordered_frame.missing_reasons.get("cnn", "NO RESULT")
                    )
                return

            stale_seconds = config.dev_test_ui.stale_after_ms / 1000.0
            if current - self._last_l5_seen > stale_seconds and not self._l5_is_stale:
                self._l5_is_stale = True
                self.cnn_panel.set_unavailable(
                    f"STALE: no completed L5 result for {config.dev_test_ui.stale_after_ms} ms"
                )

        def _render_l2_frame(self, frame):
            if frame.gate_decision is None:
                self.gate_readout.set_unavailable()
            else:
                self.gate_readout.set_decision(frame.gate_decision)
            with QSignalBlocker(self.gate_threshold):
                self.gate_threshold.set_value(
                    runtime.gate_probability_threshold,
                    pending=(
                        getattr(frame, "gate_config_revision", None) is not None
                        and getattr(frame, "gate_config_revision", None)
                        != runtime.gate_config_revision
                    ),
                )
            applied_revision = getattr(frame, "scan_config_revision", None)
            revision_pending = (
                applied_revision is not None
                and applied_revision != runtime.direction_scan_config_revision
            )
            with QSignalBlocker(self.music_dpd_rank1):
                self.music_dpd_rank1.set_enabled(
                    runtime.music_dpd_rank1_enabled, pending=revision_pending
                )
            with QSignalBlocker(self.music_noise_whitening):
                self.music_noise_whitening.set_enabled(
                    runtime.music_noise_whitening_enabled, pending=revision_pending
                )
            with QSignalBlocker(self.doa_backend):
                self.doa_backend.set_backend(runtime.doa_backend, pending=revision_pending)
            music_backend = runtime.doa_backend == "frequency_normalized_music"
            self.music_dpd_rank1.setEnabled(music_backend)
            self.music_noise_whitening.setEnabled(music_backend)
            applied_id_tracking = getattr(
                frame, "direction_id_tracking_enabled", None
            )
            with QSignalBlocker(self.srp_id_tracking):
                self.srp_id_tracking.set_enabled(
                    runtime.direction_id_tracking_enabled,
                    pending=revision_pending or (
                        applied_id_tracking is not None
                        and applied_id_tracking != runtime.direction_id_tracking_enabled
                    ),
                )
            if (
                frame.spatial_response is not None
                and frame.spatial_published_monotonic is not None
            ):
                l5_result = getattr(frame, "l5_result", None)
                probabilities = {
                    detection.track_id: detection.probability
                    for detection in (() if l5_result is None else l5_result.detections)
                    if getattr(detection, "track_id", None) is not None
                }
                snapshot = MusicPanelSnapshot(
                    frame.spatial_response,
                    getattr(frame, "directions", ()),
                    getattr(frame, "active_tracks", ()),
                    frame.spatial_published_monotonic,
                    probabilities,
                    effective_order=(
                        None if frame.search_diagnostics is None
                        else frame.search_diagnostics.effective_model_order
                    ),
                    raw_peaks=frame.candidates,
                    direction_id_tracking_enabled=(
                        True if frame.direction_id_tracking_enabled is None
                        else frame.direction_id_tracking_enabled
                    ),
                )
                window_key = (
                    frame.spatial_response.session_id,
                    frame.spatial_response.stream_epoch,
                    frame.spatial_response.window_id,
                )
                if window_key != self._last_rendered_window:
                    self.srp_polar.set_snapshot(snapshot, live=True)
                    self._last_rendered_window = window_key
                model = frame.spatial_response.model_order
                panel_state = (
                    "STALE"
                    if snapshot.age_ms > config.dev_test_ui.stale_after_ms
                    else "LIVE"
                )
                backend_name = (
                    "NN" if frame.search_diagnostics.mode == "gi_doaenet" else "MUSIC"
                )
                self._set_text(
                    self.music_status,
                    f"{backend_name} order={model.estimated_sources}  "
                    f"output={snapshot.effective_order if snapshot.effective_order is not None else '—'}  "
                    f"valid={frame.spatial_response.valid_frequency_bins}  "
                    f"status={frame.spatial_response.numerical_status}  {panel_state}",
                )
                diagnostics = frame.search_diagnostics
                search_suffix = ""
                if diagnostics is not None:
                    search_suffix = (
                        f" | {diagnostics.mode.upper()} {diagnostics.model_order.estimated_sources}"
                        f" / output {diagnostics.effective_model_order}"
                        f" | valid bins {diagnostics.valid_frequency_bins}"
                        f" | DPD {'ON' if diagnostics.dpd_rank1_enabled else 'OFF'}"
                        f" {diagnostics.selected_frequency_bins} bins"
                        f" | WHITE {diagnostics.whitening_status.upper()}"
                        f" | {diagnostics.covariance_quality.upper()}"
                        f" | ASSOC {'LMB/JPDA' if diagnostics.mode == 'gi_doaenet' else 'HUNGARIAN'}"
                    )
                dropped_reason = frame.missing_reasons.get("srp")
                state_prefix = (
                    f"STALE | {dropped_reason} | last completed"
                    if dropped_reason is not None
                    else "LIVE"
                )
                self._set_text(
                    self.srp_header,
                    f"{state_prefix} | session {frame.spatial_response.session_id[:8]} | "
                    f"epoch {frame.spatial_response.stream_epoch} | "
                    f"window {frame.spatial_response.window_id:08d} | "
                    f"sample {frame.spatial_response.decision_sample:012d} | "
                    f"age {snapshot.age_ms:03.0f} ms{search_suffix}",
                )
            elif "srp" in frame.missing_reasons:
                self.srp_header.setText(frame.missing_reasons["srp"])
                self.srp_polar.set_snapshot(None)
                self.music_status.setText(
                    "MUSIC order=—  output=—  valid=—  status=UNAVAILABLE"
                )
                self._last_rendered_window = None

        def _render_frame(self, frame, *, render_l2=True, render_l5=True):
            self.bf_panel.set_tracks(getattr(frame, "tracked_audio", ()))
            self.bf_panel.set_previews(
                frame.previews if runtime.downstream_processing_enabled else (),
                missing_reason=(
                    frame.missing_reasons.get("beamforming")
                    if runtime.downstream_processing_enabled
                    else "STOPPED BY TEST UI; L2 remains active"
                ),
            )
            if render_l5:
                self._update_l5_panel(frame)
            if render_l2:
                self._render_l2_frame(frame)
            l1 = frame.l1
            if l1 is not None:
                with QSignalBlocker(self.pre_denoise_switch):
                    self.pre_denoise_switch.setChecked(runtime.l1_pre_denoise_enabled)
                if l1.pre_denoise_enabled:
                    self._set_text(
                        self.pre_denoise_label,
                        f"预降噪: ON | 已替换7路 | 历史平均频率增益 {l1.pre_denoise_mean_gain_db:.1f} dB",
                    )
                else:
                    self._set_text(self.pre_denoise_label, "预降噪: OFF | 原始音频直通")
                with QSignalBlocker(self.countnet_switch):
                    self.countnet_switch.setChecked(runtime.l1_speaker_count_enabled)
                annotation = l1.speaker_count_annotation
                if not runtime.l1_speaker_count_enabled:
                    self._set_text(self.countnet_label, "Speech count: OFF")
                elif annotation is None:
                    self._set_text(
                        self.countnet_label,
                        "Speech count: WARMING_UP | 等待首个100 ms结果",
                    )
                elif annotation.status == "ready":
                    p = annotation.probabilities
                    level = (
                        ""
                        if annotation.input_rms_dbfs is None
                        else f"  input={annotation.input_rms_dbfs:.1f}dBFS"
                    )
                    gain = (
                        ""
                        if annotation.input_gain_db is None
                        else f" gain={annotation.input_gain_db:+.1f}dB"
                    )
                    self._set_text(
                        self.countnet_label,
                        f"Speech count={annotation.speaker_count}  "
                        f"P=[{float(p[0]):.3f},{float(p[1]):.3f},{float(p[2]):.3f}]"
                        f"{level}{gain}  end={annotation.end_sample}  "
                        f"model={annotation.model_id}",
                    )
                else:
                    suffix = "" if annotation.reason is None else f" | {annotation.reason}"
                    self._set_text(
                        self.countnet_label,
                        f"Speech count: {annotation.status.upper()}{suffix}",
                    )
                self._set_text(self.l1_header,
                    f"{frame.pipeline_status.state.upper()} | session {l1.session_id[:8]} | epoch {l1.stream_epoch} | "
                    f"sample {l1.end_sample:012d} | seq {l1.sequence_id:08d} | age 000 ms"
                )
                light_suffix = " (commanded)" if l1.light_state in {"on", "off"} else ""
                self.light_label.setText(f"状态: {l1.light_state.title()}{light_suffix}")
                manifest = runtime.scratch.current_root / "scratch_manifest.json"
                current = str(manifest) if manifest.exists() else str(runtime.scratch.current_root)
                display = "scratch/current/scratch_manifest.json" if manifest.exists() else "scratch/current"
                self._set_text(self.recording_label, f"状态: {l1.recording_state:<10} | 当前文件: {display}")
                self.recording_label.setToolTip(current)
                self._set_record_buttons(l1.recording_state)
                for index, (name, label) in enumerate(self.meter_labels):
                    rms = float(l1.rms_dbfs[index])
                    meter_value = max(-90, round(rms))
                    if self.meter_bars[index].value() != meter_value:
                        self.meter_bars[index].setValue(meter_value)
                    self._set_text(label, f"{name}\n{rms:.1f} dB")
                hop = l1.imcra_hop
                if hop is None:
                    self._set_text(self.imcra_label, "IMCRA: UNAVAILABLE")
                else:
                    levels = "  ".join(
                        f"{name} {float(value):.1f} dB"
                        for name, value in zip(
                            ("MIC0", "MIC1", "MIC2", "MIC3", "MIC4", "MIC5", "Center"),
                            hop.noise_level_db,
                            strict=True,
                        )
                    )
                    self._set_text(self.imcra_label, f"IMCRA: {hop.state.upper()} | {levels}")
            now = monotonic()
            if now - self._last_performance_refresh >= 1.0 / config.dev_test_ui.performance_refresh_hz:
                self._set_performance(frame.performance)
                self._last_performance_refresh = now

        def _set_performance(self, perf):
            if perf is None:
                text = (
                    "上一秒性能 | L2 N/A | L3 N/A | L4 离线 | L5 离线 | L6 手动离线 | "
                    "20ms窗口 0 | 丢窗 0 | 丢窗率 0.0%"
                )
            else:
                text = (
                    "上一秒性能 | "
                    f"L2 {_time(perf.l2_time_ms_last_second_avg)} | "
                    f"L3 {_time(perf.l3_time_ms_last_second_avg)} | "
                    "L4 离线 | L5 离线 | L6 手动离线 | "
                    f"20ms窗口 {perf.processed_windows_last_second} | "
                    f"丢窗 {perf.dropped_windows_last_second} | "
                    f"丢窗率 {perf.drop_rate_last_second * 100.0:.1f}%"
                )
                self.performance_bar.setToolTip(
                    "每1秒刷新；L2、L3显示实时20 ms窗口平均耗时；L4、L5在L3长音频"
                    "拼接完成后离线运行，不伪装成实时窗口性能。"
                )
            self._performance_base_text = text
            self._refresh_total_duration_text()

        def _refresh_total_duration_text(self):
            text = getattr(self, "_performance_base_text", "")
            durations = runtime.pipeline_total_durations_seconds
            offline_durations = self._offline_stage_durations_seconds

            def elapsed(stage: str) -> str:
                source = durations if stage in {"l2", "l3"} else offline_durations
                value = source.get(stage)
                return "N/A" if value is None else f"{value:.2f} s"

            text += (
                "    总处理时长 | "
                f"L2 {elapsed('l2')} | L3 {elapsed('l3')} | "
                f"L4 {elapsed('l4')} | L5 {elapsed('l5')} | L6 {elapsed('l6')}"
            )
            self.performance_bar.setToolTip(
                "左侧只统计实时L2/L3窗口性能；总处理时长中，L2、L3分别从首个20 ms"
                "窗口入队至各自排空，L3包含长音频拼接；L4、L5在点击发送后按整批"
                "顺序独立计时，L6仅在手动点击后计时。模拟输入手动暂停期间不计实时阶段时长。"
            )
            self.performance_bar.setText(text)

        def closeEvent(self, event):
            self._closing = True
            self._refresh_timer.stop()
            try:
                self._commands.shutdown(wait=True, cancel_futures=False)
            finally:
                try:
                    # Windows cannot delete a mapped playback snapshot until
                    # the audio player has released it.
                    self.preview_player.close()
                finally:
                    try:
                        self._l4_store.close()
                    finally:
                        try:
                            self._l6_store.close()
                        finally:
                            try:
                                runtime.close(delete_dev_test_ui_audio=True)
                            finally:
                                super().closeEvent(event)

        def keyPressEvent(self, event):
            if event.key() == Qt.Key.Key_F11:
                self.showNormal() if self.isFullScreen() else self.showFullScreen()
            elif event.key() == Qt.Key.Key_Escape and self.isFullScreen():
                self.showNormal()
            else:
                super().keyPressEvent(event)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.showFullScreen() if config.dev_test_ui.start_fullscreen else window.showMaximized()
    if auto_start:
        QTimer.singleShot(0, window._start_capture)
    return app, window


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input-wav", help="后台注入Test UI的7/8通道48 kHz测试WAV")
    parser.add_argument("--replay-recording", help="完整模拟录音的recording_manifest.json")
    parser.add_argument("--auto-start", action="store_true")
    args = parser.parse_args()
    app, _window = build_window(
        args.config,
        input_wav=args.input_wav,
        replay_recording=args.replay_recording,
        auto_start=args.auto_start,
    )
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
