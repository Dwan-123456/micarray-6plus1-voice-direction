from __future__ import annotations

import json
import shutil
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from PySide6.QtCore import QObject, Signal

from common.config import calibration_config_hash, config_hash, load_config
from data_management import RecordingStore, SessionMetadata
from data_management.dedicated_recording import WizardPhase
from ingest import IngestCoordinator
from layer1_input.calibration import ChannelCalibrator
from layer1_input.configuration import AudioConfig, CalibrationConfig, CdcConfig
from layer1_input.pipeline import InputPipeline
from layer1_input.imcra import Layer1Imcra
from layer1_input.sources import LiveSipeedSource
from layer1_input.serial_device import SerialDevice


class CaptureHost(QObject):
    """UAC-only host for the standalone manager; owns one authoritative input timeline."""

    STATUS_INTERVAL_SECONDS = 0.25
    CAPTURING_WIZARD_PHASES = {WizardPhase.RECORDING}
    ACTIVE_WIZARD_PHASES = {WizardPhase.RECORDING, WizardPhase.PAUSED}

    connected_changed = Signal(bool, str)
    runtime_status = Signal(object)
    wizard_status = Signal(object)
    error = Signal(str)

    def __init__(self, project_root: str | Path, service):
        super().__init__()
        self.project_root = Path(project_root).resolve()
        self.service = service
        self.config = load_config(self.project_root / "config" / "config.yaml")
        self.data_root = Path(service.data_root)
        self.recording_store = RecordingStore(self.data_root, catalog=service.catalog, config=self.config)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pipeline: InputPipeline | None = None
        self._mode = "off"
        self._started_at = 0.0
        self._session_id: str | None = None
        self._last_wizard_emit = 0.0
        self._last_status_emit = 0.0
        self._lock = threading.RLock()
        self._log_lock = threading.Lock()
        self._log_path = self.data_root / "logs" / "audio_data_manager_capture.log"

    def _make_pipeline(self) -> InputPipeline:
        cdc = CdcConfig.from_project(self.config)
        return InputPipeline(
            LiveSipeedSource(AudioConfig.from_project(self.config)),
            ChannelCalibrator(CalibrationConfig.from_project(self.config)),
            SerialDevice(cdc.port, cdc.baudrate) if cdc.enabled else None,
            owns_hotmap_source=True,
            hotmap_required=cdc.required,
            timestamp_tolerance_ms=self.config.timing.timestamp_tolerance_ms,
        )

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive() and self._pipeline is not None

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._last_status_emit = 0.0
            self._last_wizard_emit = 0.0
            self._thread = threading.Thread(target=self._run, name="audio-data-manager-capture", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> bool:
        self._stop.set()
        with self._lock:
            pipeline = self._pipeline
            thread = self._thread
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as exc:
                self._log("warning", f"停止音频输入时发生异常：{exc}")
        if thread and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))
        stopped = thread is None or not thread.is_alive()
        if not stopped:
            message = f"麦克风采集线程未能在 {timeout:g} 秒内停止"
            self._log("error", message)
            self.error.emit(message)
        return stopped

    def handle_command(self, command: str) -> None:
        try:
            if command.startswith("mode:"):
                self._mode = command.split(":", 1)[1]
                if self.connected:
                    self.recording_store.set_recording_mode(self._mode)
            elif not self.connected:
                raise RuntimeError("麦克风采集源尚未连接")
            elif command == "record":
                self.recording_store.start_recording()
            elif command == "pause":
                self.recording_store.pause_recording()
            elif command == "stop":
                if self._mode == "manual":
                    self.recording_store.pause_recording()
                elif self._mode in {"continuous", "event"}:
                    self._mode = "off"
                    self.recording_store.set_recording_mode("off")
                self._emit_status(force=True)
        except Exception as exc:
            self._log("error", f"录音控制失败（{command}）：{exc}")
            self.error.emit(str(exc))

    def _metadata(self) -> SessionMetadata:
        return SessionMetadata(
            config_hash(self.config),
            calibration_config_hash(self.config.calibration),
            geometry_version=self.config.hardware.geometry_version,
            runtime={"host": "standalone_audio_data_manager"},
            algorithm_versions={
                "capture": "layer1_uac_v1",
                "layer1_imcra": self.config.layer1_imcra.algorithm_version,
            },
        )

    def _run(self) -> None:
        pipeline: InputPipeline | None = None
        session_started = False
        run_error: Exception | None = None
        try:
            coordinator = IngestCoordinator(
                sample_rate=self.config.device.sample_rate,
                timestamp_tolerance_ms=self.config.timing.timestamp_tolerance_ms,
            )
            imcra = Layer1Imcra.from_project(self.config)
            pipeline = self._make_pipeline()
            self.recording_store.start_session(coordinator.session_id, self._metadata())
            session_started = True
            self.recording_store.set_recording_mode(self._mode)
            pipeline.start()
            with self._lock:
                self._pipeline = pipeline
                self._session_id = coordinator.session_id
                self._started_at = monotonic()
            self._log("info", f"麦克风采集已连接，会话 {coordinator.session_id}")
            self.connected_changed.emit(True, "麦克风已连接。现在可以选择录音模式，或进入测试录制向导。")
            self._emit_status(force=True)
            while not self._stop.is_set():
                audio = pipeline.read(timeout=0.1)
                if audio is None:
                    continue
                take_hotmaps = getattr(pipeline, "take_hotmap_frames", None)
                hotmap_frames = tuple(take_hotmaps()) if callable(take_hotmaps) else ()
                block = coordinator.ingest(audio, tuple(pipeline.take_health_events()))
                if self.config.layer1_imcra.enabled:
                    hops = imcra.process(block)
                    if len(hops) == 1 and hops[0].start_sample == block.start_sample and hops[0].end_sample == block.end_sample:
                        block = replace(block, imcra_hop=hops[0])
                self.recording_store.append_audio(block)
                phase = self.service.wizard.phase
                if phase in self.CAPTURING_WIZARD_PHASES:
                    try:
                        status = self.service.wizard.append(block, hotmap_frames)
                    except Exception as exc:
                        reason = f"测试录制已中止：{exc}。运行录音和麦克风采集仍在继续。"
                        status = self.service.wizard.abort(reason)
                        self._log("warning", reason)
                        self.error.emit(reason)
                        self.wizard_status.emit(status)
                    else:
                        now = monotonic()
                        if now - self._last_wizard_emit >= 0.1 or status.phase != phase:
                            self.wizard_status.emit(status)
                            self._last_wizard_emit = now
                self._emit_status()
        except Exception as exc:
            run_error = exc
            self._log("error", f"采集循环失败：{exc}")
            self.error.emit(str(exc))
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception as exc:
                    self._log("warning", f"释放音频输入时发生异常：{exc}")
            phase = self.service.wizard.phase
            if phase in self.ACTIVE_WIZARD_PHASES:
                if run_error is None:
                    reason = "麦克风已断开，本次测试录制已中止，请重新开始。"
                else:
                    reason = f"麦克风采集异常，本次测试录制已中止：{run_error}"
                status = self.service.wizard.abort(reason)
                self.wizard_status.emit(status)
            if session_started:
                try:
                    self.recording_store.stop_session("normal" if self._stop.is_set() else "input_error")
                except Exception as exc:
                    self._log("error", f"录音封存失败：{exc}")
                    self.error.emit(f"录音封存失败：{exc}")
            with self._lock:
                self._pipeline = None
                self._session_id = None
                if self._thread is threading.current_thread():
                    self._thread = None
            self._log("info", "麦克风采集已断开，运行录音封存流程结束")
            self.connected_changed.emit(False, "麦克风已断开。已有录音已安全封存。")

    def _emit_status(self, *, force: bool = False) -> None:
        now = monotonic()
        with self._lock:
            if not force and now - self._last_status_emit < self.STATUS_INTERVAL_SECONDS:
                return
            self._last_status_emit = now
            mode = self._mode
            session_id = self._session_id
            started_at = self._started_at
        root = self.data_root if self.data_root.exists() else self.data_root.parent
        self.runtime_status.emit(
            {
                "mode": mode,
                "session_id": session_id,
                "duration_seconds": max(0.0, now - started_at),
                "free_bytes": shutil.disk_usage(root).free,
            }
        )

    def _log(self, level: str, message: str) -> None:
        """Write low-volume lifecycle diagnostics for the pythonw desktop entry."""
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "level": level,
                "message": message,
            }
            with self._log_lock, self._log_path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def close(self) -> None:
        stopped = self.stop(timeout=10.0)
        if not stopped:
            raise RuntimeError("麦克风采集仍在停止，暂时不能关闭录音存储")
        self.recording_store.close()
