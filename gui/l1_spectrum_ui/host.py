from __future__ import annotations

import threading
from dataclasses import replace
from typing import Callable

from PySide6.QtCore import QObject, Signal

from common.config import ProjectConfig
from gui.dev_test_ui.meter import L1Meter
from ingest import IngestCoordinator
from layer1_input.calibration import ChannelCalibrator
from layer1_input.configuration import AudioConfig, CalibrationConfig, CdcConfig
from layer1_input.imcra import Layer1Imcra
from layer1_input.pipeline import InputPipeline
from layer1_input.pre_denoise import ImcraWienerPreDenoiser
from layer1_input.protocols import led_command
from layer1_input.serial_device import SerialDevice
from layer1_input.sources import LiveSipeedSource

from .backend import L1SpectrumAnalyzer


class L1SpectrumHost(QObject):
    """Owns the standalone UAC -> calibrated L1 -> IMCRA observation chain."""

    frame_ready = Signal(object)
    state_changed = Signal(str)
    light_state_changed = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        config: ProjectConfig,
        *,
        pipeline_factory: Callable[[], InputPipeline] | None = None,
        serial_device: SerialDevice | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self._pipeline_factory = pipeline_factory or self._make_pipeline
        cdc = CdcConfig.from_project(config)
        self._serial_device = serial_device or SerialDevice(cdc.port, cdc.baudrate)
        self._light_available = cdc.enabled or serial_device is not None
        self._light_command_lock = threading.Lock()
        self._light_thread: threading.Thread | None = None
        self._light_state = "unknown" if self._light_available else "unavailable"
        self._thread: threading.Thread | None = None
        self._pipeline: InputPipeline | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._pre_denoise_enabled = config.layer1_pre_denoise.enabled
        self._pre_denoise_latency_active = config.layer1_pre_denoise.enabled

    def _make_pipeline(self) -> InputPipeline:
        # The audio pipeline deliberately owns only UAC input. The separate
        # CDC device starts lazily only when the operator sends a light command;
        # no recording, windowing, hotmap consumer, or L2-L4 is created.
        return InputPipeline(
            LiveSipeedSource(AudioConfig.from_project(self.config)),
            ChannelCalibrator(CalibrationConfig.from_project(self.config)),
            timestamp_tolerance_ms=self.config.timing.timestamp_tolerance_ms,
        )

    @property
    def running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def pre_denoise_enabled(self) -> bool:
        with self._lock:
            return self._pre_denoise_enabled

    def set_pre_denoise_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._pre_denoise_enabled = bool(enabled)

    @property
    def light_state(self) -> str:
        with self._lock:
            return self._light_state

    def set_light(self, enabled: bool, *, report_errors: bool = True) -> None:
        """Send one official LED command without blocking the Qt event loop."""
        if not self._light_available:
            self.light_state_changed.emit("unavailable")
            if report_errors:
                self.error.emit("配置已禁用CDC串口，无法控制灯光")
            return
        with self._lock:
            if self._light_thread is not None and self._light_thread.is_alive():
                return
            self._light_state = "pending"
            self.light_state_changed.emit("pending")
            self._light_thread = threading.Thread(
                target=self._write_light,
                args=(bool(enabled), report_errors),
                name="l1-spectrum-ui-light",
                daemon=True,
            )
            self._light_thread.start()

    def _write_light(self, enabled: bool, report_errors: bool = True) -> None:
        packet = led_command(enabled)
        try:
            with self._light_command_lock:
                count = self._serial_device.write(packet)
            if count != len(packet):
                raise OSError(f"灯控命令未完整写入：{count}/{len(packet)}")
        except Exception as exc:
            state = "error" if report_errors else "unknown"
            with self._lock:
                self._light_state = state
            self.light_state_changed.emit(state)
            if report_errors:
                self.error.emit(str(exc))
            return
        state = "on" if enabled else "off"
        with self._lock:
            self._light_state = state
        self.light_state_changed.emit(state)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="l1-spectrum-ui-capture", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> bool:
        self._stop.set()
        with self._lock:
            pipeline, thread = self._pipeline, self._thread
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))
        stopped = thread is None or not thread.is_alive()
        if not stopped:
            self.error.emit(f"麦克风采集线程未能在 {timeout:g} 秒内停止")
        return stopped

    def _select_pre_denoise(self, block, pre_denoiser: ImcraWienerPreDenoiser):
        pairs = pre_denoiser.process(block)
        with self._lock:
            enabled = self._pre_denoise_enabled
            if not self._pre_denoise_latency_active:
                if not enabled:
                    return (block,)
                self._pre_denoise_latency_active = True
                return ()
        return tuple(item.denoised if enabled else item.raw for item in pairs)

    def _run(self) -> None:
        pipeline = None
        try:
            coordinator = IngestCoordinator(
                sample_rate=self.config.device.sample_rate,
                timestamp_tolerance_ms=self.config.timing.timestamp_tolerance_ms,
            )
            imcra = Layer1Imcra.from_project(self.config)
            pre_denoiser = ImcraWienerPreDenoiser.from_project(self.config)
            meter = L1Meter()
            analyzer = L1SpectrumAnalyzer(sample_rate=self.config.device.sample_rate)
            pipeline = self._pipeline_factory()
            with self._lock:
                self._pipeline = pipeline
                self._pre_denoise_latency_active = self._pre_denoise_enabled
            pipeline.start()
            # The LED default belongs to a successfully connected microphone
            # lifecycle. If UAC start fails this line is never reached, so the
            # UI neither opens CDC nor reports an unrelated light error.
            self.set_light(False, report_errors=False)
            self.state_changed.emit("RUNNING | L1 microphone + IMCRA only")
            while not self._stop.is_set():
                audio = pipeline.read(timeout=0.1)
                if audio is None:
                    continue
                block = coordinator.ingest(audio, tuple(pipeline.take_health_events()))
                hops = imcra.process(block) if self.config.layer1_imcra.enabled else ()
                if (
                    len(hops) == 1
                    and hops[0].start_sample == block.start_sample
                    and hops[0].end_sample == block.end_sample
                    and hops[0].source_sequence_ids == (block.sequence_id,)
                ):
                    block = replace(block, imcra_hop=hops[0])
                for selected in self._select_pre_denoise(block, pre_denoiser):
                    enabled = self.pre_denoise_enabled and self._pre_denoise_latency_active
                    snapshot = meter.add(
                        selected,
                        light_state=self.light_state,
                        pre_denoise_enabled=enabled,
                        pre_denoise_mean_gain_db=pre_denoiser.last_mean_gain_db if enabled else 0.0,
                    )
                    self.frame_ready.emit(analyzer.analyze(selected, snapshot))
        except Exception as exc:
            if not self._stop.is_set():
                self.error.emit(str(exc))
                self.state_changed.emit(f"ERROR | {exc}")
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            with self._lock:
                self._pipeline = None
                if self._thread is threading.current_thread():
                    self._thread = None
            self.state_changed.emit("STOPPED | L1 microphone disconnected")

    def close(self) -> None:
        self.stop()
        with self._lock:
            light_thread = self._light_thread
        if light_thread is not None and light_thread is not threading.current_thread():
            light_thread.join(timeout=2.0)
        self._serial_device.stop()
