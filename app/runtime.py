from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic, perf_counter

from common.config import ProjectConfig, load_config
from common.geometry import physical_6plus1_geometry
from gui.dev_test_ui.contracts import L2DevUiSnapshot
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
from layer2_source_detection import (
    DirectionScanConfig,
    Layer2Pipeline,
    ProbabilityGateDecision,
    ProbabilityGateState,
    SourceProbability20ms,
    SourceProbabilityState,
)
from source_counting import (
    IncrementalGccPhatSourceCounter,
    SourceCounterConfig,
    SourceCountSnapshot,
)
from windowing import WindowAssembler

from .adaptive_rate import AdaptiveRateController


@dataclass(frozen=True, slots=True)
class _QueuedWindow:
    window: object
    enqueued_monotonic: float


class ApplicationRuntime:
    """Minimal v1.4 runtime: IMCRA -> Gate -> source count -> MUSIC/ID."""

    def __init__(
        self,
        config: ProjectConfig,
        *,
        project_root: str | Path,
        pipeline: InputPipeline | None = None,
        serial_device: SerialDevice | None = None,
    ) -> None:
        self.config = config
        self.project_root = Path(project_root).resolve()
        cdc = CdcConfig.from_project(config)
        self.serial_device = serial_device or SerialDevice(cdc.port, cdc.baudrate)
        if pipeline is None:
            pipeline = InputPipeline(
                LiveSipeedSource(AudioConfig.from_project(config)),
                ChannelCalibrator(CalibrationConfig.from_project(config)),
                self.serial_device if cdc.enabled else None,
                hotmap_required=cdc.required,
                timestamp_tolerance_ms=config.timing.timestamp_tolerance_ms,
            )
        self.pipeline = pipeline
        self.latest_l1: queue.Queue[object] = queue.Queue(maxsize=1)
        self.latest_l2_dev_ui: queue.Queue[object] = queue.Queue(maxsize=1)
        self.latest_source_count: queue.Queue[object] = queue.Queue(maxsize=1)
        self.latest_windows: queue.Queue[object] = queue.Queue(maxsize=1)
        self._l2_windows: queue.Queue[object] = queue.Queue(maxsize=config.runtime.l2_queue_windows)
        self._stop = threading.Event()
        self._input_done = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._l2_thread: threading.Thread | None = None
        self._control_lock = threading.RLock()
        self._pre_denoise_lock = threading.Lock()
        self._source_count_enabled = config.source_counting.enabled
        self._music_order_follows_source_count = (
            config.source_counting.music_order_from_source_count
            and config.source_counting.enabled
        )
        self._source_count_control_revision = 0
        self._source_count_control_changed_at = 0.0
        self._source_count_applied_revision = -1
        self._current_music_effective_order: int | None = 2
        self._reset_state()
        self.light_state = "unknown"
        self.last_error: str | None = None
        self.processing_drops = 0
        self.l2_processed = 0
        self.source_count_processed = 0
        self.source_count_faults = 0
        self.source_count_last_error: str | None = None
        self._source_count_time_ms = 0.0
        self._l2_time_ms = 0.0
        self._started_at = 0.0
        self._input_exhausted = False
        audio_blocks_per_second = (
            config.device.sample_rate // config.device.block_size_samples
        )
        self._audio_cache_1s: deque[object] = deque(maxlen=audio_blocks_per_second)
        self._track_history: dict[int, dict[str, object]] = {}
        self._track_log_path = self.project_root / "tmp" / "l2_track_history.txt"
        self._last_track_log_sample = -240_000
        self._performance_enabled = True
        self._performance_lock = threading.Lock()
        self._performance_events: deque[tuple[float, str, float]] = deque(maxlen=512)
        runtime = config.runtime
        self._adaptive_enabled = runtime.adaptive_fallback_enabled
        self._adaptive_l2 = AdaptiveRateController(
            maximum_period_ms=runtime.adaptive_maximum_period_ms,
            overload_threshold_ms=runtime.adaptive_overload_threshold_ms,
            recovery_threshold_ms=runtime.adaptive_recovery_threshold_ms,
            recovery_stable_ms=runtime.adaptive_recovery_stable_ms,
        )
        self._last_l2_snapshot: L2DevUiSnapshot | None = None
        self._last_l2_control_key: tuple[object, ...] | None = None
        self._maximum_sparse_music_ms = 0.0

    @classmethod
    def from_config_path(cls, path: str | Path) -> "ApplicationRuntime":
        config_path = Path(path).resolve()
        return cls(load_config(config_path), project_root=config_path.parent.parent)

    def _reset_state(self) -> None:
        self.coordinator = IngestCoordinator(
            sample_rate=self.config.device.sample_rate,
            timestamp_tolerance_ms=self.config.timing.timestamp_tolerance_ms,
        )
        self.assembler = WindowAssembler()
        self.imcra = Layer1Imcra.from_project(self.config)
        self.pre_denoiser = ImcraWienerPreDenoiser.from_project(self.config)
        self.meter = L1Meter()
        self._layer2 = Layer2Pipeline.from_project(self.config)
        self._source_counter_config = SourceCounterConfig.from_project(self.config)
        self._source_counter = IncrementalGccPhatSourceCounter(self._source_counter_config)
        self._geometry = physical_6plus1_geometry(
            self.config.hardware.speed_of_sound_mps,
            self.config.hardware.geometry_version,
            self.config.hardware.ring_radius_m,
        )
        self._scan_config = replace(
            DirectionScanConfig.from_project(self.config),
            effective_order_limit=2,
        )
        self._scan_revision = 0
        self._gate_threshold = self.config.layer2.probability_gate.threshold
        self._gate_revision = 0
        self._id_tracking_enabled = True
        self._pre_denoise_enabled = self.config.layer1_pre_denoise.enabled
        self._pre_denoise_latency_active = self._pre_denoise_enabled
        self._source_count_applied_revision = -1
        self._current_music_effective_order = (
            1 if self._music_order_follows_source_count else 2
        )

    @staticmethod
    def _publish_latest(mailbox: queue.Queue[object], value: object) -> None:
        try:
            mailbox.put_nowait(value)
        except queue.Full:
            try:
                mailbox.get_nowait()
            except queue.Empty:
                pass
            mailbox.put_nowait(value)

    @property
    def running(self) -> bool:
        return self._capture_thread is not None and self._capture_thread.is_alive()

    @property
    def active(self) -> bool:
        return self.running or self.processing_running

    @property
    def input_exhausted(self) -> bool:
        return self._input_exhausted

    @property
    def processing_running(self) -> bool:
        return self._l2_thread is not None and self._l2_thread.is_alive()

    @property
    def source_count_running(self) -> bool:
        return self.processing_running and self.source_counting_enabled

    @property
    def run_started_monotonic(self) -> float:
        return self._started_at

    @property
    def l1_pre_denoise_enabled(self) -> bool:
        with self._pre_denoise_lock:
            return self._pre_denoise_enabled

    def set_l1_pre_denoise_enabled(self, enabled: bool) -> bool:
        if type(enabled) is not bool:
            raise ValueError("pre-denoise setting must be bool")
        with self._pre_denoise_lock:
            self._pre_denoise_enabled = enabled
        return enabled

    @property
    def gate_probability_threshold(self) -> float:
        with self._control_lock:
            return self._gate_threshold

    def set_gate_probability_threshold(self, value: float) -> float:
        value = float(value)
        if not 0 <= value <= 1:
            raise ValueError("Gate threshold must be in [0,1]")
        with self._control_lock:
            if value != self._gate_threshold:
                self._gate_threshold = value
                self._gate_revision += 1
        return value

    @property
    def direction_threshold(self) -> float:
        with self._control_lock:
            return self._scan_config.direction_threshold

    def set_direction_threshold(self, value: float) -> float:
        value = float(value)
        if not 0 <= value <= 1:
            raise ValueError("direction threshold must be in [0,1]")
        with self._control_lock:
            self._scan_config = replace(self._scan_config, direction_threshold=value)
            self._scan_revision += 1
        return value

    @property
    def music_effective_order_limit(self) -> int:
        with self._control_lock:
            return 2 if self._current_music_effective_order is None else self._current_music_effective_order

    def set_music_effective_order_limit(self, value: int) -> int:
        if value != 2:
            raise ValueError("fixed MUSIC order is 2; use the source-count follow switch")
        return 2

    @property
    def source_counting_enabled(self) -> bool:
        with self._control_lock:
            return self._source_count_enabled

    def set_source_counting_enabled(self, enabled: bool) -> bool:
        if type(enabled) is not bool:
            raise ValueError("source-counting setting must be bool")
        with self._control_lock:
            if enabled != self._source_count_enabled:
                self._source_count_enabled = enabled
                if not enabled:
                    self._music_order_follows_source_count = False
                    self._current_music_effective_order = 2
                self._source_count_control_revision += 1
                self._source_count_control_changed_at = monotonic()
        return enabled

    @property
    def music_order_follows_source_count(self) -> bool:
        with self._control_lock:
            return self._music_order_follows_source_count

    def set_music_order_follows_source_count(self, enabled: bool) -> bool:
        if type(enabled) is not bool:
            raise ValueError("MUSIC source-count-order setting must be bool")
        with self._control_lock:
            if enabled and not self._source_count_enabled:
                raise ValueError("enable source counting before following its MUSIC order")
            if enabled != self._music_order_follows_source_count:
                self._music_order_follows_source_count = enabled
                self._current_music_effective_order = 1 if enabled else 2
        return enabled

    @property
    def current_music_effective_order(self) -> int | None:
        with self._control_lock:
            return self._current_music_effective_order

    @property
    def source_count_control_changed_monotonic(self) -> float:
        with self._control_lock:
            return self._source_count_control_changed_at

    @property
    def direction_id_tracking_enabled(self) -> bool:
        return self._id_tracking_enabled

    def set_direction_id_tracking_enabled(self, value: bool) -> bool:
        if type(value) is not bool:
            raise ValueError("ID tracking setting must be bool")
        with self._control_lock:
            self._id_tracking_enabled = value
            self._scan_revision += 1
        return value

    @property
    def doa_backend(self) -> str:
        return self._scan_config.scanner_backend

    @property
    def processing_queue_depths(self) -> dict[str, int]:
        return {"l2": self._l2_windows.qsize()}

    @property
    def processing_status(self) -> dict[str, object]:
        elapsed = max(monotonic() - self._started_at, 1e-9) if self._started_at else 0.0
        return {
            "queue_depths": self.processing_queue_depths,
            "queue_capacities": {"l2": self._l2_windows.maxsize},
            "stage_alive": {
                "l1": self.running,
                "l2": self.processing_running,
                "source_count": self.source_count_running,
            },
            "completed_counts": {
                "l2": self.l2_processed,
                "source_count": self.source_count_processed,
            },
            "processing_drops": self.processing_drops,
            "l2_average_ms": self._l2_time_ms / max(self.l2_processed, 1),
            "source_count_average_ms": (
                self._source_count_time_ms / max(self.source_count_processed, 1)
            ),
            "l2_hz": self.l2_processed / elapsed if elapsed else 0.0,
            "adaptive_fallback": {
                "enabled": self._adaptive_enabled,
                "period_ms": self._adaptive_l2.period_ms,
                "stride": self._adaptive_l2.stride,
                "last_overload_reason": self._adaptive_l2.snapshot.last_overload_reason,
            },
            "last_error": self.last_error,
            "source_count_last_error": self.source_count_last_error,
            "source_count_enabled": self.source_counting_enabled,
            "music_order_follows_source_count": self.music_order_follows_source_count,
            "current_music_effective_order": self.current_music_effective_order,
        }

    @property
    def performance_monitor_enabled(self) -> bool:
        return self._performance_enabled

    def set_performance_monitor_enabled(self, enabled: bool) -> bool:
        if type(enabled) is not bool:
            raise ValueError("performance monitor setting must be bool")
        with self._performance_lock:
            self._performance_enabled = enabled
            self._performance_events.clear()
        return enabled

    @property
    def performance_snapshot(self) -> dict[str, object]:
        now = monotonic()
        with self._performance_lock:
            while self._performance_events and self._performance_events[0][0] < now - 1.0:
                self._performance_events.popleft()
            events = tuple(self._performance_events)
            enabled = self._performance_enabled
        grouped = {name: [] for name in (
            "imcra", "probability", "source_count", "source_count_fault", "music", "id",
            "queue_wait", "output", "compute", "reuse", "fault",
        )}
        for _timestamp, name, value in events:
            grouped[name].append(value)
        averages = {name: (sum(values) / len(values) if values else 0.0) for name, values in grouped.items()}
        return {
            "enabled": enabled, "window_seconds": 1.0,
            "imcra_ms": averages["imcra"], "probability_ms": averages["probability"],
            "source_count_ms": averages["source_count"],
            "music_ms": averages["music"], "id_tracking_ms": averages["id"],
            "total_ms": sum(
                averages[name]
                for name in ("imcra", "probability", "source_count", "music", "id")
            ),
            "queue_wait_ms": averages["queue_wait"],
            "source_count_frames_per_second": len(grouped["source_count"]),
            "source_count_faults_per_second": len(grouped["source_count_fault"]),
            "frames_per_second": len(grouped["output"]),
            "compute_frames_per_second": len(grouped["compute"]),
            "reused_frames_per_second": len(grouped["reuse"]),
            "faults_per_second": len(grouped["fault"]),
            "adaptive_period_ms": self._adaptive_l2.period_ms,
            "adaptive_stride": self._adaptive_l2.stride,
            "adaptive_last_overload_reason": self._adaptive_l2.snapshot.last_overload_reason,
        }

    def _record_performance(self, name: str, value_ms: float) -> None:
        with self._performance_lock:
            if self._performance_enabled:
                now = monotonic()
                self._performance_events.append((now, name, max(0.0, float(value_ms))))
                while self._performance_events and self._performance_events[0][0] < now - 1.0:
                    self._performance_events.popleft()

    def start(self) -> None:
        if self.active:
            return
        self._stop.clear()
        self._input_done.clear()
        self._input_exhausted = False
        self.last_error = None
        self.processing_drops = self.l2_processed = 0
        self.source_count_processed = self.source_count_faults = 0
        self.source_count_last_error = None
        self._l2_time_ms = 0.0
        self._source_count_time_ms = 0.0
        self._reset_state()
        self._audio_cache_1s.clear()
        self._track_history.clear()
        self._last_track_log_sample = -240_000
        self._adaptive_l2.reset()
        self._last_l2_snapshot = None
        self._last_l2_control_key = None
        self._maximum_sparse_music_ms = 0.0
        with self._performance_lock:
            self._performance_events.clear()
        while not self._l2_windows.empty():
            self._l2_windows.get_nowait()
        while not self.latest_source_count.empty():
            self.latest_source_count.get_nowait()
        self.pipeline.start()
        if self.config.device.serial_enabled:
            try:
                self.set_light(False)
            except Exception:
                self.light_state = "unknown"
        self._started_at = monotonic()
        self._capture_thread = threading.Thread(target=self._run_capture, name="l1-capture", daemon=True)
        self._l2_thread = threading.Thread(target=self._run_l2, name="l2-music-id", daemon=True)
        self._capture_thread.start()
        self._l2_thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop.set()
        self.pipeline.stop()
        limit = (
            timeout
            if timeout is not None
            else self.config.runtime.graceful_shutdown_timeout_seconds
        )
        deadline = monotonic() + max(0.0, limit)
        workers = (
            ("_capture_thread", "l1-capture"),
            ("_l2_thread", "l2-music-id"),
        )
        still_alive: list[str] = []
        for attribute, name in workers:
            worker = getattr(self, attribute)
            if worker is not None and worker.is_alive():
                worker.join(max(0.0, deadline - monotonic()))
            if worker is not None and worker.is_alive():
                still_alive.append(name)
            else:
                setattr(self, attribute, None)
        if still_alive and self.last_error is None:
            self.last_error = f"shutdown timeout: {', '.join(still_alive)} still running"

    def close(self) -> None:
        self.stop()
        try:
            self.serial_device.stop()
        except Exception:
            pass

    def set_light(self, enabled: bool) -> None:
        packet = led_command(enabled)
        count = self.serial_device.write(packet)
        if count != len(packet):
            raise OSError("light command was not written completely")
        self.light_state = "on" if enabled else "off"

    def _select_blocks(self, block) -> tuple[object, ...]:
        pairs = self.pre_denoiser.process(block)
        with self._pre_denoise_lock:
            enabled = self._pre_denoise_enabled
            if not self._pre_denoise_latency_active:
                if not enabled:
                    return (block,)
                self._pre_denoise_latency_active = True
                return ()
        return tuple(item.denoised if enabled else item.raw for item in pairs)

    def _publish_block(self, block) -> None:
        # The only retained audio is a bounded one-second in-memory ring.
        # It is cleared for every new capture and is never serialized.
        self._audio_cache_1s.append(block)
        windows = self.assembler.add(block)
        enabled = self.l1_pre_denoise_enabled and self._pre_denoise_latency_active
        snapshot = self.meter.add(
            block,
            light_state=self.light_state,
            pre_denoise_enabled=enabled,
            pre_denoise_mean_gain_db=self.pre_denoiser.last_mean_gain_db if enabled else 0.0,
        )
        self._publish_latest(self.latest_l1, snapshot)
        for window in windows:
            self._publish_latest(self.latest_windows, window)
            try:
                self._l2_windows.put_nowait(_QueuedWindow(window, monotonic()))
            except queue.Full:
                self._l2_windows.get_nowait()
                self._l2_windows.put_nowait(_QueuedWindow(window, monotonic()))
                self.processing_drops += 1

    def _flush_pre_denoiser_tail(self) -> tuple[object, ...]:
        flushed = self.pre_denoiser.flush()
        if not self._pre_denoise_latency_active:
            return ()
        enabled = self.l1_pre_denoise_enabled
        return tuple(item.denoised if enabled else item.raw for item in flushed)

    def _run_capture(self) -> None:
        try:
            while not self._stop.is_set():
                audio = self.pipeline.read(timeout=0.1)
                if audio is None:
                    if bool(getattr(self.pipeline.source, "exhausted", False)):
                        self._input_exhausted = True
                        break
                    continue
                block = self.coordinator.ingest(audio, tuple(self.pipeline.take_health_events()))
                hops = self.imcra.process(block) if self.config.layer1_imcra.enabled else ()
                if hops:
                    self._record_performance("imcra", self.imcra.last_core_ms)
                    self._record_performance("probability", self.imcra.last_probability_ms)
                if len(hops) == 1 and (hops[0].start_sample, hops[0].end_sample) == (block.start_sample, block.end_sample):
                    block = replace(block, imcra_hop=hops[0])
                for selected in self._select_blocks(block):
                    self._publish_block(selected)
        except Exception as exc:
            self.last_error = str(exc)
        finally:
            try:
                # With pre-denoise never activated, every raw block has already
                # been published immediately.  Publishing the denoiser's final
                # retained raw hop again would duplicate the last sample range,
                # break WindowAssembler continuity, and make L2/ID appear to
                # crash during stop/restart.  A latency-active chain still owns
                # one unpublished tail hop and must flush it normally.
                for block in self._flush_pre_denoiser_tail():
                    self._publish_block(block)
            finally:
                self._input_done.set()
                try:
                    self.pipeline.stop()
                except Exception:
                    pass

    @staticmethod
    def _imcra_probabilities(window) -> tuple[SourceProbability20ms, ...]:
        by_end = {hop.end_sample: hop for hop in window.imcra_hops}
        output = []
        for end in (window.doa_start_sample + 960, window.doa_end_sample):
            hop = by_end.get(end)
            if hop is None:
                return ()
            state = {
                "ready": SourceProbabilityState.READY,
                "warming_up": SourceProbabilityState.WARMING_UP,
                "invalid": SourceProbabilityState.INVALID,
            }[hop.state]
            output.append(SourceProbability20ms(
                hop.session_id, hop.stream_epoch, hop.start_sample, hop.end_sample,
                hop.array_source_probability_20ms, state, f"{hop.algorithm_version}:{hop.state}",
            ))
        return tuple(output)

    def _unavailable_source_count(
        self,
        window: object,
        *,
        error: str | None = None,
        reset: bool = True,
    ) -> SourceCountSnapshot:
        if reset:
            self._source_counter.reset()
        snapshot = SourceCountSnapshot(
            window.session_id,
            window.stream_epoch,
            window.window_id,
            window.decision_sample,
            None,
            monotonic(),
        )
        self._publish_latest(self.latest_source_count, snapshot)
        self.source_count_last_error = error
        return snapshot

    def _prepare_source_count_plan(
        self,
        window: object,
        decision: ProbabilityGateDecision,
        *,
        enabled: bool,
        follow_order: bool,
        control_revision: int,
    ) -> tuple[SourceCountSnapshot, int | None, str | None, float]:
        """Continuously count, then resolve the same-window order after Gate."""

        if self._source_count_applied_revision != control_revision:
            self._source_counter.reset()
            self._source_count_applied_revision = control_revision
        if not enabled:
            snapshot = self._unavailable_source_count(window)
            if decision.allow_srp:
                return snapshot, 2, None, 0.0
            return snapshot, None, decision.reason, 0.0

        started = perf_counter()
        try:
            snapshot = self._source_counter.process(window, self._geometry)
            expected = (
                window.session_id,
                window.stream_epoch,
                window.window_id,
                window.decision_sample,
            )
            actual = (
                snapshot.session_id,
                snapshot.stream_epoch,
                snapshot.window_id,
                snapshot.decision_sample,
            )
            if actual != expected:
                raise RuntimeError("source-count result does not match the current L2 window")
        except Exception as exc:
            elapsed_ms = (perf_counter() - started) * 1_000.0
            message = f"processing {type(exc).__name__}: {exc}"
            snapshot = self._unavailable_source_count(window, error=message)
            self.source_count_faults += 1
            self._record_performance("source_count_fault", elapsed_ms)
            if not decision.allow_srp:
                return snapshot, None, decision.reason, elapsed_ms
            return snapshot, 1 if follow_order else 2, None, elapsed_ms

        elapsed_ms = (perf_counter() - started) * 1_000.0
        self.source_count_processed += 1
        self._source_count_time_ms += elapsed_ms
        self._record_performance("source_count", elapsed_ms)
        self._publish_latest(self.latest_source_count, snapshot)
        self.source_count_last_error = None
        if not decision.allow_srp:
            return snapshot, None, decision.reason, elapsed_ms
        if not follow_order:
            return snapshot, 2, None, elapsed_ms
        order = 2 if snapshot.source_count is not None and snapshot.source_count >= 2 else 1
        return snapshot, order, None, elapsed_ms

    def _run_l2(self) -> None:
        try:
            while not self._stop.is_set() or not self._input_done.is_set() or not self._l2_windows.empty():
                try:
                    queued = self._l2_windows.get(timeout=0.1)
                except queue.Empty:
                    continue
                if isinstance(queued, _QueuedWindow):
                    window = queued.window
                    queue_wait_ms = max(0.0, (monotonic() - queued.enqueued_monotonic) * 1_000.0)
                else:  # Compatibility for a test or caller that injected a raw DecisionWindow.
                    window = queued
                    queue_wait_ms = 0.0
                with self._control_lock:
                    scan = self._scan_config
                    scan_revision = self._scan_revision
                    gate = self._gate_threshold
                    gate_revision = self._gate_revision
                    tracking = self._id_tracking_enabled
                    source_enabled = self._source_count_enabled
                    follow_order = self._music_order_follows_source_count
                    source_revision = self._source_count_control_revision
                self._record_performance("queue_wait", queue_wait_ms)
                started = perf_counter()
                decision: ProbabilityGateDecision | None = None
                control_key: tuple[object, ...] | None = None
                music_order: int | None = None
                source_count_ms = 0.0
                try:
                    decision, active_frame_count = self._layer2.evaluate_gate(
                        window,
                        self._imcra_probabilities(window),
                        gate_threshold=gate,
                        gate_config_revision=gate_revision,
                    )
                    _count_snapshot, music_order, music_skip_reason, source_count_ms = (
                        self._prepare_source_count_plan(
                            window,
                            decision,
                            enabled=source_enabled,
                            follow_order=follow_order,
                            control_revision=source_revision,
                        )
                    )
                    with self._control_lock:
                        self._current_music_effective_order = (
                            music_order
                            if follow_order
                            else 2
                        )
                    control_key = (
                        scan_revision,
                        gate_revision,
                        tracking,
                        source_revision,
                        source_enabled,
                        follow_order,
                        decision.state,
                        music_order,
                        music_skip_reason,
                    )
                    force = (
                        self._last_l2_snapshot is None
                        or self._last_l2_control_key != control_key
                        or (
                            self._last_l2_snapshot.session_id,
                            self._last_l2_snapshot.stream_epoch,
                        )
                        != (window.session_id, window.stream_epoch)
                    )
                    should_compute = (
                        True
                        if not self._adaptive_enabled
                        else self._adaptive_l2.should_compute(force=force)
                    )
                    can_reuse = (
                        decision.allow_srp
                        and music_order in {1, 2, 3}
                        and not should_compute
                        and self._last_l2_snapshot is not None
                        and self._last_l2_snapshot.spatial_response is not None
                        and self._last_l2_snapshot.spatial_response.model_order.estimated_sources
                        == music_order
                    )
                    if can_reuse:
                        sparse_started = perf_counter()
                        observe_covariance = getattr(
                            self._layer2.scanner,
                            "observe_covariance",
                            None,
                        )
                        if callable(observe_covariance):
                            observe_covariance(
                                window,
                                replace(scan, effective_order_limit=music_order),
                            )
                        sparse_music_ms = (perf_counter() - sparse_started) * 1_000.0
                        self._maximum_sparse_music_ms = max(
                            self._maximum_sparse_music_ms,
                            sparse_music_ms,
                        )
                        snapshot = self._reuse_l2_snapshot(
                            self._last_l2_snapshot,
                            window,
                            gate_decision=decision,
                            period_ms=self._adaptive_l2.period_ms,
                            queue_wait_ms=queue_wait_ms,
                        )
                        self._last_l2_snapshot = snapshot
                        self._last_l2_control_key = control_key
                        self._publish_latest(self.latest_l2_dev_ui, snapshot)
                        self._record_performance("music", sparse_music_ms)
                        self._record_performance("reuse", 0.0)
                        self._record_performance("output", 0.0)
                        self._update_track_log(snapshot.active_tracks, window.decision_sample)
                        continue
                    output = self._layer2.process_prepared(
                        window,
                        decision,
                        active_frame_count,
                        self._geometry,
                        scan,
                        music_effective_order=music_order,
                        music_skip_reason=music_skip_reason,
                        scan_config_revision=scan_revision,
                        direction_id_tracking_enabled=tracking,
                    )
                except Exception as exc:
                    self.last_error = f"L2 processing {type(exc).__name__}: {exc}"
                    if self._adaptive_enabled:
                        self._adaptive_l2.force_overload(f"l2_fault:{type(exc).__name__}")
                    can_reuse = (
                        decision is not None
                        and decision.allow_srp
                        and music_order in {1, 2, 3}
                        and self._last_l2_snapshot is not None
                        and self._last_l2_snapshot.spatial_response is not None
                        and self._last_l2_snapshot.spatial_response.model_order.estimated_sources
                        == music_order
                        and (
                            self._last_l2_snapshot.session_id,
                            self._last_l2_snapshot.stream_epoch,
                        )
                        == (window.session_id, window.stream_epoch)
                    )
                    snapshot = (
                        self._reuse_l2_snapshot(
                            self._last_l2_snapshot,
                            window,
                            gate_decision=decision,
                            period_ms=self._adaptive_l2.period_ms,
                            queue_wait_ms=queue_wait_ms,
                        )
                        if can_reuse
                        else self._fault_l2_snapshot(
                            window,
                            gate_threshold=gate,
                            gate_revision=gate_revision,
                            direction_threshold=scan.direction_threshold,
                            tracking=tracking,
                            scan_revision=scan_revision,
                            period_ms=self._adaptive_l2.period_ms,
                            queue_wait_ms=queue_wait_ms,
                            reason=f"{type(exc).__name__}:{exc}",
                        )
                    )
                    self._last_l2_snapshot = snapshot
                    self._last_l2_control_key = control_key
                    self._publish_latest(self.latest_l2_dev_ui, snapshot)
                    self._record_performance("fault", 0.0)
                    self._record_performance("reuse", 0.0)
                    self._record_performance("output", 0.0)
                    continue
                l2_wall_ms = (perf_counter() - started) * 1000.0
                self._l2_time_ms += l2_wall_ms
                self.l2_processed += 1
                music_ms = 0.0 if output.search_diagnostics is None else output.search_diagnostics.total_ms
                id_ms = output.id_tracking_ms or 0.0
                self._record_performance("music", music_ms)
                self._record_performance("id", id_ms)
                self._record_performance("compute", 0.0)
                self._record_performance("output", 0.0)
                if self._adaptive_enabled:
                    self._adaptive_l2.observe_compute(
                        queue_wait_ms=queue_wait_ms,
                        stage_ms={
                            "imcra": self.imcra.last_core_ms,
                            "probability": self.imcra.last_probability_ms,
                            "source_count": source_count_ms,
                            "music": max(music_ms, self._maximum_sparse_music_ms),
                            "id": id_ms,
                            "l2_total": l2_wall_ms,
                        },
                    )
                self._maximum_sparse_music_ms = 0.0
                self._update_track_log(output.active_tracks, window.decision_sample)
                snapshot = L2DevUiSnapshot(
                    window.session_id, window.stream_epoch, window.window_id, window.decision_sample,
                    output.spatial_response, output.candidates, output.gate_decision,
                    gate, gate_revision, scan.direction_threshold, tracking, scan_revision,
                    output.search_diagnostics, output.directions, output.active_tracks, monotonic(),
                    (
                        None
                        if output.spatial_response is not None
                        else output.music_skip_reason or output.gate_decision.reason
                    ),
                    self._adaptive_l2.period_ms, False, queue_wait_ms,
                )
                self._last_l2_snapshot = snapshot
                self._last_l2_control_key = control_key
                self._publish_latest(self.latest_l2_dev_ui, snapshot)
        except Exception as exc:
            self.last_error = str(exc)

    @staticmethod
    def _fault_l2_snapshot(
        window: object,
        *,
        gate_threshold: float,
        gate_revision: int,
        direction_threshold: float,
        tracking: bool,
        scan_revision: int,
        period_ms: int,
        queue_wait_ms: float,
        reason: str,
    ) -> L2DevUiSnapshot:
        gate = ProbabilityGateDecision(
            window.session_id,
            window.stream_epoch,
            window.window_id,
            window.decision_sample,
            "adaptive_fault_fallback_v1",
            ProbabilityGateState.UNAVAILABLE,
            None,
            None,
            None,
            gate_threshold,
            gate_revision,
            False,
            "l2_processing_fault",
            (reason,),
        )
        return L2DevUiSnapshot(
            window.session_id,
            window.stream_epoch,
            window.window_id,
            window.decision_sample,
            None,
            (),
            gate,
            gate_threshold,
            gate_revision,
            direction_threshold,
            tracking,
            scan_revision,
            None,
            (),
            (),
            monotonic(),
            f"adaptive_fault:{reason}",
            period_ms,
            True,
            queue_wait_ms,
        )

    @staticmethod
    def _reuse_l2_snapshot(
        previous: L2DevUiSnapshot | None,
        window: object,
        *,
        gate_decision: ProbabilityGateDecision | None = None,
        period_ms: int,
        queue_wait_ms: float,
    ) -> L2DevUiSnapshot:
        if previous is None:
            raise RuntimeError("adaptive reuse requires one computed L2 result")
        identity = {
            "session_id": window.session_id,
            "stream_epoch": window.stream_epoch,
            "window_id": window.window_id,
            "decision_sample": window.decision_sample,
        }
        doa = {
            **identity,
            "doa_start_sample": window.doa_start_sample,
            "doa_end_sample": window.doa_end_sample,
        }
        source_gate = previous.gate_decision if gate_decision is None else gate_decision
        gate_identity = (
            source_gate.session_id,
            source_gate.stream_epoch,
            source_gate.window_id,
            source_gate.decision_sample,
        )
        expected_identity = tuple(identity.values())
        if gate_decision is not None and gate_identity != expected_identity:
            raise ValueError("adaptive reuse Gate must match the current window")
        gate = replace(
            source_gate,
            **identity,
            reason=f"{source_gate.reason}:adaptive_reuse_previous_output",
            diagnostics=(*source_gate.diagnostics, f"reused_at_period_ms={period_ms}"),
        )
        spatial = None if previous.spatial_response is None else replace(previous.spatial_response, **doa)
        candidates = tuple(replace(item, **doa) for item in previous.candidates)

        def prediction(item):
            return replace(
                item,
                **doa,
                measured_theta_deg=None,
                is_observed=False,
                is_new_track=False,
                missed_samples=max(0, window.decision_sample - item.last_observed_sample),
            )

        directions = tuple(prediction(item) for item in previous.directions)
        active_tracks = tuple(prediction(item) for item in previous.active_tracks)
        return replace(
            previous,
            **identity,
            spatial_response=spatial,
            candidates=candidates,
            gate_decision=gate,
            directions=directions,
            active_tracks=active_tracks,
            published_monotonic=monotonic(),
            missing_reason=f"adaptive_reuse_{period_ms}ms",
            processing_period_ms=period_ms,
            reused_output=True,
            queue_wait_ms=queue_wait_ms,
        )

    def _update_track_log(self, tracks: tuple[object, ...], decision_sample: int) -> None:
        """Persist a compact, five-second-sampled text history; never audio."""

        for track in tracks:
            entry = self._track_history.setdefault(track.track_id, {
                "first": track.first_seen_sample,
                "last": track.last_observed_sample,
                "state": track.track_state,
                "trajectory": [],
            })
            entry["last"] = max(int(entry["last"]), track.last_observed_sample)
            entry["state"] = track.track_state
            trajectory = entry["trajectory"]
            if not trajectory or decision_sample - trajectory[-1][0] >= 240_000:
                trajectory.append((decision_sample, round(float(track.theta_deg), 1)))
        if decision_sample - self._last_track_log_sample < 48_000:
            return
        self._last_track_log_sample = decision_sample
        self._track_log_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["track_id\tstate\tfirst_sample\tlast_observed_sample\tduration_s\ttrajectory(sample:deg)"]
        for track_id, entry in sorted(self._track_history.items()):
            duration = (int(entry["last"]) - int(entry["first"])) / 48_000.0
            trajectory = ",".join(f"{sample}:{angle:.1f}" for sample, angle in entry["trajectory"])
            lines.append(
                f"{track_id}\t{entry['state']}\t{entry['first']}\t{entry['last']}\t"
                f"{duration:.3f}\t{trajectory}"
            )
        self._track_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
