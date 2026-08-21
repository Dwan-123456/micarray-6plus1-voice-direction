from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import monotonic, monotonic_ns
from typing import Callable

import numpy as np
import torch

from common.config import (
    DownstreamAudioWindowSpec,
    ProjectConfig,
    calibration_config_hash,
    config_hash,
    load_config,
)
from common.data_types import DecisionWindow, PipelineStatus
from common.geometry import physical_6plus1_geometry
from data_management import DecisionRecord, RecordingStore, ResultWatermark, SessionMetadata
from gui.dev_test_ui.aggregator import DevUiAggregator, PerformanceTracker
from gui.dev_test_ui.contracts import AlgorithmPerformanceSnapshot, BeamformPreview, DevUiFrame
from gui.dev_test_ui.meter import L1Meter
from gui.dev_test_ui.scratch_recorder import ScratchRecorder
from ingest import BlockFanout, IngestCoordinator
from layer1_input.calibration import ChannelCalibrator
from layer1_input.configuration import AudioConfig, CalibrationConfig, CdcConfig
from layer1_input.imcra import Layer1Imcra
from layer1_input.pre_denoise import ImcraWienerPreDenoiser
from layer1_input.pipeline import InputPipeline
from layer1_input.protocols import led_command
from layer1_input.serial_device import SerialDevice
from layer1_input.sources import LiveSipeedSource
from layer2_source_detection import (
    DirectionScanConfig, Layer2Pipeline, SourceProbability20ms, SourceProbabilityState,
)
from layer3_direction_signal import (
    L3_MODE_OPTIMIZED,
    L3_PROCESSING_MODES,
    Layer3Output,
    Layer3Processor,
)
from layer4_voice_classifier import (
    InputGainCompensationSettings,
    Layer4AudioSegment,
    Layer4Engine,
    NvidiaMarbleNetPlugin,
)
from track_audio_stream import TrackAudioStreamHub, TrackAudioWindow
from windowing import WindowAssembler

from .compute_cache import CachePartitionLimits, ComputeCache, ComputeCacheError
from .processing_contracts import (
    JoinedWindowResult,
    L2StageResult,
    L3StageResult,
    L4StageResult,
    ProcessingConfigSnapshot,
    StageState,
    WindowKey,
    WindowWorkItem,
)
from .result_joiner import JoinerCapacityError, ResultDeliveryError, ResultJoiner


_PIPELINE_EOS = object()


@dataclass(frozen=True)
class _PipelineTimingBarrier:
    generation: int


@dataclass(frozen=True, slots=True)
class _L3Work:
    work_item: WindowWorkItem
    l2: L2StageResult


@dataclass(frozen=True, slots=True)
class _L4Work:
    work_item: WindowWorkItem
    l2: L2StageResult
    l3: L3StageResult
    inputs: tuple[Layer4AudioSegment, ...]


@dataclass(slots=True)
class _RejectedAdmissionRange:
    """A compact, bounded audit trail for windows rejected before registration."""

    session_id: str
    stream_epoch: int
    first_window_id: int
    last_window_id: int
    first_decision_sample: int
    last_decision_sample: int
    reason: str


@dataclass(frozen=True, slots=True)
class _RejectedAdmission:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    reason: str

class ApplicationRuntime:
    """Single owner of the L1 → ingest → windowing application chain."""

    def __init__(
        self,
        config: ProjectConfig,
        *,
        project_root: str | Path,
        pipeline: InputPipeline | None = None,
        serial_device: SerialDevice | None = None,
        recording_store: RecordingStore | None = None,
        layer4_engine: Layer4Engine | None = None,
        dev_audio_tracker: object | None = None,
        source_probability_provider: Callable[
            [DecisionWindow], tuple[SourceProbability20ms, ...]
        ] | None = None,
    ) -> None:
        self.config = config
        self.project_root = Path(project_root).resolve()
        cdc = CdcConfig.from_project(config)
        self.serial_device = serial_device or SerialDevice(cdc.port, cdc.baudrate)
        if pipeline is None:
            source = LiveSipeedSource(AudioConfig.from_project(config))
            pipeline = InputPipeline(
                source,
                ChannelCalibrator(CalibrationConfig.from_project(config)),
                self.serial_device if cdc.enabled else None,
                owns_hotmap_source=True,
                hotmap_required=cdc.required,
                timestamp_tolerance_ms=config.timing.timestamp_tolerance_ms,
            )
        self.pipeline = pipeline
        self.coordinator = IngestCoordinator(
            sample_rate=config.device.sample_rate,
            timestamp_tolerance_ms=config.timing.timestamp_tolerance_ms,
        )
        self.assembler = WindowAssembler()
        self.imcra = Layer1Imcra.from_project(config)
        self.pre_denoiser = ImcraWienerPreDenoiser.from_project(config)
        self._pre_denoise_lock = threading.Lock()
        self._pre_denoise_enabled = config.layer1_pre_denoise.enabled
        self._pre_denoise_latency_active = config.layer1_pre_denoise.enabled
        self.fanout = BlockFanout()
        self.window_blocks = self.fanout.subscribe(2)
        self.recording_blocks = self.fanout.subscribe(2)
        self.runtime_recording_blocks = self.fanout.subscribe(2)
        self.scratch = ScratchRecorder(config.dev_test_ui.scratch_root, project_root=self.project_root)
        data_root = Path(config.paths.data_root)
        if not data_root.is_absolute():
            data_root = self.project_root / data_root
        self.recording_store = recording_store or RecordingStore(data_root, config=config)
        # Optional Test-UI-only sidecar. Production construction leaves this
        # unset, so no IDs or listening cache exist outside Development Test UI.
        self.dev_audio_tracker = dev_audio_tracker
        self._recording_session_started = False
        self.meter = L1Meter()
        self.latest_l1: queue.Queue[object] = queue.Queue(maxsize=1)
        self.latest_windows: queue.Queue[object] = queue.Queue(maxsize=1)
        self.latest_dev_ui: queue.Queue[object] = queue.Queue(maxsize=config.dev_test_ui.snapshot_mailbox_capacity)
        # Formal commits are deliberately ordered, so a burst of terminal
        # DROPPED/SKIPPED frames may follow one expensive completed L4 window.
        # Keep completed L4 UI frames on an independent latest-only mailbox so
        # that those audit frames cannot erase the newest useful CNN result.
        self.latest_l4_dev_ui: queue.Queue[object] = queue.Queue(maxsize=1)
        self._l4_ui_overwrites = 0
        self._l4_diagnostics_lock = threading.Lock()
        self._l4_actual_completed = 0
        self._l4_dropped = 0
        self._l4_skipped = 0
        self._l4_completion_times: deque[float] = deque(maxlen=512)
        # The public compatibility alias is rebound by
        # ``_reset_processing_graph`` to the formal L2 admission queue.  The
        # staged queues are independent so L2(n), L3(n-1), and L4(n-2) can run
        # concurrently without allowing two workers to mutate the same
        # stateful layer instance.
        self._l2_windows: queue.Queue[object]
        self._l3_windows: queue.Queue[object]
        self._l4_windows: queue.Queue[object]
        self._completion_results: queue.Queue[object]
        self._completion_backlog: deque[JoinedWindowResult]
        self._completion_backlog_lock = threading.Lock()
        self._completion_congested = threading.Event()
        self._commit_reorder_congested = threading.Event()
        self._completion_backlog_capacity = 1
        self._commit_pending_count = 0
        self._commit_pending_capacity = 1
        self._rejected_admission_lock = threading.Lock()
        self._rejected_admission_ranges: deque[_RejectedAdmissionRange] = deque()
        self._rejected_admission_count = 0
        self._admission_audit_overflow_count = 0
        self._last_rejected_admission: tuple[int, str] | None = None
        self._admission_permanently_closed = False
        self._preceding_gap_reasons: dict[tuple[str, int], str] = {}
        self._latest_stream_epoch_by_session: dict[str, int] = {}
        self._closed_streams_pending_cache_prune: set[tuple[str, int]] = set()
        self._timeline_gap_count = 0
        self._last_timeline_gap: tuple[int, int, str] | None = None
        self._processing_windows: queue.Queue[object]
        self._performance = PerformanceTracker(
            sample_rate=config.device.sample_rate,
            required_samples=config.timing.context_samples,
            window_count=config.dev_test_ui.performance_window_count,
            rate_seconds=config.dev_test_ui.sample_rate_window_seconds,
        )
        self._ui_aggregator = DevUiAggregator(
            self._performance, stale_after_ms=config.dev_test_ui.stale_after_ms
        )
        self._ui_lock = threading.Lock()
        self._layer2 = Layer2Pipeline.from_project(config)
        self._source_probability_provider = source_probability_provider or self._imcra_probabilities
        self._gate_config_lock = threading.Lock()
        self._gate_probability_threshold = config.layer2.probability_gate.threshold
        self._gate_config_revision = 0
        self._l3_mode_lock = threading.Lock()
        self._l3_processing_mode = L3_MODE_OPTIMIZED
        self._l3_config_revision = 0
        self._downstream_processing_enabled = threading.Event()
        self._downstream_processing_enabled.set()
        self._scan_config = DirectionScanConfig.from_project(config)
        self._scan_config_lock = threading.Lock()
        self._scan_config_revision = 0
        self._kalman_config_revision = 0
        self._direction_kalman_enabled = config.layer2.direction_kalman.enabled
        self._direction_kalman_q_scale = config.layer2.direction_kalman.process_noise_scale
        self._direction_kalman_r_scale = config.layer2.direction_kalman.measurement_noise_scale
        self._geometry = physical_6plus1_geometry(
            config.hardware.speed_of_sound_mps,
            config.hardware.geometry_version,
            config.hardware.ring_radius_m,
        )
        # The CPU path is the deterministic development integration. The same processor accepts CUDA tensors later.
        self.processing_device = self._resolve_processing_device(config)
        self._l3_cuda_stream = torch.cuda.Stream() if self.processing_device == "cuda" else None
        self._l4_cuda_stream = torch.cuda.Stream() if self.processing_device == "cuda" else None
        self._layer3 = Layer3Processor(config, device=self.processing_device)
        self.downstream_window_spec = config.downstream_audio_window
        self._l3_cache_snapshot: object | None = None
        if layer4_engine is None:
            enabled_l4 = tuple(item for item in config.layer4.models if item.enabled)
            plugins = []
            for item in enabled_l4:
                artifact = Path(item.model_artifact)
                if not artifact.is_absolute():
                    artifact = self.project_root / artifact
                    if not artifact.exists():
                        artifact = Path(__file__).resolve().parents[1] / item.model_artifact
                if item.backend not in {
                    "nvidia_marblenet_continuous_v2", "nvidia_marblenet_window_v1",
                }:
                    raise ValueError(f"unsupported Layer4 backend: {item.backend}")
                plugins.append(NvidiaMarbleNetPlugin(
                    item.model_id,
                    artifact,
                    device=self.processing_device,
                    window_spec=config.downstream_audio_window,
                ))
            primary = next(plugin for plugin in plugins if plugin.model_id == config.layer4.primary_model_id)
            shadows = tuple(plugin for plugin in plugins if plugin.model_id != config.layer4.primary_model_id)
            layer4_engine = Layer4Engine(
                primary,
                shadows,
                threshold=config.layer4.voice_probability_limit,
                input_gain_compensation=InputGainCompensationSettings(
                    **config.layer4.input_gain_compensation.model_dump()
                ),
                window_spec=config.downstream_audio_window,
            )
        self._layer4 = layer4_engine
        self.track_audio_stream = TrackAudioStreamHub(
            InputGainCompensationSettings(
                **config.layer4.input_gain_compensation.model_dump()
            ),
            context_ms=config.layer4.continuous_context_ms,
        )
        self.last_error: str | None = None
        self.processing_error: str | None = None
        self.scratch_error: str | None = None
        self.dev_audio_tracking_error: str | None = None
        self.dev_ui_error: str | None = None
        self.recording_result_error: str | None = None
        self.processing_drops = 0
        self.light_state = "unknown"
        self._stop = threading.Event()
        self._processing_abort = threading.Event()
        self._input_worker_done = threading.Event()
        self._input_exhausted = False
        self._thread: threading.Thread | None = None
        self._processing_thread: threading.Thread | None = None
        self._processing_threads: dict[str, threading.Thread] = {}
        self._stage_errors: dict[str, str | None] = {
            "l2": None, "l3": None, "l4": None, "commit": None,
        }
        self._stage_error_counts: dict[str, int] = {name: 0 for name in self._stage_errors}
        self._stage_completed_counts: dict[str, int] = {
            "l2": 0, "l3": 0, "l4": 0, "commit": 0,
        }
        self._pipeline_duration_lock = threading.RLock()
        self._pipeline_timing_generation = 0
        self._pipeline_timing_sealed = False
        self._pipeline_first_queued_ns: int | None = None
        self._pipeline_stage_finished_ns: dict[str, int | None] = {
            "l2": None, "l3": None, "l4": None,
        }
        self._project_config_hash = config_hash(config=config)
        self._calibration_hash = calibration_config_hash(config.calibration)
        self._compute_cache: ComputeCache
        self._result_joiner: ResultJoiner
        self._reset_processing_graph()
        self._lifecycle_lock = threading.RLock()

    @staticmethod
    def _resolve_processing_device(config: ProjectConfig) -> str:
        preferred = config.runtime.preferred_device.casefold()
        if preferred == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            if config.runtime.allow_cpu_fallback:
                return "cpu"
            raise RuntimeError("配置要求CUDA处理，但当前torch.cuda不可用且禁止CPU fallback")
        if preferred == "cpu":
            return "cpu"
        raise ValueError(f"未知runtime.preferred_device: {config.runtime.preferred_device}")

    def _reset_processing_graph(self) -> None:
        """Create a fresh bounded graph for one capture session.

        Queues and the joiner are deliberately session-scoped.  Reusing either
        after restart would make a new session wait behind stale EOS markers or
        stale timeline keys.
        """

        runtime = self.config.runtime
        self._l2_windows = queue.Queue(maxsize=runtime.l2_queue_windows)
        self._l3_windows = queue.Queue(maxsize=runtime.l3_queue_windows)
        self._l4_windows = queue.Queue(maxsize=runtime.l4_queue_windows)
        self._completion_results = queue.Queue(maxsize=runtime.completion_queue_windows)
        self._completion_backlog = deque()
        self._completion_backlog_capacity = runtime.completion_queue_windows
        self._commit_pending_count = 0
        self._commit_pending_capacity = (
            2 * runtime.max_inflight_windows + 2 * runtime.completion_queue_windows
        )
        self._completion_congested.clear()
        self._commit_reorder_congested.clear()
        with self._rejected_admission_lock:
            self._rejected_admission_ranges.clear()
            self._rejected_admission_count = 0
            self._admission_audit_overflow_count = 0
            self._last_rejected_admission = None
            self._admission_permanently_closed = False
            self._preceding_gap_reasons.clear()
            self._latest_stream_epoch_by_session.clear()
            self._closed_streams_pending_cache_prune.clear()
            self._timeline_gap_count = 0
            self._last_timeline_gap = None
        self._processing_windows = self._l2_windows

        total = runtime.compute_cache_max_bytes
        l2_bytes = total // 4
        l3_bytes = total // 2
        l4_bytes = total - l2_bytes - l3_bytes
        self._compute_cache = ComputeCache(
            {
                "l2": CachePartitionLimits(runtime.max_inflight_windows, l2_bytes, 8),
                "l3": CachePartitionLimits(runtime.max_inflight_windows, l3_bytes, 8),
                "l4": CachePartitionLimits(runtime.max_inflight_windows, l4_bytes, 8),
            },
            max_total_bytes=total,
        )
        self._result_joiner = ResultJoiner(
            expected_hop_samples=self.config.timing.decision_hop_samples,
            max_pending_windows=runtime.max_inflight_windows,
            max_pending_bytes=max(total, runtime.max_inflight_windows * 1_000_000),
            on_joined=self._deliver_joined,
            on_gap=self._observe_timeline_gap,
        )

    def _observe_timeline_gap(self, gap: object) -> None:
        """Compress joiner gap diagnostics; per-window drops are persisted separately."""

        with self._rejected_admission_lock:
            self._timeline_gap_count += 1
            self._last_timeline_gap = (
                int(getattr(gap, "previous_decision_sample")),
                int(getattr(gap, "next_decision_sample")),
                str(getattr(gap, "reason")),
            )

    def _observe_stream_epoch(self, key: WindowKey) -> None:
        """Advance the one-way epoch cursor and release obsolete gap metadata."""

        prune_before: int | None = None
        with self._rejected_admission_lock:
            previous = self._latest_stream_epoch_by_session.get(key.session_id)
            if previous is None or key.stream_epoch > previous:
                self._latest_stream_epoch_by_session[key.session_id] = key.stream_epoch
                for stream_key in tuple(self._preceding_gap_reasons):
                    if stream_key[0] == key.session_id and stream_key[1] < key.stream_epoch:
                        self._preceding_gap_reasons.pop(stream_key, None)
                prune_before = key.stream_epoch
        if prune_before is not None:
            self._prune_completed_stream_history(key.session_id, before_epoch=prune_before)

    def _prune_completed_stream_history(self, session_id: str, *, before_epoch: int | None = None) -> None:
        """Keep only current or genuinely in-flight epoch metadata."""

        if before_epoch is None:
            with self._rejected_admission_lock:
                before_epoch = self._latest_stream_epoch_by_session.get(session_id)
        if before_epoch is None or before_epoch <= 0:
            return
        closed = self._result_joiner.prune_completed_streams(
            session_id, before_epoch=before_epoch
        )
        with self._rejected_admission_lock:
            self._closed_streams_pending_cache_prune.update(closed)
            pending_cache_prune = tuple(self._closed_streams_pending_cache_prune)
        if pending_cache_prune:
            pruned = self._compute_cache.prune_stream_history(pending_cache_prune)
            with self._rejected_admission_lock:
                self._closed_streams_pending_cache_prune.difference_update(pruned)

    def _deliver_joined(self, result: JoinedWindowResult) -> None:
        """Deliver a joined result without ever blocking L1 admission.

        The primary completion queue and this secondary deque both have hard
        limits.  If both are saturated, ResultJoiner keeps the item in its own
        bounded retry queue and admission is temporarily rejected until the
        commit worker catches up.
        """

        try:
            self._completion_results.put_nowait(result)
            return
        except queue.Full:
            pass
        with self._completion_backlog_lock:
            if len(self._completion_backlog) < self._completion_backlog_capacity:
                self._completion_backlog.append(result)
                return
        self._completion_congested.set()
        raise queue.Full("bounded completion delivery is saturated")

    def _completion_backlog_depth(self) -> int:
        with self._completion_backlog_lock:
            return len(self._completion_backlog)

    def _record_admission_rejection(self, work_item: WindowWorkItem, reason: str) -> None:
        """Record a pre-joiner rejection without retaining its 160 ms audio."""

        window_id = work_item.key.window_id
        key = work_item.key
        audit_overflow = False
        with self._rejected_admission_lock:
            ranges = self._rejected_admission_ranges
            if (
                ranges
                and ranges[-1].session_id == key.session_id
                and ranges[-1].stream_epoch == key.stream_epoch
                and ranges[-1].last_window_id + 1 == window_id
                and ranges[-1].last_decision_sample
                + self.config.timing.decision_hop_samples
                == key.decision_sample
            ):
                ranges[-1].last_window_id = window_id
                ranges[-1].last_decision_sample = key.decision_sample
                if ranges[-1].reason != reason:
                    ranges[-1].reason = "multiple_admission_rejection_reasons"
            else:
                # Alternating accept/reject periods are normally consumed by
                # commit immediately.  If a broken commit creates 255 such
                # periods, close admission for the rest of this session so the
                # final compact range remains a hard memory bound.
                if len(ranges) >= 255:
                    self._admission_permanently_closed = True
                    self._admission_audit_overflow_count += 1
                    audit_overflow = True
                else:
                    ranges.append(
                        _RejectedAdmissionRange(
                            key.session_id,
                            key.stream_epoch,
                            window_id,
                            window_id,
                            key.decision_sample,
                            key.decision_sample,
                            reason,
                        )
                    )
            self._rejected_admission_count += 1
            self._last_rejected_admission = (window_id, reason)
            if not audit_overflow:
                self._preceding_gap_reasons[work_item.key.stream_key] = reason
        self._record_processing_drop(key)
        if audit_overflow:
            # Continuing to retain per-window identities after the compact
            # audit limit would make an unhealthy commit path unbounded. Stop
            # algorithm workers explicitly while L1 capture remains alive.
            self._processing_abort.set()
            self.processing_error = (
                "admission audit capacity exhausted; processing aborted "
                f"at window {window_id}"
            )
        else:
            self.processing_error = f"admission: {reason} (window {window_id})"

    def _consume_rejected_window(self, window_id: int) -> _RejectedAdmission | None:
        """Advance commit across a compactly audited pre-joiner rejection."""

        with self._rejected_admission_lock:
            if not self._rejected_admission_ranges:
                return None
            current = self._rejected_admission_ranges[0]
            if current.first_window_id != window_id:
                return None
            rejected = _RejectedAdmission(
                current.session_id,
                current.stream_epoch,
                current.first_window_id,
                current.first_decision_sample,
                current.reason,
            )
            if current.first_window_id == current.last_window_id:
                self._rejected_admission_ranges.popleft()
            else:
                current.first_window_id += 1
                current.first_decision_sample += self.config.timing.decision_hop_samples
            return rejected

    def _commit_rejected_admission(self, rejected: _RejectedAdmission) -> None:
        """Persist one lightweight pre-joiner drop and its formal watermark."""

        decision_sample = rejected.decision_sample
        record = DecisionRecord(
            rejected.session_id,
            rejected.stream_epoch,
            rejected.window_id,
            decision_sample,
            (decision_sample - self.config.timing.doa_window_samples, decision_sample),
            (decision_sample - self.config.timing.context_samples, decision_sample),
            "error",
            diagnostics=(f"admission_drop={rejected.reason}",),
            stage_statuses={"l2": "dropped", "l3": "dropped", "l4": "dropped"},
            terminal_reason=rejected.reason,
            config_hash=self._project_config_hash,
            calibration_version=self.config.hardware.hardware_calibration_status,
            calibration_hash=self._calibration_hash,
        )
        watermark = ResultWatermark(
            rejected.session_id,
            rejected.stream_epoch,
            decision_sample,
            ({
                "window_id": rejected.window_id,
                "decision_sample": decision_sample,
                "reason": rejected.reason,
            },),
        )
        try:
            append_with_watermark = getattr(
                self.recording_store, "append_result_with_watermark", None
            )
            if callable(append_with_watermark):
                append_with_watermark(record, watermark)
            else:
                self.recording_store.append_result(record)
                self.recording_store.advance_result_watermark(watermark)
            self.recording_result_error = None
        except Exception as exc:
            self.recording_result_error = str(exc)
            self.processing_error = f"recording: {exc}"
        self._stage_completed_counts["commit"] += 1
        self._prune_completed_stream_history(rejected.session_id)

    def _joiner_submit(self, submit: Callable[[], None]) -> bool:
        """Publish a terminal marker; delivery congestion is retryable."""

        try:
            submit()
            return True
        except ResultDeliveryError:
            # The joiner already committed the terminal result and retained it
            # for drain_ready().  This is pressure, not a compute failure.
            self._completion_congested.set()
            return True

    def _skip_disabled_downstream(self, key: WindowKey, *, include_l3: bool) -> None:
        """Terminate disabled Test-UI downstream stages as normal skipped work."""

        reason = "downstream_disabled_by_test_ui"
        if include_l3:
            l3 = L3StageResult.terminal(
                key,
                StageState.SKIPPED,
                reason,
                started_monotonic_ns=monotonic_ns(),
                finished_monotonic_ns=monotonic_ns(),
            )
            self._joiner_submit(lambda: self._result_joiner.submit_l3(l3))
            self._stage_completed_counts["l3"] += 1
        l4 = L4StageResult.terminal(
            key,
            StageState.SKIPPED,
            reason,
            started_monotonic_ns=monotonic_ns(),
            finished_monotonic_ns=monotonic_ns(),
        )
        self._joiner_submit(lambda: self._result_joiner.submit_l4(l4))
        self._stage_completed_counts["l4"] += 1
        self._record_l4_terminal(StageState.SKIPPED)

    def _drop_waiting_l2(self, reason: str) -> bool:
        try:
            displaced = self._l2_windows.get_nowait()
        except queue.Empty:
            return False
        if not isinstance(displaced, WindowWorkItem):
            return False
        self._joiner_submit(
            lambda: self._result_joiner.terminate_window(
                displaced.key, StageState.DROPPED, reason
            )
        )
        self._record_processing_drop(displaced.key)
        return True

    def _enqueue_l3_latest(self, item: _L3Work) -> bool:
        """Latest-wins enqueue; only not-yet-started L3 work may be dropped."""

        while not self._processing_abort.is_set():
            try:
                self._l3_windows.put_nowait(item)
                return True
            except queue.Full:
                try:
                    displaced = self._l3_windows.get_nowait()
                except queue.Empty:
                    continue
                if not isinstance(displaced, _L3Work):
                    continue
                now_ns = monotonic_ns()
                l3 = L3StageResult.terminal(
                    displaced.work_item.key,
                    StageState.DROPPED,
                    "l3_admission_queue_overflow",
                    started_monotonic_ns=now_ns,
                    finished_monotonic_ns=now_ns,
                )
                l4 = L4StageResult.terminal(
                    displaced.work_item.key,
                    StageState.DROPPED,
                    "l3_admission_queue_overflow",
                    started_monotonic_ns=now_ns,
                    finished_monotonic_ns=now_ns,
                )
                self._joiner_submit(lambda: self._result_joiner.submit_l3(l3))
                self._joiner_submit(lambda: self._result_joiner.submit_l4(l4))
                self._stage_completed_counts["l3"] += 1
                self._stage_completed_counts["l4"] += 1
                self._record_processing_drop(displaced.work_item.key)
        return False

    def _enqueue_l4_latest(self, item: _L4Work) -> bool:
        """Latest-wins enqueue; completed L3 remains authoritative."""

        while not self._processing_abort.is_set():
            try:
                self._l4_windows.put_nowait(item)
                return True
            except queue.Full:
                try:
                    displaced = self._l4_windows.get_nowait()
                except queue.Empty:
                    continue
                if not isinstance(displaced, _L4Work):
                    continue
                now_ns = monotonic_ns()
                l4 = L4StageResult.terminal(
                    displaced.work_item.key,
                    StageState.DROPPED,
                    "l4_admission_queue_overflow",
                    started_monotonic_ns=now_ns,
                    finished_monotonic_ns=now_ns,
                )
                self._joiner_submit(lambda: self._result_joiner.submit_l4(l4))
                self._stage_completed_counts["l4"] += 1
                self._record_processing_drop(displaced.work_item.key)
        return False

    def _put_eos_interruptibly(
        self, mailbox: queue.Queue[object], downstream_stage: str
    ) -> bool:
        """Propagate EOS while allowing abort or a dead consumer to release us."""

        while not self._processing_abort.is_set():
            try:
                mailbox.put(_PIPELINE_EOS, timeout=0.05)
                return True
            except queue.Full:
                downstream = self._processing_threads.get(downstream_stage)
                if (
                    downstream is not None
                    and downstream.ident is not None
                    and not downstream.is_alive()
                ):
                    self._processing_abort.set()
                    return False
        # During rollback/abort, consumers poll the abort event and therefore
        # do not require an EOS marker to escape an empty queue.
        try:
            mailbox.put_nowait(_PIPELINE_EOS)
            return True
        except queue.Full:
            return False

    def _put_pipeline_timing_barrier_interruptibly(
        self,
        mailbox: queue.Queue[object],
        downstream_stage: str,
        barrier: _PipelineTimingBarrier,
    ) -> bool:
        """Forward an interactive replay drain barrier without dropping work."""

        while not self._processing_abort.is_set():
            try:
                mailbox.put(barrier, timeout=0.05)
                return True
            except queue.Full:
                downstream = self._processing_threads.get(downstream_stage)
                if (
                    downstream is not None
                    and downstream.ident is not None
                    and not downstream.is_alive()
                ):
                    self._processing_abort.set()
                    return False
        return False

    def _wake_processing_workers(self) -> None:
        """Best-effort wakeup used by startup rollback and forced shutdown."""

        for mailbox in (
            self._l2_windows,
            self._l3_windows,
            self._l4_windows,
            self._completion_results,
        ):
            try:
                mailbox.put_nowait(_PIPELINE_EOS)
            except queue.Full:
                pass

    @classmethod
    def from_config_path(cls, path: str | Path) -> "ApplicationRuntime":
        config_path = Path(path).resolve()
        project_root = config_path.parent.parent
        return cls(load_config(config_path), project_root=project_root)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self.last_error is None

    @property
    def active(self) -> bool:
        return (
            self._thread is not None
            or any(thread.is_alive() for thread in self._processing_threads.values())
            or self._recording_session_started
        )

    @property
    def input_exhausted(self) -> bool:
        return self._input_exhausted

    @property
    def processing_running(self) -> bool:
        return any(thread.is_alive() for thread in self._processing_threads.values())

    def _mark_pipeline_stage_finished(self, stage: str, generation: int | None = None) -> None:
        """Freeze one stage's end-to-end duration after its input queue drains."""

        with self._pipeline_duration_lock:
            if (
                (generation is None or generation == self._pipeline_timing_generation)
                and self._pipeline_first_queued_ns is not None
                and stage in self._pipeline_stage_finished_ns
                and self._pipeline_stage_finished_ns[stage] is None
            ):
                self._pipeline_stage_finished_ns[stage] = monotonic_ns()

    def reset_pipeline_total_durations(self) -> None:
        """Start a fresh interactive replay timing generation."""

        with self._pipeline_duration_lock:
            self._pipeline_timing_generation += 1
            self._pipeline_timing_sealed = False
            self._pipeline_first_queued_ns = None
            self._pipeline_stage_finished_ns = {
                "l2": None, "l3": None, "l4": None,
            }

    def seal_pipeline_total_durations(self) -> bool:
        """Queue an ordered barrier so interactive replay timers stop after drain."""

        with self._pipeline_duration_lock:
            if self._pipeline_timing_sealed:
                return True
            generation = self._pipeline_timing_generation
        try:
            self._l2_windows.put_nowait(_PipelineTimingBarrier(generation))
        except queue.Full:
            return False
        with self._pipeline_duration_lock:
            if generation == self._pipeline_timing_generation:
                self._pipeline_timing_sealed = True
        return True

    @property
    def pipeline_total_durations_seconds(self) -> dict[str, float | None]:
        """Elapsed first-queue-to-drain time for L2/L3/L4 in the current run."""

        with self._pipeline_duration_lock:
            started_ns = self._pipeline_first_queued_ns
            finished_ns = dict(self._pipeline_stage_finished_ns)
        if started_ns is None:
            return {"l2": None, "l3": None, "l4": None}
        now_ns = monotonic_ns()
        durations: dict[str, float | None] = {}
        for stage in ("l2", "l3", "l4"):
            end_ns = finished_ns[stage]
            worker = self._processing_threads.get(stage)
            if end_ns is None and (worker is None or not worker.is_alive()):
                durations[stage] = None
            else:
                durations[stage] = max(0.0, ((end_ns or now_ns) - started_ns) / 1e9)
        return durations

    @property
    def processing_queue_depths(self) -> dict[str, int]:
        """A stable diagnostics view; callers never inspect queue internals."""

        return {
            "l2": self._l2_windows.qsize(),
            "l3": self._l3_windows.qsize(),
            "l4": self._l4_windows.qsize(),
            "completion": self._completion_results.qsize(),
        }

    @property
    def compute_cache_bytes(self) -> int:
        return self._compute_cache.current_bytes

    def _epoch_reset_reason(self, session_id: str, stream_epoch: int) -> str | None:
        for item in reversed(self.coordinator.discontinuities):
            if item.session_id == session_id and item.new_epoch == stream_epoch:
                return item.reason
        return None

    def _input_health_snapshot(self) -> dict[str, object]:
        source = getattr(self.pipeline, "source", None)
        capture = getattr(source, "capture", None)
        status_method = getattr(capture, "status", None)
        try:
            capture_status = status_method() if callable(status_method) else {}
        except Exception as exc:
            capture_status = {"last_error": f"capture status unavailable: {exc}"}
        last = self.coordinator.discontinuities[-1] if self.coordinator.discontinuities else None
        return {
            "stream_epoch": self.coordinator.stream_epoch,
            "discontinuity_count": len(self.coordinator.discontinuities),
            "last_discontinuity": None if last is None else asdict(last),
            "input_overflow_count": int(capture_status.get("input_overflow_count", 0)),
            "handoff_drop_count": int(capture_status.get("handoff_drop_count", 0)),
            "handoff_queue_depth": int(capture_status.get("handoff_queue_depth", 0)),
            "handoff_queue_capacity": int(capture_status.get("handoff_queue_capacity", 0)),
            "handoff_queue_high_water": int(capture_status.get("handoff_queue_high_water", 0)),
            "callback_status_count": int(capture_status.get("callback_status_count", 0)),
            "last_error": capture_status.get("last_error"),
        }

    @property
    def processing_status(self) -> dict[str, object]:
        snapshots = self._compute_cache.snapshots()
        joiner = self._result_joiner.snapshot()
        l4_diagnostics = self._l4_diagnostic_snapshot()
        return {
            "queue_depths": self.processing_queue_depths,
            "queue_capacities": {
                "l2": self._l2_windows.maxsize,
                "l3": self._l3_windows.maxsize,
                "l4": self._l4_windows.maxsize,
                "completion": self._completion_results.maxsize,
            },
            "stage_alive": {
                name: thread.is_alive() for name, thread in self._processing_threads.items()
            },
            "cache_bytes": sum(item.current_bytes for item in snapshots.values()),
            "cache_max_bytes": self.config.runtime.compute_cache_max_bytes,
            "l3_device_cache_bytes": int(
                getattr(self._l3_cache_snapshot, "persistent_tensor_bytes", 0)
            ),
            "l3_prepared_cache_entries": int(
                getattr(self._l3_cache_snapshot, "prepared_entries", 0)
            ),
            "l3_prepared_cache_limit": int(
                getattr(self._l3_cache_snapshot, "prepared_entry_limit", 0)
            ),
            "inflight_windows": joiner.pending_windows,
            "completed_counts": dict(self._stage_completed_counts),
            "error_counts": dict(self._stage_error_counts),
            "latest_errors": {
                **self._stage_errors,
                "recording": self.recording_result_error,
                "ui": self.dev_ui_error,
            },
            "processing_drops": self.processing_drops,
            "completion_congested": self._completion_congested.is_set(),
            "commit_reorder_congested": self._commit_reorder_congested.is_set(),
            "completion_backlog_depth": self._completion_backlog_depth(),
            "completion_backlog_capacity": self._completion_backlog_capacity,
            "commit_pending_count": self._commit_pending_count,
            "commit_pending_capacity": self._commit_pending_capacity,
            "admission_rejections": self._rejected_admission_count,
            "admission_audit_overflows": self._admission_audit_overflow_count,
            "last_admission_rejection": self._last_rejected_admission,
            "timeline_gap_count": self._timeline_gap_count,
            "last_timeline_gap": self._last_timeline_gap,
            "input_health": self._input_health_snapshot(),
            "l4_actual_completed": l4_diagnostics["actual_completed"],
            "l4_dropped": l4_diagnostics["dropped"],
            "l4_skipped": l4_diagnostics["skipped"],
            "l4_actual_hz": l4_diagnostics["actual_hz"],
            "l4_ui_mailbox_depth": self.latest_l4_dev_ui.qsize(),
            "l4_ui_mailbox_capacity": self.latest_l4_dev_ui.maxsize,
            "l4_ui_mailbox_overwrites": l4_diagnostics["ui_mailbox_overwrites"],
            "downstream_processing_enabled": self.downstream_processing_enabled,
        }

    @property
    def direction_threshold(self) -> float:
        with self._scan_config_lock:
            return self._scan_config.direction_threshold

    def set_direction_threshold(self, value: float) -> float:
        threshold = float(value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Layer 2 direction threshold必须位于[0,1]")
        with self._scan_config_lock:
            self._scan_config = replace(self._scan_config, direction_threshold=threshold)
            self._scan_config_revision += 1
        return threshold

    @property
    def music_effective_order_limit(self) -> int:
        with self._scan_config_lock:
            return self._scan_config.effective_order_limit

    def set_music_effective_order_limit(self, value: int) -> int:
        if type(value) is not int or value not in {1, 2, 3}:
            raise ValueError("MUSIC effective order limit must be 1, 2, or 3")
        with self._scan_config_lock:
            if value != self._scan_config.effective_order_limit:
                self._scan_config = replace(self._scan_config, effective_order_limit=value)
                self._scan_config_revision += 1
        return value

    @property
    def music_dpd_rank1_enabled(self) -> bool:
        with self._scan_config_lock:
            return self._scan_config.dpd_rank1_enabled

    def set_music_dpd_rank1_enabled(self, value: bool) -> bool:
        if type(value) is not bool:
            raise ValueError("L2 DPD rank-1 setting must be bool")
        with self._scan_config_lock:
            if value != self._scan_config.dpd_rank1_enabled:
                self._scan_config = replace(self._scan_config, dpd_rank1_enabled=value)
                self._scan_config_revision += 1
        return value

    @property
    def music_noise_whitening_enabled(self) -> bool:
        with self._scan_config_lock:
            return self._scan_config.noise_whitening_enabled

    def set_music_noise_whitening_enabled(self, value: bool) -> bool:
        if type(value) is not bool:
            raise ValueError("L2 IMCRA noise whitening setting must be bool")
        with self._scan_config_lock:
            if value != self._scan_config.noise_whitening_enabled:
                self._scan_config = replace(self._scan_config, noise_whitening_enabled=value)
                self._scan_config_revision += 1
        return value

    @property
    def direction_scan_config_revision(self) -> int:
        with self._scan_config_lock:
            return self._scan_config_revision

    @property
    def direction_kalman_enabled(self) -> bool:
        with self._scan_config_lock:
            return self._direction_kalman_enabled

    def set_direction_kalman_enabled(self, value: bool) -> bool:
        if type(value) is not bool:
            raise ValueError("Layer 2 Kalman setting must be bool")
        with self._scan_config_lock:
            if value != self._direction_kalman_enabled:
                self._direction_kalman_enabled = value
                self._scan_config_revision += 1
                self._kalman_config_revision += 1
        return value

    @property
    def direction_kalman_q_scale(self) -> float:
        with self._scan_config_lock:
            return self._direction_kalman_q_scale

    @property
    def direction_kalman_r_scale(self) -> float:
        with self._scan_config_lock:
            return self._direction_kalman_r_scale

    @staticmethod
    def _validate_kalman_noise_scale(value: float, name: str) -> float:
        scale = float(value)
        if (
            not np.isfinite(scale)
            or not 0.02 <= scale <= 10.0
            or (scale != 0.02 and abs(scale * 10.0 - round(scale * 10.0)) > 1.0e-9)
        ):
            raise ValueError(
                f"Layer 2 Kalman {name} scale must be 0.02..10.00 in 0.1 steps "
                "(or the 0.02 minimum)"
            )
        return round(scale, 2)

    def set_direction_kalman_q_scale(self, value: float) -> float:
        scale = self._validate_kalman_noise_scale(value, "Q")
        with self._scan_config_lock:
            if scale != self._direction_kalman_q_scale:
                self._direction_kalman_q_scale = scale
                self._scan_config_revision += 1
                self._kalman_config_revision += 1
        return scale

    def set_direction_kalman_r_scale(self, value: float) -> float:
        scale = self._validate_kalman_noise_scale(value, "R")
        with self._scan_config_lock:
            if scale != self._direction_kalman_r_scale:
                self._direction_kalman_r_scale = scale
                self._scan_config_revision += 1
                self._kalman_config_revision += 1
        return scale


    @property
    def gate_probability_threshold(self) -> float:
        with self._gate_config_lock:
            return self._gate_probability_threshold

    @property
    def gate_config_revision(self) -> int:
        with self._gate_config_lock:
            return self._gate_config_revision

    @property
    def l1_pre_denoise_enabled(self) -> bool:
        with self._pre_denoise_lock:
            return self._pre_denoise_enabled

    def set_l1_pre_denoise_enabled(self, value: bool) -> bool:
        if type(value) is not bool:
            raise ValueError("L1 pre-denoise setting must be bool")
        with self._pre_denoise_lock:
            self._pre_denoise_enabled = value
        return value

    def set_gate_probability_threshold(self, value: float) -> float:
        threshold = float(value)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("Layer 2 Gate probability threshold must be finite and in [0,1]")
        with self._gate_config_lock:
            if threshold != self._gate_probability_threshold:
                self._gate_probability_threshold = threshold
                self._gate_config_revision += 1
        return threshold

    @property
    def l4_input_gain_compensation_enabled(self) -> bool:
        return self.track_audio_stream.gain_compensation_enabled

    def set_l4_input_gain_compensation_enabled(self, value: bool) -> bool:
        return self.track_audio_stream.set_gain_compensation_enabled(value)

    @property
    def l3_processing_mode(self) -> str:
        with self._l3_mode_lock:
            return self._l3_processing_mode

    def set_l3_processing_mode(self, mode: str) -> str:
        if type(mode) is not str or mode not in L3_PROCESSING_MODES:
            raise ValueError(f"unsupported L3 processing mode: {mode!r}")
        with self._l3_mode_lock:
            if self._l3_processing_mode != mode:
                self._l3_processing_mode = mode
                self._l3_config_revision = getattr(self, "_l3_config_revision", 0) + 1
        return mode

    @property
    def downstream_processing_enabled(self) -> bool:
        """Whether L2 results are allowed to enter the L3/L4 stages."""

        return self._downstream_processing_enabled.is_set()

    def set_downstream_processing_enabled(self, value: bool) -> bool:
        """Enable or bypass L3/L4 without stopping L1/L2 or breaking joins."""

        if type(value) is not bool:
            raise ValueError("downstream processing setting must be bool")
        if value:
            self._downstream_processing_enabled.set()
        else:
            self._downstream_processing_enabled.clear()
        return value

    @staticmethod
    def _imcra_probabilities(window: DecisionWindow) -> tuple[SourceProbability20ms, ...]:
        hops = {item.end_sample: item for item in window.imcra_hops}
        output = []
        for end_sample in (window.doa_start_sample + 960, window.doa_end_sample):
            hop = hops.get(end_sample)
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

    @property
    def direction_scan_config(self) -> DirectionScanConfig:
        with self._scan_config_lock:
            return self._scan_config

    @staticmethod
    def _latest(mailbox: queue.Queue[object], value: object) -> None:
        try:
            mailbox.put_nowait(value)
        except queue.Full:
            try:
                mailbox.get_nowait()
            except queue.Empty:
                pass
            mailbox.put_nowait(value)

    def _record_processing_drop(self, key: WindowKey) -> None:
        """Record one dropped 20 ms window for cumulative and 1-second UI metrics."""

        self.processing_drops += 1
        self._performance.add_drop(key.session_id, key.stream_epoch)
        self._record_l4_terminal(StageState.DROPPED)

    def _record_l4_terminal(self, state: StageState) -> None:
        """Count actual L4 inference separately from overload terminal markers."""

        now = monotonic()
        with self._l4_diagnostics_lock:
            if state is StageState.COMPLETED:
                self._l4_actual_completed += 1
                self._l4_completion_times.append(now)
            elif state is StageState.DROPPED:
                self._l4_dropped += 1
            elif state is StageState.SKIPPED:
                self._l4_skipped += 1

    def _l4_diagnostic_snapshot(self) -> dict[str, float | int]:
        now = monotonic()
        with self._l4_diagnostics_lock:
            cutoff = now - 1.0
            while self._l4_completion_times and self._l4_completion_times[0] < cutoff:
                self._l4_completion_times.popleft()
            return {
                "actual_completed": self._l4_actual_completed,
                "dropped": self._l4_dropped,
                "skipped": self._l4_skipped,
                "actual_hz": float(len(self._l4_completion_times)),
                "ui_mailbox_overwrites": self._l4_ui_overwrites,
            }

    def _publish_completed_l4_ui(self, item: _L4Work, output: object) -> None:
        """Publish one exact-window L2/L3/L4 frame before ordered commit.

        The frame is a UI side channel only.  It carries the same immutable
        window identity at every layer and never changes formal result ordering.
        """

        l2_output = item.l2.output
        l3_output = item.l3.output
        if l2_output is None or l3_output is None or l2_output.spatial_response is None:
            return
        response = l2_output.spatial_response
        diagnostics = l2_output.search_diagnostics
        if diagnostics is None:
            return
        values = item.work_item.config.values
        scan = DirectionScanConfig(**dict(values["scan_config"]))
        candidates = tuple(getattr(l2_output, "directions", ()) or l2_output.candidates)
        previews = self._beamform_previews(l3_output, 0, len(candidates))
        window = item.work_item.window
        status = PipelineStatus(
            "running",
            window.session_id,
            window.stream_epoch,
            self.config.timing.context_samples,
            self.config.timing.context_samples,
            "Layer 4 completed",
        )
        performance = AlgorithmPerformanceSnapshot(
            window.session_id,
            window.stream_epoch,
            window.window_id,
            self.config.timing.context_samples,
            self.config.timing.context_samples,
            self.config.device.sample_rate,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        published = monotonic()
        frame = DevUiFrame(
            l1=None, spatial_response=response, candidates=candidates,
            previews=previews, tracked_audio=(), gate_decision=l2_output.gate_decision,
            gate_threshold=float(values["gate_threshold"]),
            gate_config_revision=int(values["gate_config_revision"]),
            direction_threshold=scan.direction_threshold,
            direction_kalman_enabled=bool(values["direction_kalman_enabled"]),
            direction_kalman_q_scale=float(values["direction_kalman_q_scale"]),
            direction_kalman_r_scale=float(values["direction_kalman_r_scale"]),
            scan_config_revision=int(values["scan_config_revision"]),
            pipeline_status=status, performance=performance,
            published_monotonic=published, spatial_published_monotonic=published,
            search_diagnostics=diagnostics, missing_reasons={}, l4_result=output,
            directions=getattr(l2_output, "directions", ()),
            active_tracks=getattr(l2_output, "active_tracks", ()),
        )
        with self._ui_lock:
            current_stream = getattr(self._ui_aggregator, "current_stream", None)
            if current_stream is not None and current_stream != (
                window.session_id,
                window.stream_epoch,
            ):
                return
        try:
            self.latest_l4_dev_ui.put_nowait(frame)
        except queue.Full:
            try:
                self.latest_l4_dev_ui.get_nowait()
            except queue.Empty:
                pass
            with self._l4_diagnostics_lock:
                self._l4_ui_overwrites += 1
            self.latest_l4_dev_ui.put_nowait(frame)
    def _capture_processing_config(self) -> ProcessingConfigSnapshot:
        with self._gate_config_lock:
            gate_threshold = self._gate_probability_threshold
            gate_revision = self._gate_config_revision
        with self._scan_config_lock:
            scan_config = self._scan_config
            scan_revision = self._scan_config_revision
            kalman_enabled = self._direction_kalman_enabled
            kalman_revision = self._kalman_config_revision
            q_scale = self._direction_kalman_q_scale
            r_scale = self._direction_kalman_r_scale
        with self._l3_mode_lock:
            l3_mode = self._l3_processing_mode
            l3_revision = self._l3_config_revision
        with self._pre_denoise_lock:
            audio_mode = "imcra_denoised" if (
                self._pre_denoise_enabled and self._pre_denoise_latency_active
            ) else "raw"
        return ProcessingConfigSnapshot(
            revision=(gate_revision << 42) | (scan_revision << 21) | l3_revision,
            config_hash=self._project_config_hash,
            geometry_version=self.config.hardware.geometry_version,
            audio_mode=audio_mode,
            values={
                "gate_threshold": gate_threshold,
                "gate_config_revision": gate_revision,
                "scan_config": asdict(scan_config),
                "scan_config_revision": scan_revision,
                "music_config_revision": scan_revision,
                "music_history_ms": scan_config.context_ms,
                "music_stft": {
                    "n_fft": scan_config.n_fft,
                    "win_length": scan_config.win_length,
                    "hop_length": scan_config.hop_length,
                    "window": scan_config.window,
                },
                "music_frequency_band_hz": (
                    scan_config.frequency_min_hz, scan_config.frequency_max_hz,
                ),
                "mdl": {
                    "max_age_ms": scan_config.mdl_max_age_ms,
                    "min_valid_frequency_bins": scan_config.min_valid_frequency_bins,
                    "min_cross_frequency_consistency": scan_config.min_cross_frequency_consistency,
                },
                "association_lifecycle": self.config.layer2.direction_id_tracking.model_dump(),
                "association_config_revision": 0,
                "direction_kalman_enabled": kalman_enabled,
                "kalman_config_revision": kalman_revision,
                "direction_kalman_q_scale": q_scale,
                "direction_kalman_r_scale": r_scale,
                "l3_mode": l3_mode,
                "l3_config_revision": l3_revision,
                "l4_threshold": self._layer4.threshold,
            },
        )

    def _cache_publish(self, partition: str, key: WindowKey, name: str, value: object) -> None:
        try:
            self._compute_cache.publish(partition, key, name, value)
        except ComputeCacheError:
            # The cache is an optimization only.  Its hard bounds and GPU
            # rejection must never turn a valid formal stage result into an
            # algorithm failure.
            return

    def _set_stage_error(self, stage: str, exc: BaseException) -> None:
        message = str(exc)
        self._stage_errors[stage] = message
        self._stage_error_counts[stage] += 1
        self.processing_error = f"{stage}: {message}"

    def start(self) -> None:
        with self._lifecycle_lock:
            if self.running:
                return
            if self._thread is not None or any(
                thread.is_alive() for thread in self._processing_threads.values()
            ):
                raise RuntimeError("采集仍在停止中，请稍后重试")
            self.last_error = None
            self.processing_error = None
            self.scratch_error = None
            self.dev_audio_tracking_error = None
            self.dev_ui_error = None
            self.recording_result_error = None
            self.processing_drops = 0
            with self._l4_diagnostics_lock:
                self._l4_actual_completed = 0
                self._l4_dropped = 0
                self._l4_skipped = 0
                self._l4_ui_overwrites = 0
                self._l4_completion_times.clear()
            self._input_exhausted = False
            self._layer2.reset()
            self._stop.clear()
            self._processing_abort.clear()
            self._input_worker_done.clear()
            self.coordinator = IngestCoordinator(
                sample_rate=self.config.device.sample_rate,
                timestamp_tolerance_ms=self.config.timing.timestamp_tolerance_ms,
            )
            self.assembler = WindowAssembler()
            self.imcra = Layer1Imcra.from_project(self.config)
            self.pre_denoiser = ImcraWienerPreDenoiser.from_project(self.config)
            self._layer3.clear_cache()
            self.track_audio_stream.reset()
            self._l3_cache_snapshot = None
            self._reset_processing_graph()
            self._stage_errors = {"l2": None, "l3": None, "l4": None, "commit": None}
            self._stage_error_counts = {name: 0 for name in self._stage_errors}
            self._stage_completed_counts = {name: 0 for name in self._stage_errors}
            with self._pipeline_duration_lock:
                self._pipeline_timing_generation += 1
                self._pipeline_timing_sealed = False
                self._pipeline_first_queued_ns = None
                self._pipeline_stage_finished_ns = {
                    "l2": None, "l3": None, "l4": None,
                }
            with self._pre_denoise_lock:
                self._pre_denoise_latency_active = self._pre_denoise_enabled
            if self.dev_audio_tracker is not None:
                self.dev_audio_tracker.reset()
            self._performance = PerformanceTracker(
                sample_rate=self.config.device.sample_rate,
                required_samples=self.config.timing.context_samples,
                window_count=self.config.dev_test_ui.performance_window_count,
                rate_seconds=self.config.dev_test_ui.sample_rate_window_seconds,
            )
            self._ui_aggregator = DevUiAggregator(
                self._performance, stale_after_ms=self.config.dev_test_ui.stale_after_ms
            )
            for mailbox in (
                self.latest_l1,
                self.latest_windows,
                self.latest_dev_ui,
                self.latest_l4_dev_ui,
            ):
                while True:
                    try:
                        mailbox.get_nowait()
                    except queue.Empty:
                        break
        metadata = SessionMetadata(
            config_hash(config=self.config),
            self._calibration_hash,
            config_revision=0,
            calibration_revision=0,
            calibration_version=self.config.hardware.hardware_calibration_status,
            geometry_version=self.config.hardware.geometry_version,
            runtime={
                "processing_device": self.processing_device,
                "processing_architecture": "staged_window_pipeline_v1",
                "window_key": "session_id,stream_epoch,window_id,decision_sample",
                "queue_capacities": {
                    "l2": self.config.runtime.l2_queue_windows,
                    "l3": self.config.runtime.l3_queue_windows,
                    "l4": self.config.runtime.l4_queue_windows,
                    "completion": self.config.runtime.completion_queue_windows,
                },
                "max_inflight_windows": self.config.runtime.max_inflight_windows,
                "compute_cache_max_bytes": self.config.runtime.compute_cache_max_bytes,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            algorithm_versions={
                "layer1_imcra": self.config.layer1_imcra.algorithm_version,
                "layer1_pre_denoise": self.config.layer1_pre_denoise.algorithm_version,
                "layer2": self.config.layer2.scanner_backend,
                "layer2_direction_kalman": self.config.layer2.direction_kalman.backend,
                "layer2_direction_id_tracking": self.config.layer2.direction_id_tracking.backend,
                "layer3": self.config.layer3.main_backend,
                "feature": self.config.feature.preprocessing_version,
                "layer4_primary": self.config.layer4.primary_model_id,
                "layer4_models": [item.model_id for item in self.config.layer4.models if item.enabled],
            },
        )
        pipeline_start_attempted = False
        started_threads: list[threading.Thread] = []
        try:
            self.recording_store.start_session(self.coordinator.session_id, metadata)
            self._recording_session_started = True
            self.recording_store.set_recording_mode(self.config.recording.runtime.mode)
            self._thread = threading.Thread(
                target=self._run, name="application-runtime-l1", daemon=True
            )
            self._processing_threads = {
                "l2": threading.Thread(
                    target=self._run_l2, name="application-runtime-l2", daemon=True
                ),
                "l3": threading.Thread(
                    target=self._run_l3, name="application-runtime-l3", daemon=True
                ),
                "l4": threading.Thread(
                    target=self._run_l4, name="application-runtime-l4", daemon=True
                ),
                "commit": threading.Thread(
                    target=self._run_commit, name="application-runtime-commit", daemon=True
                ),
            }
            # Compatibility for callers that previously observed one terminal
            # processing thread.  The commit worker remains that owner.
            self._processing_thread = self._processing_threads["commit"]
            for name in ("commit", "l4", "l3", "l2"):
                worker = self._processing_threads[name]
                worker.start()
                started_threads.append(worker)
            # Compute consumers are ready before the device starts producing;
            # no first audio block can race an unstarted stage queue.
            pipeline_start_attempted = True
            self.pipeline.start()
            self._thread.start()
            started_threads.append(self._thread)
        except Exception:
            # Roll back in reverse acquisition order.  Merely setting abort is
            # insufficient because workers may be sleeping on an empty queue.
            self._processing_abort.set()
            self._stop.set()
            self._input_worker_done.set()
            if pipeline_start_attempted:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
            self._wake_processing_workers()
            for worker in reversed(started_threads):
                if worker is not threading.current_thread():
                    worker.join(timeout=2.0)
            alive = {
                name: worker
                for name, worker in self._processing_threads.items()
                if worker.is_alive()
            }
            self._thread = (
                self._thread if self._thread is not None and self._thread.is_alive() else None
            )
            self._processing_threads = alive
            self._processing_thread = alive.get("commit")
            if self._recording_session_started:
                try:
                    self.recording_store.stop_session("runtime_start_failed")
                except Exception as cleanup_exc:
                    self.last_error = f"runtime start rollback recording failure: {cleanup_exc}"
                else:
                    self._recording_session_started = False
            raise

    def _select_pre_denoise(self, block) -> tuple[object, ...]:
        pairs = self.pre_denoiser.process(block)
        with self._pre_denoise_lock:
            enabled = self._pre_denoise_enabled
            if not self._pre_denoise_latency_active:
                if not enabled:
                    return (block,)
                # Everything before this point has already been published in
                # bypass mode. Start the fixed one-hop latency here and wait
                # for this exact hop's denoised replacement.
                self._pre_denoise_latency_active = True
                return ()
        return tuple(item.denoised if enabled else item.raw for item in pairs)

    def _flush_pre_denoise(self) -> tuple[object, ...]:
        pairs = self.pre_denoiser.flush()
        with self._pre_denoise_lock:
            if not self._pre_denoise_latency_active:
                return ()
            enabled = self._pre_denoise_enabled
        return tuple(item.denoised if enabled else item.raw for item in pairs)

    def _publish_l1_block(self, block, received: float) -> None:
        imcra_hops = () if block.imcra_hop is None else (block.imcra_hop,)
        self.fanout.publish(block)
        # These are the exact same object references published by fanout.
        window_block = self.window_blocks.get_nowait()
        recording_block = self.recording_blocks.get_nowait()
        runtime_recording_block = self.runtime_recording_blocks.get_nowait()
        windows = self.assembler.add(window_block, imcra_hops)
        try:
            self.scratch.append(recording_block)
        except Exception as exc:
            self.scratch_error = str(exc)
        self.recording_store.append_audio(runtime_recording_block)
        with self._pre_denoise_lock:
            enabled = self._pre_denoise_enabled and self._pre_denoise_latency_active
        snapshot = self.meter.add(
            block,
            light_state=self.light_state,
            recording_state=self.scratch.state,
            pre_denoise_enabled=enabled,
            pre_denoise_mean_gain_db=self.pre_denoiser.last_mean_gain_db if enabled else 0.0,
        )
        self._latest(self.latest_l1, snapshot)
        with self._ui_lock:
            previous_stream = getattr(self._ui_aggregator, "current_stream", None)
            next_stream = (block.session_id, block.stream_epoch)
            stream_changed = previous_stream is not None and previous_stream != next_stream
            if stream_changed:
                while True:
                    try:
                        self.latest_l4_dev_ui.get_nowait()
                    except queue.Empty:
                        break
            self._performance.add_block(block, received)
            pipeline_status = self.assembler.status
            reset_reason = self._epoch_reset_reason(block.session_id, block.stream_epoch)
            if reset_reason is not None:
                pipeline_status = replace(
                    pipeline_status,
                    message=f"{pipeline_status.message}; epoch_reset:{reset_reason}",
                )
            frame = self._ui_aggregator.update_l1(snapshot, pipeline_status)
            if stream_changed and self.dev_audio_tracker is not None:
                # A continuity break starts a new formal processing epoch, but
                # the L3 listening cache belongs to the whole capture session.
                # Re-project its archived rows immediately while L2 warms up;
                # otherwise the UI looks as if Gate recovery deleted the
                # recordings until the first new L3 result arrives.
                try:
                    retained_audio = self.dev_audio_tracker.snapshots()
                except Exception as exc:
                    self.dev_audio_tracking_error = str(exc)
                else:
                    frame = self._ui_aggregator.update_l3(
                        (),
                        "WARMING_UP: waiting for Layer 2 in the new stream epoch",
                        tracked_audio=retained_audio,
                    )
            self._latest(self.latest_dev_ui, frame)
        for window in windows:
            self._latest(self.latest_windows, window)
            self._admit_window(window)

    def _admit_window(self, window: DecisionWindow) -> bool:
        work_item = WindowWorkItem(
            WindowKey.from_window(window),
            window,
            self._capture_processing_config(),
            monotonic_ns(),
        )
        self._observe_stream_epoch(work_item.key)
        if self._processing_abort.is_set():
            self._record_admission_rejection(work_item, "processing_aborted")
            return False
        with self._rejected_admission_lock:
            permanently_closed = self._admission_permanently_closed
        if permanently_closed:
            self._record_admission_rejection(work_item, "admission_audit_capacity_closed")
            return False
        if self._completion_congested.is_set():
            self._record_admission_rejection(work_item, "completion_delivery_saturated")
            return False
        if self._commit_reorder_congested.is_set():
            self._record_admission_rejection(work_item, "commit_reorder_capacity_reached")
            return False

        # L2 is latest-wins.  Terminating an older not-yet-started item before
        # registering the replacement keeps the joiner below its hard limit.
        if self._l2_windows.full():
            self._drop_waiting_l2("l2_admission_queue_overflow")

        with self._rejected_admission_lock:
            gap_reason = self._preceding_gap_reasons.get(work_item.key.stream_key)
        if gap_reason is not None:
            snapshot = self._result_joiner.snapshot()
            has_history = any(
                key.stream_key == work_item.key.stream_key
                for key in self._result_joiner.pending_keys()
            ) or any(
                (session_id, epoch) == work_item.key.stream_key
                for session_id, epoch, _ in snapshot.committed_through
            )
            if not has_history:
                gap_reason = None

        registered = False
        # A capacity failure must never escape into the L1 capture loop.  It
        # should normally be resolved by dropping one waiting L2 item; the
        # final branch explicitly rejects the new window without retaining its
        # 160 ms samples.
        for _ in range(self._l2_windows.maxsize + 2):
            try:
                self._result_joiner.register(
                    work_item, preceding_gap_reason=gap_reason
                )
                registered = True
                break
            except JoinerCapacityError:
                if not self._drop_waiting_l2("joiner_capacity_l2_latest_wins"):
                    self._record_admission_rejection(
                        work_item, "joiner_inflight_capacity_reached"
                    )
                    return False
        if not registered:
            self._record_admission_rejection(work_item, "joiner_registration_retry_exhausted")
            return False
        with self._rejected_admission_lock:
            self._preceding_gap_reasons.pop(work_item.key.stream_key, None)

        started_timing = False
        with self._pipeline_duration_lock:
            if self._pipeline_first_queued_ns is None:
                self._pipeline_first_queued_ns = monotonic_ns()
                started_timing = True
        try:
            self._l2_windows.put_nowait(work_item)
            return True
        except queue.Full:
            if started_timing:
                with self._pipeline_duration_lock:
                    self._pipeline_first_queued_ns = None
            # A producer race should be impossible with the single L1 owner,
            # but keep it terminal and observable rather than raising.
            self._joiner_submit(
                lambda: self._result_joiner.terminate_window(
                    work_item.key,
                    StageState.DROPPED,
                    "l2_admission_queue_race",
                )
            )
            self._record_processing_drop(work_item.key)
            return False

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                audio = self.pipeline.read(timeout=0.1)
                if audio is None:
                    source = getattr(self.pipeline, "source", None)
                    if bool(getattr(source, "exhausted", False)):
                        self._input_exhausted = True
                        self._stop.set()
                        break
                    continue
                health_events = tuple(self.pipeline.take_health_events())
                block = self.coordinator.ingest(audio, health_events)
                imcra_hops = self.imcra.process(block) if self.config.layer1_imcra.enabled else ()
                if (
                    len(imcra_hops) == 1
                    and imcra_hops[0].start_sample == block.start_sample
                    and imcra_hops[0].end_sample == block.end_sample
                    and imcra_hops[0].source_sequence_ids == (block.sequence_id,)
                ):
                    block = replace(block, imcra_hop=imcra_hops[0])
                if self.dev_audio_tracker is not None:
                    try:
                        # Raw logical channel 6 is the Center microphone.  It
                        # is cached before optional L1 pre-denoise so the first
                        # L3 listening row is a true input reference.
                        self.dev_audio_tracker.append_center_reference(block, channel_index=6)
                    except Exception as exc:
                        self.dev_audio_tracking_error = f"center reference: {exc}"
                received = monotonic()
                for selected in self._select_pre_denoise(block):
                    self._publish_l1_block(selected, received)
        except Exception as exc:
            self.last_error = str(exc)
        finally:
            try:
                for selected in self._flush_pre_denoise():
                    self._publish_l1_block(selected, monotonic())
                self.scratch.close()
            finally:
                try:
                    self.pipeline.stop()
                finally:
                    self._input_worker_done.set()

    def _run_l2(self) -> None:
        drained = False
        try:
            while not self._processing_abort.is_set():
                if self._input_worker_done.is_set() and self._l2_windows.empty():
                    drained = True
                    break
                try:
                    item = self._l2_windows.get(timeout=0.1)
                except queue.Empty:
                    continue
                if isinstance(item, _PipelineTimingBarrier):
                    self._mark_pipeline_stage_finished("l2", item.generation)
                    self._put_pipeline_timing_barrier_interruptibly(
                        self._l3_windows, "l3", item
                    )
                    continue
                if not isinstance(item, WindowWorkItem):
                    continue
                # Test-UI scan controls are live operator settings. Resolve
                # them when L2 actually starts, not when a window entered a
                # potentially backlogged queue, then propagate the exact
                # applied snapshot to L3/L4, UI, and recording.
                with self._scan_config_lock:
                    live_scan_config = self._scan_config
                    live_scan_revision = self._scan_config_revision
                values = dict(item.config.values)
                values["scan_config"] = asdict(live_scan_config)
                values["scan_config_revision"] = live_scan_revision
                values["music_config_revision"] = live_scan_revision
                gate_revision = int(values["gate_config_revision"])
                l3_revision = int(values["l3_config_revision"])
                item = replace(item, config=replace(
                    item.config,
                    revision=(gate_revision << 42) | (live_scan_revision << 21) | l3_revision,
                    values=values,
                ))
                values = item.config.values
                started_ns = monotonic_ns()
                try:
                    probabilities = tuple(self._source_probability_provider(item.window))
                    self._cache_publish("l2", item.key, "source_probabilities", probabilities)
                    scan_config = DirectionScanConfig(**dict(values["scan_config"]))
                    output = self._layer2.process(
                        item.window,
                        probabilities,
                        self._geometry,
                        scan_config,
                        gate_threshold=float(values["gate_threshold"]),
                        gate_config_revision=int(values["gate_config_revision"]),
                        scan_config_revision=int(values["scan_config_revision"]),
                        direction_kalman_enabled=bool(values["direction_kalman_enabled"]),
                        direction_kalman_q_scale=float(values["direction_kalman_q_scale"]),
                        direction_kalman_r_scale=float(values["direction_kalman_r_scale"]),
                    )
                    if len(output.candidates) > 3:
                        raise RuntimeError("Layer 2 contract violation: more than 3 candidates")
                    diagnostics = self._l2_diagnostics(output, values)
                    stage = L2StageResult.completed(
                        item.key, output, started_monotonic_ns=started_ns,
                        finished_monotonic_ns=monotonic_ns(), diagnostics=diagnostics,
                    )
                    self._stage_errors["l2"] = None
                except Exception as exc:
                    self._set_stage_error("l2", exc)
                    stage = L2StageResult.terminal(
                        item.key, StageState.FAILED, "l2_failed",
                        started_monotonic_ns=started_ns,
                        finished_monotonic_ns=monotonic_ns(), error=str(exc),
                    )
                self._stage_completed_counts["l2"] += 1
                self._cache_publish("l2", item.key, "stage_result", stage)
                self._joiner_submit(lambda: self._result_joiner.submit_l2(stage))
                if stage.state is not StageState.COMPLETED:
                    self._record_l4_terminal(StageState.SKIPPED)
                    self._joiner_submit(
                        lambda: self._result_joiner.skip_missing_downstream(
                            item.key, "l2_failed"
                        )
                    )
                elif (
                    stage.output.spatial_response is None
                    and not tuple(getattr(stage.output, "directions", ()))
                ):
                    self._record_l4_terminal(StageState.SKIPPED)
                    self._joiner_submit(
                        lambda: self._result_joiner.skip_missing_downstream(
                            item.key, f"gate_{stage.output.gate_decision.state.value}"
                        )
                    )
                elif not self.downstream_processing_enabled:
                    self._skip_disabled_downstream(item.key, include_l3=True)
                else:
                    if not self._enqueue_l3_latest(_L3Work(item, stage)):
                        break
        except Exception as exc:
            self._set_stage_error("l2", exc)
            self._processing_abort.set()
        finally:
            if drained:
                self._mark_pipeline_stage_finished("l2")
            self._put_eos_interruptibly(self._l3_windows, "l3")

    def _run_l3(self) -> None:
        drained = False
        try:
            while True:
                try:
                    item = self._l3_windows.get(timeout=0.1)
                except queue.Empty:
                    if self._processing_abort.is_set():
                        break
                    continue
                if item is _PIPELINE_EOS:
                    drained = True
                    break
                if isinstance(item, _PipelineTimingBarrier):
                    self._mark_pipeline_stage_finished("l3", item.generation)
                    self._put_pipeline_timing_barrier_interruptibly(
                        self._l4_windows, "l4", item
                    )
                    continue
                if self._processing_abort.is_set():
                    break
                if not isinstance(item, _L3Work):
                    continue
                if not self.downstream_processing_enabled:
                    self._skip_disabled_downstream(
                        item.work_item.key, include_l3=True
                    )
                    continue
                started_ns = monotonic_ns()
                try:
                    assert item.l2.output is not None
                    candidates = tuple(
                        getattr(item.l2.output, "directions", ()) or item.l2.output.candidates
                    )
                    mode = str(item.work_item.config.values["l3_mode"])
                    if not candidates:
                        # A valid SRP response can have no accepted peaks.  Do
                        # not pay the 160 ms STFT/covariance preparation cost
                        # when there is no direction to synthesize.
                        output = Layer3Output(())
                    elif self._l3_cuda_stream is None:
                        prepared = self._layer3.prepare(item.work_item.window, mode=mode)
                        output = self._layer3.process_prepared(prepared, candidates, self._geometry)
                    else:
                        with torch.cuda.stream(self._l3_cuda_stream):
                            prepared = self._layer3.prepare(item.work_item.window, mode=mode)
                            output = self._layer3.process_prepared(
                                prepared, candidates, self._geometry
                            )
                        self._l3_cuda_stream.synchronize()
                    self._validate_direction_outputs(
                        "L3", candidates, output.enhanced_audio
                    )
                    probability_slots = self._context_probabilities_20ms(
                        item.work_item.window
                    )
                    audio_windows = tuple(
                        TrackAudioWindow(
                            enhanced.session_id,
                            enhanced.stream_epoch,
                            enhanced.window_id,
                            enhanced.decision_sample,
                            int(enhanced.track_id),
                            enhanced.theta_deg,
                            enhanced.enhanced_audio,
                            probability_slots,
                            enhanced.algorithm,
                        )
                        for enhanced in output.enhanced_audio
                    )
                    active_tracks = tuple(
                        getattr(item.l2.output, "active_tracks", ())
                    )
                    active_ids = tuple(
                        track.track_id for track in active_tracks
                        if track.track_state != "tentative"
                    )
                    audio_batch = self.track_audio_stream.process(
                        audio_windows,
                        active_track_ids=active_ids,
                        identity=(
                            item.work_item.key.session_id,
                            item.work_item.key.stream_epoch,
                            item.work_item.key.window_id,
                            item.work_item.key.decision_sample,
                        ),
                    )
                    l4_inputs = self._layer4_inputs_from_stream(audio_batch)
                    self._validate_direction_outputs("track audio", candidates, l4_inputs)
                    if self.dev_audio_tracker is not None:
                        try:
                            self.dev_audio_tracker.consume_stream_batch(
                                audio_batch, active_tracks=active_tracks
                            )
                            self.dev_audio_tracking_error = None
                        except Exception as exc:
                            self.dev_audio_tracking_error = str(exc)
                    stage = L3StageResult.completed(
                        item.work_item.key, output, started_monotonic_ns=started_ns,
                        finished_monotonic_ns=monotonic_ns(),
                    )
                    self._stage_errors["l3"] = None
                except Exception as exc:
                    self._set_stage_error("l3", exc)
                    stage = L3StageResult.terminal(
                        item.work_item.key, StageState.FAILED, "l3_failed",
                        started_monotonic_ns=started_ns,
                        finished_monotonic_ns=monotonic_ns(), error=str(exc),
                    )
                self._stage_completed_counts["l3"] += 1
                try:
                    self._l3_cache_snapshot = self._layer3.cache_snapshot()
                except Exception:
                    self._l3_cache_snapshot = None
                self._cache_publish("l3", item.work_item.key, "stage_result", stage)
                if stage.state is StageState.COMPLETED:
                    latest_hop_by_id = {
                        hop.track_id: hop for hop in audio_batch.emitted_hops
                        if hop.waveform is not None
                    }
                    context_by_id = {
                        context.track_id: context for context in audio_batch.continuous_audio
                    }
                    track_audio_record = tuple(
                        (
                            {
                                "track_id": source.track_id,
                                "theta_deg": source.theta_deg,
                                "backend": source.algorithm,
                                "fallback_reason": source.fallback_reason,
                                "diagnostics": list(source.diagnostics),
                                "sample_rate": 48_000,
                                "start_sample": latest_hop_by_id[source.track_id].start_sample,
                                "end_sample": latest_hop_by_id[source.track_id].end_sample,
                                "stream_kind": "id_continuous_gain_compensated",
                                "gain_compensation_enabled": (
                                    context_by_id[source.track_id].gain_diagnostic.enabled
                                ),
                            },
                            latest_hop_by_id[source.track_id].waveform,
                        )
                        for source in output.enhanced_audio
                        if source.track_id in latest_hop_by_id
                    )
                    self._cache_publish(
                        "l3", item.work_item.key, "track_audio_record", track_audio_record
                    )
                self._joiner_submit(lambda: self._result_joiner.submit_l3(stage))
                if stage.state is StageState.COMPLETED:
                    if not self._enqueue_l4_latest(
                        _L4Work(item.work_item, item.l2, stage, l4_inputs)
                    ):
                        break
                else:
                    skipped = L4StageResult.terminal(
                        item.work_item.key, StageState.SKIPPED, "l3_failed",
                        started_monotonic_ns=monotonic_ns(),
                        finished_monotonic_ns=monotonic_ns(),
                    )
                    self._record_l4_terminal(StageState.SKIPPED)
                    self._joiner_submit(lambda: self._result_joiner.submit_l4(skipped))
        except Exception as exc:
            self._set_stage_error("l3", exc)
            self._processing_abort.set()
        finally:
            if drained:
                self._mark_pipeline_stage_finished("l3")
            self._put_eos_interruptibly(self._l4_windows, "l4")

    def _run_l4(self) -> None:
        drained = False
        try:
            while True:
                try:
                    item = self._l4_windows.get(timeout=0.1)
                except queue.Empty:
                    if self._processing_abort.is_set():
                        break
                    continue
                if item is _PIPELINE_EOS:
                    drained = True
                    break
                if isinstance(item, _PipelineTimingBarrier):
                    self._mark_pipeline_stage_finished("l4", item.generation)
                    continue
                if self._processing_abort.is_set():
                    break
                if not isinstance(item, _L4Work):
                    continue
                if not self.downstream_processing_enabled:
                    self._skip_disabled_downstream(
                        item.work_item.key, include_l3=False
                    )
                    continue
                started_ns = monotonic_ns()
                try:
                    assert item.l2.output is not None and item.l3.output is not None
                    formal_directions = tuple(
                        getattr(item.l2.output, "directions", ()) or item.l2.output.candidates
                    )
                    inputs = item.inputs
                    if self._l4_cuda_stream is None:
                        output = self._layer4.process(inputs)
                    else:
                        with torch.cuda.stream(self._l4_cuda_stream):
                            output = self._layer4.process(inputs)
                        self._l4_cuda_stream.synchronize()
                    frozen_threshold = float(item.work_item.config.values["l4_threshold"])
                    if output.threshold != frozen_threshold:
                        output = self._layer4.rethreshold(output, frozen_threshold)
                    self._validate_direction_outputs(
                        "L4", formal_directions, output.detections
                    )
                    stage = L4StageResult.completed(
                        item.work_item.key, output, started_monotonic_ns=started_ns,
                        finished_monotonic_ns=monotonic_ns(),
                    )
                    submit_voice_feedback = getattr(
                        self._layer2, "submit_voice_feedback", None
                    )
                    if callable(submit_voice_feedback):
                        for detection in output.detections:
                            if detection.track_id is not None:
                                submit_voice_feedback(
                                    detection.session_id,
                                    detection.stream_epoch,
                                    detection.decision_sample,
                                    detection.track_id,
                                    detection.probability,
                                    detection.is_voice,
                                )
                    self._stage_errors["l4"] = None
                except Exception as exc:
                    self._set_stage_error("l4", exc)
                    stage = L4StageResult.terminal(
                        item.work_item.key, StageState.FAILED, "l4_failed",
                        started_monotonic_ns=started_ns,
                        finished_monotonic_ns=monotonic_ns(), error=str(exc),
                    )
                self._stage_completed_counts["l4"] += 1
                self._record_l4_terminal(stage.state)
                if stage.state is StageState.COMPLETED and stage.output is not None:
                    try:
                        self._publish_completed_l4_ui(item, stage.output)
                    except Exception as exc:
                        # This mailbox is diagnostic only; a rendering-contract
                        # failure must not change the formal L4 result.
                        self.dev_ui_error = f"L4 immediate UI: {exc}"
                self._cache_publish("l4", item.work_item.key, "stage_result", stage)
                self._joiner_submit(lambda: self._result_joiner.submit_l4(stage))
        except Exception as exc:
            self._set_stage_error("l4", exc)
            self._processing_abort.set()
        finally:
            if drained:
                self._mark_pipeline_stage_finished("l4")
            self._put_eos_interruptibly(self._completion_results, "commit")

    def _run_commit(self) -> None:
        pending: dict[int, JoinedWindowResult] = {}
        next_window_id = 0
        saw_eos = False
        soft_pending_limit = self.config.runtime.max_inflight_windows
        hard_pending_limit = self._commit_pending_capacity

        def update_pending_pressure() -> None:
            self._commit_pending_count = len(pending)
            if len(pending) >= soft_pending_limit:
                self._commit_reorder_congested.set()
            elif len(pending) <= max(1, soft_pending_limit // 2):
                self._commit_reorder_congested.clear()

        def accept(item: object) -> None:
            nonlocal saw_eos
            if item is _PIPELINE_EOS:
                saw_eos = True
                return
            if not isinstance(item, JoinedWindowResult):
                return
            if item.key.window_id in pending:
                raise RuntimeError(f"duplicate joined window id: {item.key.window_id}")
            if len(pending) >= hard_pending_limit:
                raise RuntimeError("commit pending hard capacity exceeded")
            pending[item.key.window_id] = item
            update_pending_pressure()

        def drain_fallbacks() -> bool:
            found = False
            available = hard_pending_limit - len(pending)
            if available <= 0:
                update_pending_pressure()
                return False
            with self._completion_backlog_lock:
                backlog = tuple(
                    self._completion_backlog.popleft()
                    for _ in range(min(available, len(self._completion_backlog)))
                )
            for joined in backlog:
                accept(joined)
                found = True
            available = hard_pending_limit - len(pending)
            # Callback failure means ResultJoiner retained the already-joined
            # result here.  Poll it on every commit iteration.
            for joined in self._result_joiner.drain_ready(available):
                accept(joined)
                found = True
            if self._completion_backlog_depth() < self._completion_backlog_capacity:
                self._completion_congested.clear()
            return found

        def commit_contiguous() -> bool:
            nonlocal next_window_id
            progressed = False
            while True:
                rejected = self._consume_rejected_window(next_window_id)
                if rejected is not None:
                    self._commit_rejected_admission(rejected)
                    next_window_id += 1
                    progressed = True
                    continue
                joined = pending.pop(next_window_id, None)
                if joined is None:
                    update_pending_pressure()
                    return progressed
                self._commit_joined(joined)
                next_window_id += 1
                progressed = True

        try:
            while True:
                found = drain_fallbacks()
                if len(pending) < hard_pending_limit:
                    try:
                        accept(self._completion_results.get(timeout=0.05))
                        found = True
                    except queue.Empty:
                        pass
                else:
                    self._commit_reorder_congested.set()
                    threading.Event().wait(0.01)
                commit_contiguous()
                if saw_eos:
                    # EOS is ordered after L4 production but may overtake the
                    # bounded fallback deque.  Exit only after every source and
                    # every registered joiner item is empty.
                    found = drain_fallbacks() or found
                    commit_contiguous()
                    snapshot = self._result_joiner.snapshot()
                    if (
                        snapshot.pending_windows == 0
                        and self._completion_results.empty()
                        and self._completion_backlog_depth() == 0
                    ):
                        break
                elif self._processing_abort.is_set() and not found:
                    l4 = self._processing_threads.get("l4")
                    if l4 is None or (l4.ident is not None and not l4.is_alive()):
                        # Forced shutdown normally injects EOS.  This fallback
                        # prevents a failed injection from leaving commit in an
                        # infinite get() after all producers have stopped.
                        saw_eos = True
            commit_contiguous()
            if pending:
                raise RuntimeError(
                    f"commit timeline gap before window {min(pending)}; expected {next_window_id}"
                )
            self._result_joiner.close()
        except Exception as exc:
            self._set_stage_error("commit", exc)
            self._processing_abort.set()
        finally:
            self._commit_pending_count = len(pending)

    def _l2_diagnostics(self, layer2_result, values) -> tuple[str, ...]:
        gate = layer2_result.gate_decision
        music = getattr(layer2_result, "search_diagnostics", None)
        is_music = music is not None and hasattr(music, "model_order")
        music_state = None if not is_music else getattr(music, "state", None)
        return (
            f"l2_pipeline_state={layer2_result.state.value}",
            f"l2_gate_backend={gate.backend}",
            f"l2_gate_state={gate.state.value}",
            f"l2_gate_reason={gate.reason}",
            f"l2_gate_probability_40ms={gate.probability_40ms}",
            f"l2_gate_threshold={gate.threshold}",
            f"l2_gate_config_revision={gate.config_revision}",
            f"l2_direction_kalman_backend={self.config.layer2.direction_kalman.backend}",
            f"l2_direction_kalman_enabled={values['direction_kalman_enabled']}",
            f"l2_direction_kalman_q_scale={values['direction_kalman_q_scale']}",
            f"l2_direction_kalman_r_scale={values['direction_kalman_r_scale']}",
            f"l2_direction_kalman_error={self._layer2.last_kalman_error}",
            f"l2_direction_id_tracking_backend={self.config.layer2.direction_id_tracking.backend}",
            "l2_direction_identity=authoritative_always_on",
            f"l2_direction_id_active_tracks={self._layer2.id_tracker.active_track_count}",
            f"l2_direction_id_tracking_error={self._layer2.last_id_tracking_error}",
            *(() if not is_music else (
                f"l2_music_algorithm={music.algorithm_version}",
                f"l2_music_valid_frequency_bins={music.valid_frequency_bins}",
                f"l2_music_covariance_quality={music.covariance_quality}",
                f"l2_music_model_order={music.model_order.estimated_sources}",
                f"l2_music_mdl_age_samples={music.model_order.mdl_age_samples}",
                f"l2_music_dpd_rank1_enabled={music.dpd_rank1_enabled}",
                f"l2_music_selected_frequency_bins={music.selected_frequency_bins}",
                f"l2_music_noise_whitening_enabled={music.noise_whitening_enabled}",
                f"l2_music_whitening_status={music.whitening_status}",
                f"l2_music_imcra_noise_hops={music.imcra_noise_hops}",
            )),
            *(() if music_state is None else (
                f"l2_music_rolling_state={music_state.state}",
                f"l2_music_gap_samples={music_state.gap_samples}",
                f"l2_music_state_reason={music_state.reason}",
                f"l2_music_reused_frames={music_state.reused_frames}",
                f"l2_music_added_frames={music_state.added_frames}",
                f"l2_music_removed_frames={music_state.removed_frames}",
                f"l2_music_steering_cache_rebuilt={music_state.steering_cache_rebuilt}",
            )),
            *(f"l2_gate_diagnostic={item}" for item in gate.diagnostics),
        )

    @staticmethod
    def _stage_wait_ms(current, previous_finished_ns: int) -> float:
        if current.started_monotonic_ns <= 0:
            return 0.0
        return max(0.0, (current.started_monotonic_ns - previous_finished_ns) / 1_000_000.0)

    def _commit_joined(self, joined: JoinedWindowResult) -> None:
        work = joined.work_item
        window = work.window
        values = work.config.values
        l2_output = joined.l2.output if joined.l2.state is StageState.COMPLETED else None
        l3_output = joined.l3.output if joined.l3.state is StageState.COMPLETED else None
        l4_result = joined.l4.output if joined.l4.state is StageState.COMPLETED else None
        response = None if l2_output is None else l2_output.spatial_response
        candidates = () if l2_output is None else tuple(
            getattr(l2_output, "directions", ()) or l2_output.candidates
        )
        candidate_track_ids = () if l2_output is None else tuple(
            getattr(l2_output, "candidate_track_ids", ())
        )
        candidate_track_ids = candidate_track_ids or (None,) * len(candidates)
        candidate_is_prediction = () if l2_output is None else tuple(
            getattr(l2_output, "candidate_is_prediction", ())
        )
        candidate_is_prediction = candidate_is_prediction or (False,) * len(candidates)
        candidate_is_formal = () if l2_output is None else tuple(
            getattr(l2_output, "candidate_track_is_formal", ())
        )
        candidate_is_formal = candidate_is_formal or (False,) * len(candidates)
        candidate_is_new = () if l2_output is None else tuple(
            getattr(l2_output, "candidate_track_is_new", ())
        )
        candidate_is_new = candidate_is_new or (False,) * len(candidates)
        candidate_kalman_ready = () if l2_output is None else tuple(
            getattr(l2_output, "candidate_track_is_kalman_ready", ())
        )
        candidate_kalman_ready = candidate_kalman_ready or (False,) * len(candidates)
        search_diagnostics = None if l2_output is None else l2_output.search_diagnostics
        gate_decision = None if l2_output is None else l2_output.gate_decision
        previews = () if l3_output is None else self._beamform_previews(
            l3_output, 0, len(candidates)
        )
        # Test-UI listening state is projected only after the formal result is
        # accepted below.  Slow preview disk/player work must never delay the
        # authoritative DecisionRecord/watermark path.
        tracked_audio = ()

        search_record_diagnostics = () if search_diagnostics is None else (
            f"l2_search_mode={search_diagnostics.mode}",
            f"l2_search_version={search_diagnostics.algorithm_version}",
            f"l2_search_revision={search_diagnostics.config_revision}",
            f"l2_search_iterations={search_diagnostics.iterations_used}",
            f"l2_search_stop={search_diagnostics.stop_reason}",
            f"l2_search_eligible_peaks={search_diagnostics.eligible_peak_count}",
            f"l2_search_candidate_limit={search_diagnostics.candidate_limit}",
            f"l2_search_limit_applied={search_diagnostics.candidate_limit_applied}",
            f"l2_search_dpd_rank1_enabled={search_diagnostics.dpd_rank1_enabled}",
            f"l2_search_selected_frequency_bins={search_diagnostics.selected_frequency_bins}",
            f"l2_search_noise_whitening_enabled={search_diagnostics.noise_whitening_enabled}",
            f"l2_search_whitening_status={search_diagnostics.whitening_status}",
            f"l2_search_imcra_noise_hops={search_diagnostics.imcra_noise_hops}",
            *( () if search_diagnostics.fallback_reason is None else
               (f"l2_search_fallback={search_diagnostics.fallback_reason}",) ),
        )
        gate_record = None if gate_decision is None else {
            "backend": gate_decision.backend,
            "state": gate_decision.state.value,
            "sound_present": gate_decision.sound_present,
            "reason": gate_decision.reason,
            "probability_previous_20ms": gate_decision.probability_previous_20ms,
            "probability_current_20ms": gate_decision.probability_current_20ms,
            "probability_40ms": gate_decision.probability_40ms,
            "threshold": gate_decision.threshold,
            "config_revision": gate_decision.config_revision,
            "source_hops": [
                {"start_sample": window.doa_start_sample,
                 "end_sample": window.doa_start_sample + 960,
                 "array_source_probability_20ms": gate_decision.probability_previous_20ms},
                {"start_sample": window.doa_start_sample + 960,
                 "end_sample": window.doa_end_sample,
                 "array_source_probability_20ms": gate_decision.probability_current_20ms},
            ],
            "diagnostics": list(gate_decision.diagnostics),
        }
        search_record = None if search_diagnostics is None else {
            "mode": search_diagnostics.mode,
            "algorithm_version": search_diagnostics.algorithm_version,
            "config_revision": search_diagnostics.config_revision,
            "iterations_used": search_diagnostics.iterations_used,
            "stop_reason": search_diagnostics.stop_reason,
            "remaining_weight_ratio": search_diagnostics.remaining_weight_ratio,
            "fallback_reason": search_diagnostics.fallback_reason,
            "eligible_peak_count": search_diagnostics.eligible_peak_count,
            "candidate_limit": search_diagnostics.candidate_limit,
            "candidate_limit_applied": search_diagnostics.candidate_limit_applied,
            "dpd_rank1_enabled": search_diagnostics.dpd_rank1_enabled,
            "selected_frequency_bins": search_diagnostics.selected_frequency_bins,
            "noise_whitening_enabled": search_diagnostics.noise_whitening_enabled,
            "whitening_status": search_diagnostics.whitening_status,
            "imcra_noise_hops": search_diagnostics.imcra_noise_hops,
            "evidence": [
                {"theta_deg": evidence.theta_deg,
                 "search_iteration": evidence.search_iteration,
                 "residual_raw_score": evidence.residual_raw_score,
                 "fixed_reference_normalized_score": evidence.fixed_reference_normalized_score,
                 "supporting_pairs": evidence.supporting_pairs,
                 "supporting_frequency_bins": evidence.supporting_frequency_bins,
                 "frequency_support_ratio": evidence.frequency_support_ratio,
                 "mean_plane_wave_fit": evidence.mean_plane_wave_fit,
                 "supporting_frequency_subbands": evidence.supporting_frequency_subbands,
                 "circular_concentration": evidence.circular_concentration,
                 "cluster_weight": evidence.cluster_weight}
                for evidence in search_diagnostics.evidence
            ],
        }
        # Persist only the canonical compensated 20 ms additions. The
        # overlapping raw L3 windows are transient assembler inputs and are
        # deliberately not duplicated into the recording corpus.
        track_audio_record = self._compute_cache.get(
            "l3", joined.key, "track_audio_record"
        ) or ()
        enhanced_records = tuple(item[0] for item in track_audio_record)
        enhanced_waveforms = tuple(item[1] for item in track_audio_record)
        l4_record = None if l4_result is None else {
            "primary_model_id": l4_result.primary_model_id,
            "threshold": l4_result.threshold,
            "predictions": [
                {"model_id": prediction.model_id,
                 "probabilities": prediction.probabilities.tolist(),
                 "latency_ms": prediction.latency_ms,
                 "metadata": dict(prediction.metadata)}
                for prediction in l4_result.predictions
            ],
            "input_gain_compensation": [asdict(item) for item in l4_result.input_gain_compensation],
        }
        stage_statuses = {
            "l2": joined.l2.state.value, "l3": joined.l3.state.value,
            "l4": joined.l4.state.value,
        }
        stage_timings = {
            name: float(stage.duration_ms)
            for name, stage in (("l2", joined.l2), ("l3", joined.l3), ("l4", joined.l4))
            if stage.started_monotonic_ns > 0 and stage.duration_ms is not None
        }
        stage_waits = {
            "l2": self._stage_wait_ms(joined.l2, work.accepted_monotonic_ns),
            "l3": self._stage_wait_ms(joined.l3, joined.l2.finished_monotonic_ns),
            "l4": self._stage_wait_ms(joined.l4, joined.l3.finished_monotonic_ns),
        }
        latency_ms = max(
            0.0, (joined.completed_monotonic_ns - work.accepted_monotonic_ns) / 1_000_000.0
        )
        compute_ms = sum(stage_timings.values())
        severe = {StageState.FAILED, StageState.TIMED_OUT, StageState.DROPPED, StageState.CANCELLED}
        has_severe_stage = any(
            stage.state in severe for stage in (joined.l2, joined.l3, joined.l4)
        )
        used_successful_fallback = (
            any(item.fallback_reason is not None for item in previews)
            or (
                search_diagnostics is not None
                and search_diagnostics.fallback_reason is not None
            )
        )
        status = (
            "error" if has_severe_stage
            else "degraded" if used_successful_fallback
            else "ok"
        )
        l4_diagnostics = () if l4_result is None else (
            f"l4_primary_model={l4_result.primary_model_id}",
            f"l4_threshold={l4_result.threshold}",
            *(f"l4_model_latency_ms={item.model_id}:{item.latency_ms:.3f}"
              for item in l4_result.predictions),
            *(
                f"l4_aggregation={item.model_id}:{item.metadata.get('aggregation', 'unknown')}"
                for item in l4_result.predictions
            ),
            *(
                "l4_input_gain="
                f"{item.algorithm_version}:max={item.max_applied_gain_db:.3f}:"
                f"mean={item.mean_applied_gain_db:.3f}:"
                f"segments={item.compensated_segment_count}:"
                f"peak_protection={item.peak_protection_trigger_count}"
                for item in l4_result.input_gain_compensation
            ),
        )
        terminal_diagnostics = tuple(
            f"{name}_stage={stage.state.value}:{stage.error or stage.reason}"
            for name, stage in (("l2", joined.l2), ("l3", joined.l3), ("l4", joined.l4))
            if stage.state is not StageState.COMPLETED
        )
        candidate_records = tuple(asdict(item) for item in candidates)
        active_track_records = tuple(
            asdict(item) for item in (
                () if l2_output is None else getattr(l2_output, "active_tracks", ())
            )
        )
        detection_records = tuple(
            {
                **(
                    {"track_id": int(item.track_id)}
                    if type(getattr(item, "track_id", None)) is int
                    and item.track_id > 0
                    else {}
                ),
                "theta_deg": item.theta_deg,
                "probability": item.probability,
                "is_voice": item.is_voice,
                "model_id": item.model_id,
                "threshold": l4_result.threshold,
            }
            for index, item in enumerate(() if l4_result is None else l4_result.detections)
        )
        record = DecisionRecord(
            window.session_id, window.stream_epoch, window.window_id, window.decision_sample,
            (window.doa_start_sample, window.doa_end_sample),
            (window.context_start_sample, window.context_end_sample), status,
            candidates=candidate_records,
            detections=detection_records,
            voice_direction_count=0 if l4_result is None else l4_result.voice_direction_count,
            diagnostics=joined.l2.diagnostics + search_record_diagnostics + l4_diagnostics
            + terminal_diagnostics,
            processing_latency_ms=latency_ms,
            raw_scores=None if response is None else response.raw_scores,
            normalized_scores=None if response is None else response.normalized_scores,
            gate_decision=gate_record, search_diagnostics=search_record,
            enhanced_audio=enhanced_records,
            enhanced_waveforms=enhanced_waveforms,
            l4_result=l4_record,
            stage_statuses=stage_statuses,
            stage_timings_ms=stage_timings,
            stage_queue_wait_ms=stage_waits,
            terminal_reason=joined.terminal_reason,
            active_tracks=active_track_records,
            kalman_applied=any(
                bool(values["direction_kalman_enabled"] and ready)
                for ready in candidate_kalman_ready
            ),
            config_revision=work.config.revision,
            config_hash=work.config.config_hash,
            calibration_revision=0,
            calibration_version=self.config.hardware.hardware_calibration_status,
            calibration_hash=self._calibration_hash,
            music_algorithm_version=(
                None if search_diagnostics is None else search_diagnostics.algorithm_version
            ),
            model_order=(
                None if l2_output is None or getattr(l2_output, "model_order", None) is None
                else asdict(l2_output.model_order)
            ),
            music_diagnostics=(
                None if search_diagnostics is None or not hasattr(search_diagnostics, "state") else {
                    "state": asdict(search_diagnostics.state),
                    "valid_frequency_bins": search_diagnostics.valid_frequency_bins,
                    "covariance_quality": search_diagnostics.covariance_quality,
                }
            ),
        )
        dropped_windows = ()
        if joined.state in {StageState.DROPPED, StageState.CANCELLED}:
            dropped_windows = ({
                "window_id": window.window_id,
                "decision_sample": window.decision_sample,
                "reason": joined.terminal_reason,
            },)
        watermark = ResultWatermark(
            window.session_id, window.stream_epoch, window.decision_sample, dropped_windows
        )
        try:
            append_with_watermark = getattr(
                self.recording_store, "append_result_with_watermark", None
            )
            if callable(append_with_watermark):
                append_with_watermark(record, watermark)
            else:
                # Compatibility for external stores implementing the v2 API.
                self.recording_store.append_result(record)
                self.recording_store.advance_result_watermark(watermark)
            self.recording_result_error = None
        except Exception as exc:
            # Recording is an asynchronous side effect.  A full/failing disk
            # is visible in diagnostics but cannot kill the real-time stages
            # or prevent caches from being released.
            self.recording_result_error = str(exc)
            self.processing_error = f"recording: {exc}"

        if record.status in {"ok", "degraded"} and record.voice_direction_count > 0:
            trigger_event = getattr(self.recording_store, "trigger_event", None)
            if callable(trigger_event):
                try:
                    trigger_event(record)
                except Exception as exc:
                    self.recording_result_error = f"event trigger failed: {exc}"
                    self.processing_error = f"recording: {self.recording_result_error}"

        if self.dev_audio_tracker is not None:
            try:
                # The formal stream hub already fed the exact compensated hops
                # to this cache in the L3 worker. Commit only projects its
                # immutable playback state; it never reassembles L3 windows.
                tracked_audio = self.dev_audio_tracker.snapshots()
                self.dev_audio_tracking_error = None
            except Exception as exc:
                self.dev_audio_tracking_error = str(exc)
                try:
                    tracked_audio = self.dev_audio_tracker.snapshots()
                except Exception:
                    tracked_audio = ()
        try:
            self._publish_joined_ui(
                joined, values, l2_output, l3_output, l4_result,
                response, candidates, search_diagnostics, gate_decision,
                previews, tracked_audio, compute_ms, latency_ms, stage_timings,
            )
            self.dev_ui_error = None
        except Exception as exc:
            # Development visualization is a best-effort projection of an
            # already committed formal result.  A missing/stale UI frame must
            # never stop RecordingStore, advance ordering, or the next window.
            self.dev_ui_error = str(exc)
            self.processing_error = f"ui: {exc}"
        self._stage_completed_counts["commit"] += 1
        self._compute_cache.evict_window(joined.key, "window_committed")
        self._prune_completed_stream_history(joined.key.session_id)
        if (
            not any(self._stage_errors.values())
            and self.dev_ui_error is None
            and self.recording_result_error is None
        ):
            self.processing_error = None

    def _publish_joined_ui(
        self,
        joined,
        values,
        l2_output,
        l3_output,
        l4_result,
        response,
        candidates,
        search_diagnostics,
        gate_decision,
        previews,
        tracked_audio,
        compute_ms: float,
        latency_ms: float,
        stage_timings,
    ) -> None:
        window = joined.work_item.window
        with self._ui_lock:
            current_stream = getattr(self._ui_aggregator, "current_stream", None)
            if current_stream is not None and current_stream != (
                window.session_id,
                window.stream_epoch,
            ):
                return
            self._performance.add_timing(
                window.session_id, window.stream_epoch, window.window_id,
                compute_ms, latency_ms,
                l2_ms=stage_timings.get("l2"), l3_ms=stage_timings.get("l3"),
                l4_ms=stage_timings.get("l4"), completed_monotonic=monotonic(),
                processed=joined.state is StageState.COMPLETED,
            )
            scan = DirectionScanConfig(**dict(values["scan_config"]))
            if l2_output is None:
                l2_reason = (
                    f"L2 {joined.l2.state.value.upper()}: "
                    f"{joined.l2.error or joined.l2.reason}"
                )
                if joined.l2.state is StageState.DROPPED:
                    self._ui_aggregator.report_l2_drop(l2_reason)
                else:
                    self._ui_aggregator.update_srp(None, (), l2_reason)
            elif response is None:
                unavailable_reason = gate_decision.reason
                reset_reason = self._epoch_reset_reason(
                    window.session_id, window.stream_epoch
                )
                if (
                    reset_reason is not None
                    and unavailable_reason == "upstream_probability_warming_up"
                ):
                    unavailable_reason = (
                        f"{unavailable_reason}; epoch_reset:{reset_reason}"
                    )
                self._ui_aggregator.update_srp(
                    None, (), f"UNAVAILABLE: {unavailable_reason}", gate_decision=gate_decision,
                    gate_threshold=float(values["gate_threshold"]),
                    gate_config_revision=int(values["gate_config_revision"]),
                    direction_threshold=scan.direction_threshold,
                    direction_kalman_enabled=bool(values["direction_kalman_enabled"]),
                    direction_kalman_q_scale=float(values["direction_kalman_q_scale"]),
                    direction_kalman_r_scale=float(values["direction_kalman_r_scale"]),
                    scan_config_revision=int(values["scan_config_revision"]),
                    directions=getattr(l2_output, "directions", ()),
                    active_tracks=getattr(l2_output, "active_tracks", ()),
                )
            else:
                self._ui_aggregator.update_srp(
                    response, candidates, search_diagnostics=search_diagnostics,
                    gate_decision=gate_decision,
                    gate_threshold=float(values["gate_threshold"]),
                    gate_config_revision=int(values["gate_config_revision"]),
                    direction_threshold=scan.direction_threshold,
                    direction_kalman_enabled=bool(values["direction_kalman_enabled"]),
                    direction_kalman_q_scale=float(values["direction_kalman_q_scale"]),
                    direction_kalman_r_scale=float(values["direction_kalman_r_scale"]),
                    scan_config_revision=int(values["scan_config_revision"]),
                    directions=getattr(l2_output, "directions", ()),
                    active_tracks=getattr(l2_output, "active_tracks", ()),
                )
            self._ui_aggregator.update_l3(
                previews, None if l3_output is not None else f"L3 {joined.l3.state.value}",
                tracked_audio=tracked_audio,
            )
            frame = self._ui_aggregator.update_l4(l4_result)
            self._latest(self.latest_dev_ui, frame)

    @staticmethod
    def _beamform_previews(l3_output, start: int = 0, stop: int | None = None) -> tuple[BeamformPreview, ...]:
        return tuple(
            BeamformPreview(
                item.session_id,
                item.stream_epoch,
                item.window_id,
                item.decision_sample,
                item.theta_deg,
                item.enhanced_audio,
                item.algorithm,
                item.fallback_reason,
                item.diagnostics,
                track_id=getattr(item, "track_id", None),
            )
            for item in l3_output.enhanced_audio[start:stop]
        )

    @staticmethod
    def _layer4_audio_segments(
        l3_output,
        stop: int | None = None,
        probabilities_20ms: tuple[float | None, ...] = (),
    ) -> tuple[Layer4AudioSegment, ...]:
        """Adapt the formal immutable L3 audio batch to L4's audio contract."""
        return tuple(
            Layer4AudioSegment(
                item.session_id,
                item.stream_epoch,
                item.window_id,
                item.decision_sample,
                item.theta_deg,
                item.sample_rate,
                item.enhanced_audio,
                probabilities_20ms,
                getattr(item, "track_id", None),
            )
            for item in l3_output.enhanced_audio[:stop]
        )

    def _context_probabilities_20ms(self, window: DecisionWindow) -> tuple[float | None, ...]:
        spec = getattr(
            self,
            "downstream_window_spec",
            getattr(
                getattr(self, "_layer3", None),
                "window_spec",
                None,
            ),
        )
        if spec is None:
            spec = DownstreamAudioWindowSpec(160, 7_680, 8, 17, 2_560)
        hops_by_interval = {
            (hop.start_sample, hop.end_sample): hop for hop in window.imcra_hops
        }
        probabilities = tuple(
            (
                None
                if (hop := hops_by_interval.get((start, start + 960))) is None
                or hop.state != "ready"
                else hop.array_source_probability_20ms
            )
            for start in range(
                window.context_end_sample - spec.samples,
                window.context_end_sample,
                960,
            )
        )
        expected = spec.decision_hops
        if len(probabilities) != expected:
            raise RuntimeError(
                f"L4 requires exactly {expected} context-aligned 20 ms probability slots"
            )
        return probabilities

    def _layer4_inputs_from_output(self, window, l3_output, formal_count: int):
        return self._layer4_audio_segments(
            l3_output, formal_count, self._context_probabilities_20ms(window)
        )

    @staticmethod
    def _layer4_inputs_from_stream(batch) -> tuple[Layer4AudioSegment, ...]:
        """Adapt immutable compensated continuous tracks to the L4 contract."""
        return tuple(
            Layer4AudioSegment(
                item.session_id,
                item.stream_epoch,
                item.window_id,
                item.decision_sample,
                item.theta_deg,
                48_000,
                item.waveform,
                item.probabilities_20ms,
                item.track_id,
                item.effective_start_sample,
                item.effective_end_sample,
                True,
                item.gain_diagnostic,
            )
            for item in batch.continuous_audio
        )

    @staticmethod
    def _validate_direction_outputs(layer: str, candidates, outputs) -> None:
        """Enforce exact, ordered one-output-per-candidate stage contracts."""

        candidates = tuple(candidates)
        outputs = tuple(outputs)
        if len(outputs) != len(candidates):
            raise RuntimeError(
                f"{layer} contract violation: expected {len(candidates)} ordered outputs, "
                f"received {len(outputs)}"
            )
        for index, (candidate, output) in enumerate(zip(candidates, outputs, strict=True)):
            delta = abs(
                ((float(output.theta_deg) - float(candidate.theta_deg) + 180.0) % 360.0)
                - 180.0
            )
            if delta > 1e-6:
                raise RuntimeError(
                    f"{layer} contract violation: output {index} angle "
                    f"{output.theta_deg} does not match candidate {candidate.theta_deg}"
                )
            if getattr(output, "track_id", None) != getattr(candidate, "track_id", None):
                raise RuntimeError(
                    f"{layer} contract violation: output {index} track_id "
                    f"{getattr(output, 'track_id', None)} does not match candidate "
                    f"{getattr(candidate, 'track_id', None)}"
                )

    def _process_l3(self, window, candidates):
        """Run L3 exactly once for the formal, already-smoothed L2 candidates."""
        candidates = tuple(candidates)
        with self._l3_mode_lock:
            mode = self._l3_processing_mode
        l3_output = self._layer3.process(window, candidates, self._geometry, mode=mode)
        formal_count = len(candidates)
        self._validate_direction_outputs("L3", candidates, l3_output.enhanced_audio)
        formal_previews = self._beamform_previews(l3_output, 0, formal_count)
        l4_inputs = self._layer4_inputs_from_output(window, l3_output, formal_count)
        return formal_previews, l4_inputs

    def set_light(self, enabled: bool) -> None:
        packet = led_command(enabled)
        try:
            count = self.serial_device.write(packet)
        except Exception:
            self.light_state = "error"
            raise
        if count != len(packet):
            self.light_state = "error"
            raise OSError(f"灯控命令未完整写入：{count}/{len(packet)}")
        self.light_state = "on" if enabled else "off"

    def begin_scratch(self) -> None:
        self.scratch.record()

    def pause_or_resume_scratch(self) -> None:
        if self.scratch.state == "recording":
            self.scratch.pause()
        elif self.scratch.state == "paused":
            self.scratch.resume()
        else:
            raise RuntimeError("当前录音状态不能暂停或继续")

    def finish_scratch(self) -> Path:
        return self.scratch.finish()

    def begin_runtime_recording(self) -> None:
        self.recording_store.start_recording()

    def pause_runtime_recording(self) -> None:
        self.recording_store.pause_recording()

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop.set()
            thread = self._thread
            workers = dict(self._processing_threads)
        # Stop acquisition first.  Its finally block flushes the delayed L1
        # denoiser and then marks input complete.  The four processing workers
        # subsequently propagate EOS while draining every admitted window.
        try:
            self.pipeline.stop()
        except Exception as exc:
            self.last_error = f"input stop failed: {exc}"
        timeout = self.config.runtime.graceful_shutdown_timeout_seconds
        deadline = monotonic() + timeout

        def join_until(worker: threading.Thread | None) -> None:
            if worker is None or worker is threading.current_thread():
                return
            worker.join(timeout=max(0.0, deadline - monotonic()))

        join_until(thread)
        for name in ("l2", "l3", "l4", "commit"):
            join_until(workers.get(name))
        alive = {
            name: worker for name, worker in workers.items() if worker.is_alive()
        }
        if alive:
            self._processing_abort.set()
            # Convert every registered queued/in-flight window into an
            # explicit terminal record before asking the commit worker to
            # finish.  A late stage result will be rejected by the joiner and
            # cannot overwrite this cancellation.
            for key in self._result_joiner.pending_keys():
                try:
                    self._joiner_submit(
                        lambda key=key: self._result_joiner.terminate_window(
                            key, StageState.CANCELLED, "graceful_shutdown_timeout"
                        )
                    )
                except Exception as exc:
                    self._set_stage_error("commit", exc)
            self._wake_processing_workers()
            # A large bounded offline queue can leave hundreds of joined
            # terminal records for commit.  Give forced cancellation one
            # complete configured shutdown interval instead of a fixed one
            # second, then report failure only if a worker is still alive.
            forced_deadline = monotonic() + timeout
            for name in ("l2", "l3", "l4", "commit"):
                worker = workers.get(name)
                if worker is not None and worker is not threading.current_thread() and worker.is_alive():
                    worker.join(timeout=max(0.0, forced_deadline - monotonic()))
            alive = {
                name: worker for name, worker in workers.items() if worker.is_alive()
            }
            if alive:
                self.last_error = (
                    "processing workers did not stop after graceful drain and forced cancellation: "
                    f"{','.join(sorted(alive))}"
                )
        input_alive = thread is not None and thread.is_alive()
        with self._lifecycle_lock:
            if input_alive:
                self.last_error = f"runtime input worker did not stop within {timeout:.1f} seconds"
            else:
                self._thread = None
            if not alive:
                self._processing_threads = {}
                self._processing_thread = None
        if not alive and not input_alive:
            # Workers own these queues.  Once every worker has exited, stale
            # work/EOS markers can be released safely so STOP telemetry reads
            # zero and retained 320 ms windows do not occupy memory.
            for mailbox in (
                self._l2_windows,
                self._l3_windows,
                self._l4_windows,
                self._completion_results,
            ):
                while True:
                    try:
                        mailbox.get_nowait()
                    except queue.Empty:
                        break
        try:
            self.scratch.close()
        except Exception as exc:
            self.scratch_error = f"scratch finalize failed: {exc}"
        # Never close RecordingStore while a stage or commit worker can still
        # append to it.  A later stop/close call can finalize after the worker
        # exits, and the explicit error remains visible to the UI.
        if self._recording_session_started and not alive and not input_alive:
            try:
                self.recording_store.stop_session("normal" if self.last_error is None else "runtime_error")
            except Exception as exc:
                self.last_error = f"runtime recording finalize failed: {exc}"
            else:
                self._recording_session_started = False

    def close(self, *, delete_dev_test_ui_audio: bool = False) -> None:
        self.stop()
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(
                "cannot close RecordingStore while the input worker is still alive"
            )
        if any(thread.is_alive() for thread in self._processing_threads.values()):
            raise RuntimeError(
                "cannot close RecordingStore while processing workers are still alive"
            )
        close_error: BaseException | None = None
        # A light command can lazily open CDC before capture starts.  In that
        # state InputPipeline does not own an active hotmap lifecycle, so its
        # stop() has nothing to close; runtime shutdown must still release the
        # independently opened control port.
        stop_serial = getattr(self.serial_device, "stop", None)
        if callable(stop_serial):
            try:
                stop_serial()
            except BaseException as exc:
                close_error = exc
        try:
            self.scratch.shutdown(delete_files=delete_dev_test_ui_audio)
        except BaseException as exc:
            close_error = exc
        try:
            self.recording_store.close()
        except BaseException as exc:
            close_error = close_error or exc
        if delete_dev_test_ui_audio:
            # Drop bounded formal previews and UI snapshot references.
            for mailbox in (
                self.latest_l1, self.latest_windows, self.latest_dev_ui,
                self.latest_l4_dev_ui,
                self._l2_windows, self._l3_windows, self._l4_windows,
                self._completion_results,
            ):
                while True:
                    try:
                        mailbox.get_nowait()
                    except queue.Empty:
                        break
            self._ui_aggregator = DevUiAggregator(
                self._performance, stale_after_ms=self.config.dev_test_ui.stale_after_ms
            )
        if not self._processing_threads:
            self._compute_cache.clear("runtime_close")
            self._layer3.clear_cache()
            self.track_audio_stream.reset()
        if self.dev_audio_tracker is not None:
            try:
                self.dev_audio_tracker.close(delete_files=delete_dev_test_ui_audio)
            except BaseException as exc:
                close_error = close_error or exc
        if close_error is not None:
            raise close_error
