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
from gui.dev_test_ui.contracts import (
    AlgorithmPerformanceSnapshot, BeamformPreview, DevUiFrame, L2DevUiSnapshot,
)
from gui.dev_test_ui.meter import L1Meter
from gui.dev_test_ui.scratch_recorder import ScratchRecorder
from ingest import BlockFanout, IngestCoordinator
from layer1_input.calibration import ChannelCalibrator
from layer1_input.configuration import AudioConfig, CalibrationConfig, CdcConfig
from layer1_input.imcra import Layer1Imcra
from layer1_input.pre_denoise import ImcraWienerPreDenoiser
from layer1_input.speaker_count import AsyncSpeakerCounter
from layer1_input.pipeline import InputPipeline
from layer1_input.protocols import led_command
from layer1_input.serial_device import SerialDevice
from layer1_input.sources import LiveSipeedSource
from layer2_source_detection import (
    DirectionScanConfig, Layer2Pipeline, SourceProbability20ms, SourceProbabilityState,
)
from layer3_direction_signal import (
    L3_MODE_DS_BASELINE,
    L3_PROCESSING_MODES,
    Layer3Output,
    Layer3Processor,
)
from layer4_speech_separation import (
    DirectionCountSpeakerClassifier,
    MossFormer2Backend,
    TigerBackend,
)
from layer4_speech_separation.offline import OfflineLayer4Pipeline
from layer5_voice_classifier import (
    InputGainCompensationSettings,
    Layer5AudioSegment,
    Layer5Engine,
    NvidiaMarbleNetPlugin,
)
from layer6_speaker_consolidation import CampPlusEmbedder, DnsMosScorer, OfflineLayer6Pipeline
from track_audio_stream import TrackAudioStreamHub, TrackAudioWindow, TrackVoiceAnnotation
from windowing import WindowAssembler

from .compute_cache import CachePartitionLimits, ComputeCache, ComputeCacheError
from .processing_contracts import (
    JoinedWindowResult,
    L2StageResult,
    L3StageResult,
    L5StageResult,
    ProcessingConfigSnapshot,
    StageState,
    WindowKey,
    WindowWorkItem,
)
from .result_joiner import JoinerCapacityError, ResultDeliveryError, ResultJoiner


_PIPELINE_EOS = object()
_CONFIGURED_DRAIN_TIMEOUT = object()
_CONFIRMED_BACKFILL_SAMPLES = 48_000
_CONFIRMED_BACKFILL_HISTORY_WINDOWS = 60


@dataclass(frozen=True)
class _PipelineTimingBarrier:
    generation: int


@dataclass(frozen=True, slots=True)
class _L3Work:
    work_item: WindowWorkItem
    l2: L2StageResult


@dataclass(frozen=True, slots=True)
class _L3PreparedWork:
    work: _L3Work
    started_ns: int
    candidates: tuple[object, ...]
    result: object


@dataclass(frozen=True, slots=True)
class _L3HostWork:
    work: _L3Work
    started_ns: int
    candidates: tuple[object, ...]
    result: object
    ready_event: object | None


@dataclass(frozen=True, slots=True)
class _L5Work:
    work_item: WindowWorkItem
    l2: L2StageResult
    l3: L3StageResult


@dataclass(frozen=True, slots=True)
class _ConfirmedBackfillWork:
    track: object
    windows: tuple[DecisionWindow, ...]
    processing_mode: str
    l2_direction_count: int


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
        layer5_engine: Layer5Engine | None = None,
        layer3_backfill_processor: object | None = None,
        dev_audio_tracker: object | None = None,
        source_probability_provider: Callable[
            [DecisionWindow], tuple[SourceProbability20ms, ...]
        ] | None = None,
        speaker_count_model: object | None = None,
        ephemeral_live_capture: bool = False,
        development_center_l4_bypass: bool = False,
    ) -> None:
        self.config = config
        self.project_root = Path(project_root).resolve()
        # These L1-L3 tensors are small (7 channels, at most 513 bins). Letting
        # every operation fan out to every logical core creates more scheduler
        # work than arithmetic and makes concurrent L2/L3 stages fight over the
        # same pool. L3 preparation, beamforming and stitching have independent
        # bounded pipeline workers instead.
        torch.set_num_threads(config.runtime.torch_cpu_threads)
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
        self.speaker_counter = AsyncSpeakerCounter(
            config.layer1_speaker_count,
            project_root=self.project_root,
            model=speaker_count_model,
            timestamp_tolerance_ms=config.timing.timestamp_tolerance_ms,
        )
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
        # Live microphone sessions in Development Test UI are intentionally
        # temporary and never enter the formal RecordingStore.
        self._ephemeral_live_capture = bool(ephemeral_live_capture)
        self._development_center_l4_bypass = bool(development_center_l4_bypass)
        self._development_l4_reference_track_id = 0
        self._ephemeral_recording_active = False
        self._ephemeral_transition_lock = threading.RLock()
        self._recording_session_started = False
        self.meter = L1Meter()
        self.latest_l1: queue.Queue[object] = queue.Queue(maxsize=1)
        self.latest_windows: queue.Queue[object] = queue.Queue(maxsize=1)
        self.latest_dev_ui: queue.Queue[object] = queue.Queue(maxsize=config.dev_test_ui.snapshot_mailbox_capacity)
        # L2 visualization must follow the L2 worker, not the slower ordered
        # L2/L3/L5 commit path. This latest-only side channel never changes
        # formal recording or cross-layer ordering.
        self.latest_l2_dev_ui: queue.Queue[object] = queue.Queue(maxsize=1)
        self._l2_ui_overwrites = 0
        # Formal commits are deliberately ordered, so a burst of terminal
        # DROPPED/SKIPPED frames may follow one expensive completed L5 window.
        # Keep completed L5 UI frames on an independent latest-only mailbox so
        # that those audit frames cannot erase the newest useful CNN result.
        self.latest_l5_dev_ui: queue.Queue[object] = queue.Queue(maxsize=1)
        self._l5_ui_overwrites = 0
        self._l5_diagnostics_lock = threading.Lock()
        self._l5_actual_completed = 0
        self._l5_dropped = 0
        self._l5_skipped = 0
        self._l5_completion_times: deque[float] = deque(maxlen=512)
        # The public compatibility alias is rebound by
        # ``_reset_processing_graph`` to the formal L2 admission queue.  The
        # staged queues are independent so L2(n), L3(n-1), and L5(n-2) can run
        # concurrently without allowing two workers to mutate the same
        # stateful layer instance.
        self._l2_windows: queue.Queue[object]
        self._l3_windows: queue.Queue[object]
        self._l3_prepared_windows: queue.Queue[object]
        self._l3_host_windows: queue.Queue[object]
        self._l5_windows: queue.Queue[object]
        self._confirmed_backfill_work: queue.Queue[object]
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
        self._layer2_state_lock = threading.RLock()
        self._source_probability_provider = source_probability_provider or self._imcra_probabilities
        self._gate_config_lock = threading.Lock()
        self._gate_probability_threshold = config.layer2.probability_gate.threshold
        self._gate_config_revision = 0
        self._l3_mode_lock = threading.Lock()
        self._l3_processing_mode = L3_MODE_DS_BASELINE
        self._l3_config_revision = 0
        self._downstream_processing_enabled = threading.Event()
        if not self._ephemeral_live_capture:
            self._downstream_processing_enabled.set()
        self._scan_config = DirectionScanConfig.from_project(config)
        self._scan_config_lock = threading.Lock()
        self._scan_config_revision = 0
        self._kalman_config_revision = 0
        # Compatibility snapshot fields only. Circular IMM is intrinsic to the
        # ID tracker and no longer has an independent runtime switch or Q/R UI.
        self._direction_kalman_enabled = True
        self._direction_id_tracking_enabled = True
        self._direction_kalman_q_scale = 1.0
        self._direction_kalman_r_scale = 1.0
        self._geometry = physical_6plus1_geometry(
            config.hardware.speed_of_sound_mps,
            config.hardware.geometry_version,
            config.hardware.ring_radius_m,
        )
        # L1 and L2 remain CPU stages. L3, stopped/offline L4 and lightweight
        # L5 have independent device policies so the separator can own CUDA
        # without forcing every realtime stage onto the GPU.
        self.l3_device = self._resolve_layer_device(config, "l3_device")
        self.l4_device = self._resolve_layer_device(config, "l4_device")
        self.l5_device = self._resolve_layer_device(config, "l5_device")
        # Compatibility alias consumed by existing Test UI/status clients. It
        # has always described the realtime L3 device in user-facing text.
        self.processing_device = self.l3_device
        self._l3_cuda_stream = torch.cuda.Stream() if self.l3_device == "cuda" else None
        self._l5_cuda_stream = torch.cuda.Stream() if self.l5_device == "cuda" else None
        self._layer3 = Layer3Processor(config, device=self.l3_device)
        # Historical recovery owns an independent processor/cache so it never
        # mutates the realtime L3 rolling state. It is intentionally one
        # low-throughput worker because it runs only once per newly confirmed ID.
        self._layer3_backfill = (
            layer3_backfill_processor
            if layer3_backfill_processor is not None
            else Layer3Processor(config, device=self.l3_device)
        )
        self.downstream_window_spec = config.downstream_audio_window
        self._l3_cache_snapshot: object | None = None
        if layer5_engine is None:
            enabled_l5 = tuple(item for item in config.layer5.models if item.enabled)
            plugins = []
            for item in enabled_l5:
                artifact = Path(item.model_artifact)
                if not artifact.is_absolute():
                    artifact = self.project_root / artifact
                    if not artifact.exists():
                        artifact = Path(__file__).resolve().parents[1] / item.model_artifact
                if item.backend not in {
                    "nvidia_marblenet_continuous_v2", "nvidia_marblenet_window_v1",
                }:
                    raise ValueError(f"unsupported Layer5 backend: {item.backend}")
                plugins.append(NvidiaMarbleNetPlugin(
                    item.model_id,
                    artifact,
                    device=self.l5_device,
                    window_spec=config.downstream_audio_window,
                ))
            primary = next(plugin for plugin in plugins if plugin.model_id == config.layer5.primary_model_id)
            shadows = tuple(plugin for plugin in plugins if plugin.model_id != config.layer5.primary_model_id)
            layer5_engine = Layer5Engine(
                primary,
                shadows,
                threshold=config.layer5.voice_probability_limit,
                input_gain_compensation=InputGainCompensationSettings(
                    **config.layer5.input_gain_compensation.model_dump()
                ),
                window_spec=config.downstream_audio_window,
            )
        self._layer5 = layer5_engine
        self._dnsmos_scorer: DnsMosScorer | None = None
        self.track_audio_stream = TrackAudioStreamHub(
            InputGainCompensationSettings(
                **config.layer5.input_gain_compensation.model_dump()
            ),
            context_ms=config.layer5.continuous_context_ms,
            minimum_output_seconds=(
                config.dev_test_ui.minimum_listening_track_seconds
                if dev_audio_tracker is not None
                else 0.0
            ),
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
        self._discard_processing = threading.Event()
        self._input_worker_done = threading.Event()
        self._input_exhausted = False
        self._thread: threading.Thread | None = None
        self._processing_thread: threading.Thread | None = None
        self._processing_threads: dict[str, threading.Thread] = {}
        self._confirmed_backfill_lock = threading.RLock()
        self._confirmed_backfill_history: deque[DecisionWindow] = deque(
            maxlen=_CONFIRMED_BACKFILL_HISTORY_WINDOWS
        )
        self._confirmed_backfill_ids: set[tuple[str, int, int]] = set()
        self._stage_errors: dict[str, str | None] = {
            "l2": None, "l3": None, "l5": None, "commit": None,
        }
        self._stage_error_counts: dict[str, int] = {name: 0 for name in self._stage_errors}
        self._stage_completed_counts: dict[str, int] = {
            "l2": 0, "l3": 0, "l5": 0, "commit": 0,
        }
        self._pipeline_duration_lock = threading.RLock()
        self._pipeline_timing_generation = 0
        self._pipeline_timing_sealed = False
        self._pipeline_first_queued_ns: int | None = None
        self._pipeline_stage_finished_ns: dict[str, int | None] = {
            "l2": None, "l3": None, "l5": None,
        }
        self._pipeline_stage_pause_reasons: dict[str, set[str]] = {
            "l2": set(), "l3": set(), "l5": set(),
        }
        self._pipeline_stage_pause_started_ns: dict[str, int | None] = {
            "l2": None, "l3": None, "l5": None,
        }
        self._pipeline_stage_excluded_ns: dict[str, int] = {
            "l2": 0, "l3": 0, "l5": 0,
        }
        self._project_config_hash = config_hash(config=config)
        self._calibration_hash = calibration_config_hash(config.calibration)
        self._compute_cache: ComputeCache
        self._result_joiner: ResultJoiner
        self._reset_processing_graph()
        self._lifecycle_lock = threading.RLock()

    @staticmethod
    def _resolve_layer_device(config: ProjectConfig, field_name: str) -> str:
        configured = getattr(config.runtime, field_name)
        preferred = (configured or config.runtime.preferred_device).casefold()
        if preferred == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            if config.runtime.allow_cpu_fallback:
                return "cpu"
            raise RuntimeError(
                f"配置runtime.{field_name}=cuda，但当前torch.cuda不可用且禁止CPU fallback"
            )
        if preferred == "cpu":
            return "cpu"
        raise ValueError(f"未知runtime.{field_name}: {preferred}")

    def _reset_processing_graph(self) -> None:
        """Create a fresh bounded graph for one capture session.

        Queues and the joiner are deliberately session-scoped.  Reusing either
        after restart would make a new session wait behind stale EOS markers or
        stale timeline keys.
        """

        runtime = self.config.runtime
        self._l2_windows = queue.Queue(maxsize=runtime.l2_queue_windows)
        self._l3_windows = queue.Queue(maxsize=runtime.l3_queue_windows)
        self._l3_prepared_windows = queue.Queue(maxsize=runtime.l3_queue_windows)
        # CUDA submission may overlap one GPU batch with one CPU stitching
        # batch, but must never enqueue the full external L3 backlog as an
        # unbounded device graph. Two micro-batches provide double buffering
        # and explicit GPU-to-host backpressure.
        self._l3_host_windows = queue.Queue(
            maxsize=2 * runtime.l3_cuda_microbatch_windows
        )
        self._l5_windows = queue.Queue(maxsize=runtime.l5_queue_windows)
        # At most three authoritative directions are published per L2 window.
        # One bounded slot per possible first-confirmed ID is sufficient and
        # prevents historical work from becoming an unbounded side queue.
        self._confirmed_backfill_work = queue.Queue(maxsize=3)
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
        l5_bytes = total - l2_bytes - l3_bytes
        self._compute_cache = ComputeCache(
            {
                "l2": CachePartitionLimits(runtime.max_inflight_windows, l2_bytes, 8),
                "l3": CachePartitionLimits(runtime.max_inflight_windows, l3_bytes, 8),
                "l5": CachePartitionLimits(runtime.max_inflight_windows, l5_bytes, 8),
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
            stage_statuses={"l2": "dropped", "l3": "dropped", "l5": "dropped"},
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
        if not self._recording_session_started:
            self._stage_completed_counts["commit"] += 1
            self._prune_completed_stream_history(rejected.session_id)
            return
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

    def _skip_disabled_downstream(
        self,
        key: WindowKey,
        *,
        include_l3: bool,
        reason: str = "downstream_disabled_by_test_ui",
    ) -> None:
        """Terminate disabled Test-UI downstream stages as normal skipped work."""

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
        l5 = L5StageResult.terminal(
            key,
            StageState.SKIPPED,
            reason,
            started_monotonic_ns=monotonic_ns(),
            finished_monotonic_ns=monotonic_ns(),
        )
        self._joiner_submit(lambda: self._result_joiner.submit_l5(l5))
        self._stage_completed_counts["l5"] += 1
        self._record_l5_terminal(StageState.SKIPPED)

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
                l5 = L5StageResult.terminal(
                    displaced.work_item.key,
                    StageState.DROPPED,
                    "l3_admission_queue_overflow",
                    started_monotonic_ns=now_ns,
                    finished_monotonic_ns=now_ns,
                )
                self._joiner_submit(lambda: self._result_joiner.submit_l3(l3))
                self._joiner_submit(lambda: self._result_joiner.submit_l5(l5))
                self._stage_completed_counts["l3"] += 1
                self._stage_completed_counts["l5"] += 1
                self._record_processing_drop(displaced.work_item.key)
        return False

    def _enqueue_l5_latest(self, item: _L5Work) -> bool:
        """Latest-wins enqueue; completed L3 remains authoritative."""

        while not self._processing_abort.is_set():
            try:
                self._l5_windows.put_nowait(item)
                return True
            except queue.Full:
                try:
                    displaced = self._l5_windows.get_nowait()
                except queue.Empty:
                    continue
                if not isinstance(displaced, _L5Work):
                    continue
                now_ns = monotonic_ns()
                l5 = L5StageResult.terminal(
                    displaced.work_item.key,
                    StageState.DROPPED,
                    "l5_admission_queue_overflow",
                    started_monotonic_ns=now_ns,
                    finished_monotonic_ns=now_ns,
                )
                self._joiner_submit(lambda: self._result_joiner.submit_l5(l5))
                self._stage_completed_counts["l5"] += 1
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
            self._l3_prepared_windows,
            self._l3_host_windows,
            self._l5_windows,
            self._confirmed_backfill_work,
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
                finished_ns = monotonic_ns()
                pause_started_ns = self._pipeline_stage_pause_started_ns[stage]
                if pause_started_ns is not None:
                    self._pipeline_stage_excluded_ns[stage] += max(
                        0, finished_ns - pause_started_ns
                    )
                    self._pipeline_stage_pause_started_ns[stage] = None
                self._pipeline_stage_finished_ns[stage] = finished_ns

    def reset_pipeline_total_durations(self) -> None:
        """Start a fresh interactive replay timing generation."""

        with self._pipeline_duration_lock:
            self._pipeline_timing_generation += 1
            self._pipeline_timing_sealed = False
            self._pipeline_first_queued_ns = None
            self._pipeline_stage_finished_ns = {
                "l2": None, "l3": None, "l5": None,
            }
            self._pipeline_stage_pause_started_ns = {
                "l2": None, "l3": None, "l5": None,
            }
            self._pipeline_stage_excluded_ns = {
                "l2": 0, "l3": 0, "l5": 0,
            }

    def set_pipeline_timing_paused(
        self,
        reason: str,
        paused: bool,
        *,
        stages: tuple[str, ...] = ("l2", "l3", "l5"),
    ) -> None:
        """Exclude an operator-controlled pause from selected stage timers."""

        if not reason:
            raise ValueError("pipeline timing pause reason cannot be empty")
        if type(paused) is not bool:
            raise ValueError("pipeline timing paused setting must be bool")
        if not stages or any(stage not in self._pipeline_stage_finished_ns for stage in stages):
            raise ValueError("pipeline timing stages must be a non-empty L2/L3/L5 subset")
        now_ns = monotonic_ns()
        with self._pipeline_duration_lock:
            for stage in stages:
                reasons = self._pipeline_stage_pause_reasons[stage]
                if paused:
                    if reason in reasons:
                        continue
                    if (
                        not reasons
                        and self._pipeline_first_queued_ns is not None
                        and self._pipeline_stage_finished_ns[stage] is None
                    ):
                        self._pipeline_stage_pause_started_ns[stage] = now_ns
                    reasons.add(reason)
                else:
                    if reason not in reasons:
                        continue
                    reasons.remove(reason)
                    pause_started_ns = self._pipeline_stage_pause_started_ns[stage]
                    if not reasons and pause_started_ns is not None:
                        self._pipeline_stage_excluded_ns[stage] += max(
                            0, now_ns - pause_started_ns
                        )
                        self._pipeline_stage_pause_started_ns[stage] = None

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
        """Elapsed first-queue-to-drain time for L2/L3/L5 in the current run."""

        with self._pipeline_duration_lock:
            started_ns = self._pipeline_first_queued_ns
            finished_ns = dict(self._pipeline_stage_finished_ns)
            excluded_ns = dict(self._pipeline_stage_excluded_ns)
            pause_started_ns = dict(self._pipeline_stage_pause_started_ns)
        if started_ns is None:
            return {"l2": None, "l3": None, "l5": None}
        now_ns = monotonic_ns()
        durations: dict[str, float | None] = {}
        for stage in ("l2", "l3", "l5"):
            end_ns = finished_ns[stage]
            worker = self._processing_threads.get(stage)
            if end_ns is None and (worker is None or not worker.is_alive()):
                durations[stage] = None
            else:
                effective_end_ns = end_ns or now_ns
                excluded = excluded_ns[stage]
                if pause_started_ns[stage] is not None:
                    excluded += max(0, effective_end_ns - pause_started_ns[stage])
                durations[stage] = max(
                    0.0, (effective_end_ns - started_ns - excluded) / 1e9
                )
        return durations

    @property
    def processing_queue_depths(self) -> dict[str, int]:
        """A stable diagnostics view; callers never inspect queue internals."""

        return {
            "l2": self._l2_windows.qsize(),
            "l3": self._l3_windows.qsize(),
            "l3_prepared": self._l3_prepared_windows.qsize(),
            "l3_host": self._l3_host_windows.qsize(),
            "l5": self._l5_windows.qsize(),
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
        l5_diagnostics = self._l5_diagnostic_snapshot()
        return {
            "devices": {
                "l1": "cpu",
                "l2": "cpu",
                "l3": self.l3_device,
                "l4": self.l4_device,
                "l5": self.l5_device,
            },
            "queue_depths": self.processing_queue_depths,
            "queue_capacities": {
                "l2": self._l2_windows.maxsize,
                "l3": self._l3_windows.maxsize,
                "l3_prepared": self._l3_prepared_windows.maxsize,
                "l3_host": self._l3_host_windows.maxsize,
                "l5": self._l5_windows.maxsize,
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
            "l5_actual_completed": l5_diagnostics["actual_completed"],
            "l5_dropped": l5_diagnostics["dropped"],
            "l5_skipped": l5_diagnostics["skipped"],
            "l5_actual_hz": l5_diagnostics["actual_hz"],
            "l5_ui_mailbox_depth": self.latest_l5_dev_ui.qsize(),
            "l5_ui_mailbox_capacity": self.latest_l5_dev_ui.maxsize,
            "l5_ui_mailbox_overwrites": l5_diagnostics["ui_mailbox_overwrites"],
            "l2_ui_mailbox_depth": self.latest_l2_dev_ui.qsize(),
            "l2_ui_mailbox_capacity": self.latest_l2_dev_ui.maxsize,
            "l2_ui_mailbox_overwrites": self._l2_ui_overwrites,
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
    def doa_backend(self) -> str:
        with self._scan_config_lock:
            return self._scan_config.scanner_backend

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
    def l5_input_gain_compensation_enabled(self) -> bool:
        return self.track_audio_stream.gain_compensation_enabled

    def set_l5_input_gain_compensation_enabled(self, value: bool) -> bool:
        return self.track_audio_stream.set_gain_compensation_enabled(value)

    def set_development_l4_reference_track(self, track_id: int) -> int:
        resolved = int(track_id)
        if resolved not in {0, -1, -2, -3}:
            raise ValueError("unsupported Development UI L4 reference track")
        self._development_l4_reference_track_id = resolved
        return resolved

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
        """Whether L2 results are allowed to enter the L3/L5 stages."""

        return self._downstream_processing_enabled.is_set()

    def set_downstream_processing_enabled(self, value: bool) -> bool:
        """Enable or bypass L3/L5 without stopping L1/L2 or breaking joins."""

        if type(value) is not bool:
            raise ValueError("downstream processing setting must be bool")
        if value and self._ephemeral_live_capture and not self._ephemeral_recording_active:
            value = False
        if value:
            self._downstream_processing_enabled.set()
        else:
            self._downstream_processing_enabled.clear()
        return value

    @property
    def runtime_recording_active(self) -> bool:
        if self._ephemeral_live_capture:
            return self._ephemeral_recording_active
        return bool(self.recording_store.manual_active)

    @property
    def runtime_recording_mode(self) -> str:
        return "temporary" if self._ephemeral_live_capture else str(self.recording_store.mode)

    @property
    def l1_speaker_count_enabled(self) -> bool:
        return self.speaker_counter.enabled

    def set_l1_speaker_count_enabled(self, value: bool) -> bool:
        return self.speaker_counter.set_enabled(value)

    @property
    def l1_speaker_count_model_error(self) -> str | None:
        return self.speaker_counter.model_error

    @property
    def direction_id_tracking_enabled(self) -> bool:
        with self._scan_config_lock:
            return self._direction_id_tracking_enabled

    def set_direction_id_tracking_enabled(self, value: bool) -> bool:
        if type(value) is not bool:
            raise ValueError("L2 direction ID tracking setting must be bool")
        with self._scan_config_lock:
            if value != self._direction_id_tracking_enabled:
                self._direction_id_tracking_enabled = value
                self._scan_config_revision += 1
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
        self._record_l5_terminal(StageState.DROPPED)

    def _record_l5_terminal(self, state: StageState) -> None:
        """Count actual L5 inference separately from overload terminal markers."""

        now = monotonic()
        with self._l5_diagnostics_lock:
            if state is StageState.COMPLETED:
                self._l5_actual_completed += 1
                self._l5_completion_times.append(now)
            elif state is StageState.DROPPED:
                self._l5_dropped += 1
            elif state is StageState.SKIPPED:
                self._l5_skipped += 1

    def _l5_diagnostic_snapshot(self) -> dict[str, float | int]:
        now = monotonic()
        with self._l5_diagnostics_lock:
            cutoff = now - 1.0
            while self._l5_completion_times and self._l5_completion_times[0] < cutoff:
                self._l5_completion_times.popleft()
            return {
                "actual_completed": self._l5_actual_completed,
                "dropped": self._l5_dropped,
                "skipped": self._l5_skipped,
                "actual_hz": float(len(self._l5_completion_times)),
                "ui_mailbox_overwrites": self._l5_ui_overwrites,
            }

    def _publish_completed_l5_ui(self, item: _L5Work, output: object) -> None:
        """Publish one exact-window L2/L3/L5 frame before ordered commit.

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
            "Layer 5 completed",
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
        tracked_audio = ()
        if self.dev_audio_tracker is not None:
            try:
                tracked_audio = self.dev_audio_tracker.snapshots()
            except Exception as exc:
                self.dev_audio_tracking_error = str(exc)
        frame = DevUiFrame(
            l1=None, spatial_response=response, candidates=candidates,
            previews=previews, tracked_audio=tracked_audio,
            gate_decision=l2_output.gate_decision,
            gate_threshold=float(values["gate_threshold"]),
            gate_config_revision=int(values["gate_config_revision"]),
            direction_threshold=scan.direction_threshold,
            direction_kalman_enabled=bool(values["direction_kalman_enabled"]),
            direction_id_tracking_enabled=bool(
                values["direction_id_tracking_enabled"]
            ),
            direction_kalman_q_scale=float(values["direction_kalman_q_scale"]),
            direction_kalman_r_scale=float(values["direction_kalman_r_scale"]),
            scan_config_revision=int(values["scan_config_revision"]),
            pipeline_status=status, performance=performance,
            published_monotonic=published, spatial_published_monotonic=published,
            search_diagnostics=diagnostics, missing_reasons={}, l5_result=output,
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
            self.latest_l5_dev_ui.put_nowait(frame)
        except queue.Full:
            try:
                self.latest_l5_dev_ui.get_nowait()
            except queue.Empty:
                pass
            with self._l5_diagnostics_lock:
                self._l5_ui_overwrites += 1
            self.latest_l5_dev_ui.put_nowait(frame)

    def _publish_latest_l2_ui(
        self, item: WindowWorkItem, stage: L2StageResult,
    ) -> None:
        """Publish L2 immediately without waiting for L3/L5/ordered commit."""

        values = item.config.values
        scan = DirectionScanConfig(**dict(values["scan_config"]))
        output = stage.output if stage.state is StageState.COMPLETED else None
        response = None if output is None else output.spatial_response
        candidates = () if output is None or response is None else tuple(
            getattr(output, "directions", ()) or output.candidates
        )
        missing_reason = None
        if output is None:
            missing_reason = (
                f"L2 {stage.state.value.upper()}: {stage.error or stage.reason}"
            )
        elif response is None:
            missing_reason = f"UNAVAILABLE: {output.gate_decision.reason}"
        snapshot = L2DevUiSnapshot(
            item.key.session_id,
            item.key.stream_epoch,
            item.key.window_id,
            item.key.decision_sample,
            response,
            candidates,
            None if output is None else output.gate_decision,
            float(values["gate_threshold"]),
            int(values["gate_config_revision"]),
            scan.direction_threshold,
            bool(values["direction_kalman_enabled"]),
            bool(values["direction_id_tracking_enabled"]),
            float(values["direction_kalman_q_scale"]),
            float(values["direction_kalman_r_scale"]),
            int(values["scan_config_revision"]),
            None if output is None else output.search_diagnostics,
            () if output is None else tuple(getattr(output, "directions", ())),
            () if output is None else tuple(getattr(output, "active_tracks", ())),
            monotonic(),
            missing_reason,
        )
        try:
            self.latest_l2_dev_ui.put_nowait(snapshot)
        except queue.Full:
            try:
                self.latest_l2_dev_ui.get_nowait()
            except queue.Empty:
                pass
            self._l2_ui_overwrites += 1
            self.latest_l2_dev_ui.put_nowait(snapshot)

    @staticmethod
    def _track_voice_annotations(
        inputs: tuple[Layer5AudioSegment, ...],
        output: object,
    ) -> tuple[TrackVoiceAnnotation, ...]:
        detections = tuple(getattr(output, "detections", ()))
        if len(inputs) != len(detections):
            raise ValueError("L5 results must align with continuous track audio inputs")
        threshold = float(getattr(output, "threshold"))
        annotations: list[TrackVoiceAnnotation] = []
        for audio, detection in zip(inputs, detections, strict=True):
            if (
                audio.track_id is None
                or detection.track_id != audio.track_id
                or (
                    detection.session_id, detection.stream_epoch,
                    detection.window_id, detection.decision_sample,
                ) != (
                    audio.session_id, audio.stream_epoch,
                    audio.window_id, audio.decision_sample,
                )
                or not np.isclose(
                    detection.theta_deg, audio.theta_deg, atol=1e-6, rtol=0.0
                )
            ):
                raise ValueError("L5 annotation identity/order/theta mismatch")
            end_sample = int(audio.effective_end_sample)
            annotations.append(TrackVoiceAnnotation(
                audio.session_id,
                audio.stream_epoch,
                audio.window_id,
                audio.decision_sample,
                int(audio.track_id),
                end_sample - 960,
                end_sample,
                float(detection.probability),
                bool(detection.is_voice),
                str(detection.model_id),
                threshold,
            ))
        return tuple(annotations)

    def _capture_processing_config(self) -> ProcessingConfigSnapshot:
        with self._gate_config_lock:
            gate_threshold = self._gate_probability_threshold
            gate_revision = self._gate_config_revision
        with self._scan_config_lock:
            scan_config = self._scan_config
            scan_revision = self._scan_config_revision
            kalman_enabled = self._direction_kalman_enabled
            id_tracking_enabled = self._direction_id_tracking_enabled
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
                "music_order": {
                    "source": "test_ui_manual",
                    "value": scan_config.effective_order_limit,
                    "min_valid_frequency_bins": scan_config.min_valid_frequency_bins,
                },
                "association_lifecycle": self.config.layer2.direction_id_tracking.model_dump(),
                "association_config_revision": 0,
                "direction_kalman_enabled": kalman_enabled,
                "direction_id_tracking_enabled": id_tracking_enabled,
                "kalman_config_revision": kalman_revision,
                "direction_kalman_q_scale": q_scale,
                "direction_kalman_r_scale": r_scale,
                "l3_mode": l3_mode,
                "l3_config_revision": l3_revision,
                "l5_threshold": self._layer5.threshold,
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
            self._l2_ui_overwrites = 0
            with self._l5_diagnostics_lock:
                self._l5_actual_completed = 0
                self._l5_dropped = 0
                self._l5_skipped = 0
                self._l5_ui_overwrites = 0
                self._l5_completion_times.clear()
            self._input_exhausted = False
            if self._ephemeral_live_capture:
                self._ephemeral_recording_active = False
                self._downstream_processing_enabled.clear()
            with self._layer2_state_lock:
                self._layer2.reset()
            self._stop.clear()
            self._processing_abort.clear()
            self._discard_processing.clear()
            self._input_worker_done.clear()
            self.coordinator = IngestCoordinator(
                sample_rate=self.config.device.sample_rate,
                timestamp_tolerance_ms=self.config.timing.timestamp_tolerance_ms,
            )
            self.assembler = WindowAssembler()
            self.imcra = Layer1Imcra.from_project(self.config)
            self.pre_denoiser = ImcraWienerPreDenoiser.from_project(self.config)
            self._layer3.clear_cache()
            self._layer3_backfill.clear_cache()
            self.track_audio_stream.reset()
            with self._confirmed_backfill_lock:
                self._confirmed_backfill_history.clear()
                self._confirmed_backfill_ids.clear()
            self._l3_cache_snapshot = None
            self._reset_processing_graph()
            self._stage_errors = {"l2": None, "l3": None, "l5": None, "commit": None}
            self._stage_error_counts = {name: 0 for name in self._stage_errors}
            self._stage_completed_counts = {name: 0 for name in self._stage_errors}
            with self._pipeline_duration_lock:
                self._pipeline_timing_generation += 1
                self._pipeline_timing_sealed = False
                self._pipeline_first_queued_ns = None
                self._pipeline_stage_finished_ns = {
                    "l2": None, "l3": None, "l5": None,
                }
                self._pipeline_stage_pause_started_ns = {
                    "l2": None, "l3": None, "l5": None,
                }
                self._pipeline_stage_excluded_ns = {
                    "l2": 0, "l3": 0, "l5": 0,
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
                self.latest_l2_dev_ui,
                self.latest_l5_dev_ui,
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
                "devices": {
                    "l1": "cpu",
                    "l2": "cpu",
                    "l3": self.l3_device,
                    "l4": self.l4_device,
                    "l5": self.l5_device,
                },
                "processing_architecture": "staged_window_pipeline_v1",
                "window_key": "session_id,stream_epoch,window_id,decision_sample",
                "queue_capacities": {
                    "l2": self.config.runtime.l2_queue_windows,
                    "l3": self.config.runtime.l3_queue_windows,
                    "l5": self.config.runtime.l5_queue_windows,
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
                "layer2_direction_tracker": self.config.layer2.direction_id_tracking.backend,
                "layer2_direction_id_tracking": self.config.layer2.direction_id_tracking.backend,
                "layer3": self.config.layer3.main_backend,
                "feature": self.config.feature.preprocessing_version,
                "layer5_primary": self.config.layer5.primary_model_id,
                "layer5_models": [item.model_id for item in self.config.layer5.models if item.enabled],
            },
        )
        pipeline_start_attempted = False
        started_threads: list[threading.Thread] = []
        try:
            if not self._ephemeral_live_capture:
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
                "l5": threading.Thread(
                    target=self._run_l5, name="application-runtime-l5", daemon=True
                ),
                "backfill": threading.Thread(
                    target=self._run_confirmed_backfill,
                    name="application-runtime-l3-backfill",
                    daemon=True,
                ),
                "commit": threading.Thread(
                    target=self._run_commit, name="application-runtime-commit", daemon=True
                ),
            }
            self._processing_threads["l3_host"] = threading.Thread(
                target=self._run_l3_host,
                name="application-runtime-l3-host",
                daemon=True,
            )
            if self._l3_cuda_stream is None:
                self._processing_threads["l3_prepare"] = threading.Thread(
                    target=self._run_l3_prepare,
                    name="application-runtime-l3-prepare",
                    daemon=True,
                )
            # Compatibility for callers that previously observed one terminal
            # processing thread.  The commit worker remains that owner.
            self._processing_thread = self._processing_threads["commit"]
            start_order = ["commit", "l5", "backfill"]
            start_order.append("l3_host")
            start_order.append("l3")
            if "l3_prepare" in self._processing_threads:
                start_order.append("l3_prepare")
            start_order.append("l2")
            for name in start_order:
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

    def _select_pre_denoise_with_mode(self, block) -> tuple[tuple[object, bool], ...]:
        pairs = self.pre_denoiser.process(block)
        with self._pre_denoise_lock:
            enabled = self._pre_denoise_enabled
            if not self._pre_denoise_latency_active:
                if not enabled:
                    return ((block, False),)
                # Everything before this point has already been published in
                # bypass mode. Start the fixed one-hop latency here and wait
                # for this exact hop's denoised replacement.
                self._pre_denoise_latency_active = True
                return ()
        return tuple(
            (item.denoised if enabled else item.raw, enabled)
            for item in pairs
        )

    def _select_pre_denoise(self, block) -> tuple[object, ...]:
        return tuple(item for item, _denoised in self._select_pre_denoise_with_mode(block))

    def _flush_pre_denoise_with_mode(self) -> tuple[tuple[object, bool], ...]:
        pairs = self.pre_denoiser.flush()
        with self._pre_denoise_lock:
            if not self._pre_denoise_latency_active:
                return ()
            enabled = self._pre_denoise_enabled
        return tuple(
            (item.denoised if enabled else item.raw, enabled)
            for item in pairs
        )

    def _flush_pre_denoise(self) -> tuple[object, ...]:
        return tuple(item for item, _denoised in self._flush_pre_denoise_with_mode())

    def _publish_l1_selection(self, block, received: float, *, denoised: bool) -> None:
        if (
            denoised
            and self.dev_audio_tracker is not None
            and not (
                self._ephemeral_live_capture
                and self._ephemeral_recording_active
                and not self._development_center_l4_bypass
            )
        ):
            try:
                self.dev_audio_tracker.append_imcra_center_reference(
                    block, channel_index=6,
                )
                self.dev_audio_tracker.append_imcra_hardware_mix_reference(
                    block, channel_index=7,
                )
            except Exception as exc:
                self.dev_audio_tracking_error = f"IMCRA center reference: {exc}"
        self._publish_l1_block(block, received)

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
        if self._recording_session_started:
            self.recording_store.append_audio(runtime_recording_block)
        with self._pre_denoise_lock:
            enabled = self._pre_denoise_enabled and self._pre_denoise_latency_active
        snapshot = self.meter.add(
            block,
            light_state=self.light_state,
            recording_state=self.scratch.state,
            pre_denoise_enabled=enabled,
            pre_denoise_mean_gain_db=self.pre_denoiser.last_mean_gain_db if enabled else 0.0,
            speaker_count_annotation=(
                annotation
                if (
                    (annotation := self.speaker_counter.latest()) is not None
                    and (annotation.session_id, annotation.stream_epoch)
                    == (block.session_id, block.stream_epoch)
                    and annotation.end_sample <= block.end_sample
                )
                else None
            ),
        )
        self._latest(self.latest_l1, snapshot)
        with self._ui_lock:
            previous_stream = getattr(self._ui_aggregator, "current_stream", None)
            next_stream = (block.session_id, block.stream_epoch)
            stream_changed = previous_stream is not None and previous_stream != next_stream
            if stream_changed:
                while True:
                    try:
                        self.latest_l5_dev_ui.get_nowait()
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
            if self._development_center_l4_bypass and self.dev_audio_tracker is not None:
                try:
                    retained_audio = self.dev_audio_tracker.snapshots()
                except Exception as exc:
                    self.dev_audio_tracking_error = str(exc)
                else:
                    frame = self._ui_aggregator.update_l3(
                        (),
                        "L2/L3 disabled: reference audio goes directly to L4",
                        tracked_audio=retained_audio,
                    )
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
            if not self._development_center_l4_bypass:
                self._admit_window(window)

    def _admit_window(self, window: DecisionWindow) -> bool:
        self._remember_confirmed_backfill_window(window)
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
                started_ns = monotonic_ns()
                self._pipeline_first_queued_ns = started_ns
                for stage, reasons in self._pipeline_stage_pause_reasons.items():
                    if reasons:
                        self._pipeline_stage_pause_started_ns[stage] = started_ns
                started_timing = True
        try:
            self._l2_windows.put_nowait(work_item)
            return True
        except queue.Full:
            if started_timing:
                with self._pipeline_duration_lock:
                    self._pipeline_first_queued_ns = None
                    self._pipeline_stage_pause_started_ns = {
                        "l2": None, "l3": None, "l5": None,
                    }
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

    def _remember_confirmed_backfill_window(self, window: DecisionWindow) -> None:
        """Retain only 1.2 s of immutable L1 windows for confirmed-ID recovery."""

        stream = (window.session_id, window.stream_epoch)
        with self._confirmed_backfill_lock:
            if self._confirmed_backfill_history and (
                self._confirmed_backfill_history[-1].session_id,
                self._confirmed_backfill_history[-1].stream_epoch,
            ) != stream:
                self._confirmed_backfill_history.clear()
            self._confirmed_backfill_history.append(window)

    def _schedule_confirmed_backfill(
        self,
        item: WindowWorkItem,
        active_tracks: tuple[object, ...],
        *,
        processing_mode: str,
        l2_direction_count: int,
    ) -> None:
        """Queue one historical BF request at an ID's first confirmed result."""

        if not self.downstream_processing_enabled:
            return
        for track in active_tracks:
            if str(getattr(track, "track_state", "")) != "confirmed":
                continue
            key = (item.key.session_id, item.key.stream_epoch, int(track.track_id))
            with self._confirmed_backfill_lock:
                if key in self._confirmed_backfill_ids:
                    continue
                target_start = max(
                    0, item.key.decision_sample - _CONFIRMED_BACKFILL_SAMPLES
                )
                windows = tuple(
                    window
                    for window in self._confirmed_backfill_history
                    if (window.session_id, window.stream_epoch) == key[:2]
                    and window.decision_sample <= item.key.decision_sample
                    # Realtime L3 already owns every slot from the ID's first
                    # tentative observation onward. Historical recovery only
                    # fills audio before that authoritative ID existed.
                    and window.decision_sample
                    < int(getattr(track, "first_seen_sample"))
                    and window.decision_sample - 2 * self.config.timing.decision_hop_samples
                    >= target_start
                )
                self._confirmed_backfill_ids.add(key)
            if not windows:
                continue
            work = _ConfirmedBackfillWork(
                track,
                windows,
                processing_mode,
                max(1, min(3, int(l2_direction_count))),
            )
            while not self._processing_abort.is_set():
                try:
                    self._confirmed_backfill_work.put(work, timeout=0.05)
                    break
                except queue.Full:
                    worker = self._processing_threads.get("backfill")
                    if worker is not None and worker.ident is not None and not worker.is_alive():
                        self._processing_abort.set()
                        break

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
                # Non-blocking enqueue only. CountNet owns resampling, buffering,
                # model loading and inference on its independent L1 worker.
                self.speaker_counter.submit(block)
                if (
                    self.dev_audio_tracker is not None
                    and not (
                        self._ephemeral_live_capture
                        and self._ephemeral_recording_active
                        and not self._development_center_l4_bypass
                    )
                ):
                    try:
                        # Raw logical channel 6 is the Center microphone.  It
                        # is cached before optional L1 pre-denoise so the first
                        # L3 listening row is a true input reference.
                        self.dev_audio_tracker.append_center_reference(block, channel_index=6)
                        self.dev_audio_tracker.append_hardware_mix_reference(
                            block, channel_index=7,
                        )
                    except Exception as exc:
                        self.dev_audio_tracking_error = f"center reference: {exc}"
                received = monotonic()
                for selected, denoised in self._select_pre_denoise_with_mode(block):
                    self._publish_l1_selection(selected, received, denoised=denoised)
        except Exception as exc:
            self.last_error = str(exc)
        finally:
            try:
                for selected, denoised in self._flush_pre_denoise_with_mode():
                    self._publish_l1_selection(selected, monotonic(), denoised=denoised)
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
                # applied snapshot to L3/L5, UI, and recording.
                with self._scan_config_lock:
                    live_scan_config = self._scan_config
                    live_scan_revision = self._scan_config_revision
                    live_id_tracking_enabled = self._direction_id_tracking_enabled
                values = dict(item.config.values)
                values["scan_config"] = asdict(live_scan_config)
                values["scan_config_revision"] = live_scan_revision
                values["music_config_revision"] = live_scan_revision
                values["direction_id_tracking_enabled"] = live_id_tracking_enabled
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
                    with self._layer2_state_lock:
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
                            direction_id_tracking_enabled=bool(
                                values["direction_id_tracking_enabled"]
                            ),
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
                try:
                    self._publish_latest_l2_ui(item, stage)
                except Exception as exc:
                    # This mailbox is visualization-only. A rejected UI
                    # projection must not turn a valid formal L2 result into
                    # an algorithm or recording failure.
                    self.dev_ui_error = f"latest L2 UI: {exc}"
                self._joiner_submit(lambda: self._result_joiner.submit_l2(stage))
                if (
                    stage.state is StageState.COMPLETED
                    and bool(values["direction_id_tracking_enabled"])
                ):
                    l2_directions = tuple(
                        getattr(stage.output, "directions", ())
                        or stage.output.candidates
                    )
                    active_tracks = tuple(
                        getattr(stage.output, "active_tracks", ())
                    )
                    try:
                        self.track_audio_stream.observe_l2(
                            identity=(
                                item.key.session_id,
                                item.key.stream_epoch,
                                item.key.window_id,
                                item.key.decision_sample,
                            ),
                            active_tracks=active_tracks,
                            processing_mode=str(values["l3_mode"]),
                            l2_direction_count=len(l2_directions),
                        )
                        self._schedule_confirmed_backfill(
                            item,
                            active_tracks,
                            processing_mode=str(values["l3_mode"]),
                            l2_direction_count=len(l2_directions),
                        )
                    except Exception as exc:
                        self.dev_audio_tracking_error = f"L2 timeline: {exc}"
                if stage.state is not StageState.COMPLETED:
                    self._record_l5_terminal(StageState.SKIPPED)
                    self._joiner_submit(
                        lambda: self._result_joiner.skip_missing_downstream(
                            item.key, "l2_failed"
                        )
                    )
                elif (
                    stage.output.spatial_response is None
                    and not tuple(getattr(stage.output, "directions", ()))
                ):
                    self._record_l5_terminal(StageState.SKIPPED)
                    self._joiner_submit(
                        lambda: self._result_joiner.skip_missing_downstream(
                            item.key, f"gate_{stage.output.gate_decision.state.value}"
                        )
                    )
                elif not self.downstream_processing_enabled:
                    self._skip_disabled_downstream(item.key, include_l3=True)
                elif not bool(values["direction_id_tracking_enabled"]):
                    self._skip_disabled_downstream(
                        item.key,
                        include_l3=True,
                        reason="direction_id_tracking_disabled_by_test_ui",
                    )
                else:
                    if not self._enqueue_l3_latest(_L3Work(item, stage)):
                        break
        except Exception as exc:
            self._set_stage_error("l2", exc)
            self._processing_abort.set()
        finally:
            if drained:
                self._mark_pipeline_stage_finished("l2")
            self._put_eos_interruptibly(self._confirmed_backfill_work, "backfill")
            self._put_eos_interruptibly(self._l3_windows, "l3")

    def _run_confirmed_backfill(self) -> None:
        """Recover only missing pre-confirmation L3 slots on a side worker."""

        try:
            while True:
                try:
                    item = self._confirmed_backfill_work.get(timeout=0.1)
                except queue.Empty:
                    if self._processing_abort.is_set():
                        break
                    continue
                if item is _PIPELINE_EOS:
                    break
                if not isinstance(item, _ConfirmedBackfillWork):
                    continue
                try:
                    self._process_confirmed_backfill(item)
                except Exception as exc:
                    # Historical recovery is additive. A failure must remain
                    # visible, but cannot invalidate realtime L1-L3 or stop a
                    # later ID from attempting its own recovery.
                    self.dev_audio_tracking_error = f"confirmed L3 backfill: {exc}"
                finally:
                    self._layer3_backfill.clear_cache()
        finally:
            try:
                self._layer3_backfill.clear_cache()
            except Exception:
                pass

    def _process_confirmed_backfill(self, item: _ConfirmedBackfillWork) -> None:
        track = item.track
        decisions = self.track_audio_stream.missing_backfill_decisions(
            session_id=str(getattr(track, "session_id")),
            stream_epoch=int(getattr(track, "stream_epoch")),
            track_id=int(getattr(track, "track_id")),
            processing_mode=item.processing_mode,
            decision_samples=tuple(window.decision_sample for window in item.windows),
        )
        wanted = set(decisions)
        audio_windows: list[TrackAudioWindow] = []
        for window in item.windows:
            if self._processing_abort.is_set():
                break
            if window.decision_sample not in wanted:
                continue
            theta = float(getattr(track, "theta_deg"))
            candidate = replace(
                track,
                window_id=window.window_id,
                decision_sample=window.decision_sample,
                doa_start_sample=window.doa_start_sample,
                doa_end_sample=window.doa_end_sample,
                rank=1,
                measured_theta_deg=theta,
                theta_deg=theta,
                track_state="confirmed",
                is_observed=True,
                is_new_track=False,
                first_seen_sample=min(
                    int(getattr(track, "first_seen_sample")), window.decision_sample,
                ),
                last_observed_sample=window.decision_sample,
                missed_samples=0,
            )
            output = self._layer3_backfill.process(
                window, (candidate,), self._geometry, mode=item.processing_mode,
            )
            self._validate_direction_outputs(
                "L3 backfill", (candidate,), output.enhanced_audio
            )
            enhanced = output.enhanced_audio[0]
            audio_windows.append(TrackAudioWindow(
                enhanced.session_id,
                enhanced.stream_epoch,
                enhanced.window_id,
                enhanced.decision_sample,
                int(enhanced.track_id),
                enhanced.theta_deg,
                enhanced.enhanced_audio,
                self._context_probabilities_20ms(window),
                item.processing_mode,
            ))
        missing = self.track_audio_stream.missing_backfill_windows(
            tuple(audio_windows)
        )
        inserted = self.track_audio_stream.insert_backfill(
            missing, l2_direction_count=item.l2_direction_count,
        )
        if inserted and self.dev_audio_tracker is not None:
            self.dev_audio_tracker.prepend_backfill_hops(
                inserted,
                authoritative_track=track,
                processing_mode=missing[0].processing_mode,
            )

    def _complete_l3_work(
        self,
        item: _L3Work,
        started_ns: int,
        candidates: tuple[object, ...],
        result: object,
    ) -> bool:
        """Finish host validation, ID stitching and publication for one L3 window."""
        try:
            if isinstance(result, Exception):
                raise result
            if not isinstance(result, Layer3Output):
                result = self._layer3.finalize_host_output(result)
            output = result
            assert item.l2.output is not None
            self._validate_direction_outputs("L3", candidates, output.enhanced_audio)
            probability_slots = self._context_probabilities_20ms(item.work_item.window)
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
                    str(item.work_item.config.values["l3_mode"]),
                )
                for enhanced in output.enhanced_audio
            )
            active_tracks = tuple(getattr(item.l2.output, "active_tracks", ()))
            active_ids = tuple(track.track_id for track in active_tracks)
            context_track_ids: tuple[int, ...] | None
            if self.dev_audio_tracker is None:
                # Realtime L5 is deliberately deferred; the Hub archive owns
                # the full exact-ID stream for offline L4/L5, so no rolling
                # 3.2 s context needs to be rebuilt for every 20 ms window.
                context_track_ids = ()
            else:
                required_contexts = getattr(
                    self.dev_audio_tracker, "required_context_track_ids", None
                )
                context_track_ids = (
                    tuple(required_contexts(
                        item.work_item.key.session_id,
                        item.work_item.key.stream_epoch,
                        active_tracks,
                    ))
                    if callable(required_contexts)
                    else None
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
                l2_direction_count=len(candidates),
                context_track_ids=context_track_ids,
            )
            if self.dev_audio_tracker is not None:
                try:
                    self.dev_audio_tracker.consume_stream_batch(
                        audio_batch, active_tracks=active_tracks
                    )
                    self.dev_audio_tracking_error = None
                except Exception as exc:
                    self.dev_audio_tracking_error = str(exc)
            stage = L3StageResult.completed(
                item.work_item.key,
                output,
                started_monotonic_ns=started_ns,
                finished_monotonic_ns=monotonic_ns(),
            )
            self._stage_errors["l3"] = None
        except Exception as exc:
            self._set_stage_error("l3", exc)
            if self._l3_cuda_stream is not None:
                self._layer3.clear_cache()
            stage = L3StageResult.terminal(
                item.work_item.key,
                StageState.FAILED,
                "l3_failed",
                started_monotonic_ns=started_ns,
                finished_monotonic_ns=monotonic_ns(),
                error=str(exc),
            )
        self._stage_completed_counts["l3"] += 1
        try:
            self._l3_cache_snapshot = self._layer3.cache_snapshot()
        except Exception:
            self._l3_cache_snapshot = None
        self._cache_publish("l3", item.work_item.key, "stage_result", stage)
        if stage.state is StageState.COMPLETED:
            latest_hop_by_id = {
                hop.track_id: hop
                for hop in audio_batch.emitted_hops
                if hop.waveform is not None
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
                            self.track_audio_stream.gain_compensation_enabled
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
            return self._enqueue_l5_latest(_L5Work(item.work_item, item.l2, stage))
        skipped = L5StageResult.terminal(
            item.work_item.key,
            StageState.SKIPPED,
            "l3_failed",
            started_monotonic_ns=monotonic_ns(),
            finished_monotonic_ns=monotonic_ns(),
        )
        self._record_l5_terminal(StageState.SKIPPED)
        self._joiner_submit(lambda: self._result_joiner.submit_l5(skipped))
        return True

    def _run_l3_prepare(self) -> None:
        """Prepare ordered CPU L3 contexts ahead of candidate-dependent BF."""

        try:
            while True:
                try:
                    item = self._l3_windows.get(timeout=0.1)
                except queue.Empty:
                    if self._processing_abort.is_set():
                        break
                    continue
                if item is _PIPELINE_EOS:
                    break
                if isinstance(item, _PipelineTimingBarrier):
                    self._put_pipeline_timing_barrier_interruptibly(
                        self._l3_prepared_windows, "l3", item
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
                candidates: tuple[object, ...] = ()
                try:
                    assert item.l2.output is not None
                    candidates = tuple(
                        getattr(item.l2.output, "directions", ())
                        or item.l2.output.candidates
                    )
                    if not candidates:
                        result: object = Layer3Output(())
                    else:
                        mode = str(item.work_item.config.values["l3_mode"])
                        prepare_kwargs: dict[str, object] = {"mode": mode}
                        if isinstance(self._layer3, Layer3Processor):
                            prepare_kwargs["defer_device_validation"] = True
                        result = self._layer3.prepare(
                            item.work_item.window, **prepare_kwargs
                        )
                except Exception as exc:
                    result = exc
                prepared = _L3PreparedWork(
                    item, started_ns, candidates, result
                )
                while not self._processing_abort.is_set():
                    try:
                        self._l3_prepared_windows.put(prepared, timeout=0.05)
                        break
                    except queue.Full:
                        downstream = self._processing_threads.get("l3")
                        if downstream is not None and not downstream.is_alive():
                            self._processing_abort.set()
                            break
        except Exception as exc:
            self._set_stage_error("l3", exc)
            self._processing_abort.set()
        finally:
            self._put_eos_interruptibly(self._l3_prepared_windows, "l3")

    def _run_l3(self) -> None:
        deferred_items: deque[object] = deque()
        input_queue = (
            self._l3_windows
            if self._l3_cuda_stream is not None
            else self._l3_prepared_windows
        )
        try:
            while True:
                candidates: tuple[object, ...] = ()
                try:
                    item = (
                        deferred_items.popleft()
                        if deferred_items
                        else input_queue.get(timeout=0.1)
                    )
                except queue.Empty:
                    if self._processing_abort.is_set():
                        break
                    continue
                if item is _PIPELINE_EOS:
                    break
                if isinstance(item, _PipelineTimingBarrier):
                    self._put_pipeline_timing_barrier_interruptibly(
                        self._l3_host_windows, "l3_host", item
                    )
                    continue
                if self._processing_abort.is_set():
                    break
                if self._l3_cuda_stream is None:
                    if not isinstance(item, _L3PreparedWork):
                        continue
                    result = item.result
                    if not isinstance(result, (Layer3Output, Exception)):
                        try:
                            result = self._layer3.process_prepared(
                                result, item.candidates, self._geometry
                            )
                        except Exception as exc:
                            result = exc
                    host_work = _L3HostWork(
                        item.work,
                        item.started_ns,
                        item.candidates,
                        result,
                        None,
                    )
                    while not self._processing_abort.is_set():
                        try:
                            self._l3_host_windows.put(host_work, timeout=0.05)
                            break
                        except queue.Full:
                            host = self._processing_threads.get("l3_host")
                            if host is not None and not host.is_alive():
                                self._processing_abort.set()
                                break
                    continue
                if not isinstance(item, _L3Work):
                    continue
                if not self.downstream_processing_enabled:
                    self._skip_disabled_downstream(
                        item.work_item.key, include_l3=True
                    )
                    continue
                cuda_batch_api = all(callable(getattr(self._layer3, name, None)) for name in (
                    "process_prepared_device", "stage_host_transfer", "finalize_host_output",
                ))
                if (
                    self._l3_cuda_stream is not None
                    and cuda_batch_api
                ):
                    # CUDA kernels here are intentionally small.  When L3 is
                    # backlogged, submit several complete windows and perform
                    # one stream synchronization/host handoff for the group.
                    # The production profile never waits for a future item;
                    # only an already-observed backlog is coalesced.
                    works = [item]
                    deadline = monotonic() + (
                        self.config.runtime.l3_cuda_batch_wait_ms / 1_000.0
                    )
                    while len(works) < self.config.runtime.l3_cuda_microbatch_windows:
                        try:
                            following = self._l3_windows.get_nowait()
                        except queue.Empty:
                            remaining = deadline - monotonic()
                            if remaining <= 0:
                                break
                            try:
                                following = self._l3_windows.get(timeout=remaining)
                            except queue.Empty:
                                break
                        if isinstance(following, _L3Work):
                            works.append(following)
                        else:
                            deferred_items.append(following)
                            break

                    staged: list[tuple[_L3Work, int, tuple[object, ...], object]] = []
                    with torch.cuda.stream(self._l3_cuda_stream):
                        for work in works:
                            work_started_ns = monotonic_ns()
                            try:
                                assert work.l2.output is not None
                                work_candidates = tuple(
                                    getattr(work.l2.output, "directions", ())
                                    or work.l2.output.candidates
                                )
                                if not work_candidates:
                                    result: object = Layer3Output(())
                                else:
                                    mode = str(work.work_item.config.values["l3_mode"])
                                    prepared = self._layer3.prepare(
                                        work.work_item.window,
                                        mode=mode,
                                        defer_device_validation=True,
                                    )
                                    device_output = self._layer3.process_prepared_device(
                                        prepared, work_candidates, self._geometry
                                    )
                                    result = self._layer3.stage_host_transfer(device_output)
                            except Exception as exc:
                                work_candidates = ()
                                result = exc
                            staged.append((
                                work, work_started_ns, work_candidates, result,
                            ))
                        ready_event = torch.cuda.Event()
                        ready_event.record(self._l3_cuda_stream)
                    for work, work_started_ns, work_candidates, result in staged:
                        while not self._processing_abort.is_set():
                            try:
                                self._l3_host_windows.put(
                                    _L3HostWork(
                                        work,
                                        work_started_ns,
                                        work_candidates,
                                        result,
                                        ready_event,
                                    ),
                                    timeout=0.05,
                                )
                                break
                            except queue.Full:
                                host = self._processing_threads.get("l3_host")
                                if host is not None and not host.is_alive():
                                    self._processing_abort.set()
                                    break
                    continue

                started_ns = monotonic_ns()
                try:
                    assert item.l2.output is not None
                    candidates = tuple(
                        getattr(item.l2.output, "directions", ())
                        or item.l2.output.candidates
                    )
                    mode = str(item.work_item.config.values["l3_mode"])
                    if not candidates:
                        # A valid SRP response can have no accepted peaks.  Do
                        # not pay the 160 ms STFT/covariance preparation cost
                        # when there is no direction to synthesize.
                        output = Layer3Output(())
                    else:
                        # Runtime validates the final host waveform and public
                        # direction contract below. Re-scanning every trusted
                        # intermediate tensor (STFT, covariance, weights and
                        # ISTFT) repeats the same O(n) work and, on CUDA,
                        # introduces synchronization points.
                        prepared = self._layer3.prepare(
                            item.work_item.window,
                            mode=mode,
                            defer_device_validation=True,
                        )
                        output = self._layer3.process_prepared(prepared, candidates, self._geometry)
                    if not self._complete_l3_work(item, started_ns, candidates, output):
                        break
                    continue
                except Exception as exc:
                    if not self._complete_l3_work(
                        item, started_ns, candidates, exc
                    ):
                        break
        except Exception as exc:
            self._set_stage_error("l3", exc)
            self._processing_abort.set()
        finally:
            self._put_eos_interruptibly(self._l3_host_windows, "l3_host")

    def _run_l3_host(self) -> None:
        """Finalize asynchronous CUDA L3 results and run CPU-only stitching."""
        drained = False
        try:
            while True:
                try:
                    item = self._l3_host_windows.get(timeout=0.1)
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
                        self._l5_windows, "l5", item
                    )
                    continue
                if self._processing_abort.is_set():
                    break
                if not isinstance(item, _L3HostWork):
                    continue
                result = item.result
                try:
                    if item.ready_event is not None:
                        item.ready_event.synchronize()
                except Exception as exc:
                    result = exc
                if not self._complete_l3_work(
                    item.work,
                    item.started_ns,
                    item.candidates,
                    result,
                ):
                    break
        except Exception as exc:
            self._set_stage_error("l3", exc)
            self._processing_abort.set()
        finally:
            if drained:
                self._mark_pipeline_stage_finished("l3")
            self._put_eos_interruptibly(self._l5_windows, "l5")

    def _run_l5(self) -> None:
        drained = False
        try:
            while True:
                try:
                    item = self._l5_windows.get(timeout=0.1)
                except queue.Empty:
                    if self._processing_abort.is_set():
                        break
                    continue
                if item is _PIPELINE_EOS:
                    drained = True
                    break
                if isinstance(item, _PipelineTimingBarrier):
                    self._mark_pipeline_stage_finished("l5", item.generation)
                    continue
                if self._processing_abort.is_set():
                    break
                if not isinstance(item, _L5Work):
                    continue
                if not self.downstream_processing_enabled:
                    self._skip_disabled_downstream(
                        item.work_item.key, include_l3=False
                    )
                    continue
                started_ns = monotonic_ns()
                # L5 is deliberately deferred until TrackAudioStreamHub has
                # sealed the complete ID streams and offline L4 has performed
                # speaker-count routing/separation.  Keeping this explicit
                # terminal stage preserves ordered per-window audit records
                # without executing the CNN on partial real-time context.
                stage = L5StageResult.terminal(
                    item.work_item.key, StageState.SKIPPED, "offline_after_l4",
                    started_monotonic_ns=started_ns,
                    finished_monotonic_ns=monotonic_ns(),
                )
                self._stage_errors["l5"] = None
                self._stage_completed_counts["l5"] += 1
                self._record_l5_terminal(stage.state)
                self._cache_publish("l5", item.work_item.key, "stage_result", stage)
                self._joiner_submit(lambda: self._result_joiner.submit_l5(stage))
        except Exception as exc:
            self._set_stage_error("l5", exc)
            self._processing_abort.set()
        finally:
            if drained:
                self._mark_pipeline_stage_finished("l5")
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
                if self._discard_processing.is_set():
                    pending.clear()
                    update_pending_pressure()
                    break
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
                    # EOS is ordered after L5 production but may overtake the
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
                    l5 = self._processing_threads.get("l5")
                    if l5 is None or (l5.ident is not None and not l5.is_alive()):
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
            f"l2_chain_backend={dict(values['scan_config'])['scanner_backend']}",
            f"l2_gate_backend={gate.backend}",
            f"l2_gate_state={gate.state.value}",
            f"l2_gate_reason={gate.reason}",
            f"l2_gate_probability_40ms={gate.probability_40ms}",
            f"l2_gate_threshold={gate.threshold}",
            f"l2_gate_config_revision={gate.config_revision}",
            f"l2_direction_tracker_backend={self.config.layer2.direction_id_tracking.backend}",
            "l2_direction_imm=intrinsic",
            f"l2_direction_id_tracking_backend={getattr(self._layer2.id_tracker, 'backend', self.config.layer2.direction_id_tracking.backend)}",
            f"l2_direction_id_tracking_enabled={values['direction_id_tracking_enabled']}",
            f"l2_direction_identity={'authoritative' if values['direction_id_tracking_enabled'] else 'raw_doa_peaks'}",
            f"l2_direction_id_active_tracks={self._layer2.id_tracker.active_track_count}",
            f"l2_direction_id_tracking_error={self._layer2.last_id_tracking_error}",
            *(() if not is_music else (
                f"l2_music_algorithm={music.algorithm_version}",
                f"l2_music_valid_frequency_bins={music.valid_frequency_bins}",
                f"l2_music_covariance_quality={music.covariance_quality}",
                f"l2_music_manual_order={music.model_order.estimated_sources}",
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
        l5_result = joined.l5.output if joined.l5.state is StageState.COMPLETED else None
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
        track_voice_annotations = self._compute_cache.get(
            "l5", joined.key, "track_voice_annotations"
        ) or ()
        annotation_by_id = {item.track_id: item for item in track_voice_annotations}
        if len(annotation_by_id) != len(track_voice_annotations):
            raise ValueError("L5 track voice annotations contain duplicate IDs")
        enhanced_records = tuple(
            {
                **item[0],
                **(
                    {
                        "l5_voice_20ms": asdict(annotation_by_id[int(item[0]["track_id"])])
                    }
                    if int(item[0]["track_id"]) in annotation_by_id
                    else {}
                ),
            }
            for item in track_audio_record
        )
        for audio_meta in enhanced_records:
            annotation = audio_meta.get("l5_voice_20ms")
            if annotation is not None and (
                int(annotation["start_sample"]) != int(audio_meta["start_sample"])
                or int(annotation["end_sample"]) != int(audio_meta["end_sample"])
            ):
                raise ValueError("L5 voice result is not aligned to its 20 ms audio hop")
        enhanced_waveforms = tuple(item[1] for item in track_audio_record)
        l5_record = None if l5_result is None else {
            "primary_model_id": l5_result.primary_model_id,
            "threshold": l5_result.threshold,
            "predictions": [
                {"model_id": prediction.model_id,
                 "probabilities": prediction.probabilities.tolist(),
                 "latency_ms": prediction.latency_ms,
                 "metadata": dict(prediction.metadata)}
                for prediction in l5_result.predictions
            ],
            "input_gain_compensation": [asdict(item) for item in l5_result.input_gain_compensation],
        }
        stage_statuses = {
            "l2": joined.l2.state.value, "l3": joined.l3.state.value,
            "l5": joined.l5.state.value,
        }
        stage_timings = {
            name: float(stage.duration_ms)
            for name, stage in (("l2", joined.l2), ("l3", joined.l3), ("l5", joined.l5))
            if stage.started_monotonic_ns > 0 and stage.duration_ms is not None
        }
        stage_waits = {
            "l2": self._stage_wait_ms(joined.l2, work.accepted_monotonic_ns),
            "l3": self._stage_wait_ms(joined.l3, joined.l2.finished_monotonic_ns),
            "l5": self._stage_wait_ms(joined.l5, joined.l3.finished_monotonic_ns),
        }
        latency_ms = max(
            0.0, (joined.completed_monotonic_ns - work.accepted_monotonic_ns) / 1_000_000.0
        )
        compute_ms = sum(stage_timings.values())
        severe = {StageState.FAILED, StageState.TIMED_OUT, StageState.DROPPED, StageState.CANCELLED}
        has_severe_stage = any(
            stage.state in severe for stage in (joined.l2, joined.l3, joined.l5)
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
        l5_diagnostics = () if l5_result is None else (
            f"l5_primary_model={l5_result.primary_model_id}",
            f"l5_threshold={l5_result.threshold}",
            *(f"l5_model_latency_ms={item.model_id}:{item.latency_ms:.3f}"
              for item in l5_result.predictions),
            *(
                f"l5_aggregation={item.model_id}:{item.metadata.get('aggregation', 'unknown')}"
                for item in l5_result.predictions
            ),
            *(
                "l5_input_gain="
                f"{item.algorithm_version}:max={item.max_applied_gain_db:.3f}:"
                f"mean={item.mean_applied_gain_db:.3f}:"
                f"segments={item.compensated_segment_count}:"
                f"peak_protection={item.peak_protection_trigger_count}"
                for item in l5_result.input_gain_compensation
            ),
        )
        terminal_diagnostics = tuple(
            f"{name}_stage={stage.state.value}:{stage.error or stage.reason}"
            for name, stage in (("l2", joined.l2), ("l3", joined.l3), ("l5", joined.l5))
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
                "threshold": l5_result.threshold,
            }
            for index, item in enumerate(() if l5_result is None else l5_result.detections)
        )
        record = DecisionRecord(
            window.session_id, window.stream_epoch, window.window_id, window.decision_sample,
            (window.doa_start_sample, window.doa_end_sample),
            (window.context_start_sample, window.context_end_sample), status,
            candidates=candidate_records,
            detections=detection_records,
            voice_direction_count=0 if l5_result is None else l5_result.voice_direction_count,
            diagnostics=joined.l2.diagnostics + search_record_diagnostics + l5_diagnostics
            + terminal_diagnostics,
            processing_latency_ms=latency_ms,
            raw_scores=None if response is None else response.raw_scores,
            normalized_scores=None if response is None else response.normalized_scores,
            gate_decision=gate_record, search_diagnostics=search_record,
            enhanced_audio=enhanced_records,
            enhanced_waveforms=enhanced_waveforms,
            l5_result=l5_record,
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
        if self._recording_session_started:
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
                joined, values, l2_output, l3_output, l5_result,
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
        l5_result,
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
                l5_ms=stage_timings.get("l5"), completed_monotonic=monotonic(),
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
                    direction_id_tracking_enabled=bool(
                        values["direction_id_tracking_enabled"]
                    ),
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
                    direction_id_tracking_enabled=bool(
                        values["direction_id_tracking_enabled"]
                    ),
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
            frame = self._ui_aggregator.update_l5(l5_result)
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
                f"L5 requires exactly {expected} context-aligned 20 ms probability slots"
            )
        return probabilities

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
        self._validate_direction_outputs("L3", candidates, l3_output.enhanced_audio)
        return self._beamform_previews(l3_output, 0, len(candidates))

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
        if not self._ephemeral_live_capture:
            self.recording_store.start_recording()
            return
        if not self.running:
            raise RuntimeError("采集未运行，不能开始正式录音")
        with self._ephemeral_transition_lock:
            if self._ephemeral_recording_active:
                return
            # Preserve only the already-warmed IMCRA estimator. MUSIC rolling
            # state and every L2 ID/tracker lifecycle restart at this boundary.
            self._ephemeral_recording_active = True
            with self._layer2_state_lock:
                self._layer2.reset()
            self.track_audio_stream.reset()
            with self._confirmed_backfill_lock:
                self._confirmed_backfill_history.clear()
                self._confirmed_backfill_ids.clear()
            while True:
                try:
                    self._confirmed_backfill_work.get_nowait()
                except queue.Empty:
                    break
            if self.dev_audio_tracker is not None:
                self.dev_audio_tracker.reset()
            self._downstream_processing_enabled.set()

    def pause_runtime_recording(self) -> None:
        if not self._ephemeral_live_capture:
            self.recording_store.pause_recording()
            return
        with self._ephemeral_transition_lock:
            self._downstream_processing_enabled.clear()
            self._ephemeral_recording_active = False

    def stop(
        self,
        *,
        drain_timeout_seconds: float | None | object = _CONFIGURED_DRAIN_TIMEOUT,
    ) -> None:
        """Stop input and drain staged work.

        ``None`` is reserved for finite replay input whose complete admitted
        timeline must be processed before sealing. Live/manual shutdown keeps
        the configured bounded deadline so a failed device or worker cannot
        hang the application indefinitely.
        """
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
        timeout = (
            self.config.runtime.graceful_shutdown_timeout_seconds
            if drain_timeout_seconds is _CONFIGURED_DRAIN_TIMEOUT
            else drain_timeout_seconds
        )
        if timeout is not None:
            timeout = float(timeout)
            if timeout <= 0.0:
                raise ValueError("drain timeout must be positive or None")
        deadline = None if timeout is None else monotonic() + timeout

        def join_until(worker: threading.Thread | None) -> None:
            if worker is None or worker is threading.current_thread():
                return
            worker.join(
                timeout=None if deadline is None else max(0.0, deadline - monotonic())
            )

        join_until(thread)
        for name in ("l2", "backfill", "l3_prepare", "l3", "l3_host", "l5", "commit"):
            join_until(workers.get(name))
        alive = {
            name: worker for name, worker in workers.items() if worker.is_alive()
        }
        if alive:
            assert timeout is not None
            self._processing_abort.set()
            pending_before_cancel = len(self._result_joiner.pending_keys())
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
            for name in ("l2", "backfill", "l3_prepare", "l3", "l3_host", "l5", "commit"):
                worker = workers.get(name)
                if worker is not None and worker is not threading.current_thread() and worker.is_alive():
                    worker.join(timeout=max(0.0, forced_deadline - monotonic()))
            alive = {
                name: worker for name, worker in workers.items() if worker.is_alive()
            }
            if pending_before_cancel:
                self.last_error = (
                    f"processing drain timed out after {timeout:.1f} seconds; "
                    f"cancelled {pending_before_cancel} pending windows"
                )
            if alive:
                self.last_error = (
                    "processing workers did not stop after graceful drain and forced cancellation: "
                    f"{','.join(sorted(alive))}"
                )
        input_alive = thread is not None and thread.is_alive()
        with self._lifecycle_lock:
            if input_alive:
                self.last_error = (
                    "runtime input worker did not stop during complete replay drain"
                    if timeout is None
                    else f"runtime input worker did not stop within {timeout:.1f} seconds"
                )
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
                self._l3_prepared_windows,
                self._l3_host_windows,
                self._l5_windows,
                self._confirmed_backfill_work,
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
        if not alive and not input_alive:
            allowed_l4_track_keys: set[tuple[str, int, int]] | None = None
            if self.dev_audio_tracker is not None:
                # Fail closed: if final UI filtering cannot complete, do not
                # let an invisible Hub-only track leak into offline L4.
                allowed_l4_track_keys = set()
                try:
                    self.dev_audio_tracker.append_terminal_hops(
                        self.track_audio_stream.finalize_missing_hops()
                    )
                    retained = self.dev_audio_tracker.finalize_capture()
                    allowed_l4_track_keys = {
                        (item.session_id, item.stream_epoch, item.track_id)
                        for item in retained
                        if item.track_id > 0
                    }
                except Exception as exc:
                    self.dev_audio_tracking_error = f"Test UI audio finalize failed: {exc}"
            self.track_audio_stream.seal(
                allowed_track_keys=allowed_l4_track_keys,
            )
            if self._ephemeral_live_capture:
                self._ephemeral_recording_active = False
                self._downstream_processing_enabled.clear()
        if self._recording_session_started and not alive and not input_alive:
            try:
                self.recording_store.stop_session("normal" if self.last_error is None else "runtime_error")
            except Exception as exc:
                self.last_error = f"runtime recording finalize failed: {exc}"
            else:
                self._recording_session_started = False

    def discard_pending_for_replay(self) -> dict[str, int]:
        """Abort the current replay pass and discard every uncommitted item.

        This is intentionally separate from :meth:`stop`: normal/EOF stop
        drains admitted work so its duration and offline audio remain
        authoritative, while an operator-requested replay is a hard epoch
        boundary.  A currently executing kernel cannot be pre-empted safely,
        but all waiting queues are emptied immediately and its late result is
        discarded before a fresh Runtime graph may start.
        """

        with self._lifecycle_lock:
            self._stop.set()
            self._discard_processing.set()
            self._processing_abort.set()
            self._input_worker_done.set()
            thread = self._thread
            workers = dict(self._processing_threads)

        discarded = {
            "l2": 0,
            "l3": 0,
            "l5": 0,
            "backfill": 0,
            "completion": 0,
            "completion_backlog": 0,
            "joiner_pending": len(self._result_joiner.pending_keys()),
        }

        def clear_waiting_queues() -> None:
            for name, mailbox in (
                ("l2", self._l2_windows),
                ("l3", self._l3_windows),
                ("l3", self._l3_prepared_windows),
                ("l3", self._l3_host_windows),
                ("l5", self._l5_windows),
                ("backfill", self._confirmed_backfill_work),
                ("completion", self._completion_results),
            ):
                while True:
                    try:
                        mailbox.get_nowait()
                        discarded[name] += 1
                    except queue.Empty:
                        break
            with self._completion_backlog_lock:
                discarded["completion_backlog"] += len(self._completion_backlog)
                self._completion_backlog.clear()

        # Drop queued L3 work before waiting for the one item that may already
        # be inside a CPU/CUDA kernel. Pausing the source in the UI prevents
        # new capture, and pipeline.stop wakes the input thread immediately.
        clear_waiting_queues()
        try:
            self.pipeline.stop()
        except Exception as exc:
            self.last_error = f"replay discard input stop failed: {exc}"
        self._wake_processing_workers()

        timeout = float(self.config.runtime.graceful_shutdown_timeout_seconds)
        deadline = monotonic() + timeout

        def join_until(worker: threading.Thread | None) -> None:
            if worker is None or worker is threading.current_thread():
                return
            worker.join(timeout=max(0.0, deadline - monotonic()))

        join_until(thread)
        for name in ("l2", "backfill", "l3_prepare", "l3", "l3_host", "l5", "commit"):
            join_until(workers.get(name))
        alive = {
            name: worker for name, worker in workers.items() if worker.is_alive()
        }
        input_alive = thread is not None and thread.is_alive()
        if input_alive or alive:
            names = (["input"] if input_alive else []) + sorted(alive)
            self.last_error = (
                f"replay discard did not stop workers within {timeout:.1f} seconds: "
                f"{','.join(names)}"
            )
            raise RuntimeError(self.last_error)

        # A just-finished in-flight worker may have published after the first
        # clear. Remove that late result as well before start() replaces the
        # graph, Hub, caches and per-run timing generation.
        clear_waiting_queues()
        with self._lifecycle_lock:
            self._thread = None
            self._processing_threads = {}
            self._processing_thread = None
        self._compute_cache.clear("replay_restart_discard")
        self._layer3.clear_cache()
        self._layer3_backfill.clear_cache()
        self.track_audio_stream.reset()
        with self._confirmed_backfill_lock:
            self._confirmed_backfill_history.clear()
            self._confirmed_backfill_ids.clear()
        if self.dev_audio_tracker is not None:
            self.dev_audio_tracker.reset()
        if self._recording_session_started:
            try:
                self.recording_store.stop_session("replay_restarted_discarded")
            except Exception as exc:
                self.last_error = f"replay discard recording finalize failed: {exc}"
                raise RuntimeError(self.last_error) from exc
            self._recording_session_started = False
        self.last_error = None
        self.processing_error = None
        return discarded

    @property
    def offline_l4_sources(self):
        """Sealed Hub output; populated only after a fully drained stop."""

        if self.running or any(thread.is_alive() for thread in self._processing_threads.values()):
            raise RuntimeError("offline L4 requires capture stop and complete processing drain")
        if self._development_center_l4_bypass:
            if self.dev_audio_tracker is None:
                return ()
            source = self.dev_audio_tracker.reference_l4_source(
                self._development_l4_reference_track_id,
            )
            return () if source is None else (source,)
        return self.track_audio_stream.sealed_tracks

    def run_offline_l4(self, pipeline):
        """Run a configured offline L4/L5 pipeline on the sealed Hub package."""

        sources = self.offline_l4_sources
        if not sources:
            raise RuntimeError("TrackAudioStreamHub has no sealed long audio")
        process_sealed = getattr(pipeline, "process_sealed", None)
        if not callable(process_sealed):
            raise TypeError("offline Layer4 pipeline must provide process_sealed")
        return process_sealed(sources)

    def build_offline_l4_pipeline(self, backend_id: str | None = None) -> OfflineLayer4Pipeline:
        """Create the configured two-step L3→L4→L5 pipeline for UI/batch callers."""

        if not self.config.layer4.enabled:
            raise RuntimeError("Layer4 is disabled in project config")
        selected = backend_id or self.config.layer4.default_backend
        artifacts = {
            "mossformer2_ss_16k": self.config.layer4.mossformer2_artifact,
            "tiger_speech_16k": self.config.layer4.tiger_artifact,
        }
        if selected not in artifacts:
            raise ValueError(f"unsupported offline Layer4 backend: {selected}")
        if self.l3_device == self.l4_device == "cuda":
            if self.running or any(
                thread.is_alive() for thread in self._processing_threads.values()
            ):
                raise RuntimeError("offline L4 requires CUDA L3 to stop and drain first")
            if self._l3_cuda_stream is not None:
                self._l3_cuda_stream.synchronize()
            self._layer3.clear_cache()
            self._l3_cache_snapshot = None
            torch.cuda.empty_cache()
        artifact = Path(artifacts[selected])
        if not artifact.is_absolute():
            artifact = self.project_root / artifact
        backend = (
            MossFormer2Backend(artifact, device=self.l4_device)
            if selected == "mossformer2_ss_16k"
            else TigerBackend(artifact, device=self.l4_device)
        )
        return OfflineLayer4Pipeline(
            speaker_counter=DirectionCountSpeakerClassifier(),
            backends={selected: backend},
            layer5=self._layer5,
            quality_scorer=self._get_dnsmos_scorer(),
            default_backend=selected,
        )

    def _get_dnsmos_scorer(self) -> DnsMosScorer:
        if self._dnsmos_scorer is None:
            artifact = Path(self.config.layer6.dnsmos_artifact)
            if not artifact.is_absolute():
                artifact = self.project_root / artifact
            self._dnsmos_scorer = DnsMosScorer(artifact)
        return self._dnsmos_scorer

    def build_offline_l6_pipeline(self) -> OfflineLayer6Pipeline:
        """Load the CPU-only models for the next post-L4/L5 L6 job."""

        if not self.config.layer6.enabled:
            raise RuntimeError("Layer6 is disabled in project config")
        campplus = Path(self.config.layer6.campplus_artifact)
        if not campplus.is_absolute():
            campplus = self.project_root / campplus
        return OfflineLayer6Pipeline(
            CampPlusEmbedder(campplus), self.config.layer6,
        )

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
                self.latest_l2_dev_ui, self.latest_l5_dev_ui,
                self._l2_windows, self._l3_windows, self._l3_prepared_windows,
                self._l3_host_windows,
                self._l5_windows,
                self._confirmed_backfill_work,
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
            self._layer3_backfill.clear_cache()
            self.track_audio_stream.reset()
            with self._confirmed_backfill_lock:
                self._confirmed_backfill_history.clear()
                self._confirmed_backfill_ids.clear()
        if self.dev_audio_tracker is not None:
            try:
                self.dev_audio_tracker.close(delete_files=delete_dev_test_ui_audio)
            except BaseException as exc:
                close_error = close_error or exc
        if close_error is not None:
            raise close_error
