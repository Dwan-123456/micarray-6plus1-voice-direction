from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.runtime import ApplicationRuntime  # noqa: E402
from common.config import ProjectConfig, load_config  # noqa: E402
from layer1_input.calibration import ChannelCalibrator  # noqa: E402
from layer1_input.configuration import CalibrationConfig  # noqa: E402
from layer1_input.pipeline import InputPipeline  # noqa: E402
from layer1_input.recording_replay import RecordingReplaySource  # noqa: E402
from layer3_direction_signal import L3_MODE_DS_BASELINE  # noqa: E402


SCHEMA_VERSION = "realtime_l456_benchmark_v1"
SAMPLE_RATE = 48_000
DEFAULT_CONFIG = PROJECT_ROOT / "config/config.yaml"


def _chunk_seconds(value: str | int) -> int:
    try:
        seconds = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("chunk seconds must be an integer") from exc
    if not 3 <= seconds <= 15:
        raise argparse.ArgumentTypeError("chunk seconds must be in the inclusive range 3..15")
    return seconds


def _benchmark_config(
    config_path: str | Path,
    *,
    chunk_seconds: int,
    ephemeral_data_root: str | Path,
) -> ProjectConfig:
    """Freeze the measured workflow without mutating the project YAML."""

    seconds = _chunk_seconds(chunk_seconds)
    raw = load_config(config_path, environ={}).model_dump(mode="python")
    raw["paths"]["data_root"] = str(Path(ephemeral_data_root).resolve())
    raw["layer1_pre_denoise"]["enabled"] = False
    raw["layer1_speaker_count"]["enabled"] = False
    raw["layer2"]["scanner_backend"] = "frequency_normalized_music"
    raw["layer2"]["dpd_rank1_enabled"] = False
    raw["layer2"]["noise_whitening_enabled"] = True
    raw["layer4"]["enabled"] = True
    raw["layer4"]["default_backend"] = "mossformer2_ss_16k"
    raw["layer4"]["streaming"]["enabled"] = True
    raw["layer4"]["streaming"]["chunk_seconds"] = seconds
    raw["layer5"]["input_gain_compensation"]["enabled"] = False
    raw["layer6"]["enabled"] = True
    # The Runtime is also constructed in ephemeral mode. Keeping the formal
    # recording policy OFF makes the no-corpus-write contract explicit twice.
    raw["recording"]["runtime"]["mode"] = "off"
    return ProjectConfig.model_validate(raw)


def _current_rss_bytes() -> int:
    """Return resident memory without adding a psutil runtime dependency."""

    if os.name == "nt":
        from ctypes import wintypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = kernel32.GetCurrentProcess()
        succeeded = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        return int(counters.WorkingSetSize) if succeeded else 0
    statm = Path("/proc/self/statm")
    if statm.exists():
        resident_pages = int(statm.read_text(encoding="ascii").split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return 0


class _RssMonitor:
    def __init__(self, *, interval_seconds: float = 0.05) -> None:
        if interval_seconds <= 0.0:
            raise ValueError("RSS sampling interval must be positive")
        self.interval_seconds = float(interval_seconds)
        self.baseline_bytes = _current_rss_bytes()
        self.peak_bytes = self.baseline_bytes
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="realtime-l456-benchmark-rss",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_bytes = max(self.peak_bytes, _current_rss_bytes())

    def stop(self) -> None:
        self._stop.set()
        worker = self._thread
        if worker is not None:
            worker.join(timeout=max(1.0, self.interval_seconds * 4.0))
        self.peak_bytes = max(self.peak_bytes, _current_rss_bytes())


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return {}


class _ObservedMetrics:
    """Collect latest-only previews and sampled queue diagnostics."""

    def __init__(self) -> None:
        self.queue_high_water: dict[str, int] = {}
        self.observed_revisions: list[int] = []
        self.first_preview_wall_seconds: float | None = None
        self.first_preview_source_seconds: float | None = None
        self.first_preview_revision: int | None = None
        self.latest_snapshot: object | None = None
        self.latest_revision = 0
        self.final_preview_observed = False
        self.final_processing_status: dict[str, Any] = {}
        self.latest_preview_lag_seconds: float | None = None
        self.maximum_preview_lag_seconds: float | None = None

    @staticmethod
    def _get(value: object, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    def observe(
        self,
        runtime: object,
        *,
        elapsed_seconds: float,
        source_seconds: float,
    ) -> None:
        status = _mapping(getattr(runtime, "processing_status", {}))
        self.final_processing_status = status
        queue_depths = _mapping(status.get("queue_depths", {}))
        for name, depth in queue_depths.items():
            self.queue_high_water[name] = max(
                self.queue_high_water.get(name, 0), int(depth)
            )
        layer456 = _mapping(status.get("layer456_stream", {}))
        layer456_depth = int(layer456.get("queued_blocks", 0))
        self.queue_high_water["layer456"] = max(
            self.queue_high_water.get("layer456", 0), layer456_depth
        )
        self.latest_revision = max(
            self.latest_revision, int(layer456.get("latest_revision", 0))
        )

        mailbox = getattr(runtime, "latest_realtime_postprocessing", None)
        if mailbox is None:
            return
        while True:
            try:
                snapshot = mailbox.get_nowait()
            except queue.Empty:
                break
            revision = int(self._get(snapshot, "revision", 0))
            if revision > 0 and revision not in self.observed_revisions:
                self.observed_revisions.append(revision)
            self.latest_revision = max(self.latest_revision, revision)
            self.latest_snapshot = snapshot
            self.final_preview_observed = self.final_preview_observed or bool(
                self._get(snapshot, "is_final", False)
            )
            if self.first_preview_wall_seconds is None:
                self.first_preview_wall_seconds = float(elapsed_seconds)
                self.first_preview_source_seconds = float(source_seconds)
                self.first_preview_revision = revision
            valid_through = int(
                self._get(snapshot, "valid_through_sample_48k", 0)
            )
            lag = max(0.0, float(source_seconds) - valid_through / SAMPLE_RATE)
            self.latest_preview_lag_seconds = lag
            self.maximum_preview_lag_seconds = max(
                lag,
                self.maximum_preview_lag_seconds or 0.0,
            )


def _stage_durations(snapshot: object | None) -> dict[str, float]:
    output = {
        "model_load": 0.0,
        "l4": 0.0,
        "dnsmos": 0.0,
        "l5": 0.0,
        "l6": 0.0,
        "snapshot": 0.0,
    }
    if snapshot is None:
        return output
    raw = _ObservedMetrics._get(snapshot, "stage_durations_seconds", ())
    values = dict(raw) if not isinstance(raw, Mapping) else raw
    for stage, value in values.items():
        output[str(stage)] = float(value)
    return output


def _l4_workload(snapshot: object | None) -> dict[str, object]:
    processed = tuple(
        _ObservedMetrics._get(snapshot, "l4_processed", ())
        if snapshot is not None
        else ()
    )
    track_requests: dict[tuple[str, int, int], int] = {}
    paths: set[str] = set()
    for item in processed:
        source = _ObservedMetrics._get(item, "source")
        identity = (
            str(_ObservedMetrics._get(source, "session_id", "")),
            int(_ObservedMetrics._get(source, "stream_epoch", 0)),
            int(_ObservedMetrics._get(source, "track_id", 0)),
        )
        metadata = _mapping(_ObservedMetrics._get(item, "metadata", {}))
        track_requests[identity] = max(
            track_requests.get(identity, 0),
            int(metadata.get("realtime_mf2_request_count", 0)),
        )
        paths.add(str(_ObservedMetrics._get(item, "path", "unknown")))
    return {
        "track_count": len(track_requests),
        "output_count": len(processed),
        "paths": sorted(paths),
        "mf2_request_count": sum(track_requests.values()),
    }


def _track_end_samples(values: Sequence[object]) -> dict[tuple[str, int, int], int]:
    """Collapse sealed sources or branched preview outputs to exact track ends."""

    ends: dict[tuple[str, int, int], int] = {}
    for value in values:
        source = _ObservedMetrics._get(value, "source", value)
        identity = (
            str(_ObservedMetrics._get(source, "session_id", "")),
            int(_ObservedMetrics._get(source, "stream_epoch", -1)),
            int(_ObservedMetrics._get(source, "track_id", 0)),
        )
        end_sample = int(_ObservedMetrics._get(source, "end_sample", -1))
        if not identity[0] or identity[1] < 0 or identity[2] <= 0 or end_sample < 0:
            raise ValueError("benchmark L4 track identity or end sample is invalid")
        ends[identity] = max(ends.get(identity, end_sample), end_sample)
    return ends


def _preview_covers_sealed_sources(
    snapshot: object | None,
    sealed_sources: Sequence[object],
) -> bool:
    if snapshot is None or not sealed_sources:
        return False
    preview = tuple(_ObservedMetrics._get(snapshot, "l4_processed", ()))
    return _track_end_samples(preview) == _track_end_samples(sealed_sources)


def _json_track_ends(values: Mapping[tuple[str, int, int], int]) -> dict[str, int]:
    return {
        f"{session_id}:{stream_epoch}:{track_id}": end_sample
        for (session_id, stream_epoch, track_id), end_sample in sorted(values.items())
    }


def _gpu_peaks(device: str) -> tuple[int, int]:
    if device != "cuda" or not torch.cuda.is_available():
        return 0, 0
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())


def _configure_runtime(runtime: ApplicationRuntime) -> None:
    runtime.set_l1_pre_denoise_enabled(False)
    runtime.set_l1_speaker_count_enabled(False)
    runtime.set_music_dpd_rank1_enabled(False)
    runtime.set_music_noise_whitening_enabled(True)
    runtime.set_direction_id_tracking_enabled(True)
    runtime.set_l3_processing_mode(L3_MODE_DS_BASELINE)
    runtime.set_l5_input_gain_compensation_enabled(False)


def run_benchmark(
    recording_manifest: str | Path,
    *,
    chunk_seconds: int = 10,
    config_path: str | Path = DEFAULT_CONFIG,
    poll_interval_seconds: float = 0.02,
    playback_timeout_seconds: float | None = None,
    include_canonical: bool = True,
) -> dict[str, object]:
    """Replay one verified recording through Runtime and return JSON-safe metrics."""

    manifest_path = Path(recording_manifest).resolve(strict=True)
    if manifest_path.name not in {"recording_manifest.json", "session_manifest.json"}:
        raise ValueError("benchmark input must be a recording/session manifest")
    seconds = _chunk_seconds(chunk_seconds)
    if poll_interval_seconds <= 0.0:
        raise ValueError("poll interval must be positive")
    if playback_timeout_seconds is not None and playback_timeout_seconds <= 0.0:
        raise ValueError("playback timeout must be positive or None")

    total_started = time.perf_counter()
    process_cpu_started = time.process_time()
    rss = _RssMonitor()
    rss.start()
    runtime: ApplicationRuntime | None = None
    metrics = _ObservedMetrics()
    stop_seconds = 0.0
    playback_seconds = 0.0
    replay_status: object | None = None
    offline_source_count = 0
    errors: list[str] = []
    result: dict[str, object] | None = None
    pre_stop_status: dict[str, Any] = {}
    canonical: dict[str, object] = {
        "included": bool(include_canonical),
        "success": not include_canonical,
    }

    def close_runtime_before_temporary_data() -> None:
        """Release SQLite/model handles before Windows removes the temp tree."""

        nonlocal runtime
        current = runtime
        runtime = None
        if current is None:
            return
        try:
            current.close(delete_dev_test_ui_audio=False)
        except Exception as exc:
            if str(exc) not in errors:
                errors.append(f"runtime close: {exc}")

    try:
        with ExitStack() as cleanup:
            temporary = cleanup.enter_context(
                tempfile.TemporaryDirectory(prefix="micarray-l456-benchmark-")
            )
            config = _benchmark_config(
                config_path,
                chunk_seconds=seconds,
                ephemeral_data_root=Path(temporary) / "data",
            )
            replay = RecordingReplaySource(
                manifest_path,
                logical_channel_map=config.device.logical_channel_map,
                block_size=config.device.block_size_samples,
                autoplay=False,
            )
            pipeline = InputPipeline(
                replay,
                ChannelCalibrator(CalibrationConfig.from_project(config)),
                timestamp_tolerance_ms=config.timing.timestamp_tolerance_ms,
            )
            runtime = ApplicationRuntime(
                config,
                project_root=PROJECT_ROOT,
                pipeline=pipeline,
                ephemeral_live_capture=True,
            )
            # ExitStack callbacks run in reverse order, so Runtime releases
            # catalog.sqlite before TemporaryDirectory removes the data tree.
            cleanup.callback(close_runtime_before_temporary_data)
            _configure_runtime(runtime)
            if runtime.l4_device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            runtime.start()
            # The paused source provides an exact boundary: enable the
            # temporary downstream graph before sample zero can be consumed.
            runtime.begin_runtime_recording()
            playback_started = time.perf_counter()
            replay.resume()
            initial_status = replay.status()
            audio_duration_seconds = float(initial_status.total_seconds)
            timeout = playback_timeout_seconds
            if timeout is None:
                timeout = audio_duration_seconds + max(30.0, 2.0 * seconds)
            playback_deadline = playback_started + timeout

            while True:
                replay_status = replay.status()
                elapsed = time.perf_counter() - playback_started
                metrics.observe(
                    runtime,
                    elapsed_seconds=elapsed,
                    source_seconds=float(replay_status.current_seconds),
                )
                if replay_status.state == "ended":
                    break
                if runtime.last_error is not None:
                    raise RuntimeError(f"runtime input failed: {runtime.last_error}")
                if time.perf_counter() >= playback_deadline:
                    raise TimeoutError(
                        f"recording replay did not reach EOF within {timeout:.1f} seconds"
                    )
                time.sleep(poll_interval_seconds)

            playback_seconds = time.perf_counter() - playback_started
            pre_stop_status = _mapping(runtime.processing_status)
            stop_started = time.perf_counter()
            # Finite replay benchmark requires a complete, unbounded drain;
            # a configured live timeout would invalidate downstream timings.
            runtime.stop(drain_timeout_seconds=None)
            stop_seconds = time.perf_counter() - stop_started
            metrics.observe(
                runtime,
                elapsed_seconds=time.perf_counter() - playback_started,
                source_seconds=float(replay_status.current_seconds),
            )
            try:
                offline_sources = tuple(runtime.offline_l4_sources)
                offline_source_count = len(offline_sources)
            except Exception as exc:
                offline_sources = ()
                errors.append(f"offline L4 seal: {exc}")
            progressive_total_seconds = time.perf_counter() - total_started
            progressive_cpu_seconds = time.process_time() - process_cpu_started
            progressive_rss_peak_bytes = max(rss.peak_bytes, _current_rss_bytes())
            progressive_gpu_allocated, progressive_gpu_reserved = _gpu_peaks(
                runtime.l4_device
            )
            if include_canonical and offline_sources:
                if runtime.l4_device == "cuda" and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                canonical_started = time.perf_counter()
                try:
                    build_started = time.perf_counter()
                    pipeline = runtime.build_offline_l4_pipeline()
                    build_l4_seconds = time.perf_counter() - build_started
                    l4_started = time.perf_counter()
                    canonical_l4 = tuple(
                        pipeline.process_l4_sealed(
                            offline_sources,
                            merge_candidates=False,
                        )
                    )
                    l4_seconds = time.perf_counter() - l4_started
                    l5_started = time.perf_counter()
                    canonical_l5 = tuple(pipeline.process_l5_sealed(canonical_l4))
                    l5_seconds = time.perf_counter() - l5_started
                    build_l6_started = time.perf_counter()
                    l6_pipeline = runtime.build_offline_l6_pipeline()
                    build_l6_seconds = time.perf_counter() - build_l6_started
                    l6_started = time.perf_counter()
                    canonical_l6 = l6_pipeline.process(canonical_l5)
                    l6_seconds = time.perf_counter() - l6_started
                    canonical_gpu_allocated, canonical_gpu_reserved = _gpu_peaks(
                        runtime.l4_device
                    )
                    canonical = {
                        "included": True,
                        "success": True,
                        "build_l4_seconds": build_l4_seconds,
                        "l4_seconds": l4_seconds,
                        "l5_seconds": l5_seconds,
                        "build_l6_seconds": build_l6_seconds,
                        "l6_seconds": l6_seconds,
                        "total_seconds": time.perf_counter() - canonical_started,
                        "l4_output_count": len(canonical_l4),
                        "l5_output_count": len(canonical_l5),
                        "speaker_count": int(
                            _ObservedMetrics._get(canonical_l6, "speaker_count", 0)
                        ),
                        "gpu_peak_allocated_bytes": canonical_gpu_allocated,
                        "gpu_peak_reserved_bytes": canonical_gpu_reserved,
                    }
                except Exception as exc:
                    canonical = {
                        "included": True,
                        "success": False,
                        "total_seconds": time.perf_counter() - canonical_started,
                        "error": str(exc),
                    }
                    errors.append(f"canonical L4-L6: {exc}")
            status = runtime.processing_status
            realtime = _mapping(status.get("layer456_stream", {}))
            for value in (
                runtime.last_error,
                runtime.processing_error,
                runtime.dev_ui_error,
                realtime.get("error"),
            ):
                if value and str(value) not in errors:
                    errors.append(str(value))

            final_gpu_allocated, final_gpu_reserved = _gpu_peaks(runtime.l4_device)
            canonical_gpu_allocated = int(
                canonical.get("gpu_peak_allocated_bytes", final_gpu_allocated)
            )
            canonical_gpu_reserved = int(
                canonical.get("gpu_peak_reserved_bytes", final_gpu_reserved)
            )
            gpu_allocated = max(progressive_gpu_allocated, canonical_gpu_allocated)
            gpu_reserved = max(progressive_gpu_reserved, canonical_gpu_reserved)
            replay_status = replay.status()
            snapshot = metrics.latest_snapshot
            valid_through_sample = int(
                _ObservedMetrics._get(snapshot, "valid_through_sample_48k", 0)
            )
            sealed_track_ends = _track_end_samples(offline_sources)
            preview_values = tuple(
                _ObservedMetrics._get(snapshot, "l4_processed", ())
                if snapshot is not None
                else ()
            )
            preview_track_ends = _track_end_samples(preview_values)
            preview_covers_sealed = _preview_covers_sealed_sources(
                snapshot, offline_sources,
            )
            final_layer456 = _mapping(status.get("layer456_stream", {}))
            input_health = _mapping(status.get("input_health", {}))
            drops = {
                "layer456": int(final_layer456.get("dropped_blocks", 0)),
                "runtime_processing": int(status.get("processing_drops", 0)),
                "input_handoff": int(input_health.get("handoff_drop_count", 0)),
            }
            submitted = int(final_layer456.get("submitted_blocks", 0))
            processed_blocks = int(final_layer456.get("processed_blocks", 0))
            successful = (
                not errors
                and metrics.latest_revision > 0
                and metrics.final_preview_observed
                and final_layer456.get("state") == "final"
                and submitted == processed_blocks
                and not any(drops.values())
                and offline_source_count > 0
                and preview_covers_sealed
                and bool(canonical.get("success"))
            )
            result = {
                "schema_version": SCHEMA_VERSION,
                "success": successful,
                "recording_manifest": str(manifest_path),
                "recording_display_name": replay.display_name,
                "chunk_seconds": seconds,
                "workflow": {
                    "pre_denoise": False,
                    "speaker_count_model": False,
                    "localization": "frequency_normalized_music",
                    "direction_id_tracking": True,
                    "noise_whitening": True,
                    "dpd_rank1": False,
                    "layer3": L3_MODE_DS_BASELINE,
                    "input_gain_compensation": False,
                    "layer4": "mossformer2_ss_16k",
                    "ephemeral": True,
                },
                "devices": dict(status.get("devices", {})),
                "audio": {
                    "sample_rate": SAMPLE_RATE,
                    "total_samples": int(replay_status.total_samples),
                    "duration_seconds": float(replay_status.total_seconds),
                },
                "preview": {
                    "first_latency_seconds": metrics.first_preview_wall_seconds,
                    "source_seconds_at_first_preview": metrics.first_preview_source_seconds,
                    "latest_lag_seconds": metrics.latest_preview_lag_seconds,
                    "maximum_lag_seconds": metrics.maximum_preview_lag_seconds,
                    "first_revision": metrics.first_preview_revision,
                    "latest_revision": metrics.latest_revision,
                    "observed_revisions": metrics.observed_revisions,
                    "final_observed": metrics.final_preview_observed,
                    "valid_through_sample_48k": valid_through_sample,
                    "valid_through_seconds": valid_through_sample / SAMPLE_RATE,
                    "covers_all_sealed_track_ends": preview_covers_sealed,
                    "track_ends_48k": _json_track_ends(preview_track_ends),
                    "sealed_track_ends_48k": _json_track_ends(sealed_track_ends),
                },
                "stage_durations_seconds": _stage_durations(snapshot),
                "l4_workload": _l4_workload(snapshot),
                "pipeline_total_durations_seconds": runtime.pipeline_total_durations_seconds,
                "queues": {
                    "sampling_interval_seconds": float(poll_interval_seconds),
                    "observed_high_water": metrics.queue_high_water,
                    "pre_stop": _mapping(pre_stop_status.get("queue_depths", {})),
                    "pre_stop_layer456": _mapping(
                        pre_stop_status.get("layer456_stream", {})
                    ),
                    "final_layer456": final_layer456,
                },
                "drops": drops,
                "timing": {
                    "playback_wall_seconds": playback_seconds,
                    "stop_seconds": stop_seconds,
                    "progressive_total_wall_seconds": progressive_total_seconds,
                    "total_wall_seconds": time.perf_counter() - total_started,
                },
                "resources": {
                    "rss_baseline_bytes": rss.baseline_bytes,
                    "rss_peak_bytes": max(rss.peak_bytes, _current_rss_bytes()),
                    "progressive_rss_peak_bytes": progressive_rss_peak_bytes,
                    "gpu_peak_allocated_bytes": gpu_allocated,
                    "gpu_peak_reserved_bytes": gpu_reserved,
                    "progressive_gpu_peak_allocated_bytes": (
                        progressive_gpu_allocated
                    ),
                    "progressive_gpu_peak_reserved_bytes": progressive_gpu_reserved,
                    "progressive_process_cpu_seconds": progressive_cpu_seconds,
                    "process_cpu_seconds": time.process_time() - process_cpu_started,
                    "effective_cpu_cores": (
                        (time.process_time() - process_cpu_started)
                        / max(time.perf_counter() - total_started, 1e-9)
                    ),
                },
                "offline_sources_count": offline_source_count,
                "canonical": canonical,
                "ui_load_included": False,
                "errors": errors,
            }
            return result
    finally:
        if runtime is not None:
            try:
                runtime.close(delete_dev_test_ui_audio=False)
            except Exception as exc:
                if str(exc) not in errors:
                    errors.append(f"runtime close: {exc}")
        rss.stop()
        if result is not None:
            result["success"] = bool(result.get("success")) and not errors
            result["errors"] = errors
            resources = result.get("resources")
            if isinstance(resources, dict):
                resources["rss_peak_bytes"] = rss.peak_bytes
                resources["process_cpu_seconds"] = (
                    time.process_time() - process_cpu_started
                )
                resources["effective_cpu_cores"] = (
                    resources["process_cpu_seconds"]
                    / max(time.perf_counter() - total_started, 1e-9)
                )
            timing = result.get("timing")
            if isinstance(timing, dict):
                timing["total_wall_seconds"] = time.perf_counter() - total_started


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay one recording through the progressive L1-L6 runtime and emit JSON metrics."
    )
    parser.add_argument("recording_manifest", type=Path)
    parser.add_argument(
        "--chunk-seconds",
        type=_chunk_seconds,
        default=10,
        metavar="3..15",
        help="integer L4-L6 cadence in seconds; odd values are valid (default: 10)",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--poll-interval-ms", type=float, default=20.0)
    parser.add_argument("--playback-timeout-seconds", type=float)
    parser.add_argument(
        "--progressive-only",
        action="store_true",
        help="skip the post-stop canonical L4/L5/L6 correction",
    )
    parser.add_argument("--output", type=Path, help="optional JSON file; stdout is always emitted")
    return parser


def _emit(result: Mapping[str, object], output: Path | None) -> None:
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if output is not None:
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_benchmark(
            args.recording_manifest,
            chunk_seconds=args.chunk_seconds,
            config_path=args.config,
            poll_interval_seconds=args.poll_interval_ms / 1_000.0,
            playback_timeout_seconds=args.playback_timeout_seconds,
            include_canonical=not args.progressive_only,
        )
    except Exception as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "success": False,
            "recording_manifest": str(args.recording_manifest.resolve()),
            "chunk_seconds": args.chunk_seconds,
            "error": str(exc),
        }
    _emit(result, args.output)
    return 0 if bool(result.get("success")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
