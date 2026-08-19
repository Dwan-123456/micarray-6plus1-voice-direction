from __future__ import annotations

from collections import deque
from time import monotonic

import numpy as np

from common.data_types import CandidateDirection, IngestedAudioBlock, PipelineStatus, SpatialResponse, TrackedDirection

from .contracts import (
    AlgorithmPerformanceSnapshot, BeamformPreview, DevUiFrame, L1MeterSnapshot,
    TrackedAudioSnapshot,
)
from layer2_source_detection.music import MusicDiagnostics
from layer2_source_detection.probability_gate import ProbabilityGateDecision
from layer4_voice_classifier.contracts import Layer4Result


class PerformanceTracker:
    """Epoch-scoped timing statistics for the Development Test UI."""

    def __init__(self, *, sample_rate: int, required_samples: int, window_count: int, rate_seconds: int):
        self.sample_rate = sample_rate
        self.required_samples = required_samples
        self.window_count = window_count
        self.rate_seconds = rate_seconds
        self._key: tuple[str, int] | None = None
        self._rates: deque[tuple[float, int]] = deque()
        self._compute: deque[float] = deque(maxlen=window_count)
        self._latency: deque[float] = deque(maxlen=window_count)
        self._stage_timings: deque[
            tuple[float, float | None, float | None, float | None]
        ] = deque(maxlen=window_count)
        self._current_window: int | None = None

    def _ensure_epoch(self, session_id: str, epoch: int) -> None:
        key = (session_id, epoch)
        if key != self._key:
            self._key = key
            self._rates.clear()
            self._compute.clear()
            self._latency.clear()
            self._stage_timings.clear()
            self._current_window = None

    def add_block(self, block: IngestedAudioBlock, received_monotonic: float) -> None:
        self._ensure_epoch(block.session_id, block.stream_epoch)
        self._rates.append((received_monotonic, block.end_sample))
        cutoff = received_monotonic - self.rate_seconds
        while len(self._rates) > 1 and self._rates[0][0] < cutoff:
            self._rates.popleft()

    def add_timing(
        self,
        session_id: str,
        epoch: int,
        window_id: int,
        compute_ms: float,
        latency_ms: float,
        *,
        l2_ms: float | None = None,
        l3_ms: float | None = None,
        l4_ms: float | None = None,
        completed_monotonic: float | None = None,
    ) -> None:
        self._ensure_epoch(session_id, epoch)
        if latency_ms < compute_ms:
            latency_ms = compute_ms
        self._compute.append(float(compute_ms))
        self._latency.append(float(latency_ms))
        stage_values = tuple(
            None if value is None else float(value)
            for value in (l2_ms, l3_ms, l4_ms)
        )
        if any(value is not None and (not np.isfinite(value) or value < 0.0) for value in stage_values):
            raise ValueError("layer timing must be non-negative finite or None")
        completed = monotonic() if completed_monotonic is None else float(completed_monotonic)
        if not np.isfinite(completed):
            raise ValueError("completed_monotonic must be finite")
        self._stage_timings.append((completed, *stage_values))
        self._current_window = window_id

    def _last_second_stage_statistics(
        self, now_monotonic: float,
    ) -> tuple[
        tuple[float | None, float],
        tuple[float | None, float],
        tuple[float | None, float],
    ]:
        cutoff = now_monotonic - 1.0
        while self._stage_timings and self._stage_timings[0][0] < cutoff:
            self._stage_timings.popleft()
        statistics: list[tuple[float | None, float]] = []
        for index in range(1, 4):
            values = [item[index] for item in self._stage_timings if item[index] is not None]
            average = None if not values else float(np.mean(values, dtype=np.float64))
            # The observation window is exactly one second, so the number of
            # completed stage executions in it is also the effective rate in Hz.
            statistics.append((average, float(len(values))))
        return tuple(statistics)

    def snapshot(self, status: PipelineStatus) -> AlgorithmPerformanceSnapshot:
        self._ensure_epoch(status.session_id, status.stream_epoch)
        (l2_avg, l2_hz), (l3_avg, l3_hz), (l4_avg, l4_hz) = (
            self._last_second_stage_statistics(monotonic())
        )
        observed = None
        if len(self._rates) >= 2:
            elapsed = self._rates[-1][0] - self._rates[0][0]
            if elapsed >= 1.0:
                observed = (self._rates[-1][1] - self._rates[0][1]) / elapsed
        compute = np.asarray(self._compute, dtype=np.float64)
        latency = np.asarray(self._latency, dtype=np.float64)
        return AlgorithmPerformanceSnapshot(
            status.session_id,
            status.stream_epoch,
            self._current_window,
            status.buffered_samples,
            status.required_samples,
            self.sample_rate,
            observed,
            None if not len(compute) else float(compute[-1]),
            None if not len(compute) else float(np.percentile(compute, 50)),
            None if not len(compute) else float(np.percentile(compute, 95)),
            None if not len(latency) else float(latency[-1]),
            None if not len(latency) else float(np.percentile(latency, 50)),
            None if not len(latency) else float(np.percentile(latency, 95)),
            l2_time_ms_last_second_avg=l2_avg,
            l3_time_ms_last_second_avg=l3_avg,
            l4_time_ms_last_second_avg=l4_avg,
            l2_refresh_hz_last_second=l2_hz,
            l3_refresh_hz_last_second=l3_hz,
            l4_refresh_hz_last_second=l4_hz,
        )


class DevUiAggregator:
    """Builds one immutable, window-consistent latest-value diagnostic frame."""

    def __init__(self, performance: PerformanceTracker, *, stale_after_ms: int = 500):
        self.performance = performance
        self.stale_after_ms = stale_after_ms
        self._l1: L1MeterSnapshot | None = None
        self._response: SpatialResponse | None = None
        self._candidates: tuple[CandidateDirection, ...] = ()
        self._directions: tuple[TrackedDirection, ...] = ()
        self._active_tracks: tuple[TrackedDirection, ...] = ()
        self._previews: tuple[BeamformPreview, ...] = ()
        self._tracked_audio: tuple[TrackedAudioSnapshot, ...] = ()
        self._gate_decision: ProbabilityGateDecision | None = None
        self._gate_threshold: float | None = None
        self._gate_config_revision: int | None = None
        self._direction_threshold: float | None = None
        self._direction_kalman_enabled: bool | None = None
        self._direction_kalman_q_scale: float | None = None
        self._direction_kalman_r_scale: float | None = None
        self._scan_config_revision: int | None = None
        self._status: PipelineStatus | None = None
        self._srp_error: str | None = None
        self._l3_error: str | None = None
        self._spatial_published: float | None = None
        self._search_diagnostics: MusicDiagnostics | None = None
        self._l4_result: Layer4Result | None = None

    @property
    def current_stream(self) -> tuple[str, int] | None:
        """Return the L1-authoritative stream currently rendered by the UI."""

        if self._status is None:
            return None
        return self._status.session_id, self._status.stream_epoch

    def update_l1(self, l1: L1MeterSnapshot, status: PipelineStatus) -> DevUiFrame:
        previous_stream = (
            None if self._status is None else (self._status.session_id, self._status.stream_epoch)
        )
        self._l1, self._status = l1, status
        current_stream = (status.session_id, status.stream_epoch)
        if previous_stream is not None and previous_stream != current_stream:
            self._clear_window_state(clear_tracked_audio=True)
            self._srp_error = (
                "WARMING_UP: awaiting the first Layer 2 result for the new stream epoch; "
                f"{status.message}"
            )
            return self.frame()
        retained_keys = {
            (item.session_id, item.stream_epoch)
            for item in (self._response, self._gate_decision)
            if item is not None
        }
        if retained_keys and retained_keys != {(status.session_id, status.stream_epoch)}:
            self._clear_window_state(clear_tracked_audio=True)
        elif (
            (self._response is not None or self._gate_decision is not None)
            and self._spatial_published is not None
            and l1.end_sample > (
                self._response.decision_sample
                if self._response is not None
                else self._gate_decision.decision_sample
            )
            and (monotonic() - self._spatial_published) * 1_000.0 > self.stale_after_ms
        ):
            # L2 freshness is window-scoped, while the L3 listening rows are
            # session-scoped recordings.  A Gate/IMCRA unavailable interval
            # must not make already cached audio disappear from the UI.
            self._clear_window_state(clear_tracked_audio=False)
            self._srp_error = (
                "UNAVAILABLE / TIMEOUT: Layer 2 result exceeded "
                f"{self.stale_after_ms} ms aggregation limit"
            )
        return self.frame()

    def _clear_window_state(self, *, clear_tracked_audio: bool) -> None:
        self._response, self._candidates, self._previews = None, (), ()
        if clear_tracked_audio:
            self._tracked_audio = ()
        self._directions = self._active_tracks = ()
        self._gate_decision = None
        self._gate_threshold = self._gate_config_revision = None
        self._direction_threshold = None
        self._direction_kalman_enabled = None
        self._direction_kalman_q_scale = self._direction_kalman_r_scale = None
        self._scan_config_revision = None
        self._search_diagnostics = None
        self._l4_result = None
        self._spatial_published = None
        self._srp_error = None
        self._l3_error = None

    def update_srp(
        self, response: SpatialResponse | None, candidates: tuple[CandidateDirection, ...], error: str | None = None,
        search_diagnostics: MusicDiagnostics | None = None,
        gate_decision: ProbabilityGateDecision | None = None,
        gate_threshold: float | None = None,
        gate_config_revision: int | None = None,
        direction_threshold: float | None = None,
        direction_kalman_enabled: bool | None = None,
        direction_kalman_q_scale: float | None = None,
        direction_kalman_r_scale: float | None = None,
        scan_config_revision: int | None = None,
        directions: tuple[TrackedDirection, ...] = (),
        active_tracks: tuple[TrackedDirection, ...] = (),
    ) -> DevUiFrame:
        result_keys = {
            (item.session_id, item.stream_epoch)
            for item in (response, gate_decision)
            if item is not None
        }
        if len(result_keys) > 1:
            raise ValueError("SRP, Noise Estimation and Gate results must belong to one stream")
        if self._status is not None and result_keys and result_keys != {
            (self._status.session_id, self._status.stream_epoch)
        }:
            return self.frame()

        # A new Layer 2 result supersedes every window-derived L3 view.  Apply
        # the replacement transactionally: if the immutable frame contract
        # rejects a mixed window/configuration, restore the last valid frame
        # instead of leaving the aggregator poisoned for all later refreshes.
        previous = (
            self._response,
            self._candidates,
            self._directions,
            self._active_tracks,
            self._srp_error,
            self._gate_decision,
            self._gate_threshold,
            self._gate_config_revision,
            self._direction_threshold,
            self._direction_kalman_enabled,
            self._direction_kalman_q_scale,
            self._direction_kalman_r_scale,
            self._scan_config_revision,
            self._search_diagnostics,
            self._previews,
            self._tracked_audio,
            self._l3_error,
            self._l4_result,
            self._spatial_published,
        )
        self._response, self._candidates, self._srp_error = response, tuple(candidates), error
        self._directions = tuple(directions)
        self._active_tracks = tuple(active_tracks)
        self._gate_decision = gate_decision
        self._gate_threshold = gate_threshold
        self._gate_config_revision = gate_config_revision
        self._direction_threshold = direction_threshold
        self._direction_kalman_enabled = direction_kalman_enabled
        self._direction_kalman_q_scale = direction_kalman_q_scale
        self._direction_kalman_r_scale = direction_kalman_r_scale
        self._scan_config_revision = scan_config_revision
        self._search_diagnostics = search_diagnostics
        # A new SRP/Gate result supersedes only window-derived previews.  The
        # listening recordings live for the complete capture session and are
        # updated independently by update_l3().
        self._previews, self._l3_error = (), None
        self._l4_result = None
        self._spatial_published = monotonic()
        try:
            return self.frame()
        except Exception:
            (
                self._response,
                self._candidates,
                self._directions,
                self._active_tracks,
                self._srp_error,
                self._gate_decision,
                self._gate_threshold,
                self._gate_config_revision,
                self._direction_threshold,
                self._direction_kalman_enabled,
                self._direction_kalman_q_scale,
                self._direction_kalman_r_scale,
                self._scan_config_revision,
                self._search_diagnostics,
                self._previews,
                self._tracked_audio,
                self._l3_error,
                self._l4_result,
                self._spatial_published,
            ) = previous
            raise

    def update_l3(
        self, previews: tuple[BeamformPreview, ...], error: str | None = None,
        tracked_audio: tuple[TrackedAudioSnapshot, ...] = (),
    ) -> DevUiFrame:
        result_keys = {
            (item.session_id, item.stream_epoch)
            for item in (*previews, *tracked_audio)
        }
        if len(result_keys) > 1:
            raise ValueError("Layer 3 previews must belong to one stream")
        if self._status is not None and result_keys and result_keys != {
            (self._status.session_id, self._status.stream_epoch)
        }:
            return self.frame()
        self._previews = tuple(previews)
        self._tracked_audio = tuple(tracked_audio)
        self._l3_error = error
        return self.frame()

    def update_l4(self, result: Layer4Result | None) -> DevUiFrame:
        if result is not None:
            detection_streams = {
                (item.session_id, item.stream_epoch) for item in result.detections
            }
            if len(detection_streams) > 1:
                raise ValueError("Layer 4 detections must belong to one stream")
            # L1 owns the current stream and SRP owns the current window.  A
            # late ordered commit must be a no-op rather than poisoning the
            # new epoch's warming-up frame with an old L4 result.
            if self._status is None or self._response is None:
                return self.frame()
            expected_stream = (self._status.session_id, self._status.stream_epoch)
            if detection_streams and detection_streams != {expected_stream}:
                return self.frame()
            expected_window = (
                self._response.session_id,
                self._response.stream_epoch,
                self._response.window_id,
                self._response.decision_sample,
            )
            if any(
                (
                    item.session_id,
                    item.stream_epoch,
                    item.window_id,
                    item.decision_sample,
                )
                != expected_window
                for item in result.detections
            ):
                return self.frame()
        self._l4_result = result
        return self.frame()

    def frame(self) -> DevUiFrame:
        if self._status is None:
            raise RuntimeError("聚合器尚未收到PipelineStatus")
        missing = {}
        if self._l4_result is None:
            missing["cnn"] = "NO CANDIDATE" if not self._previews else "Layer 4 result unavailable"
        if not self._previews:
            missing["beamforming"] = self._l3_error or "NO CANDIDATE"
        if self._srp_error:
            missing["srp"] = self._srp_error
        return DevUiFrame(
            l1=self._l1,
            spatial_response=self._response,
            candidates=self._candidates,
            previews=self._previews,
            tracked_audio=self._tracked_audio,
            gate_decision=self._gate_decision,
            gate_threshold=self._gate_threshold,
            gate_config_revision=self._gate_config_revision,
            direction_threshold=self._direction_threshold,
            direction_kalman_enabled=self._direction_kalman_enabled,
            direction_kalman_q_scale=self._direction_kalman_q_scale,
            direction_kalman_r_scale=self._direction_kalman_r_scale,
            scan_config_revision=self._scan_config_revision,
            pipeline_status=self._status,
            performance=self.performance.snapshot(self._status),
            published_monotonic=monotonic(),
            spatial_published_monotonic=self._spatial_published,
            search_diagnostics=self._search_diagnostics,
            missing_reasons=missing,
            l4_result=self._l4_result,
            directions=self._directions,
            active_tracks=self._active_tracks,
        )
