from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import threading

import numpy as np

from layer4_voice_classifier.gain_compensation import (
    InputGainCompensationDiagnostic,
    InputGainCompensationSettings,
    SegmentGainDiagnostic,
    compensate_l4_input,
)

from .contracts import ContinuousTrackAudio, TrackAudioBatch, TrackAudioHop, TrackAudioWindow


_HOP_SAMPLES = 960
_CROSSFADE_SAMPLES = 96


@dataclass(slots=True)
class _TrackState:
    processing_mode: str
    last_source_decision: int | None = None
    last_emitted_end: int | None = None
    previous_gain_db: float = 0.0
    future_source_decision: int | None = None
    future_audio: np.ndarray | None = None
    audio: deque[np.ndarray] = field(default_factory=deque)
    probabilities: deque[float | None] = field(default_factory=deque)
    diagnostics: deque[SegmentGainDiagnostic] = field(default_factory=deque)


def _combined_diagnostic(
    settings: InputGainCompensationSettings,
    segments: tuple[SegmentGainDiagnostic, ...],
    final_gain_db: float,
) -> InputGainCompensationDiagnostic:
    gains = tuple(item.applied_gain_db for item in segments)
    return InputGainCompensationDiagnostic(
        settings.algorithm_version,
        settings.enabled,
        segments,
        max(gains, default=0.0),
        float(np.mean(gains)) if gains else 0.0,
        sum(value > 1.0e-12 for value in gains),
        sum(item.peak_protection_triggered for item in segments),
        final_gain_db,
    )


class TrackAudioStreamHub:
    """Build one compensated continuous 48 kHz stream per exact L2 track ID."""

    def __init__(
        self,
        settings: InputGainCompensationSettings,
        *,
        context_ms: int = 3_200,
    ) -> None:
        if context_ms < 60 or context_ms % 20:
            raise ValueError("continuous track context must be >=60 ms on the 20 ms grid")
        self.settings = settings
        self.max_hops = context_ms // 20
        self._tracks: dict[tuple[str, int, int], _TrackState] = {}
        self._lock = threading.RLock()

    def reset(self) -> None:
        with self._lock:
            self._tracks.clear()

    @property
    def gain_compensation_enabled(self) -> bool:
        with self._lock:
            return self.settings.enabled

    def set_gain_compensation_enabled(self, enabled: bool) -> bool:
        if type(enabled) is not bool:
            raise ValueError("gain compensation enabled state must be bool")
        with self._lock:
            if enabled != self.settings.enabled:
                self.settings = replace(self.settings, enabled=enabled)
            return self.settings.enabled

    @staticmethod
    def _extract_hop(window: TrackAudioWindow, source_decision: int) -> tuple[np.ndarray, float | None] | None:
        # Emit [source_decision-40 ms, source_decision-20 ms): it is inside
        # the centered L3 reconstruction, remains on the IMCRA 20 ms grid and
        # therefore has one unambiguous probability slot.
        start_sample = source_decision - 2 * _HOP_SAMPLES
        relative = start_sample - (window.decision_sample - len(window.waveform))
        if relative < 0 or relative + _HOP_SAMPLES > len(window.waveform):
            return None
        probability_index = relative // _HOP_SAMPLES
        if relative % _HOP_SAMPLES or probability_index >= len(window.probabilities_20ms):
            return None
        audio = np.ascontiguousarray(
            window.waveform[relative:relative + _HOP_SAMPLES], dtype=np.float32
        )
        return audio, window.probabilities_20ms[probability_index]

    def _reset_track_audio(self, state: _TrackState) -> None:
        state.audio.clear()
        state.probabilities.clear()
        state.diagnostics.clear()
        state.previous_gain_db = 0.0
        state.last_emitted_end = None
        state.future_source_decision = None
        state.future_audio = None

    @staticmethod
    def _join_from_overlapping_window(
        previous_estimate: np.ndarray,
        current_estimate: np.ndarray,
    ) -> np.ndarray:
        """Move BF weights between windows without a periodic 20 ms seam."""

        previous = np.asarray(previous_estimate, dtype=np.float32)
        current = np.asarray(current_estimate, dtype=np.float32)
        if previous.shape != (_HOP_SAMPLES,) or current.shape != (_HOP_SAMPLES,):
            raise ValueError("track-audio crossfade requires two complete 20 ms hops")
        output = current.copy()
        phase = np.linspace(
            0.0, np.pi / 2.0, _CROSSFADE_SAMPLES, dtype=np.float32,
        )
        old_weight = np.cos(phase) ** 2
        new_weight = np.sin(phase) ** 2
        output[:_CROSSFADE_SAMPLES] = (
            previous[:_CROSSFADE_SAMPLES] * old_weight
            + current[:_CROSSFADE_SAMPLES] * new_weight
        )
        return np.ascontiguousarray(output, dtype=np.float32)

    def _append(
        self,
        key: tuple[str, int, int],
        state: _TrackState,
        source_decision: int,
        audio: np.ndarray,
        probability: float | None,
    ) -> TrackAudioHop:
        start_sample = source_decision - 2 * _HOP_SAMPLES
        end_sample = source_decision - _HOP_SAMPLES
        if state.last_emitted_end is not None and start_sample != state.last_emitted_end:
            raise ValueError("continuous track audio must advance by exactly one 20 ms hop")
        compensated, diagnostic = compensate_l4_input(
            audio,
            (probability,),
            self.settings,
            segment_count=1,
            initial_gain_db=state.previous_gain_db,
        )
        state.previous_gain_db = diagnostic.final_gain_db
        state.audio.append(compensated)
        state.probabilities.append(probability)
        state.diagnostics.append(diagnostic.segments[0])
        while len(state.audio) > self.max_hops:
            state.audio.popleft()
            state.probabilities.popleft()
            state.diagnostics.popleft()
        state.last_emitted_end = end_sample
        return TrackAudioHop(
            key[0], key[1], key[2], start_sample, end_sample,
            compensated, probability, True,
        )

    def process(
        self,
        windows: tuple[TrackAudioWindow, ...],
        *,
        active_track_ids: tuple[int, ...],
        identity: tuple[str, int, int, int],
    ) -> TrackAudioBatch:
        session_id, stream_epoch, window_id, decision_sample = identity
        windows = tuple(windows)
        ids = tuple(item.track_id for item in windows)
        if len(ids) != len(set(ids)):
            raise ValueError("track-audio input IDs must be unique")
        if any(
            (item.session_id, item.stream_epoch, item.window_id, item.decision_sample) != identity
            for item in windows
        ):
            raise ValueError("track-audio windows must belong to one exact WindowKey")
        active = tuple(int(item) for item in active_track_ids)
        if len(active) != len(set(active)) or any(item <= 0 for item in active):
            raise ValueError("active track IDs must be unique positive integers")

        emitted: list[TrackAudioHop] = []
        contexts: list[ContinuousTrackAudio] = []
        with self._lock:
            active_keys = {(session_id, stream_epoch, item) for item in active}
            for key in tuple(self._tracks):
                if key[:2] != (session_id, stream_epoch) or key not in active_keys:
                    del self._tracks[key]

            for window in windows:
                key = (session_id, stream_epoch, window.track_id)
                state = self._tracks.get(key)
                if state is None:
                    state = _TrackState(window.processing_mode)
                    self._tracks[key] = state
                elif state.processing_mode != window.processing_mode:
                    self._reset_track_audio(state)
                    state.processing_mode = window.processing_mode
                    state.last_source_decision = None
                previous = state.last_source_decision
                if previous is not None and window.decision_sample <= previous:
                    raise ValueError("track-audio windows must be strictly ordered per ID")
                first = window.decision_sample if previous is None else previous + _HOP_SAMPLES
                if first < window.decision_sample - len(window.waveform) + 2 * _HOP_SAMPLES:
                    # The gap exceeds the current L3 overlap. Preserve its
                    # duration for listening, but restart CNN context rather
                    # than treating invented silence as observed audio.
                    gap_end = window.decision_sample - len(window.waveform) + 2 * _HOP_SAMPLES
                    while first < gap_end:
                        emitted.append(TrackAudioHop(
                            session_id, stream_epoch, window.track_id,
                            first - 2 * _HOP_SAMPLES, first - _HOP_SAMPLES,
                            None, None, False,
                        ))
                        first += _HOP_SAMPLES
                    self._reset_track_audio(state)
                for source_decision in range(first, window.decision_sample + 1, _HOP_SAMPLES):
                    recovered = self._extract_hop(window, source_decision)
                    if recovered is None:
                        self._reset_track_audio(state)
                        continue
                    audio, probability = recovered
                    if (
                        state.future_audio is not None
                        and state.future_source_decision == source_decision
                    ):
                        audio = self._join_from_overlapping_window(
                            state.future_audio, audio,
                        )
                    emitted.append(self._append(
                        key, state, source_decision, audio, probability,
                    ))
                state.last_source_decision = window.decision_sample
                # The current L3 window also contains the immediately following
                # 20 ms interval.  Keep that overlapping estimate so the next
                # window can begin with the same BF solution that ended this
                # emitted hop, then move to its newer solution over 2 ms.
                future_source_decision = window.decision_sample + _HOP_SAMPLES
                future = self._extract_hop(window, future_source_decision)
                state.future_source_decision = future_source_decision
                state.future_audio = None if future is None else future[0]
                if not state.audio:
                    continue
                waveform = np.ascontiguousarray(np.concatenate(tuple(state.audio)), dtype=np.float32)
                effective_end = int(state.last_emitted_end)
                segments = tuple(state.diagnostics)
                contexts.append(ContinuousTrackAudio(
                    session_id,
                    stream_epoch,
                    window_id,
                    decision_sample,
                    window.track_id,
                    window.theta_deg,
                    effective_end - len(waveform),
                    effective_end,
                    waveform,
                    tuple(state.probabilities),
                    _combined_diagnostic(self.settings, segments, state.previous_gain_db),
                    state.processing_mode,
                ))
        return TrackAudioBatch(
            session_id, stream_epoch, window_id, decision_sample,
            tuple(emitted), tuple(contexts), active,
        )
