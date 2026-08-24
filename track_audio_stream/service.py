from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import hashlib
import threading

import numpy as np

from layer5_voice_classifier.gain_compensation import (
    InputGainCompensationDiagnostic,
    InputGainCompensationSettings,
    SegmentGainDiagnostic,
    compensate_l5_input,
)
from layer4_speech_separation.contracts import Layer4LongAudioInput

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


@dataclass(frozen=True, slots=True)
class _ArchivedHop:
    start_sample: int
    end_sample: int
    theta_deg: float
    l2_direction_count: int
    waveform: np.ndarray


@dataclass(slots=True)
class _L2TrackTimeline:
    processing_mode: str
    first_start_sample: int
    end_sample: int
    theta_deg: float
    ever_formal: bool = False
    direction_counts: dict[int, int] = field(default_factory=dict)


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
        minimum_output_seconds: float = 0.0,
    ) -> None:
        if context_ms < 60 or context_ms % 20:
            raise ValueError("continuous track context must be >=60 ms on the 20 ms grid")
        self.settings = settings
        self.max_hops = context_ms // 20
        minimum_seconds = float(minimum_output_seconds)
        if not np.isfinite(minimum_seconds) or minimum_seconds < 0.0:
            raise ValueError("minimum output duration must be finite and non-negative")
        self.minimum_output_samples = round(minimum_seconds * 48_000)
        self._tracks: dict[tuple[str, int, int], _TrackState] = {}
        self._archive: dict[tuple[str, int, int], list[_ArchivedHop]] = {}
        self._archive_modes: dict[tuple[str, int, int], str] = {}
        self._l2_timelines: dict[tuple[str, int, int], _L2TrackTimeline] = {}
        self._emitted_ends: dict[tuple[str, int, int], int] = {}
        self._sealed: tuple[Layer4LongAudioInput, ...] = ()
        self._lock = threading.RLock()

    def reset(self) -> None:
        with self._lock:
            self._tracks.clear()
            self._archive.clear()
            self._archive_modes.clear()
            self._l2_timelines.clear()
            self._emitted_ends.clear()
            self._sealed = ()

    def observe_l2(
        self,
        *,
        identity: tuple[str, int, int, int],
        active_tracks: tuple[object, ...],
        processing_mode: str,
        l2_direction_count: int,
    ) -> None:
        """Record the authoritative ID timeline independently of BF throughput."""

        session_id, stream_epoch, _window_id, decision_sample = identity
        if not session_id or min(stream_epoch, decision_sample) < 0:
            raise ValueError("invalid L2 track timeline identity")
        if not processing_mode:
            raise ValueError("L2 track timeline processing mode must be non-empty")
        if type(l2_direction_count) is not int or l2_direction_count not in {0, 1, 2, 3}:
            raise ValueError("L2 direction count must be 0, 1, 2 or 3")
        start_sample = decision_sample - 2 * _HOP_SAMPLES
        end_sample = decision_sample - _HOP_SAMPLES
        if start_sample < 0:
            return
        with self._lock:
            for track in active_tracks:
                track_state = str(getattr(track, "track_state", ""))
                if track_state not in {"tentative", "confirmed", "coasting"}:
                    raise ValueError("invalid authoritative L2 track state")
                track_id = int(getattr(track, "track_id"))
                theta_deg = float(getattr(track, "theta_deg"))
                if track_id <= 0 or not np.isfinite(theta_deg) or not 0.0 <= theta_deg < 360.0:
                    raise ValueError("invalid authoritative L2 track")
                key = (session_id, stream_epoch, track_id)
                timeline = self._l2_timelines.get(key)
                if timeline is None or timeline.processing_mode != processing_mode:
                    timeline = _L2TrackTimeline(
                        processing_mode, start_sample, end_sample, theta_deg,
                        track_state in {"confirmed", "coasting"},
                    )
                    self._l2_timelines[key] = timeline
                else:
                    if end_sample <= timeline.end_sample:
                        raise ValueError("L2 track timeline must be strictly ordered per ID")
                    timeline.end_sample = end_sample
                    timeline.theta_deg = theta_deg
                    timeline.ever_formal |= track_state in {"confirmed", "coasting"}
                timeline.direction_counts[end_sample] = l2_direction_count

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
        theta_deg: float,
        l2_direction_count: int,
    ) -> TrackAudioHop:
        start_sample = source_decision - 2 * _HOP_SAMPLES
        end_sample = source_decision - _HOP_SAMPLES
        if state.last_emitted_end is not None and start_sample != state.last_emitted_end:
            raise ValueError("continuous track audio must advance by exactly one 20 ms hop")
        compensated, diagnostic = compensate_l5_input(
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
        self._archive.setdefault(key, []).append(_ArchivedHop(
            start_sample, end_sample, theta_deg, l2_direction_count,
            np.frombuffer(compensated.tobytes(), dtype=np.float32),
        ))
        self._archive_modes[key] = state.processing_mode
        self._emitted_ends[key] = end_sample
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
        l2_direction_count: int | None = None,
        context_track_ids: tuple[int, ...] | None = None,
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
        requested_contexts = (
            None
            if context_track_ids is None
            else tuple(int(item) for item in context_track_ids)
        )
        if requested_contexts is not None and (
            len(requested_contexts) != len(set(requested_contexts))
            or any(item <= 0 for item in requested_contexts)
        ):
            raise ValueError("context track IDs must be unique positive integers")
        direction_count = len(windows) if l2_direction_count is None else l2_direction_count
        if type(direction_count) is not int or direction_count not in {0, 1, 2, 3}:
            raise ValueError("L2 direction count must be 0, 1, 2 or 3")

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
                    # L3 modes are isolated listening experiments. Once the
                    # UI switches mode, the old-mode waveform is no longer
                    # displayed and must not remain in the offline L4 archive
                    # under the same authoritative ID.
                    self._archive.pop(key, None)
                    self._archive_modes.pop(key, None)
                    self._emitted_ends.pop(key, None)
                previous = state.last_source_decision
                if previous is not None and window.decision_sample <= previous:
                    raise ValueError("track-audio windows must be strictly ordered per ID")
                timeline = self._l2_timelines.get(key)
                first = window.decision_sample if previous is None else previous + _HOP_SAMPLES
                if (
                    previous is None
                    and timeline is not None
                    and timeline.processing_mode == window.processing_mode
                ):
                    first = timeline.first_start_sample + 2 * _HOP_SAMPLES
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
                        self._emitted_ends[key] = first - _HOP_SAMPLES
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
                        window.theta_deg, direction_count,
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
                if not state.audio or (
                    requested_contexts is not None
                    and window.track_id not in requested_contexts
                ):
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

    def finalize_missing_hops(self) -> tuple[TrackAudioHop, ...]:
        """Emit exact-duration silent tail slots through each L2-authoritative end."""

        emitted: list[TrackAudioHop] = []
        with self._lock:
            for key, timeline in sorted(self._l2_timelines.items()):
                if not self._archive.get(key):
                    continue
                if self._archive_modes.get(key) != timeline.processing_mode:
                    continue
                cursor = self._emitted_ends.get(key, timeline.first_start_sample)
                while cursor < timeline.end_sample:
                    hop = TrackAudioHop(
                        key[0], key[1], key[2], cursor, cursor + _HOP_SAMPLES,
                        None, None, False,
                    )
                    emitted.append(hop)
                    cursor += _HOP_SAMPLES
                self._emitted_ends[key] = cursor
        return tuple(emitted)

    def seal(
        self,
        *,
        allowed_track_keys: set[tuple[str, int, int]] | None = None,
    ) -> tuple[Layer4LongAudioInput, ...]:
        """Freeze only the authoritative L2 IDs retained by the Test UI.

        ``allowed_track_keys`` is intentionally an exact identity allow-list.
        When the Development Test UI has hidden and deleted a short or mostly
        silent listening track, the Hub archive for that same identity is
        deleted as well so it cannot reappear in an offline L4 submission.
        ``None`` skips only the UI allow-list; the authoritative L2 timeline
        still prevents a never-confirmed tentative ID from being published.
        """

        outputs: list[Layer4LongAudioInput] = []
        with self._lock:
            discarded: list[tuple[str, int, int]] = []
            for (session_id, epoch, track_id), hops in sorted(self._archive.items()):
                key = (session_id, epoch, track_id)
                if allowed_track_keys is not None and key not in allowed_track_keys:
                    discarded.append(key)
                    continue
                if not hops:
                    continue
                timeline = self._l2_timelines.get(key)
                if timeline is not None and not timeline.ever_formal:
                    discarded.append(key)
                    continue
                if timeline is not None and self._archive_modes.get(key) == timeline.processing_mode:
                    output_start = timeline.first_start_sample
                    output_end = timeline.end_sample
                    output_theta = timeline.theta_deg
                else:
                    output_start = hops[0].start_sample
                    output_end = hops[-1].end_sample
                    output_theta = hops[-1].theta_deg
                if output_end - output_start < self.minimum_output_samples:
                    discarded.append(key)
                    continue
                audio: list[np.ndarray] = []
                direction_counts: list[tuple[int, int]] = []
                cursor = output_start
                for hop in hops:
                    if hop.start_sample < cursor:
                        raise ValueError("archived L3 track hops must not overlap")
                    gap = hop.start_sample - cursor
                    if gap % _HOP_SAMPLES:
                        raise ValueError("archived L3 track gaps must align to 20 ms hops")
                    if gap:
                        audio.append(np.zeros(gap, dtype=np.float32))
                        direction_counts.extend(
                            (
                                end_sample,
                                0 if timeline is None else timeline.direction_counts.get(end_sample, 0),
                            )
                            for end_sample in range(
                                cursor + _HOP_SAMPLES,
                                hop.start_sample + _HOP_SAMPLES,
                                _HOP_SAMPLES,
                            )
                        )
                    audio.append(hop.waveform)
                    direction_counts.append((
                        hop.end_sample,
                        hop.l2_direction_count if timeline is None else timeline.direction_counts.get(
                            hop.end_sample, hop.l2_direction_count,
                        ),
                    ))
                    cursor = hop.end_sample
                if cursor < output_end:
                    audio.append(np.zeros(output_end - cursor, dtype=np.float32))
                    direction_counts.extend(
                        (end_sample, timeline.direction_counts.get(end_sample, 0))
                        for end_sample in range(
                            cursor + _HOP_SAMPLES,
                            output_end + _HOP_SAMPLES,
                            _HOP_SAMPLES,
                        )
                    )
                waveform = np.ascontiguousarray(np.concatenate(audio), dtype=np.float32)
                digest = hashlib.sha256(waveform.tobytes()).hexdigest()
                outputs.append(Layer4LongAudioInput(
                    asset_id=(
                        f"{session_id}:epoch{epoch}:track{track_id}:"
                        f"start{output_start}"
                    ),
                    sha256=digest,
                    session_id=session_id,
                    stream_epoch=epoch,
                    track_id=track_id,
                    theta_deg=output_theta,
                    start_sample=output_start,
                    sample_rate=48_000,
                    waveform=waveform,
                    l2_direction_counts=tuple(direction_counts),
                ))
            for key in discarded:
                self._archive.pop(key, None)
                self._archive_modes.pop(key, None)
                self._l2_timelines.pop(key, None)
                self._emitted_ends.pop(key, None)
                self._tracks.pop(key, None)
            self._sealed = tuple(outputs)
            return self._sealed

    @property
    def sealed_tracks(self) -> tuple[Layer4LongAudioInput, ...]:
        with self._lock:
            return self._sealed
