from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
import threading

import numpy as np

from common.disk_audio import (
    DiskAudioStore,
    DiskFloat32Spool,
    DiskUInt8Series,
    DiskUInt8Timeline,
)

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
    waveform: np.ndarray | None = None


@dataclass(slots=True)
class _L2TrackTimeline:
    processing_mode: str
    first_start_sample: int
    end_sample: int
    theta_deg: float
    ever_formal: bool = False
    active: bool = True
    inactive_since_sample: int | None = None
    observed_through_sample: int = 0
    birth_theta_deg: float | None = None


@dataclass(frozen=True, slots=True)
class _StreamingChunkPlan:
    key: tuple[str, int, int]
    start_sample: int
    end_sample: int
    theta_deg: float
    direction_counts: tuple[int, ...]
    waveform_spool: DiskFloat32Spool
    waveform_origin_sample: int


class _DirectionCountSeries(Sequence[tuple[int, int]]):
    """Lazy 20 ms direction-count history backed by one byte per hop."""

    _disk_direction_trusted = True

    def __init__(
        self,
        values: DiskUInt8Series,
        *,
        start_sample: int,
        hop_samples: int = _HOP_SAMPLES,
        maximum_count: int = 3,
    ) -> None:
        self.values = values
        self.start_sample = int(start_sample)
        self.hop_samples = int(hop_samples)
        self.maximum_count = int(maximum_count)

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for index, count in enumerate(self.values):
            yield self.start_sample + (index + 1) * self.hop_samples, int(count)

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return tuple(self)[index]
        resolved = index + len(self) if index < 0 else index
        if resolved < 0 or resolved >= len(self):
            raise IndexError("direction-count history index out of range")
        return (
            self.start_sample + (resolved + 1) * self.hop_samples,
            int(self.values[resolved]),
        )

    def __eq__(self, other: object) -> bool:
        try:
            return tuple(self) == tuple(other)  # type: ignore[arg-type]
        except TypeError:
            return False


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
        ended_track_grace_ms: int = 1_000,
        spool_min_free_bytes: int = 0,
    ) -> None:
        if context_ms < 60 or context_ms % 20:
            raise ValueError("continuous track context must be >=60 ms on the 20 ms grid")
        self.settings = settings
        self.max_hops = context_ms // 20
        minimum_seconds = float(minimum_output_seconds)
        if not np.isfinite(minimum_seconds) or minimum_seconds < 0.0:
            raise ValueError("minimum output duration must be finite and non-negative")
        self.minimum_output_samples = round(minimum_seconds * 48_000)
        if (
            type(ended_track_grace_ms) is not int
            or ended_track_grace_ms < 0
            or ended_track_grace_ms % 20
        ):
            raise ValueError("ended track grace must be a non-negative 20 ms multiple")
        self.ended_track_grace_samples = ended_track_grace_ms * 48
        if type(spool_min_free_bytes) is not int or spool_min_free_bytes < 0:
            raise ValueError("track archive disk reserve must be a non-negative integer")
        self.spool_min_free_bytes = spool_min_free_bytes
        self._tracks: dict[tuple[str, int, int], _TrackState] = {}
        self._archive: dict[tuple[str, int, int], list[_ArchivedHop]] = {}
        self._archive_store = DiskAudioStore(
            prefix="micarray_track_hub_",
            minimum_free_bytes=self.spool_min_free_bytes,
        )
        self._archive_audio: dict[
            tuple[str, int, int], DiskFloat32Spool
        ] = {}
        self._audio_presence: dict[
            tuple[str, int, int], DiskUInt8Timeline
        ] = {}
        self._direction_counts: dict[
            tuple[str, int, int], DiskUInt8Timeline
        ] = {}
        self._archive_origins: dict[tuple[str, int, int], int] = {}
        self._archive_modes: dict[tuple[str, int, int], str] = {}
        self._l2_timelines: dict[tuple[str, int, int], _L2TrackTimeline] = {}
        self._active_timeline_keys: set[tuple[str, int, int]] = set()
        self._emitted_ends: dict[tuple[str, int, int], int] = {}
        self._streaming_cursors: dict[tuple[str, int, int], int] = {}
        self._streaming_claims: dict[
            tuple[str, int, int], tuple[int, int, str]
        ] = {}
        self._streaming_finalization_claims: set[tuple[str, int, int]] = set()
        self._streaming_finalized_tracks: set[tuple[str, int, int]] = set()
        self._sealed: tuple[Layer4LongAudioInput, ...] = ()
        self._seal_complete = False
        self._lock = threading.RLock()

    def reset(self) -> None:
        with self._lock:
            previous_store = self._archive_store
            self._archive_store = DiskAudioStore(
                prefix="micarray_track_hub_",
                minimum_free_bytes=self.spool_min_free_bytes,
            )
            self._tracks.clear()
            self._archive.clear()
            self._archive_audio.clear()
            self._audio_presence.clear()
            self._direction_counts.clear()
            self._archive_origins.clear()
            self._archive_modes.clear()
            self._l2_timelines.clear()
            self._active_timeline_keys.clear()
            self._emitted_ends.clear()
            self._streaming_cursors.clear()
            self._streaming_claims.clear()
            self._streaming_finalization_claims.clear()
            self._streaming_finalized_tracks.clear()
            self._sealed = ()
            self._seal_complete = False
        previous_store.retire()

    @staticmethod
    def _close_spool(spool: object | None) -> None:
        if spool is None:
            return
        spool.close()  # type: ignore[attr-defined]

    def _clear_archive_generation(
        self,
        key: tuple[str, int, int],
        *,
        clear_timeline: bool,
    ) -> None:
        """Drop one mutable mode generation without touching published snapshots."""

        self._archive.pop(key, None)
        self._close_spool(self._archive_audio.pop(key, None))
        self._close_spool(self._audio_presence.pop(key, None))
        self._close_spool(self._direction_counts.pop(key, None))
        self._archive_origins.pop(key, None)
        self._archive_modes.pop(key, None)
        self._emitted_ends.pop(key, None)
        self._streaming_cursors.pop(key, None)
        self._streaming_claims.pop(key, None)
        self._streaming_finalization_claims.discard(key)
        self._streaming_finalized_tracks.discard(key)
        if clear_timeline:
            self._l2_timelines.pop(key, None)
            self._active_timeline_keys.discard(key)

    def _archive_origin(
        self,
        key: tuple[str, int, int],
        start_sample: int | None = None,
    ) -> int:
        origin = self._archive_origins.get(key)
        if origin is None:
            if start_sample is None or start_sample < 0 or start_sample % _HOP_SAMPLES:
                raise ValueError("track archive origin must be a non-negative 20 ms boundary")
            origin = start_sample
            self._archive_origins[key] = origin
        elif start_sample is not None and start_sample < origin:
            self._prepend_archive_origin(key, start_sample)
            origin = start_sample
        return origin

    def _prepend_archive_origin(
        self,
        key: tuple[str, int, int],
        new_origin: int,
    ) -> None:
        """Rebase one short pre-confirmation history without loading the track."""

        old_origin = self._archive_origins.get(key)
        if (
            old_origin is None
            or new_origin < 0
            or new_origin >= old_origin
            or (old_origin - new_origin) % _HOP_SAMPLES
        ):
            raise ValueError("track archive prepend origin is invalid")
        delta_samples = old_origin - new_origin
        delta_slots = delta_samples // _HOP_SAMPLES

        old_audio = self._archive_audio.get(key)
        if old_audio is not None:
            replacement = self._archive_store.create_spool(
                f"session_{key[0]}_epoch_{key[1]}_track_{key[2]}_rebased"
            )
            zeros = np.zeros(min(delta_samples, 1_048_576), dtype=np.float32)
            remaining = delta_samples
            while remaining:
                count = min(remaining, len(zeros))
                replacement.append(zeros[:count])
                remaining -= count
            if old_audio.length_samples:
                old_audio.copy_range_to(
                    0,
                    old_audio.length_samples,
                    replacement,
                    destination_start_sample=delta_samples,
                )
            old_audio.close()
            self._archive_audio[key] = replacement

        for timelines in (self._direction_counts, self._audio_presence):
            old_timeline = timelines.get(key)
            if old_timeline is None:
                continue
            payload = (
                old_timeline.read_range(0, old_timeline.disk_bytes)
                if old_timeline.disk_bytes
                else b""
            )
            replacement_timeline = self._archive_store.create_u8_timeline(
                f"session_{key[0]}_epoch_{key[1]}_track_{key[2]}_rebased_meta"
            )
            if payload:
                replacement_timeline.write_range(delta_slots, payload)
            old_timeline.close()
            timelines[key] = replacement_timeline
        self._archive_origins[key] = new_origin

    def _relative_sample(
        self,
        key: tuple[str, int, int],
        sample: int,
    ) -> int:
        relative = sample - self._archive_origin(key, sample)
        if relative < 0 or relative % _HOP_SAMPLES:
            raise ValueError("track archive sample must align to its disk origin")
        return relative

    def _audio_spool(self, key: tuple[str, int, int]) -> DiskFloat32Spool:
        spool = self._archive_audio.get(key)
        if spool is None:
            spool = self._archive_store.create_spool(
                f"session_{key[0]}_epoch_{key[1]}_track_{key[2]}"
            )
            self._archive_audio[key] = spool
        return spool

    def _direction_spool(self, key: tuple[str, int, int]) -> DiskUInt8Timeline:
        spool = self._direction_counts.get(key)
        if spool is None:
            spool = self._archive_store.create_u8_timeline(
                f"session_{key[0]}_epoch_{key[1]}_track_{key[2]}_directions"
            )
            self._direction_counts[key] = spool
        return spool

    def _presence_spool(self, key: tuple[str, int, int]) -> DiskUInt8Timeline:
        spool = self._audio_presence.get(key)
        if spool is None:
            spool = self._archive_store.create_u8_timeline(
                f"session_{key[0]}_epoch_{key[1]}_track_{key[2]}_presence"
            )
            self._audio_presence[key] = spool
        return spool

    def _slot_index(self, key: tuple[str, int, int], start_sample: int) -> int:
        return self._relative_sample(key, start_sample) // _HOP_SAMPLES

    def _has_audio(self, key: tuple[str, int, int], start_sample: int) -> bool:
        presence = self._audio_presence.get(key)
        origin = self._archive_origins.get(key)
        if presence is None or origin is None or start_sample < origin:
            return False
        relative = start_sample - origin
        if relative % _HOP_SAMPLES:
            raise ValueError("track archive slot must align to its disk origin")
        return bool(presence.get(relative // _HOP_SAMPLES, 0))

    def _direction_series(
        self,
        key: tuple[str, int, int],
        start_sample: int,
        end_sample: int,
    ) -> _DirectionCountSeries:
        return _DirectionCountSeries(
            self._direction_spool(key).series(
                self._slot_index(key, start_sample),
                self._slot_index(key, end_sample),
            ),
            start_sample=start_sample,
        )

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
        observed: list[tuple[str, int, float]] = []
        for track in active_tracks:
            track_state = str(getattr(track, "track_state", ""))
            if track_state not in {"tentative", "confirmed", "coasting"}:
                raise ValueError("invalid authoritative L2 track state")
            track_id = int(getattr(track, "track_id"))
            theta_deg = float(getattr(track, "theta_deg"))
            if track_id <= 0 or not np.isfinite(theta_deg) or not 0.0 <= theta_deg < 360.0:
                raise ValueError("invalid authoritative L2 track")
            observed.append((track_state, track_id, theta_deg))
        if len({track_id for _, track_id, _ in observed}) != len(observed):
            raise ValueError("authoritative L2 track IDs must be unique")

        active_keys = {
            (session_id, stream_epoch, track_id)
            for _, track_id, _ in observed
        }
        with self._lock:
            for key in tuple(self._active_timeline_keys):
                timeline = self._l2_timelines.get(key)
                if timeline is None:
                    self._active_timeline_keys.discard(key)
                    continue
                if (
                    key[:2] != (session_id, stream_epoch)
                    or timeline.processing_mode != processing_mode
                ):
                    continue
                timeline.observed_through_sample = max(
                    timeline.observed_through_sample,
                    decision_sample,
                )
                if key not in active_keys and timeline.active:
                    timeline.active = False
                    timeline.inactive_since_sample = decision_sample
                elif (
                    key not in active_keys
                    and not timeline.active
                    and self._timeline_ready_for_early_finalization(timeline)
                ):
                    self._active_timeline_keys.discard(key)

            for track_state, track_id, theta_deg in observed:
                key = (session_id, stream_epoch, track_id)
                timeline = self._l2_timelines.get(key)
                if timeline is None or timeline.processing_mode != processing_mode:
                    # A mode change starts a new audio experiment under the
                    # same authoritative ID.  Any incremental consumer cursor
                    # belongs to the discarded old-mode archive and must not
                    # hide the beginning of the replacement stream.
                    if timeline is not None or (
                        key in self._archive_modes
                        and self._archive_modes[key] != processing_mode
                    ):
                        self._clear_archive_generation(key, clear_timeline=False)
                    timeline = _L2TrackTimeline(
                        processing_mode, start_sample, end_sample, theta_deg,
                        track_state in {"confirmed", "coasting"},
                        observed_through_sample=decision_sample,
                        birth_theta_deg=theta_deg,
                    )
                    self._l2_timelines[key] = timeline
                    self._archive_origin(key, start_sample)
                    self._archive_modes[key] = processing_mode
                else:
                    if end_sample <= timeline.end_sample:
                        raise ValueError("L2 track timeline must be strictly ordered per ID")
                    timeline.end_sample = end_sample
                    timeline.theta_deg = theta_deg
                    timeline.ever_formal |= track_state in {"confirmed", "coasting"}
                    timeline.active = True
                    timeline.inactive_since_sample = None
                    timeline.observed_through_sample = decision_sample
                self._direction_spool(key).set(
                    self._slot_index(key, start_sample),
                    l2_direction_count,
                )
                self._active_timeline_keys.add(key)

    def track_birth_theta_deg(
        self, session_id: str, stream_epoch: int, track_id: int
    ) -> float | None:
        """Return the first authoritative L2 angle recorded for one exact ID."""

        if not session_id or min(stream_epoch, track_id) < 0 or track_id == 0:
            raise ValueError("invalid L2 track identity")
        with self._lock:
            timeline = self._l2_timelines.get((session_id, stream_epoch, track_id))
            if timeline is None:
                return None
            theta = timeline.birth_theta_deg
            return timeline.theta_deg if theta is None else theta

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
        # Persist the canonical audio and metadata before advancing any in-memory
        # state.  A disk-full/short-write failure is therefore retryable at the
        # same absolute hop without corrupting gain continuity or cursors.
        relative_start = self._relative_sample(key, start_sample)
        self._audio_spool(key).write_at(relative_start, compensated)
        direction_spool = self._direction_spool(key)
        slot_index = relative_start // _HOP_SAMPLES
        if direction_spool.get(slot_index, 0) == 0:
            direction_spool.set(slot_index, l2_direction_count)
        self._presence_spool(key).set(slot_index, 1)

        state.previous_gain_db = diagnostic.final_gain_db
        state.audio.append(compensated)
        state.probabilities.append(probability)
        state.diagnostics.append(diagnostic.segments[0])
        while len(state.audio) > self.max_hops:
            state.audio.popleft()
            state.probabilities.popleft()
            state.diagnostics.popleft()
        state.last_emitted_end = end_sample
        archive = self._archive.setdefault(key, [])
        archive.append(_ArchivedHop(
            start_sample, end_sample, theta_deg, l2_direction_count, compensated,
        ))
        if len(archive) > self.max_hops:
            del archive[:-self.max_hops]
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
                    if (
                        key in self._archive_modes
                        and self._archive_modes[key] != window.processing_mode
                    ):
                        self._clear_archive_generation(key, clear_timeline=False)
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
                    # observe_l2() normally rotates the generation first and
                    # writes the new mode's direction slot.  Do not erase that
                    # freshly prepared metadata when the first L3 window lands.
                    if self._archive_modes.get(key) != window.processing_mode:
                        self._clear_archive_generation(key, clear_timeline=False)
                        first_start = window.decision_sample - 2 * _HOP_SAMPLES
                        self._archive_origin(key, first_start)
                        self._archive_modes[key] = window.processing_mode
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

    def missing_backfill_windows(
        self,
        windows: tuple[TrackAudioWindow, ...],
    ) -> tuple[TrackAudioWindow, ...]:
        """Return only historical canonical slots not already produced live."""

        windows = tuple(sorted(windows, key=lambda item: item.decision_sample))
        if not windows:
            return ()
        key = (windows[0].session_id, windows[0].stream_epoch, windows[0].track_id)
        mode = windows[0].processing_mode
        if any(
            (item.session_id, item.stream_epoch, item.track_id) != key
            or item.processing_mode != mode
            for item in windows
        ):
            raise ValueError("backfill windows must belong to one exact ID and L3 mode")
        with self._lock:
            if self._archive_modes.get(key, mode) != mode:
                return windows
            return tuple(
                item
                for item in windows
                if not self._has_audio(
                    key,
                    item.decision_sample - 2 * _HOP_SAMPLES,
                )
            )

    def missing_backfill_decisions(
        self,
        *,
        session_id: str,
        stream_epoch: int,
        track_id: int,
        processing_mode: str,
        decision_samples: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Filter raw historical decisions before the expensive BF pass."""

        key = (str(session_id), int(stream_epoch), int(track_id))
        if not processing_mode:
            raise ValueError("backfill L3 mode must be non-empty")
        with self._lock:
            if self._archive_modes.get(key, processing_mode) != processing_mode:
                return tuple(int(item) for item in decision_samples)
            return tuple(
                int(decision)
                for decision in decision_samples
                if not self._has_audio(
                    key,
                    int(decision) - 2 * _HOP_SAMPLES,
                )
            )

    def insert_backfill(
        self,
        windows: tuple[TrackAudioWindow, ...],
        *,
        l2_direction_count: int,
    ) -> tuple[TrackAudioHop, ...]:
        """Insert missing historical BF hops without replacing live results.

        Backfill uses the same absolute 20 ms grid as realtime stitching.  An
        already archived live hop always wins an overlap; only absent slots are
        compensated and inserted before final L4 sealing.
        """

        if type(l2_direction_count) is not int or l2_direction_count not in {0, 1, 2, 3}:
            raise ValueError("L2 direction count must be 0, 1, 2 or 3")
        windows = tuple(sorted(windows, key=lambda item: item.decision_sample))
        if not windows:
            return ()
        key = (windows[0].session_id, windows[0].stream_epoch, windows[0].track_id)
        mode = windows[0].processing_mode
        if any(
            (item.session_id, item.stream_epoch, item.track_id) != key
            or item.processing_mode != mode
            for item in windows
        ):
            raise ValueError("backfill windows must belong to one exact ID and L3 mode")

        emitted: list[TrackAudioHop] = []
        with self._lock:
            timeline = self._l2_timelines.get(key)
            if timeline is None or not timeline.ever_formal:
                raise ValueError("backfill requires an already confirmed authoritative ID")
            if self._archive_modes.get(key, mode) != mode:
                return ()
            self._archive_origin(key, timeline.first_start_sample)
            self._archive_modes[key] = mode
            archive = self._archive.setdefault(key, [])
            previous_gain_db = 0.0
            previous_end: int | None = None
            for window in windows:
                source_decision = window.decision_sample
                start_sample = source_decision - 2 * _HOP_SAMPLES
                end_sample = source_decision - _HOP_SAMPLES
                if self._has_audio(key, start_sample):
                    continue
                recovered = self._extract_hop(window, source_decision)
                if recovered is None:
                    continue
                audio, probability = recovered
                if previous_end is not None and start_sample != previous_end:
                    previous_gain_db = 0.0
                compensated, diagnostic = compensate_l5_input(
                    audio,
                    (probability,),
                    self.settings,
                    segment_count=1,
                    initial_gain_db=previous_gain_db,
                )
                previous_gain_db = diagnostic.final_gain_db
                archived = _ArchivedHop(
                    start_sample,
                    end_sample,
                    window.theta_deg,
                    l2_direction_count,
                    compensated,
                )
                relative_start = self._relative_sample(key, start_sample)
                self._audio_spool(key).write_at(relative_start, compensated)
                direction_spool = self._direction_spool(key)
                direction_index = relative_start // _HOP_SAMPLES
                if direction_spool.get(direction_index, 0) == 0:
                    direction_spool.set(direction_index, l2_direction_count)
                self._presence_spool(key).set(direction_index, 1)
                archive.append(archived)
                previous_end = end_sample
                timeline.first_start_sample = min(timeline.first_start_sample, start_sample)
                emitted.append(TrackAudioHop(
                    key[0], key[1], key[2], start_sample, end_sample,
                    compensated, probability, True,
                ))
            archive.sort(key=lambda item: item.start_sample)
            if len(archive) > self.max_hops:
                del archive[:-self.max_hops]
            if archive:
                self._archive_modes[key] = mode
            return tuple(emitted)

    @staticmethod
    def _build_streaming_chunk(
        plan: _StreamingChunkPlan,
    ) -> Layer4LongAudioInput:
        """Materialize one immutable interval using the final seal gap rules."""

        key = plan.key
        start_sample = plan.start_sample
        end_sample = plan.end_sample

        if (
            start_sample < 0
            or end_sample <= start_sample
            or start_sample % _HOP_SAMPLES
            or (end_sample - start_sample) % _HOP_SAMPLES
        ):
            raise ValueError("streaming L4 chunk must align to complete 20 ms hops")

        slot_count = (end_sample - start_sample) // _HOP_SAMPLES
        if len(plan.direction_counts) != slot_count:
            raise ValueError("streaming L4 direction history does not cover its chunk")
        direction_counts = tuple(
            (
                start_sample + (index + 1) * _HOP_SAMPLES,
                int(count),
            )
            for index, count in enumerate(plan.direction_counts)
        )

        # A claim is a content-addressed immutable retry token.  Copy only this
        # bounded 3-15 second interval into its own short-lived store instead of
        # exposing the mutable capture archive mmap.
        relative_start = start_sample - plan.waveform_origin_sample
        relative_end = end_sample - plan.waveform_origin_sample
        snapshot_store = DiskAudioStore(
            prefix="micarray_track_chunk_",
            minimum_free_bytes=plan.waveform_spool.owner.minimum_free_bytes,
        )
        snapshot = snapshot_store.create_spool("immutable_chunk")
        try:
            plan.waveform_spool.ensure_length(relative_end)
            plan.waveform_spool.copy_range_to(
                relative_start,
                relative_end,
                snapshot,
            )
            snapshot.close()
            waveform = snapshot.view(0, relative_end - relative_start)
            digest = snapshot.digest(0, relative_end - relative_start)
        except BaseException:
            snapshot_store.retire()
            raise
        snapshot_store.retire()
        session_id, stream_epoch, track_id = key
        return Layer4LongAudioInput(
            asset_id=(
                f"{session_id}:epoch{stream_epoch}:track{track_id}:"
                f"start{start_sample}"
            ),
            sha256=digest,
            session_id=session_id,
            stream_epoch=stream_epoch,
            track_id=track_id,
            theta_deg=plan.theta_deg,
            start_sample=start_sample,
            sample_rate=48_000,
            waveform=waveform,
            l2_direction_counts=direction_counts,
        )

    def _timeline_ready_for_early_finalization(
        self,
        timeline: _L2TrackTimeline,
    ) -> bool:
        return (
            not timeline.active
            and timeline.inactive_since_sample is not None
            and timeline.observed_through_sample - timeline.inactive_since_sample
            >= self.ended_track_grace_samples
        )

    def claim_streaming_chunks(
        self,
        *,
        chunk_samples: int,
        ready_track_keys: set[tuple[str, int, int]] | None = None,
        flush: bool = False,
        max_chunks: int = 1,
    ) -> tuple[Layer4LongAudioInput, ...]:
        """Claim bounded L4-ready chunks without advancing committed cursors.

        Normal calls stop at the Hub's processed-audio watermark, so a missing
        realtime L3 result is not converted to silence while it may still
        arrive.  ``flush=True`` advances through the authoritative L2 end and
        emits a final shorter interval when it still contains complete 20 ms
        hops.  Callers may withhold a newly confirmed ID through
        ``ready_track_keys`` until its asynchronous historical backfill has
        completed; no cursor is created for a withheld ID.  A claimed chunk
        remains outstanding until :meth:`resolve_streaming_chunk` accepts or
        releases it, so downstream admission failure cannot lose its interval.
        """

        if (
            type(chunk_samples) is not int
            or chunk_samples <= 0
            or chunk_samples % _HOP_SAMPLES
        ):
            raise ValueError("streaming chunk_samples must be a positive 20 ms multiple")
        if type(flush) is not bool:
            raise ValueError("streaming flush must be bool")
        if type(max_chunks) is not int or max_chunks <= 0:
            raise ValueError("streaming max_chunks must be a positive integer")
        ready = None if ready_track_keys is None else frozenset(ready_track_keys)
        if ready is not None and any(
            not isinstance(key, tuple)
            or len(key) != 3
            or not isinstance(key[0], str)
            or not key[0]
            or type(key[1]) is not int
            or key[1] < 0
            or type(key[2]) is not int
            or key[2] <= 0
            for key in ready
        ):
            raise ValueError("ready_track_keys must contain valid exact track identities")

        plans: list[_StreamingChunkPlan] = []
        with self._lock:
            for key, timeline in sorted(self._l2_timelines.items()):
                if len(plans) >= max_chunks:
                    break
                if not timeline.ever_formal or (ready is not None and key not in ready):
                    continue
                if key in self._streaming_claims:
                    continue
                waveform_spool = self._archive_audio.get(key)
                if waveform_spool is None:
                    continue
                if self._archive_modes.get(key) != timeline.processing_mode:
                    self._streaming_cursors.pop(key, None)
                    self._streaming_claims.pop(key, None)
                    continue
                if (
                    timeline.first_start_sample < 0
                    or timeline.first_start_sample % _HOP_SAMPLES
                    or (timeline.end_sample - timeline.first_start_sample) % _HOP_SAMPLES
                ):
                    raise ValueError("authoritative L2 track timeline must align to 20 ms")

                cursor = self._streaming_cursors.get(
                    key, timeline.first_start_sample,
                )
                if cursor < timeline.first_start_sample:
                    cursor = timeline.first_start_sample
                if (
                    cursor > timeline.end_sample
                    or (cursor - timeline.first_start_sample) % _HOP_SAMPLES
                ):
                    raise ValueError("streaming L4 cursor is outside its authoritative timeline")

                processed_end = self._emitted_ends.get(
                    key, timeline.first_start_sample,
                )
                ended_ready = self._timeline_ready_for_early_finalization(timeline)
                available_end = (
                    timeline.end_sample
                    if flush
                    else min(timeline.end_sample, processed_end)
                )
                if available_end <= cursor:
                    continue

                if available_end - cursor >= chunk_samples:
                    chunk_end = cursor + chunk_samples
                elif (flush or ended_ready) and available_end > cursor:
                    chunk_end = available_end
                else:
                    continue
                origin = self._archive_origin(key, timeline.first_start_sample)
                direction_counts = tuple(
                    self._direction_spool(key).read_range(
                        self._slot_index(key, cursor),
                        self._slot_index(key, chunk_end),
                    )
                )
                plans.append(_StreamingChunkPlan(
                    key,
                    cursor,
                    chunk_end,
                    timeline.theta_deg,
                    direction_counts,
                    waveform_spool,
                    origin,
                ))
                # Empty digest reserves this exact interval while expensive
                # concatenation, validation and SHA run outside the Hub lock.
                self._streaming_claims[key] = (cursor, chunk_end, "")

        outputs: list[Layer4LongAudioInput] = []
        try:
            for plan in plans:
                source = self._build_streaming_chunk(plan)
                with self._lock:
                    reservation = (
                        source.start_sample,
                        source.end_sample,
                        "",
                    )
                    if self._streaming_claims.get(plan.key) != reservation:
                        raise RuntimeError(
                            "streaming L4 claim changed while materializing"
                        )
                    self._streaming_claims[plan.key] = (
                        source.start_sample,
                        source.end_sample,
                        source.sha256,
                    )
                outputs.append(source)
        except BaseException:
            with self._lock:
                for plan in plans:
                    claim = self._streaming_claims.get(plan.key)
                    if claim is not None and claim[:2] == (
                        plan.start_sample,
                        plan.end_sample,
                    ):
                        self._streaming_claims.pop(plan.key, None)
            raise
        return tuple(outputs)

    def resolve_streaming_chunk(
        self,
        source: Layer4LongAudioInput,
        *,
        accepted: bool,
    ) -> None:
        """Commit one admitted claim or release it for an exact retry."""

        if not isinstance(source, Layer4LongAudioInput):
            raise TypeError("streaming claim resolution requires Layer4LongAudioInput")
        if type(accepted) is not bool:
            raise ValueError("streaming claim accepted state must be bool")
        key = (source.session_id, source.stream_epoch, source.track_id)
        claim = (source.start_sample, source.end_sample, source.sha256)
        with self._lock:
            if self._streaming_claims.get(key) != claim:
                raise ValueError("streaming chunk is not the outstanding claim for its ID")
            if accepted:
                cursor = self._streaming_cursors.get(key, source.start_sample)
                if cursor != source.start_sample:
                    raise ValueError("accepted streaming chunk does not start at its cursor")
                self._streaming_cursors[key] = source.end_sample
                hops = self._archive.get(key)
                if hops is not None and len(hops) > self.max_hops:
                    del hops[:-self.max_hops]
            self._streaming_claims.pop(key, None)

    def claim_streaming_finalizations(
        self,
        *,
        ready_track_keys: set[tuple[str, int, int]] | None = None,
        flush: bool = False,
        max_tracks: int = 1,
    ) -> tuple[tuple[str, int, int], ...]:
        """Claim tracks whose complete audio cursor can now be finalized.

        Normal operation finalizes an ended ID after a short inactivity grace,
        spreading sub-chunk tail work across capture time. ``flush=True`` also
        includes the tracks that remain active at global stop. Claims use the
        same admission/ack discipline as audio chunks.
        """

        if type(flush) is not bool:
            raise ValueError("streaming flush must be bool")
        if type(max_tracks) is not int or max_tracks <= 0:
            raise ValueError("streaming max_tracks must be a positive integer")
        ready = None if ready_track_keys is None else frozenset(ready_track_keys)
        claimed: list[tuple[str, int, int]] = []
        with self._lock:
            for key, timeline in sorted(self._l2_timelines.items()):
                if len(claimed) >= max_tracks:
                    break
                if (
                    not timeline.ever_formal
                    or (ready is not None and key not in ready)
                    or key in self._streaming_finalized_tracks
                    or key in self._streaming_finalization_claims
                    or key in self._streaming_claims
                ):
                    continue
                if not flush and not self._timeline_ready_for_early_finalization(timeline):
                    continue
                if key not in self._archive_audio:
                    continue
                if self._archive_modes.get(key) != timeline.processing_mode:
                    continue
                cursor = self._streaming_cursors.get(
                    key,
                    timeline.first_start_sample,
                )
                if cursor < timeline.end_sample:
                    continue
                if cursor > timeline.end_sample:
                    raise ValueError("streaming finalization cursor exceeds track end")
                self._streaming_finalization_claims.add(key)
                claimed.append(key)
        return tuple(claimed)

    def resolve_streaming_finalization(
        self,
        key: tuple[str, int, int],
        *,
        accepted: bool,
    ) -> None:
        """Commit one admitted per-track finalization or release it for retry."""

        if (
            not isinstance(key, tuple)
            or len(key) != 3
            or not isinstance(key[0], str)
            or type(key[1]) is not int
            or type(key[2]) is not int
        ):
            raise ValueError("invalid streaming finalization identity")
        if type(accepted) is not bool:
            raise ValueError("streaming finalization accepted state must be bool")
        with self._lock:
            if key not in self._streaming_finalization_claims:
                raise ValueError("streaming track is not claimed for finalization")
            self._streaming_finalization_claims.remove(key)
            if accepted:
                self._streaming_finalized_tracks.add(key)
                # Authoritative timeline + disk presence are now the source of
                # truth. Ended-track hop descriptors are no longer needed.
                archive = self._archive.get(key)
                if archive is not None and key in self._l2_timelines:
                    archive.clear()

    def take_streaming_chunks(
        self,
        *,
        chunk_samples: int,
        ready_track_keys: set[tuple[str, int, int]] | None = None,
        flush: bool = False,
    ) -> tuple[Layer4LongAudioInput, ...]:
        """Compatibility helper that claims and immediately accepts all chunks."""

        outputs: list[Layer4LongAudioInput] = []
        while True:
            claimed = self.claim_streaming_chunks(
                chunk_samples=chunk_samples,
                ready_track_keys=ready_track_keys,
                flush=flush,
                max_chunks=64,
            )
            if not claimed:
                break
            for source in claimed:
                self.resolve_streaming_chunk(source, accepted=True)
            outputs.extend(claimed)
        return tuple(outputs)

    def finalize_missing_hops(self) -> tuple[TrackAudioHop, ...]:
        """Emit exact-duration silent tail slots through each L2-authoritative end."""

        emitted: list[TrackAudioHop] = []
        with self._lock:
            for key, timeline in sorted(self._l2_timelines.items()):
                if key not in self._archive_audio:
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
            if self._seal_complete:
                return self._sealed
            discarded: list[tuple[str, int, int]] = []
            keys = sorted(set(self._archive) | set(self._archive_audio))
            for session_id, epoch, track_id in keys:
                key = (session_id, epoch, track_id)
                hops = self._archive.get(key, [])
                if allowed_track_keys is not None and key not in allowed_track_keys:
                    discarded.append(key)
                    continue
                timeline = self._l2_timelines.get(key)
                if not hops and timeline is None:
                    continue
                if timeline is not None and not timeline.ever_formal:
                    discarded.append(key)
                    continue
                spool = self._archive_audio.get(key)
                authoritative = (
                    timeline is not None
                    and self._archive_modes.get(key) == timeline.processing_mode
                )
                if authoritative:
                    output_start = timeline.first_start_sample
                    output_end = timeline.end_sample
                    output_theta = timeline.theta_deg
                elif spool is not None:
                    output_start = self._archive_origin(key)
                    output_end = output_start + spool.length_samples
                    output_theta = hops[-1].theta_deg
                else:
                    output_start = hops[0].start_sample
                    output_end = hops[-1].end_sample
                    output_theta = hops[-1].theta_deg
                if output_end - output_start < self.minimum_output_samples:
                    discarded.append(key)
                    continue
                direction_history: Sequence[tuple[int, int]]
                if timeline is not None or spool is not None:
                    # Keep the 20 ms history lazy; constructing 90,000 Python
                    # tuples for a 30 minute track defeats disk-backed audio.
                    direction_history = self._direction_series(
                        key,
                        output_start,
                        output_end,
                    )
                else:
                    direction_counts: list[tuple[int, int]] = []
                    cursor = output_start
                    for hop in hops:
                        if hop.start_sample < cursor:
                            raise ValueError("archived L3 track hops must not overlap")
                        gap = hop.start_sample - cursor
                        if gap % _HOP_SAMPLES:
                            raise ValueError("archived L3 track gaps must align to 20 ms hops")
                        direction_counts.extend(
                            (end, 0)
                            for end in range(
                                cursor + _HOP_SAMPLES,
                                hop.start_sample + _HOP_SAMPLES,
                                _HOP_SAMPLES,
                            )
                        )
                        direction_counts.append((hop.end_sample, hop.l2_direction_count))
                        cursor = hop.end_sample
                    direction_counts.extend(
                        (end, 0)
                        for end in range(
                            cursor + _HOP_SAMPLES,
                            output_end + _HOP_SAMPLES,
                            _HOP_SAMPLES,
                        )
                    )
                    direction_history = tuple(direction_counts)
                if spool is None:
                    self._archive_origin(key, output_start)
                    spool = self._audio_spool(key)
                    for hop in hops:
                        if hop.waveform is None:
                            raise RuntimeError("legacy in-memory archive hop has no waveform")
                        spool.write_at(
                            self._relative_sample(key, hop.start_sample),
                            hop.waveform,
                        )
                origin = self._archive_origin(key, output_start)
                relative_start = output_start - origin
                relative_end = output_end - origin
                spool.ensure_length(relative_end)
                # seal() is terminal for this archive generation. Closing it
                # freezes the exact bytes before publishing a read-only mmap.
                spool.close()
                waveform = spool.view(relative_start, relative_end)
                digest = spool.digest(relative_start, relative_end)
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
                    l2_direction_counts=direction_history,
                ))
            for key in discarded:
                self._clear_archive_generation(key, clear_timeline=True)
                self._tracks.pop(key, None)
            self._sealed = tuple(outputs)
            self._seal_complete = True
            return self._sealed

    @property
    def sealed_tracks(self) -> tuple[Layer4LongAudioInput, ...]:
        with self._lock:
            return self._sealed
