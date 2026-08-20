from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import threading

import numpy as np

from .contracts import BeamformPreview, TrackedAudioSnapshot


_HOP_SAMPLES = 960
_CROSSFADE_SAMPLES = 480
_EDGE_FADE_SAMPLES = 240


@dataclass(slots=True)
class _Track:
    track_id: int
    theta_deg: float
    score: float
    last_seen_sample: int
    state: str = "active"
    segment_index: int = 0
    segment_samples: int = 0
    segments: list[Path] = field(default_factory=list)
    pending_waveform: np.ndarray | None = None
    pending_decision_sample: int | None = None
    last_emitted_decision_sample: int | None = None
    fade_in_next: bool = True
    authoritative_id: bool = False
    envelope_peaks: deque[float] = field(default_factory=deque)
    stream_key: tuple[str, int] | None = None


class AudioIdTracker:
    """Test-UI-only greedy IDs and bounded L3 listening caches.

    The input is a completed L3 preview batch. Kalman-ready temporary IDs and
    formal IDs own cached audio; first-hit temporary and ID-less points do not.
    exist only for listening in Development Test UI and are never fed back to
    L2, L3, L4, or RecordingStore.
    """

    def __init__(
        self,
        cache_root: str | Path,
        *,
        project_root: str | Path,
        wait_seconds: float = 3.0,
        association_distance_deg: float = 30.0,
        formal_id_rollover_distance_deg: float = 20.0,
        formal_id_rollover_wait_seconds: float = 5.0,
        segment_seconds: float = 10.0,
        retained_segments: int = 3,
        max_ended_tracks: int = 8,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        root = Path(cache_root)
        self.cache_root = (self.project_root / root).resolve() if not root.is_absolute() else root.resolve()
        if self.cache_root == self.project_root or self.project_root not in self.cache_root.parents:
            raise ValueError("Test UI audio cache must be a child of project_root")
        self.wait_samples = round(float(wait_seconds) * 48_000)
        self.association_distance_deg = float(association_distance_deg)
        self.formal_id_rollover_distance_deg = float(formal_id_rollover_distance_deg)
        self.formal_id_rollover_wait_samples = round(
            float(formal_id_rollover_wait_seconds) * 48_000
        )
        self.segment_samples = round(float(segment_seconds) * 48_000)
        self.retained_segments = int(retained_segments)
        self.max_ended_tracks = int(max_ended_tracks)
        if min(
            self.wait_samples,
            self.formal_id_rollover_wait_samples,
            self.segment_samples,
            self.retained_segments,
            self.max_ended_tracks,
        ) <= 0:
            raise ValueError("invalid Test UI tracker retention configuration")
        if not 0.0 < self.formal_id_rollover_distance_deg <= 180.0:
            raise ValueError("formal ID rollover association distance must be in (0,180]")
        self._lock = threading.RLock()
        self._stream: tuple[str, int] | None = None
        self._processing_mode: str | None = None
        self._tracks: dict[int, _Track] = {}
        self._formal_aliases: dict[int, int] = {}
        self._reference_track: _Track | None = None
        self._reference_stream = None
        self._reference_stream_path: Path | None = None
        self._next_track_id = 1
        self._snapshot_counter = 0
        self.reset()

    @staticmethod
    def _distance(left: float, right: float) -> float:
        delta = abs(float(left) - float(right)) % 360.0
        return min(delta, 360.0 - delta)

    def reset(self) -> None:
        with self._lock:
            self._close_reference_stream()
            if self.cache_root.exists():
                shutil.rmtree(self.cache_root)
            self.cache_root.mkdir(parents=True, exist_ok=True)
            (self.cache_root / "playback").mkdir(exist_ok=True)
            self._stream = None
            self._processing_mode = None
            self._tracks.clear()
            self._formal_aliases.clear()
            self._reference_track = None
            self._next_track_id = 1
            self._snapshot_counter = 0

    def _ensure_stream(self, stream: tuple[str, int]) -> None:
        """Move to a new epoch without deleting this capture session's audio."""

        if self._stream is None:
            self._stream = stream
            if self._reference_track is not None:
                self._reference_track.stream_key = stream
            return
        if self._stream == stream:
            return
        if self._stream[0] != stream[0]:
            # A new source start owns a new cache lifecycle.  Runtime.start()
            # normally performs this reset first; keep the tracker safe for
            # direct callers as well.
            self.reset()
            self._stream = stream
            return

        # An epoch change is a continuity boundary inside the same capture
        # session.  Close live directional runs, but keep every playable file
        # and row.  L2 private IDs restart per epoch, so aliases must not cross
        # the boundary and future rows receive session-unique UI IDs.
        self._close_reference_stream()
        for track in self._tracks.values():
            if track.state != "ended":
                self._flush_pending(track, fade_out=True)
                track.state = "ended"
        self._formal_aliases.clear()
        self._stream = stream
        self._processing_mode = None
        if self._reference_track is not None:
            self._reference_track.stream_key = stream

    def _close_reference_stream(self) -> None:
        stream = self._reference_stream
        self._reference_stream = None
        self._reference_stream_path = None
        if stream is not None:
            stream.flush()
            stream.close()

    def append_center_reference(self, block, *, channel_index: int = 6) -> None:
        """Cache the raw logical Center microphone without entering L2/L3."""
        samples = np.asarray(block.samples, dtype=np.float32)
        if (
            samples.ndim != 2
            or not 0 <= int(channel_index) < samples.shape[1]
            or len(samples) <= 0
            or len(samples) % _HOP_SAMPLES
        ):
            raise ValueError("Center reference block must contain whole 20 ms logical hops")
        stream_identity = (block.session_id, block.stream_epoch)
        with self._lock:
            self._ensure_stream(stream_identity)
            if self._reference_track is None:
                self._reference_track = _Track(
                    0,
                    0.0,
                    0.0,
                    int(block.end_sample),
                    authoritative_id=True,
                    stream_key=stream_identity,
                )
            track = self._reference_track
            if track.segment_samples >= self.segment_samples:
                self._close_reference_stream()
            path = self._segment_path(track)
            if self._reference_stream_path != path:
                self._close_reference_stream()
                self._reference_stream = path.open("ab", buffering=1 << 20)
                self._reference_stream_path = path
            center = np.ascontiguousarray(samples[:, int(channel_index)], dtype=np.float32)
            self._reference_stream.write(center.tobytes())
            track.segment_samples += len(center)
            track.last_seen_sample = int(block.end_sample)
            self._update_envelope(track, center)

    def _segment_path(self, track: _Track) -> Path:
        directory = self.cache_root / f"track_{track.track_id:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        if not track.segments or track.segment_samples >= self.segment_samples:
            path = directory / f"segment_{track.segment_index:06d}.f32"
            track.segment_index += 1
            track.segment_samples = 0
            track.segments.append(path)
            # Center Mic is the full-session input reference.  It remains
            # segmented for bounded writes but is never rolled off before the
            # user stops capture; close() still deletes the whole UI cache.
            if track.track_id != 0:
                while len(track.segments) > self.retained_segments:
                    track.segments.pop(0).unlink(missing_ok=True)
        return track.segments[-1]

    @staticmethod
    def _edge_fade(audio: np.ndarray, *, fade_in: bool, fade_out: bool) -> np.ndarray:
        output = np.asarray(audio, dtype=np.float32).copy()
        ramp = np.sin(
            np.linspace(0.0, np.pi / 2.0, _EDGE_FADE_SAMPLES, dtype=np.float32)
        ) ** 2
        if fade_in:
            output[:_EDGE_FADE_SAMPLES] *= ramp
        if fade_out:
            output[-_EDGE_FADE_SAMPLES:] *= ramp[::-1]
        return output

    def _append_audio(self, track: _Track, audio: np.ndarray) -> None:
        audio = np.asarray(audio, dtype=np.float32)
        if (
            audio.ndim != 1
            or len(audio) <= 0
            or len(audio) % _HOP_SAMPLES
            or not np.isfinite(audio).all()
        ):
            raise ValueError("Test UI L3 audio block must contain finite whole 20 ms hops")
        offset = 0
        while offset < len(audio):
            path = self._segment_path(track)
            available = self.segment_samples - track.segment_samples
            count = min(available, len(audio) - offset)
            with path.open("ab") as stream:
                stream.write(np.ascontiguousarray(audio[offset:offset + count]).tobytes())
            track.segment_samples += count
            offset += count
        self._update_envelope(track, audio)

    def _update_envelope(self, track: _Track, audio: np.ndarray) -> None:
        hops = np.asarray(audio, dtype=np.float32).reshape(-1, _HOP_SAMPLES)
        track.envelope_peaks.extend(
            float(item) for item in np.max(np.abs(hops), axis=1)
        )
        if track.track_id == 0:
            return
        maximum = self.retained_segments * self.segment_samples // _HOP_SAMPLES
        while len(track.envelope_peaks) > maximum:
            track.envelope_peaks.popleft()

    @staticmethod
    def _stable_hop(waveform: np.ndarray) -> np.ndarray:
        return np.asarray(waveform[-1440:-480], dtype=np.float32).copy()

    @staticmethod
    def _recover_hop(
        current: np.ndarray,
        current_decision: int,
        wanted_decision: int,
    ) -> np.ndarray | None:
        """Recover one canonical hop from the current 160 ms L3 context."""
        offset = int(current_decision) - int(wanted_decision)
        start = len(current) - 1_440 - offset
        stop = len(current) - 480 - offset
        if start < 0 or stop > len(current):
            return None
        return np.asarray(current[start:stop], dtype=np.float32).copy()

    def _append_timeline_hop(
        self,
        track: _Track,
        audio: np.ndarray | None,
        decision_sample: int,
        *,
        fade_out: bool = False,
    ) -> None:
        """Append exactly one 20 ms position on the absolute sample timeline."""
        decision_sample = int(decision_sample)
        previous = track.last_emitted_decision_sample
        if previous is not None:
            if decision_sample <= previous:
                return
            if decision_sample != previous + _HOP_SAMPLES:
                raise ValueError("Test UI listening timeline must advance in 20 ms hops")
        if audio is None:
            output = np.zeros(_HOP_SAMPLES, dtype=np.float32)
            track.fade_in_next = True
        else:
            output = self._edge_fade(
                audio,
                fade_in=track.fade_in_next,
                fade_out=fade_out,
            )
            track.fade_in_next = bool(fade_out)
        self._append_audio(track, output)
        track.last_emitted_decision_sample = decision_sample

    def _append_timeline_block(
        self,
        track: _Track,
        hops: list[np.ndarray | None],
        first_decision_sample: int,
    ) -> None:
        """Write a variable-duration timeline fill as one contiguous block."""
        if not hops:
            return
        first_decision_sample = int(first_decision_sample)
        previous = track.last_emitted_decision_sample
        if previous is not None and first_decision_sample != previous + _HOP_SAMPLES:
            raise ValueError("Test UI listening block does not follow the absolute timeline")
        prepared: list[np.ndarray] = []
        for hop in hops:
            if hop is None:
                prepared.append(np.zeros(_HOP_SAMPLES, dtype=np.float32))
                track.fade_in_next = True
            else:
                prepared.append(self._edge_fade(
                    hop, fade_in=track.fade_in_next, fade_out=False,
                ))
                track.fade_in_next = False
        self._append_audio(track, np.concatenate(prepared))
        track.last_emitted_decision_sample = (
            first_decision_sample + (len(hops) - 1) * _HOP_SAMPLES
        )

    def _fill_until(
        self,
        track: _Track,
        current: np.ndarray,
        current_decision: int,
        final_decision: int,
    ) -> None:
        """Recover skipped hops from current context; pad older holes with silence."""
        if track.last_emitted_decision_sample is None:
            return
        wanted = track.last_emitted_decision_sample + _HOP_SAMPLES
        first_wanted = wanted
        hops: list[np.ndarray | None] = []
        while wanted <= final_decision:
            hops.append(self._recover_hop(current, current_decision, wanted))
            wanted += _HOP_SAMPLES
        context_hops = len(current) // _HOP_SAMPLES
        if any(hop is None for hop in hops) and len(hops) >= context_hops:
            # The stable canonical slices cannot cover this discontinuity.
            # Preserve the absolute duration, but use the complete current
            # 160 ms L3 output for the newest 8 missing hops instead of
            # replacing that whole region with silence.  Only any still older
            # portion of a >160 ms hole remains exact-duration silence.
            complete = np.asarray(current, dtype=np.float32).reshape(context_hops, _HOP_SAMPLES)
            hops[-context_hops:] = [np.ascontiguousarray(item) for item in complete]
        self._append_timeline_block(track, hops, first_wanted)

    def _accept_preview(self, track: _Track, preview: BeamformPreview) -> None:
        """Finalize prior audio and preserve the absolute decision-sample timeline."""
        current = np.asarray(preview.waveform, dtype=np.float32)
        decision_sample = int(preview.decision_sample)
        previous = track.pending_waveform
        previous_decision = track.pending_decision_sample
        if previous is not None and previous_decision is not None:
            delta = decision_sample - previous_decision
            if delta <= 0 or delta % _HOP_SAMPLES:
                raise ValueError("Test UI L3 decisions must increase on the 20 ms sample grid")
            audio = self._stable_hop(previous)
            aligned_start = len(current) - 960 - delta
            aligned_stop = len(current) - 480 - delta
            if aligned_start >= 0 and aligned_stop <= len(current):
                aligned = np.asarray(current[aligned_start:aligned_stop], dtype=np.float32)
                phase = np.linspace(
                    0.0, np.pi / 2.0, _CROSSFADE_SAMPLES, dtype=np.float32,
                )
                old_weight = np.cos(phase) ** 2
                new_weight = np.sin(phase) ** 2
                audio[-_CROSSFADE_SAMPLES:] = (
                    audio[-_CROSSFADE_SAMPLES:] * old_weight + aligned * new_weight
                )
            self._append_timeline_hop(
                track,
                audio,
                previous_decision,
                fade_out=aligned_start < 0,
            )
            # A delayed L3 result may represent several skipped 20 ms
            # decisions.  Recover every still-covered hop from this 160 ms
            # waveform; older, unrecoverable positions become exact silence.
            self._fill_until(
                track, current, decision_sample, decision_sample - _HOP_SAMPLES,
            )
        elif track.last_emitted_decision_sample is not None:
            self._fill_until(
                track, current, decision_sample, decision_sample - _HOP_SAMPLES,
            )
        track.pending_waveform = current
        track.pending_decision_sample = decision_sample

    def _flush_pending(self, track: _Track, *, fade_out: bool = True) -> None:
        """Close a detected run smoothly before a gap or track end."""
        if track.pending_waveform is None:
            return
        audio = self._stable_hop(track.pending_waveform)
        self._append_timeline_hop(
            track,
            audio,
            int(track.pending_decision_sample),
            fade_out=fade_out,
        )
        track.pending_waveform = None
        track.pending_decision_sample = None

    def _advance_missing(self, track: _Track, decision_sample: int) -> None:
        """Keep coasting/ended duration honest instead of deleting silent time."""
        self._flush_pending(track, fade_out=True)
        if track.last_emitted_decision_sample is None:
            return
        wanted = track.last_emitted_decision_sample + _HOP_SAMPLES
        count = max(0, (int(decision_sample) - wanted) // _HOP_SAMPLES + 1)
        self._append_timeline_block(track, [None] * count, wanted)

    @staticmethod
    def _cached_samples(track: _Track) -> int:
        return sum(path.stat().st_size // np.dtype(np.float32).itemsize for path in track.segments if path.exists())

    def _reference_cached_samples(self) -> int:
        track = self._reference_track
        if track is None:
            return 0
        prior = sum(
            path.stat().st_size // np.dtype(np.float32).itemsize
            for path in track.segments[:-1] if path.exists()
        )
        return prior + track.segment_samples

    def _new_track(
        self,
        decision_sample: int,
        preview: BeamformPreview,
        score: float,
        *,
        track_id: int | None = None,
    ) -> _Track:
        resolved_id = self._next_track_id if track_id is None else int(track_id)
        track = _Track(
            resolved_id,
            float(preview.theta_deg),
            float(score),
            int(decision_sample),
            authoritative_id=track_id is not None,
            stream_key=self._stream,
        )
        self._next_track_id = max(self._next_track_id + (track_id is None), resolved_id + 1)
        self._tracks[track.track_id] = track
        self._accept_preview(track, preview)
        return track

    def _resolve_formal_track_id(
        self,
        formal_id: int,
        preview: BeamformPreview,
        decision_sample: int,
        reserved_track_ids: set[int],
    ) -> int:
        """Map an L2 ID rollover onto one unambiguous continuous listening ID."""
        aliased = self._formal_aliases.get(formal_id)
        if aliased is not None and aliased in self._tracks:
            if aliased not in reserved_track_ids or aliased == formal_id:
                return aliased
            self._formal_aliases.pop(formal_id, None)
        existing = self._tracks.get(formal_id)
        if (
            existing is not None
            and existing.stream_key == self._stream
            and existing.state != "ended"
        ):
            self._formal_aliases[formal_id] = formal_id
            return formal_id
        if existing is not None:
            resolved = self._next_track_id
            while resolved in self._tracks:
                resolved += 1
            self._formal_aliases[formal_id] = resolved
            return resolved
        candidates = tuple(
            track.track_id
            for track in self._tracks.values()
            if track.authoritative_id
            and track.stream_key == self._stream
            and track.track_id not in reserved_track_ids
            and 0 < int(decision_sample) - track.last_seen_sample
            <= self.formal_id_rollover_wait_samples
            and self._distance(track.theta_deg, preview.theta_deg)
            <= self.formal_id_rollover_distance_deg
        )
        # Ambiguous proximity is deliberately not merged: preserving two
        # sources is safer than joining unrelated audio into one listening ID.
        resolved = candidates[0] if len(candidates) == 1 else formal_id
        self._formal_aliases[formal_id] = resolved
        return resolved

    def update(
        self,
        window,
        candidates,
        previews,
        *,
        track_ids=(),
        prediction_flags=(),
        formal_flags=(),
        kalman_ready_flags=(),
    ) -> tuple[TrackedAudioSnapshot, ...]:
        """Append L3 audio for formal or Kalman-ready temporary L2 IDs."""
        candidates, previews = tuple(candidates), tuple(previews)
        if len(candidates) != len(previews):
            raise ValueError("Test UI tracker requires one L3 preview per formal candidate")
        track_ids = tuple(track_ids)
        prediction_flags = tuple(prediction_flags)
        formal_flags = tuple(formal_flags)
        kalman_ready_flags = tuple(kalman_ready_flags)
        if not track_ids:
            track_ids = (None,) * len(previews)
        if not prediction_flags:
            prediction_flags = (False,) * len(previews)
        if formal_flags and len(formal_flags) != len(previews):
            raise ValueError("Test UI formal-ID flags must align with L3 previews")
        if kalman_ready_flags and len(kalman_ready_flags) != len(previews):
            raise ValueError("Test UI Kalman-ready flags must align with L3 previews")
        if len(track_ids) != len(previews) or len(prediction_flags) != len(previews):
            raise ValueError("Test UI tracker metadata must align with L3 previews")
        # Runtime supplies both flags. A first-hit temporary ID is excluded;
        # its cache begins exactly when L2 declares the per-ID Kalman state
        # ready, and continues seamlessly if that same ID becomes formal.
        if formal_flags or kalman_ready_flags:
            if not formal_flags:
                formal_flags = (False,) * len(previews)
            if not kalman_ready_flags:
                kalman_ready_flags = (False,) * len(previews)
            accepted_indices = tuple(
                index for index, (is_formal, is_ready) in enumerate(
                    zip(formal_flags, kalman_ready_flags, strict=True)
                )
                if (bool(is_formal) or bool(is_ready)) and track_ids[index] is not None
            )
            candidates = tuple(candidates[index] for index in accepted_indices)
            previews = tuple(previews[index] for index in accepted_indices)
            track_ids = tuple(track_ids[index] for index in accepted_indices)
            prediction_flags = tuple(prediction_flags[index] for index in accepted_indices)
        backend_modes = {
            "ds_baseline": "ds_baseline",
            "subband_robust_baseline": "subband_robust_baseline",
        }
        preview_modes = {
            backend_modes.get(item.runtime_backend, "optimized") for item in previews
        }
        if len(preview_modes) > 1:
            raise ValueError("Test UI tracker cannot combine different L3 modes in one window")
        incoming_mode = None if not preview_modes else next(iter(preview_modes))
        stream = (window.session_id, window.stream_epoch)
        with self._lock:
            self._ensure_stream(stream)
            if (
                incoming_mode is not None
                and self._processing_mode is not None
                and incoming_mode != self._processing_mode
            ):
                self.reset()
                self._stream = stream
            if incoming_mode is not None:
                self._processing_mode = incoming_mode

            live_tracks = tuple(item for item in self._tracks.values() if item.state != "ended")
            matched_tracks: set[int] = set()
            matched_previews: set[int] = set()

            incoming_formal_ids = tuple(
                None if item is None else int(item) for item in track_ids
            )
            reserved_track_ids = {
                resolved
                for formal_id in incoming_formal_ids if formal_id is not None
                for resolved in (
                    self._formal_aliases.get(formal_id, formal_id),
                )
                if resolved in self._tracks
            }

            # L2's private IDs are authoritative whenever available.  This
            # prevents a second 30-degree UI association from splitting one
            # formal source into multiple listening files.
            for index, formal_id in enumerate(track_ids):
                if formal_id is None:
                    continue
                formal_id = int(formal_id)
                listening_id = self._resolve_formal_track_id(
                    formal_id,
                    previews[index],
                    int(window.decision_sample),
                    (reserved_track_ids | matched_tracks) - {
                        self._formal_aliases.get(formal_id, formal_id),
                    },
                )
                track = self._tracks.get(listening_id)
                if track is None:
                    track = self._new_track(
                        window.decision_sample,
                        previews[index],
                        candidates[index].normalized_score,
                        track_id=listening_id,
                    )
                    self._formal_aliases[formal_id] = listening_id
                else:
                    preview = previews[index]
                    track.theta_deg = float(preview.theta_deg)
                    track.score = float(candidates[index].normalized_score)
                    track.last_seen_sample = int(window.decision_sample)
                    track.state = "active"
                    track.authoritative_id = True
                    self._accept_preview(track, preview)
                matched_tracks.add(listening_id)
                matched_previews.add(index)

            pairs = sorted(
                (
                    (self._distance(track.theta_deg, preview.theta_deg), track.track_id, index)
                    for track in live_tracks if not track.authoritative_id
                    for index, preview in enumerate(previews) if track_ids[index] is None
                ),
                key=lambda item: item[0],
            )
            for distance, track_id, preview_index in pairs:
                if distance > self.association_distance_deg:
                    break
                if track_id in matched_tracks or preview_index in matched_previews:
                    continue
                track = self._tracks[track_id]
                preview = previews[preview_index]
                candidate = candidates[preview_index]
                track.theta_deg = float(preview.theta_deg)
                track.score = float(candidate.normalized_score)
                track.last_seen_sample = int(window.decision_sample)
                track.state = "active"
                self._accept_preview(track, preview)
                matched_tracks.add(track_id)
                matched_previews.add(preview_index)

            for index, preview in enumerate(previews):
                if index not in matched_previews:
                    track = self._new_track(
                        window.decision_sample,
                        preview,
                        candidates[index].normalized_score,
                    )
                    matched_tracks.add(track.track_id)

            for track in live_tracks:
                if track.track_id in matched_tracks:
                    continue
                self._advance_missing(track, int(window.decision_sample))
                elapsed = int(window.decision_sample) - track.last_seen_sample
                track.state = "ended" if elapsed >= self.wait_samples else "coasting"

            # A visible row must remain playable for the complete capture
            # session.  Do not prune ENDED tracks behind the UI; reset()/close()
            # release every row and file together at the next session boundary.
            return self.snapshots()

    def snapshots(self) -> tuple[TrackedAudioSnapshot, ...]:
        with self._lock:
            if self._stream is None:
                return ()
            tracks = tuple(
                TrackedAudioSnapshot(
                    self._stream[0], self._stream[1], track.track_id, track.state,
                    track.theta_deg, track.score, self._cached_samples(track),
                    waveform_envelope=tuple(track.envelope_peaks),
                )
                for track in sorted(self._tracks.values(), key=lambda item: item.track_id)
            )
            reference_samples = self._reference_cached_samples()
            reference = () if reference_samples <= 0 else (TrackedAudioSnapshot(
                self._stream[0], self._stream[1], 0, "active",
                0.0, 0.0, reference_samples,
                waveform_envelope=tuple(self._reference_track.envelope_peaks),
            ),)
            return reference + tracks

    def audio_cache_path(self, track_id: int) -> Path | None:
        """Create a stable bounded snapshot so playback never reads a live file."""
        with self._lock:
            track = self._reference_track if int(track_id) == 0 else self._tracks.get(int(track_id))
            if int(track_id) == 0 and self._reference_stream is not None:
                self._reference_stream.flush()
            if track is None or not any(path.exists() and path.stat().st_size for path in track.segments):
                return None
            self._snapshot_counter += 1
            output = self.cache_root / "playback" / f"track_{track.track_id:03d}_{self._snapshot_counter:06d}.f32"
            with output.open("wb") as target:
                for path in track.segments:
                    if path.exists():
                        with path.open("rb") as source:
                            shutil.copyfileobj(source, target, length=1 << 20)
            return output

    def close(self, *, delete_files: bool = True) -> None:
        with self._lock:
            self._close_reference_stream()
            self._tracks.clear()
            self._formal_aliases.clear()
            self._reference_track = None
            self._stream = None
            if delete_files and self.cache_root.exists():
                shutil.rmtree(self.cache_root)
