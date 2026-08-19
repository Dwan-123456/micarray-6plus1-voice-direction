from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product

from common.data_types import CandidateDirection

from .direction_smoothing import circular_delta_deg


_SAMPLE_RATE = 48_000
_HOP_SAMPLES = 960


@dataclass(frozen=True, slots=True)
class DirectionIdTrackingV2Config:
    association_gate_deg: float = 20.0
    max_tracks: int = 4
    max_formal_tracks: int = 2
    kalman_min_matches: int = 2
    kalman_match_window_samples: int = _SAMPLE_RATE
    confirmation_age_samples: int = 2 * _SAMPLE_RATE
    confirmation_min_matches: int = 5
    formal_lease_samples: int = 3 * _SAMPLE_RATE
    provisional_hold_samples: int = 3 * _SAMPLE_RATE
    short_prediction_samples: int = _SAMPLE_RATE // 2
    human_confidence_threshold: float = 0.60


@dataclass(slots=True)
class _Track:
    track_id: int
    theta_deg: float
    created_sample: int
    last_observed_sample: int
    last_norm: float
    match_samples: deque[int]
    total_matches: int = 1
    confirmation_matches: int = 1
    voice_mass: float = 0.0
    nonvoice_mass: float = 0.0
    voice_hits: int = 0
    formal: bool = False
    lease_expiry_sample: int | None = None

    @property
    def human_confidence(self) -> float:
        return (1.0 + self.voice_mass) / (2.0 + self.voice_mass + self.nonvoice_mass)


@dataclass(frozen=True, slots=True)
class _PublishedPoint:
    track_id: int
    theta_deg: float


class DirectionIdTrackerV2:
    """Four-hypothesis private tracker with L4-controlled speaker semantics."""

    backend = "confidence_id_tracker_v2"

    def __init__(
        self, config: DirectionIdTrackingV2Config = DirectionIdTrackingV2Config()
    ) -> None:
        self.config = config
        self._stream_key: tuple[str, int] | None = None
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1
        self._last_decision_sample: int | None = None
        self._last_assignments: tuple[int, ...] = ()
        self._last_assignment_is_new: tuple[bool, ...] = ()
        self._published: deque[tuple[int, tuple[_PublishedPoint, ...]]] = deque(maxlen=512)

    @property
    def active_track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tracks))

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    @property
    def last_assignments(self) -> tuple[int, ...]:
        return self._last_assignments

    @property
    def last_assignment_is_new(self) -> tuple[bool, ...]:
        return self._last_assignment_is_new

    @property
    def kalman_ready_track_ids(self) -> tuple[int, ...]:
        if self._last_decision_sample is None:
            return ()
        return tuple(
            track_id for track_id, track in sorted(self._tracks.items())
            if len(track.match_samples) >= self.config.kalman_min_matches
            and track.match_samples[-1] - track.match_samples[-2]
            <= self.config.kalman_match_window_samples
        )

    @property
    def prediction_track_ids(self) -> tuple[int, ...]:
        if self._last_decision_sample is None:
            return ()
        now = self._last_decision_sample
        return tuple(
            track_id for track_id, track in sorted(self._tracks.items())
            if track_id in self.kalman_ready_track_ids
            and now > track.last_observed_sample
            and (
                (track.formal and track.lease_expiry_sample is not None
                 and now <= track.lease_expiry_sample)
                or (not track.formal
                    and now - track.last_observed_sample <= self.config.short_prediction_samples)
            )
        )

    def reset(self) -> None:
        self._stream_key = None
        self._tracks.clear()
        self._next_track_id = 1
        self._last_decision_sample = None
        self._last_assignments = ()
        self._last_assignment_is_new = ()
        self._published.clear()

    def prepare_stream(self, session_id: str, stream_epoch: int) -> None:
        key = (session_id, stream_epoch)
        if key != self._stream_key:
            self.reset()
            self._stream_key = key

    def confirmed_track_ids_at(
        self, decision_sample: int, *, require_advance: bool = True
    ) -> tuple[int, ...]:
        if self._last_decision_sample is None:
            return ()
        if require_advance and decision_sample <= self._last_decision_sample:
            raise ValueError("ID tracking decision_sample must advance monotonically")
        return tuple(
            track_id for track_id, track in sorted(self._tracks.items())
            if track.formal and track.lease_expiry_sample is not None
            and decision_sample <= track.lease_expiry_sample
        )

    @property
    def confirmed_track_ids(self) -> tuple[int, ...]:
        if self._last_decision_sample is None:
            return ()
        return self.confirmed_track_ids_at(self._last_decision_sample, require_advance=False)

    def update(
        self,
        session_id: str,
        stream_epoch: int,
        decision_sample: int,
        candidates: tuple[CandidateDirection, ...],
        *,
        existing_formal_only: bool = False,
        predicted_angles: dict[int, float] | None = None,
    ) -> tuple[CandidateDirection, ...]:
        self.prepare_stream(session_id, stream_epoch)
        if self._last_decision_sample is not None and decision_sample <= self._last_decision_sample:
            raise ValueError("ID tracking decision_sample must advance monotonically")
        self._expire(decision_sample)
        candidates = tuple(candidates)
        predicted_angles = {} if predicted_angles is None else dict(predicted_angles)
        eligible = {
            track_id for track_id, track in self._tracks.items()
            if not existing_formal_only or track.formal
        }
        assignment = self._associate(candidates, eligible, predicted_angles)
        protected_ids = {track_id for track_id in assignment if track_id is not None}
        output: list[CandidateDirection] = []
        ids: list[int] = []
        new_flags: list[bool] = []
        confirmation_end_by_track: dict[int, int] = {}
        for candidate, track_id in zip(candidates, assignment, strict=True):
            is_new = False
            if track_id is None:
                if existing_formal_only:
                    continue
                self._make_room(protected_ids)
                if len(self._tracks) >= self.config.max_tracks:
                    continue
                track_id = self._next_track_id
                self._next_track_id += 1
                track = _Track(
                    track_id, candidate.theta_deg, decision_sample, decision_sample,
                    candidate.normalized_score, deque((decision_sample,), maxlen=16),
                )
                self._tracks[track_id] = track
                is_new = True
            else:
                track = self._tracks[track_id]
                track.theta_deg = candidate.theta_deg
                track.last_observed_sample = decision_sample
                track.last_norm = candidate.normalized_score
                track.total_matches += 1
                track.match_samples.append(decision_sample)
                confirmation_end = track.created_sample + self.config.confirmation_age_samples
                confirmation_end_by_track[track_id] = confirmation_end
                if decision_sample <= confirmation_end:
                    track.confirmation_matches += 1
            output.append(candidate)
            ids.append(track_id)
            new_flags.append(is_new)
        self._promote(decision_sample)
        self._last_assignments = tuple(ids)
        self._last_assignment_is_new = tuple(new_flags)
        self._last_decision_sample = decision_sample
        return tuple(output)

    def select_public(
        self,
        candidates: tuple[CandidateDirection, ...],
        track_ids: tuple[int, ...],
        prediction_flags: tuple[bool, ...],
        min_distance_deg: float,
    ) -> tuple[tuple[CandidateDirection, ...], tuple[int, ...], tuple[bool, ...]]:
        items = list(zip(candidates, track_ids, prediction_flags, strict=True))
        items.sort(key=lambda item: (-self._confidence(item[1]), -item[0].normalized_score, item[1]))
        chosen: list[tuple[CandidateDirection, int, bool]] = []
        for item in items:
            if any(abs(circular_delta_deg(item[0].theta_deg, old[0].theta_deg)) < min_distance_deg
                   for old in chosen):
                continue
            chosen.append(item)
            if len(chosen) == 3:
                break
        return (
            tuple(item[0] for item in chosen),
            tuple(item[1] for item in chosen),
            tuple(item[2] for item in chosen),
        )

    def record_published(
        self, decision_sample: int, candidates: tuple[CandidateDirection, ...],
        track_ids: tuple[int, ...]
    ) -> None:
        self._published.append((decision_sample, tuple(
            _PublishedPoint(track_id, candidate.theta_deg)
            for candidate, track_id in zip(candidates, track_ids, strict=True)
            if track_id in self._tracks
        )))

    def apply_classification_feedback(
        self, session_id: str, stream_epoch: int, decision_sample: int,
        theta_deg: float, probability: float, is_voice: bool,
    ) -> int | None:
        if (session_id, stream_epoch) != self._stream_key:
            return None
        points = next((points for sample, points in reversed(self._published)
                       if sample == decision_sample), ())
        matches = [point for point in points
                   if abs(circular_delta_deg(theta_deg, point.theta_deg))
                   <= self.config.association_gate_deg]
        if len(matches) != 1:
            return None
        track = self._tracks.get(matches[0].track_id)
        if track is None:
            return None
        if is_voice:
            track.voice_hits += 1
            track.voice_mass += float(probability)
            # A positive semantic result clears prior negative evidence. L2
            # never hides an angle based on L4, because L4 can only classify
            # angles that L2 continues to publish.
            track.nonvoice_mass = 0.0
            if track.formal:
                track.lease_expiry_sample = max(
                    track.lease_expiry_sample or decision_sample,
                    decision_sample + self.config.formal_lease_samples,
                )
        else:
            track.nonvoice_mass += 1.0 - float(probability)
        self._promote(self._last_decision_sample or decision_sample)
        return track.track_id

    def apply_voice_feedback(
        self, session_id: str, stream_epoch: int, decision_sample: int, theta_deg: float
    ) -> int | None:
        return self.apply_classification_feedback(
            session_id, stream_epoch, decision_sample, theta_deg, 1.0, True
        )

    def _promote(self, now: int) -> None:
        formal_count = sum(track.formal for track in self._tracks.values())
        eligible = sorted(
            (track for track in self._tracks.values() if not track.formal),
            key=lambda track: (-self._confidence(track.track_id), track.track_id),
        )
        for track in eligible:
            if formal_count >= self.config.max_formal_tracks:
                break
            if (now - track.created_sample >= self.config.confirmation_age_samples
                    and track.confirmation_matches >= self.config.confirmation_min_matches
                    and track.voice_hits >= 1
                    and track.human_confidence >= self.config.human_confidence_threshold):
                track.formal = True
                track.lease_expiry_sample = now + self.config.formal_lease_samples
                formal_count += 1

    def _expire(self, now: int) -> None:
        for track_id, track in tuple(self._tracks.items()):
            if track.formal:
                if track.lease_expiry_sample is None or now > track.lease_expiry_sample:
                    del self._tracks[track_id]
            elif now - track.last_observed_sample > self.config.provisional_hold_samples:
                del self._tracks[track_id]

    def _make_room(self, protected_ids: set[int]) -> None:
        if len(self._tracks) < self.config.max_tracks:
            return
        removable = [
            track for track in self._tracks.values()
            if not track.formal and track.track_id not in protected_ids
        ]
        if not removable:
            return
        victim = min(removable, key=lambda track: (
            self._confidence(track.track_id), track.last_observed_sample, track.track_id
        ))
        del self._tracks[victim.track_id]

    def _confidence(self, track_id: int) -> float:
        track = self._tracks[track_id]
        persistence = min(track.total_matches / 5.0, 1.0)
        semantic = track.human_confidence
        score = 0.45 * track.last_norm + 0.30 * semantic + 0.20 * persistence
        if track.formal:
            score += 0.15
        return score

    def _associate(
        self, candidates: tuple[CandidateDirection, ...], eligible: set[int],
        predicted: dict[int, float]
    ) -> tuple[int | None, ...]:
        if not candidates:
            return ()
        track_ids = tuple(sorted(eligible))
        options: list[tuple[int | None, ...]] = []
        costs: list[dict[int, float]] = []
        for candidate in candidates:
            item_costs = {
                track_id: abs(circular_delta_deg(
                    candidate.theta_deg, predicted.get(track_id, self._tracks[track_id].theta_deg)
                )) for track_id in track_ids
            }
            viable = tuple(track_id for track_id in track_ids
                           if item_costs[track_id] <= self.config.association_gate_deg)
            options.append((None, *viable))
            costs.append(item_costs)
        best: tuple[int | None, ...] | None = None
        best_key: tuple[object, ...] | None = None
        for assignment in product(*options):
            assigned = tuple(item for item in assignment if item is not None)
            if len(assigned) != len(set(assigned)):
                continue
            total = sum(costs[i][track_id] for i, track_id in enumerate(assignment)
                        if track_id is not None)
            key = (-len(assigned), round(total, 9), tuple(
                1_000_000 if item is None else item for item in assignment
            ))
            if best_key is None or key < best_key:
                best_key, best = key, assignment
        return best or (None,) * len(candidates)
