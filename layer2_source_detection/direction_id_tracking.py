from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product

from common.data_types import CandidateDirection

from .direction_smoothing import circular_delta_deg


_HOP_SAMPLES = 960
_PUBLISHED_HISTORY_LIMIT = 256


@dataclass(frozen=True, slots=True)
class DirectionIdTrackingConfig:
    association_gate_deg: float = 20.0
    prediction_association_gate_deg: float = 30.0
    max_missed_windows: int = 150
    confirmation_min_age_windows: int = 150
    confirmation_min_matches: int = 5
    prediction_hold_windows: int = 150


@dataclass(slots=True)
class _IdTrack:
    track_id: int
    theta_deg: float
    created_decision_sample: int
    missed_windows: int = 0
    confirmation_matches: int = 1
    voice_confirmed_during_confirmation: bool = False
    prediction_eligible: bool = False
    formalized_decision_sample: int | None = None
    voice_expiry_sample: int | None = None


@dataclass(frozen=True, slots=True)
class _PublishedTrackPoint:
    track_id: int
    theta_deg: float


class DirectionIdTracker:
    """Assign private IDs and own their L4-confirmed three-second voice lease."""

    backend = "circular_id_tracker_v4"

    def __init__(self, config: DirectionIdTrackingConfig = DirectionIdTrackingConfig()) -> None:
        self.config = config
        self._stream_key: tuple[str, int] | None = None
        self._tracks: dict[int, _IdTrack] = {}
        self._next_track_id = 1
        self._last_assignments: tuple[int, ...] = ()
        self._last_assignment_is_new: tuple[bool, ...] = ()
        self._last_decision_sample: int | None = None
        self._published_history: deque[
            tuple[int, tuple[_PublishedTrackPoint, ...]]
        ] = deque(maxlen=_PUBLISHED_HISTORY_LIMIT)

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    @property
    def active_track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tracks))

    @property
    def last_assignments(self) -> tuple[int, ...]:
        """Diagnostic-only IDs; they are intentionally absent from public DTOs."""
        return self._last_assignments

    @property
    def last_assignment_is_new(self) -> tuple[bool, ...]:
        """Whether each last assignment was first created in that update."""
        return self._last_assignment_is_new

    @property
    def confirmed_track_ids(self) -> tuple[int, ...]:
        """All currently live formal IDs at the latest processed audio sample."""
        if self._last_decision_sample is None:
            return ()
        return self.confirmed_track_ids_at(self._last_decision_sample, require_advance=False)

    def confirmed_track_ids_at(
        self, decision_sample: int, *, require_advance: bool = True
    ) -> tuple[int, ...]:
        """Formal IDs whose L4-controlled voice lease includes this sample."""
        if self._last_decision_sample is None:
            return ()
        if require_advance and decision_sample <= self._last_decision_sample:
            raise ValueError("ID tracking decision_sample must advance monotonically")
        return tuple(
            track_id for track_id, track in sorted(self._tracks.items())
            if track.prediction_eligible
            and track.voice_expiry_sample is not None
            and decision_sample <= track.voice_expiry_sample
        )

    @property
    def prediction_track_ids(self) -> tuple[int, ...]:
        if self._last_decision_sample is None:
            return ()
        return tuple(
            track_id for track_id, track in sorted(self._tracks.items())
            if track.prediction_eligible
            and track.voice_expiry_sample is not None
            and self._last_decision_sample <= track.voice_expiry_sample
            and track.missed_windows > 0
        )

    def reset(self) -> None:
        self._stream_key = None
        self._tracks.clear()
        self._next_track_id = 1
        self._last_assignments = ()
        self._last_assignment_is_new = ()
        self._last_decision_sample = None
        self._published_history.clear()

    def prepare_stream(self, session_id: str, stream_epoch: int) -> None:
        key = (session_id, stream_epoch)
        if key != self._stream_key:
            self.reset()
            self._stream_key = key

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
        """Associate observations; angle observations never extend a formal lease."""
        candidates = tuple(candidates)
        self.prepare_stream(session_id, stream_epoch)
        if self._last_decision_sample is None:
            elapsed_windows = 1
        else:
            delta = decision_sample - self._last_decision_sample
            if delta <= 0:
                raise ValueError("ID tracking decision_sample must advance monotonically")
            elapsed_windows = max(1, (delta + _HOP_SAMPLES - 1) // _HOP_SAMPLES)

        self._expire_tracks(decision_sample, elapsed_windows)
        eligible_ids = None
        if existing_formal_only:
            eligible_ids = set(self.confirmed_track_ids_at(decision_sample, require_advance=False))
        assignment = self._associate(candidates, eligible_ids, predicted_angles)
        matched = {item for item in assignment if item is not None}
        for track_id, track in tuple(self._tracks.items()):
            if track_id not in matched:
                track.missed_windows += elapsed_windows

        accepted: list[CandidateDirection] = []
        ids: list[int] = []
        is_new: list[bool] = []
        for candidate, track_id in zip(candidates, assignment, strict=True):
            if track_id is None:
                if existing_formal_only:
                    continue
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[track_id] = _IdTrack(
                    track_id, candidate.theta_deg, decision_sample
                )
                assignment_is_new = True
            else:
                track = self._tracks[track_id]
                track.theta_deg = candidate.theta_deg
                track.missed_windows = 0
                if not track.prediction_eligible and not existing_formal_only:
                    track.confirmation_matches += 1
                assignment_is_new = False
            accepted.append(candidate)
            ids.append(track_id)
            is_new.append(assignment_is_new)

        if not existing_formal_only:
            self._promote_or_expire_provisional(decision_sample, accepted, ids, is_new)
        self._last_assignments = tuple(ids)
        self._last_assignment_is_new = tuple(is_new)
        self._last_decision_sample = decision_sample
        return tuple(accepted)

    def record_published(
        self,
        decision_sample: int,
        candidates: tuple[CandidateDirection, ...],
        track_ids: tuple[int, ...],
    ) -> None:
        """Remember the private ID behind each public angle for delayed L4 feedback."""
        if len(candidates) != len(track_ids):
            raise ValueError("published L2 candidates and private IDs must align")
        points = tuple(
            _PublishedTrackPoint(track_id, candidate.theta_deg)
            for candidate, track_id in zip(candidates, track_ids, strict=True)
            if track_id in self._tracks
        )
        self._published_history.append((decision_sample, points))

    def apply_voice_feedback(
        self,
        session_id: str,
        stream_epoch: int,
        decision_sample: int,
        theta_deg: float,
    ) -> int | None:
        """Extend one uniquely matching live formal ID from an L4 voice point."""
        if (session_id, stream_epoch) != self._stream_key:
            return None
        history = next(
            (points for sample, points in reversed(self._published_history)
             if sample == decision_sample),
            (),
        )
        matches = tuple(
            point for point in history
            if abs(circular_delta_deg(theta_deg, point.theta_deg))
            <= self.config.association_gate_deg
        )
        if len(matches) != 1:
            return None
        track = self._tracks.get(matches[0].track_id)
        if track is None or self._last_decision_sample is None:
            return None
        if not track.prediction_eligible:
            confirmation_end = (
                track.created_decision_sample
                + self.config.confirmation_min_age_windows * _HOP_SAMPLES
            )
            if not track.created_decision_sample <= decision_sample <= confirmation_end:
                return None
            track.voice_confirmed_during_confirmation = True
            return track.track_id
        if (
            track.voice_expiry_sample is None
            or self._last_decision_sample > track.voice_expiry_sample
        ):
            return None
        extension = decision_sample + self.config.prediction_hold_windows * _HOP_SAMPLES
        track.voice_expiry_sample = max(track.voice_expiry_sample, extension)
        return track.track_id

    def _expire_tracks(self, decision_sample: int, elapsed_windows: int) -> None:
        for track_id, track in tuple(self._tracks.items()):
            if track.prediction_eligible:
                if track.voice_expiry_sample is None or decision_sample > track.voice_expiry_sample:
                    del self._tracks[track_id]
            elif track.missed_windows + elapsed_windows > self.config.max_missed_windows:
                del self._tracks[track_id]

    def _promote_or_expire_provisional(
        self,
        decision_sample: int,
        accepted: list[CandidateDirection],
        ids: list[int],
        is_new: list[bool],
    ) -> None:
        confirmation_samples = self.config.confirmation_min_age_windows * _HOP_SAMPLES
        lease_samples = self.config.prediction_hold_windows * _HOP_SAMPLES
        for track_id, track in tuple(self._tracks.items()):
            if track.prediction_eligible:
                continue
            age_samples = decision_sample - track.created_decision_sample
            if age_samples < confirmation_samples:
                continue
            if (
                track.confirmation_matches >= self.config.confirmation_min_matches
                and track.voice_confirmed_during_confirmation
            ):
                track.prediction_eligible = True
                track.formalized_decision_sample = decision_sample
                track.voice_expiry_sample = decision_sample + lease_samples
                continue
            if track_id in ids:
                index = ids.index(track_id)
                replacement_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[replacement_id] = _IdTrack(
                    replacement_id, accepted[index].theta_deg, decision_sample
                )
                ids[index] = replacement_id
                is_new[index] = True
            del self._tracks[track_id]

    def _associate(
        self,
        candidates: tuple[CandidateDirection, ...],
        eligible_ids: set[int] | None = None,
        predicted_angles: dict[int, float] | None = None,
    ) -> tuple[int | None, ...]:
        if not candidates:
            return ()
        track_ids = tuple(
            track_id for track_id in sorted(self._tracks)
            if eligible_ids is None or track_id in eligible_ids
        )
        predicted_angles = {} if predicted_angles is None else dict(predicted_angles)
        options = []
        association_costs: list[dict[int, float]] = []
        for candidate in candidates:
            costs = {
                track_id: abs(circular_delta_deg(
                    candidate.theta_deg,
                    predicted_angles.get(track_id, self._tracks[track_id].theta_deg),
                ))
                for track_id in track_ids
            }
            viable = sorted(
                (
                    (distance, track_id)
                    for track_id, distance in costs.items()
                    if distance <= (
                        self.config.prediction_association_gate_deg
                        if track_id in predicted_angles
                        else self.config.association_gate_deg
                    )
                ),
                key=lambda item: (item[0], item[1]),
            )
            # Keep every valid edge for the two-source one-to-one assignment.
            # With one observation this chooses its nearest forecast; with
            # two observations it chooses the minimum-total-distance pairing
            # and never assigns one private ID twice.
            options.append((None, *(track_id for _distance, track_id in viable)))
            association_costs.append(costs)
        best = None
        best_key = None
        for assignment in product(*options):
            assigned = tuple(item for item in assignment if item is not None)
            if len(assigned) != len(set(assigned)):
                continue
            cost = sum(
                association_costs[index][track_id]
                for index, track_id in enumerate(assignment)
                if track_id is not None
            )
            key = (-len(assigned), round(cost, 12), tuple(
                self._next_track_id + 10 if item is None else item for item in assignment
            ))
            if best_key is None or key < best_key:
                best_key, best = key, assignment
        return tuple(best or (None,) * len(candidates))
