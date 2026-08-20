from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from threading import Lock

from common.angle import circular_distance_deg
from common.config import ProjectConfig
from common.data_types import CandidateDirection, DecisionWindow, ModelOrderEstimate, SpatialResponse, TrackedDirection
from common.geometry import MicGeometry

from .configuration import DirectionScanConfig
from .global_tracker import GlobalDirectionTracker, GlobalTrackerConfig
from .interface import DetailedDirectionScanner
from .music import MusicDiagnostics, MusicStateDiagnostic, RollingNormMusicScanner
from .probability_gate import ProbabilityGate, ProbabilityGateDecision, ProbabilityGateState, SourceProbability20ms


class Layer2ExecutionState(str, Enum):
    BLOCKED = "blocked"
    PROCESSED = "processed"


def _select_l3_directions(
    observed: tuple[TrackedDirection, ...],
    active: tuple[TrackedDirection, ...],
    *,
    limit: int = 3,
    minimum_separation_deg: float = 50.0,
    voice_confirmed_coasting_ids: frozenset[int] | None = None,
) -> tuple[TrackedDirection, ...]:
    """Keep L4-confirmed human tracks continuous while their geometry TTL is alive."""

    observed_confirmed = tuple(item for item in observed if item.track_state == "confirmed")
    observed_by_id = {item.track_id: item for item in observed_confirmed}
    active_by_id = {item.track_id: item for item in active}
    if voice_confirmed_coasting_ids is None:
        # Test/helper compatibility: without an explicit semantic set, retain
        # the historical observed-first ordering and then append all coasting.
        prioritized = list(observed_confirmed)
        prioritized.extend(sorted(
            (
                item for item in active
                if item.track_state == "coasting" and item.track_id not in observed_by_id
            ),
            key=lambda item: (item.missed_samples, -item.normalized_score, item.track_id),
        ))
    else:
        # In production an L4-confirmed human ID owns its L3 slot throughout
        # the three-second geometry TTL. A transient MUSIC peak must not take
        # that slot and turn a missing observation into a silent audio hop.
        human: list[TrackedDirection] = []
        for track_id in sorted(voice_confirmed_coasting_ids):
            item = observed_by_id.get(track_id, active_by_id.get(track_id))
            if item is not None and item.track_state in {"confirmed", "coasting"}:
                human.append(item)
        human_ids = {item.track_id for item in human}
        prioritized = human + [
            item for item in observed_confirmed if item.track_id not in human_ids
        ]

    selected: list[TrackedDirection] = []
    for item in prioritized:
        if len(selected) >= limit:
            break
        if all(
            circular_distance_deg(item.theta_deg, existing.theta_deg)
            >= minimum_separation_deg
            for existing in selected
        ):
            selected.append(item)
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class Layer2PipelineResult:
    """One window of authoritative L2 output.

    ``directions`` is the public contract. ``candidates`` is a compatibility
    projection with identical angles/order and never creates another identity.
    """

    state: Layer2ExecutionState
    gate_decision: ProbabilityGateDecision
    spatial_response: SpatialResponse | None
    candidates: tuple[CandidateDirection, ...]
    search_diagnostics: MusicDiagnostics | None
    candidate_track_ids: tuple[int | None, ...] = ()
    candidate_is_prediction: tuple[bool, ...] = ()
    candidate_track_is_formal: tuple[bool, ...] = ()
    candidate_track_is_new: tuple[bool, ...] = ()
    candidate_track_is_kalman_ready: tuple[bool, ...] = ()
    directions: tuple[TrackedDirection, ...] = ()
    active_tracks: tuple[TrackedDirection, ...] = ()
    model_order: ModelOrderEstimate | None = None
    music_state: MusicStateDiagnostic | None = None

    def __post_init__(self) -> None:
        identity = (self.gate_decision.session_id, self.gate_decision.stream_epoch,
                    self.gate_decision.window_id, self.gate_decision.decision_sample)
        candidates, directions, active = tuple(self.candidates), tuple(self.directions), tuple(self.active_tracks)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "active_tracks", active)
        track_ids = tuple(self.candidate_track_ids) or tuple(item.track_id for item in directions)
        prediction = tuple(self.candidate_is_prediction) or tuple(not item.is_observed for item in directions)
        formal = tuple(self.candidate_track_is_formal) or tuple(
            item.track_state in {"confirmed", "coasting"} for item in directions
        )
        is_new = tuple(self.candidate_track_is_new) or tuple(item.is_new_track for item in directions)
        kalman = tuple(self.candidate_track_is_kalman_ready) or tuple(item.kalman_applied for item in directions)
        if not directions:
            track_ids = track_ids or (None,) * len(candidates)
            prediction = prediction or (False,) * len(candidates)
            formal = formal or (False,) * len(candidates)
            is_new = is_new or (False,) * len(candidates)
            kalman = kalman or (False,) * len(candidates)
        metadata = (track_ids, prediction, formal, is_new, kalman)
        if any(len(items) != len(candidates) for items in metadata):
            raise ValueError("L2 candidate metadata must align with candidates")
        if any(item is not None and (type(item) is not int or item <= 0) for item in track_ids):
            raise ValueError("L2 track IDs must be positive integers or None")
        concrete_ids = tuple(item for item in track_ids if item is not None)
        if len(set(concrete_ids)) != len(concrete_ids):
            raise ValueError("L2 track IDs must be unique within a window")
        if any(type(item) is not bool for values in metadata[1:] for item in values):
            raise TypeError("L2 candidate state flags must be bool")
        for name, value in zip(("candidate_track_ids", "candidate_is_prediction",
                                "candidate_track_is_formal", "candidate_track_is_new",
                                "candidate_track_is_kalman_ready"), metadata, strict=True):
            object.__setattr__(self, name, value)
        if directions:
            if len(directions) != len(candidates) or track_ids != tuple(item.track_id for item in directions):
                raise ValueError("public directions must align with candidate projections")
            if any(circular_distance_deg(d.theta_deg, c.theta_deg) > 1e-6
                   for d, c in zip(directions, candidates, strict=True)):
                raise ValueError("candidate projections must use public direction angles")
        active_ids = tuple(item.track_id for item in active)
        if len(set(active_ids)) != len(active_ids):
            raise ValueError("active track IDs must be unique")
        if directions and not set(concrete_ids).issubset(active_ids):
            raise ValueError("published directions must be present in active_tracks")
        if len(candidates) > 3:
            raise ValueError("L2 cannot publish more than 3 directions")
        if any(circular_distance_deg(candidates[i].theta_deg, candidates[j].theta_deg) < 50.0
               for i in range(len(candidates)) for j in range(i + 1, len(candidates))):
            raise ValueError("published directions require 50-degree circular separation")
        if any((item.session_id, item.stream_epoch, item.window_id, item.decision_sample) != identity
               for item in (*candidates, *directions, *active)):
            raise ValueError("all L2 output objects must belong to the same window")
        if self.state is Layer2ExecutionState.BLOCKED:
            if self.gate_decision.allow_srp:
                raise ValueError("blocked L2 result requires a closed probability Gate")
            if self.spatial_response is not None or candidates or self.search_diagnostics is not None:
                raise ValueError("blocked L2 result cannot contain MUSIC observations")
            return
        if self.gate_decision.state is not ProbabilityGateState.OPEN:
            raise ValueError("processed L2 result requires an open probability Gate")
        if self.spatial_response is None or self.search_diagnostics is None:
            raise ValueError("processed L2 result requires complete MUSIC output")
        response_identity = (self.spatial_response.session_id, self.spatial_response.stream_epoch,
                             self.spatial_response.window_id, self.spatial_response.decision_sample)
        if response_identity != identity:
            raise ValueError("MUSIC response and Gate must belong to the same window")


class Layer2Pipeline:
    """Probability Gate -> rolling NormMUSIC -> permanent global ID tracker."""

    def __init__(self, gate: ProbabilityGate, scanner: DetailedDirectionScanner,
                 tracker: GlobalDirectionTracker | None = None) -> None:
        self.gate, self.scanner = gate, scanner
        self.id_tracker = tracker or GlobalDirectionTracker()
        self.last_kalman_error: str | None = None
        self.last_id_tracking_error: str | None = None
        self._voice_feedback: deque[tuple[str, int, int, int, float, bool]] = deque(maxlen=4096)
        self._voice_feedback_lock = Lock()

    @classmethod
    def from_project(cls, config: ProjectConfig, *, scanner: DetailedDirectionScanner | None = None) -> "Layer2Pipeline":
        if config.layer2.scanner_backend != "frequency_normalized_music":
            raise ValueError(f"unsupported L2 scanner backend: {config.layer2.scanner_backend}")
        if config.layer2.probability_gate.backend != ProbabilityGate.backend:
            raise ValueError(f"unsupported L2 probability Gate backend: {config.layer2.probability_gate.backend}")
        tracking = config.layer2.direction_id_tracking
        return cls(ProbabilityGate(), scanner or RollingNormMusicScanner(), GlobalDirectionTracker(
            GlobalTrackerConfig(
                association_gate_deg=tracking.association_gate_deg,
                max_velocity_dps=tracking.max_velocity_dps,
                confirmation_observations=tracking.confirmation_observations,
                confirmation_window_samples=tracking.confirmation_window_ms * 48,
                coasting_ttl_samples=tracking.coasting_ttl_ms * 48,
                miss_cost=tracking.miss_cost,
                birth_cost=tracking.birth_cost,
            )
        ))

    def reset(self) -> None:
        reset_scanner = getattr(self.scanner, "reset", None)
        if callable(reset_scanner):
            reset_scanner()
        self.id_tracker.reset()
        with self._voice_feedback_lock:
            self._voice_feedback.clear()
        self.last_kalman_error = self.last_id_tracking_error = None

    def submit_voice_feedback(
        self,
        session_id: str,
        stream_epoch: int,
        decision_sample: int,
        track_id: int,
        probability: float,
        is_voice: bool,
    ) -> bool:
        if (
            not session_id or min(stream_epoch, decision_sample) < 0
            or type(track_id) is not int or track_id <= 0
            or type(is_voice) is not bool
            or not isfinite(probability) or not 0.0 <= probability <= 1.0
        ):
            return False
        with self._voice_feedback_lock:
            self._voice_feedback.append(
                (session_id, stream_epoch, decision_sample, track_id, float(probability), is_voice)
            )
        return True

    def _drain_voice_feedback(self) -> None:
        with self._voice_feedback_lock:
            pending = tuple(self._voice_feedback)
            self._voice_feedback.clear()
        for item in pending:
            self.id_tracker.apply_voice_feedback(*item)

    def process(self, window: DecisionWindow, probabilities: tuple[SourceProbability20ms, ...],
                geometry: MicGeometry, scan_config: DirectionScanConfig, *, gate_threshold: float,
                gate_config_revision: int, scan_config_revision: int = 0,
                direction_kalman_enabled: bool = False,
                direction_kalman_q_scale: float = 1.0,
                direction_kalman_r_scale: float = 1.0) -> Layer2PipelineResult:
        if type(direction_kalman_enabled) is not bool:
            raise TypeError("L2 Kalman switch must be bool")
        self._drain_voice_feedback()
        voice_confirmed_ids = self.id_tracker.voice_confirmed_track_ids(
            window.session_id, window.stream_epoch, window.decision_sample
        )
        decision = self.gate.evaluate(window, probabilities, threshold=gate_threshold,
                                      config_revision=gate_config_revision)
        if voice_confirmed_ids and decision.state is ProbabilityGateState.CLOSED:
            decision = replace(
                decision,
                state=ProbabilityGateState.OPEN,
                sound_present=True,
                reason="voice_confirmed_id_force_open",
                diagnostics=decision.diagnostics + (
                    "voice_confirmed_id_force_open=true",
                    "force_open_requires_l4_voice_confirmations=2",
                ),
            )
        response: SpatialResponse | None = None
        diagnostics: MusicDiagnostics | None = None
        observations: tuple[CandidateDirection, ...] = ()
        if decision.allow_srp:
            response, observations, diagnostics = self.scanner.scan_detailed(
                window, geometry, scan_config, scan_config_revision)
        observed_directions, active = self.id_tracker.update(
            window.session_id, window.stream_epoch, window.decision_sample, observations,
            window_id=window.window_id, doa_start_sample=window.doa_start_sample,
            doa_end_sample=window.doa_end_sample, kalman_enabled=direction_kalman_enabled,
            q_scale=direction_kalman_q_scale, r_scale=direction_kalman_r_scale,
            allow_births=True if diagnostics is None else diagnostics.births_allowed)
        self.last_id_tracking_error = self.last_kalman_error = None
        directions = (
            _select_l3_directions(
                observed_directions,
                active,
                voice_confirmed_coasting_ids=voice_confirmed_ids,
            )
            if decision.allow_srp else ()
        )
        candidates = tuple(CandidateDirection(
            item.session_id, item.stream_epoch, item.window_id, item.decision_sample,
            item.doa_start_sample, item.doa_end_sample, item.theta_deg,
            item.raw_score, item.normalized_score) for item in directions)
        state = Layer2ExecutionState.PROCESSED if decision.allow_srp else Layer2ExecutionState.BLOCKED
        return Layer2PipelineResult(
            state, decision, response, candidates, diagnostics,
            tuple(item.track_id for item in directions),
            tuple(not item.is_observed for item in directions),
            tuple(item.track_state in {"confirmed", "coasting"} for item in directions),
            tuple(item.is_new_track for item in directions),
            tuple(item.kalman_applied for item in directions), directions, active,
            getattr(self.scanner, "model_order", None),
            getattr(self.scanner, "last_state_diagnostic", None))
