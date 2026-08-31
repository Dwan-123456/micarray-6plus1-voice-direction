from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from time import perf_counter

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
    MUSIC_SKIPPED = "music_skipped"
    PROCESSED = "processed"


def _select_l3_directions(
    observed: tuple[TrackedDirection, ...],
    active: tuple[TrackedDirection, ...],
    *,
    limit: int = 3,
    minimum_separation_deg: float = 50.0,
) -> tuple[TrackedDirection, ...]:
    """Publish observed starts plus formal/coasting IDs using authoritative L2 state."""

    observed_confirmed = tuple(item for item in observed if item.track_state == "confirmed")
    observed_tentative = tuple(item for item in observed if item.track_state == "tentative")
    observed_by_id = {item.track_id: item for item in (*observed_confirmed, *observed_tentative)}
    prioritized = list(observed_confirmed)
    # Tentative observations are published immediately for low-latency L2
    # diagnostics. They remain lower priority than confirmed tracks.
    prioritized.extend(observed_tentative)
    prioritized.extend(sorted(
        (
            item for item in active
            if item.track_state == "coasting"
            and item.track_id not in observed_by_id
        ),
        key=lambda item: (item.missed_samples, -item.normalized_score, item.track_id),
    ))

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
    direction_id_tracking_enabled: bool = True
    id_tracking_ms: float | None = None
    music_effective_order: int | None = None
    music_skip_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.direction_id_tracking_enabled) is not bool:
            raise TypeError("L2 direction ID tracking flag must be bool")
        if self.id_tracking_ms is not None and (
            not isfinite(self.id_tracking_ms) or self.id_tracking_ms < 0.0
        ):
            raise ValueError("L2 ID tracking timing must be non-negative finite or None")
        if self.music_effective_order is not None and (
            type(self.music_effective_order) is not int
            or self.music_effective_order not in {0, 1, 2, 3}
        ):
            raise ValueError("effective MUSIC order must be 0, 1, 2, 3, or None")
        if self.music_skip_reason is not None and not self.music_skip_reason:
            raise ValueError("MUSIC skip reason cannot be empty")
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
        if self.state is Layer2ExecutionState.MUSIC_SKIPPED:
            if self.gate_decision.state is not ProbabilityGateState.OPEN:
                raise ValueError("MUSIC-skipped L2 result requires an open probability Gate")
            if self.music_effective_order not in {0, None} or self.music_skip_reason is None:
                raise ValueError("MUSIC-skipped result requires order 0/None and a reason")
            if self.spatial_response is not None or self.search_diagnostics is not None:
                raise ValueError("MUSIC-skipped result cannot contain a MUSIC spectrum")
            if any(item.is_observed for item in directions):
                raise ValueError("MUSIC-skipped result can publish only coasting predictions")
            return
        if self.gate_decision.state is not ProbabilityGateState.OPEN:
            if self.spatial_response is not None or self.search_diagnostics is not None:
                raise ValueError("closed-Gate prediction output cannot contain MUSIC observations")
            if not directions or any(
                item.track_state != "coasting" or item.is_observed for item in directions
            ):
                raise ValueError("closed-Gate processed output requires only coasting predictions")
            return
        if self.spatial_response is None or self.search_diagnostics is None:
            raise ValueError("processed L2 result requires complete MUSIC output")
        if self.music_effective_order not in {1, 2, 3} or self.music_skip_reason is not None:
            raise ValueError("processed open-Gate result requires an applied MUSIC order")
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
        self.last_id_tracking_error: str | None = None
        self._direction_id_tracking_enabled = True
        self._gate_activity_key: tuple[str, int, int] | None = None
        self._consecutive_gate_open_hops = 0

    @classmethod
    def from_project(cls, config: ProjectConfig, *, scanner: DetailedDirectionScanner | None = None) -> "Layer2Pipeline":
        if config.layer2.probability_gate.backend != ProbabilityGate.backend:
            raise ValueError(f"unsupported L2 probability Gate backend: {config.layer2.probability_gate.backend}")
        tracking = config.layer2.direction_id_tracking
        doa_scanner = scanner or RollingNormMusicScanner()
        tracker_config = GlobalTrackerConfig(
            **tracking.model_dump(exclude={"confirmation_window_ms", "tentative_ttl_ms", "coasting_ttl_ms"}),
            confirmation_window_samples=tracking.confirmation_window_ms * 48,
            tentative_ttl_samples=tracking.tentative_ttl_ms * 48,
            coasting_ttl_samples=tracking.coasting_ttl_ms * 48,
        )
        return cls(
            ProbabilityGate(), doa_scanner, GlobalDirectionTracker(tracker_config),
        )

    def reset(self) -> None:
        reset_scanner = getattr(self.scanner, "reset", None)
        if callable(reset_scanner):
            reset_scanner()
        self.id_tracker.reset()
        self.last_id_tracking_error = None
        self._direction_id_tracking_enabled = True
        self._gate_activity_key = None
        self._consecutive_gate_open_hops = 0

    def _update_gate_activity(
        self, window: DecisionWindow, decision: ProbabilityGateDecision
    ) -> int:
        previous = self._gate_activity_key
        sample_delta = 0 if previous is None else window.decision_sample - previous[2]
        continuous = (
            previous is not None
            and previous[:2] == (window.session_id, window.stream_epoch)
            and sample_delta > 0
            and sample_delta % 960 == 0
        )
        if decision.state is not ProbabilityGateState.OPEN:
            self._consecutive_gate_open_hops = 0
        elif not continuous:
            self._consecutive_gate_open_hops = 1
        else:
            # Adaptive L2 may compute only one of every N contiguous 20 ms
            # windows and explicitly reuses the preceding Gate/output on the
            # skipped windows.  Count those reused hops here so a 40--200 ms
            # fallback period cannot keep MUSIC birth warm-up stuck at one.
            self._consecutive_gate_open_hops += sample_delta // 960
        self._gate_activity_key = (
            window.session_id,
            window.stream_epoch,
            window.decision_sample,
        )
        return self._consecutive_gate_open_hops

    def evaluate_gate(
        self,
        window: DecisionWindow,
        probabilities: tuple[SourceProbability20ms, ...],
        *,
        gate_threshold: float,
        gate_config_revision: int,
    ) -> tuple[ProbabilityGateDecision, int]:
        """Evaluate the current Gate before any source-count or MUSIC work."""

        decision = self.gate.evaluate(
            window,
            probabilities,
            threshold=gate_threshold,
            config_revision=gate_config_revision,
        )
        return decision, self._update_gate_activity(window, decision)

    def process_prepared(
        self,
        window: DecisionWindow,
        decision: ProbabilityGateDecision,
        active_frame_count: int,
        geometry: MicGeometry,
        scan_config: DirectionScanConfig,
        *,
        music_effective_order: int | None,
        music_skip_reason: str | None = None,
        scan_config_revision: int = 0,
        direction_id_tracking_enabled: bool = True,
    ) -> Layer2PipelineResult:
        """Run MUSIC/tracking after the Gate and source-count plan are fixed."""

        identity = (
            window.session_id,
            window.stream_epoch,
            window.window_id,
            window.decision_sample,
        )
        decision_identity = (
            decision.session_id,
            decision.stream_epoch,
            decision.window_id,
            decision.decision_sample,
        )
        if decision_identity != identity or active_frame_count < 0:
            raise ValueError("prepared L2 Gate does not match the current window")
        if type(direction_id_tracking_enabled) is not bool:
            raise TypeError("L2 direction ID tracking switch must be bool")
        if decision.allow_srp:
            if music_effective_order not in {0, 1, 2, 3, None}:
                raise ValueError("prepared MUSIC order must be 0..3 or None")
            if music_effective_order in {0, None} and not music_skip_reason:
                raise ValueError("skipped MUSIC requires a reason")
            if music_effective_order in {1, 2, 3} and music_skip_reason is not None:
                raise ValueError("applied MUSIC order cannot have a skip reason")
        else:
            music_effective_order = None
            music_skip_reason = decision.reason

        id_tracking_ms: float | None = 0.0 if direction_id_tracking_enabled else None
        if direction_id_tracking_enabled != self._direction_id_tracking_enabled:
            # This transition runs on the single L2 worker, so tracker state is
            # never reset concurrently with an update. Re-enabling starts a
            # fresh authoritative identity epoch for the same audio stream.
            self.id_tracker.reset()
            self._direction_id_tracking_enabled = direction_id_tracking_enabled

        required_active_frames = scan_config.context_ms // 20
        response: SpatialResponse | None = None
        diagnostics: MusicDiagnostics | None = None
        observations: tuple[CandidateDirection, ...] = ()
        music_state: MusicStateDiagnostic | None = None
        if decision.allow_srp and music_effective_order in {1, 2, 3}:
            effective_scan = replace(
                scan_config,
                effective_order_limit=music_effective_order,
            )
            response, observations, diagnostics = self.scanner.scan_detailed(
                window,
                geometry,
                effective_scan,
                scan_config_revision,
            )
            warm = active_frame_count >= required_active_frames
            diagnostics = replace(
                diagnostics,
                births_allowed=diagnostics.births_allowed and warm,
                active_frame_count=active_frame_count,
                birth_required_active_frames=required_active_frames,
            )
            music_state = getattr(self.scanner, "last_state_diagnostic", None)
            # Before one complete continuously-open covariance context exists,
            # the spectrum remains diagnostic-only and cannot update/create IDs.
            if not warm:
                observations = ()
        else:
            reset_scanner = getattr(self.scanner, "reset", None)
            if callable(reset_scanner):
                reset_scanner()
            if decision.allow_srp:
                # A Gate-open window with count 0/None has no active MUSIC
                # covariance context.  The next positive order must warm from
                # one instead of inheriting Gate-only time spent while skipped.
                self._consecutive_gate_open_hops = 0

        if not direction_id_tracking_enabled:
            self.last_id_tracking_error = None
            state = (
                Layer2ExecutionState.MUSIC_SKIPPED
                if decision.allow_srp and music_effective_order in {0, None}
                else Layer2ExecutionState.PROCESSED
                if decision.allow_srp
                else Layer2ExecutionState.BLOCKED
            )
            return Layer2PipelineResult(
                state,
                decision,
                response,
                observations,
                diagnostics,
                model_order=None if diagnostics is None else diagnostics.model_order,
                music_state=music_state,
                direction_id_tracking_enabled=False,
                id_tracking_ms=None,
                music_effective_order=music_effective_order,
                music_skip_reason=music_skip_reason,
            )

        id_started = perf_counter()
        observed_directions, active = self.id_tracker.update(
            window.session_id,
            window.stream_epoch,
            window.decision_sample,
            observations,
            window_id=window.window_id,
            doa_start_sample=window.doa_start_sample,
            doa_end_sample=window.doa_end_sample,
            allow_births=diagnostics is not None and diagnostics.births_allowed,
        )
        assert id_tracking_ms is not None
        id_tracking_ms += (perf_counter() - id_started) * 1_000.0
        self.last_id_tracking_error = None
        directions = _select_l3_directions(observed_directions, active)
        candidates = tuple(
            CandidateDirection(
                item.session_id,
                item.stream_epoch,
                item.window_id,
                item.decision_sample,
                item.doa_start_sample,
                item.doa_end_sample,
                item.theta_deg,
                item.raw_score,
                item.normalized_score,
            )
            for item in directions
        )
        if decision.allow_srp and music_effective_order in {0, None}:
            state = Layer2ExecutionState.MUSIC_SKIPPED
        elif decision.allow_srp or directions:
            state = Layer2ExecutionState.PROCESSED
        else:
            state = Layer2ExecutionState.BLOCKED
        return Layer2PipelineResult(
            state,
            decision,
            response,
            candidates,
            diagnostics,
            tuple(item.track_id for item in directions),
            tuple(not item.is_observed for item in directions),
            tuple(item.track_state in {"confirmed", "coasting"} for item in directions),
            tuple(item.is_new_track for item in directions),
            tuple(item.kalman_applied for item in directions),
            directions,
            active,
            None if diagnostics is None else diagnostics.model_order,
            music_state,
            True,
            id_tracking_ms=id_tracking_ms,
            music_effective_order=music_effective_order,
            music_skip_reason=music_skip_reason,
        )

    def process(
        self,
        window: DecisionWindow,
        probabilities: tuple[SourceProbability20ms, ...],
        geometry: MicGeometry,
        scan_config: DirectionScanConfig,
        *,
        gate_threshold: float,
        gate_config_revision: int,
        scan_config_revision: int = 0,
        direction_kalman_enabled: bool = True,
        direction_kalman_q_scale: float = 1.0,
        direction_kalman_r_scale: float = 1.0,
        direction_id_tracking_enabled: bool = True,
    ) -> Layer2PipelineResult:
        # Compatibility-only arguments retained for older Runtime callers.
        del direction_kalman_enabled, direction_kalman_q_scale, direction_kalman_r_scale
        decision, active_frame_count = self.evaluate_gate(
            window,
            probabilities,
            gate_threshold=gate_threshold,
            gate_config_revision=gate_config_revision,
        )
        return self.process_prepared(
            window,
            decision,
            active_frame_count,
            geometry,
            scan_config,
            music_effective_order=(scan_config.effective_order_limit if decision.allow_srp else None),
            music_skip_reason=None,
            scan_config_revision=scan_config_revision,
            direction_id_tracking_enabled=direction_id_tracking_enabled,
        )
