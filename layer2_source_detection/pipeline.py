from __future__ import annotations

import queue
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from common.angle import circular_distance_deg
from common.config import ProjectConfig
from common.data_types import CandidateDirection, DecisionWindow, SpatialResponse
from common.geometry import MicGeometry

from .configuration import DirectionScanConfig
from .circular_kalman import CircularKalmanConfig, CircularKalmanFilter
from .circular_kalman_v2 import CircularKalmanFilterV2, CircularKalmanV2Config
from .candidates import rank_observation_indices
from .direction_id_tracking import DirectionIdTracker, DirectionIdTrackingConfig
from .direction_id_tracking_v2 import DirectionIdTrackerV2, DirectionIdTrackingV2Config
from .direction_smoothing import DirectionSmoothingError
from .interface import DetailedDirectionScanner
from .iterative import CandidateSearchDiagnostics
from .probability_gate import (
    ProbabilityGate,
    ProbabilityGateDecision,
    ProbabilityGateState,
    SourceProbability20ms,
)
from .srp_phat import SrpPhatScanner


class Layer2ExecutionState(str, Enum):
    BLOCKED = "blocked"
    PROCESSED = "processed"


@dataclass(frozen=True, slots=True)
class Layer2PipelineResult:
    state: Layer2ExecutionState
    gate_decision: ProbabilityGateDecision
    spatial_response: SpatialResponse | None
    candidates: tuple[CandidateDirection, ...]
    search_diagnostics: CandidateSearchDiagnostics | None
    # Private L2 identities are exposed only to the local runtime sidecar.
    # They are deliberately not added to CandidateDirection, which remains
    # the public inter-layer DTO.
    candidate_track_ids: tuple[int | None, ...] = ()
    candidate_is_prediction: tuple[bool, ...] = ()
    candidate_track_is_formal: tuple[bool, ...] = ()
    candidate_track_is_new: tuple[bool, ...] = ()
    candidate_track_is_kalman_ready: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        identity = (
            self.gate_decision.session_id,
            self.gate_decision.stream_epoch,
            self.gate_decision.window_id,
            self.gate_decision.decision_sample,
        )
        object.__setattr__(self, "candidates", tuple(self.candidates))
        track_ids = tuple(self.candidate_track_ids)
        prediction_flags = tuple(self.candidate_is_prediction)
        formal_flags = tuple(self.candidate_track_is_formal)
        new_flags = tuple(self.candidate_track_is_new)
        kalman_ready_flags = tuple(self.candidate_track_is_kalman_ready)
        if not track_ids:
            track_ids = (None,) * len(self.candidates)
        if not prediction_flags:
            prediction_flags = (False,) * len(self.candidates)
        if not formal_flags:
            formal_flags = (False,) * len(self.candidates)
        if not new_flags:
            new_flags = (False,) * len(self.candidates)
        if not kalman_ready_flags:
            kalman_ready_flags = (False,) * len(self.candidates)
        if (
            len(track_ids) != len(self.candidates)
            or len(prediction_flags) != len(self.candidates)
            or len(formal_flags) != len(self.candidates)
            or len(new_flags) != len(self.candidates)
            or len(kalman_ready_flags) != len(self.candidates)
        ):
            raise ValueError("Layer 2 private candidate metadata must align with candidates")
        if any(item is not None and (type(item) is not int or item <= 0) for item in track_ids):
            raise ValueError("Layer 2 private track IDs must be positive integers or None")
        if len({item for item in track_ids if item is not None}) != sum(
            item is not None for item in track_ids
        ):
            raise ValueError("Layer 2 private track IDs must be unique within a window")
        if any(type(item) is not bool for item in (
            *prediction_flags, *formal_flags, *new_flags, *kalman_ready_flags,
        )):
            raise TypeError("Layer 2 prediction/formal flags must be bool")
        if any(is_formal and track_id is None for is_formal, track_id in zip(formal_flags, track_ids)):
            raise ValueError("a formal Layer 2 candidate requires a private track ID")
        if any(ready and track_id is None for ready, track_id in zip(kalman_ready_flags, track_ids)):
            raise ValueError("a Kalman-ready Layer 2 candidate requires a private track ID")
        object.__setattr__(self, "candidate_track_ids", track_ids)
        object.__setattr__(self, "candidate_is_prediction", prediction_flags)
        object.__setattr__(self, "candidate_track_is_formal", formal_flags)
        object.__setattr__(self, "candidate_track_is_new", new_flags)
        object.__setattr__(self, "candidate_track_is_kalman_ready", kalman_ready_flags)
        if len(self.candidates) > 3:
            raise ValueError("Layer 2 result cannot publish more than 3 candidates")
        if any(
            circular_distance_deg(self.candidates[left].theta_deg, self.candidates[right].theta_deg)
            < 45.0
            for left in range(len(self.candidates))
            for right in range(left + 1, len(self.candidates))
        ):
            raise ValueError(
                "all Layer 2 source points must be separated by at least 45 circular degrees"
            )
        if any(
            (item.session_id, item.stream_epoch, item.window_id, item.decision_sample) != identity
            for item in self.candidates
        ):
            raise ValueError("Layer 2 candidates must belong to the same window")
        if self.state is Layer2ExecutionState.BLOCKED:
            if self.gate_decision.allow_srp:
                raise ValueError("blocked Layer 2 result requires a closed probability Gate")
            if self.spatial_response is not None or self.candidates or self.search_diagnostics is not None:
                raise ValueError("blocked Layer 2 result cannot contain SRP output")
            return
        if self.gate_decision.state is not ProbabilityGateState.OPEN:
            raise ValueError("processed Layer 2 result requires an open probability Gate")
        if self.spatial_response is None or self.search_diagnostics is None:
            raise ValueError("processed Layer 2 result requires complete SRP output")
        response_identity = (
            self.spatial_response.session_id,
            self.spatial_response.stream_epoch,
            self.spatial_response.window_id,
            self.spatial_response.decision_sample,
        )
        if response_identity != identity:
            raise ValueError("SRP response and probability Gate must belong to the same window")


@dataclass(frozen=True, slots=True)
class _VoiceDirectionFeedback:
    session_id: str
    stream_epoch: int
    decision_sample: int
    theta_deg: float
    probability: float = 1.0
    is_voice: bool = True


class Layer2Pipeline:
    """Probability Gate -> SRP-PHAT -> optional private IDs -> optional per-ID Kalman."""

    def __init__(
        self,
        gate: ProbabilityGate,
        scanner: DetailedDirectionScanner,
        kalman_filter: CircularKalmanFilter | CircularKalmanFilterV2 | None = None,
        id_tracker: DirectionIdTracker | DirectionIdTrackerV2 | None = None,
    ) -> None:
        self.gate = gate
        self.scanner = scanner
        self.kalman_filter = kalman_filter or CircularKalmanFilter()
        self.id_tracker = id_tracker or DirectionIdTracker()
        self.last_kalman_error: str | None = None
        self.last_id_tracking_error: str | None = None
        self._kalman_active = False
        self._id_tracking_active = False
        self._voice_feedback: queue.Queue[_VoiceDirectionFeedback] = queue.Queue(maxsize=256)
        self.voice_feedback_applied = 0
        self.voice_feedback_rejected = 0
        self.voice_feedback_dropped = 0

    @classmethod
    def from_project(
        cls,
        config: ProjectConfig,
        *,
        scanner: DetailedDirectionScanner | None = None,
    ) -> "Layer2Pipeline":
        if config.layer2.probability_gate.backend != ProbabilityGate.backend:
            raise ValueError(
                f"unsupported Layer 2 probability Gate backend: {config.layer2.probability_gate.backend}"
            )
        kalman = config.layer2.direction_kalman
        tracking = config.layer2.direction_id_tracking
        if kalman.backend == CircularKalmanFilterV2.backend:
            kalman_filter = CircularKalmanFilterV2(CircularKalmanV2Config(
                process_angle_std_deg=kalman.process_angle_std_deg,
                process_velocity_std_dps=kalman.process_velocity_std_dps,
                measurement_std_deg=kalman.measurement_std_deg,
                velocity_half_life_seconds=kalman.velocity_half_life_seconds,
                max_velocity_dps=kalman.max_velocity_dps,
                prediction_freeze_std_deg=kalman.prediction_freeze_std_deg,
                innovation_gate_deg=tracking.association_gate_deg,
            ))
        else:
            kalman_filter = CircularKalmanFilter(CircularKalmanConfig(
                process_angle_std_deg=kalman.process_angle_std_deg,
                process_velocity_std_dps=kalman.process_velocity_std_dps,
                measurement_std_deg=kalman.measurement_std_deg,
                max_missed_windows=kalman.max_missed_windows,
            ))
        if tracking.backend == DirectionIdTrackerV2.backend:
            id_tracker = DirectionIdTrackerV2(DirectionIdTrackingV2Config(
                association_gate_deg=tracking.association_gate_deg,
                confirmation_age_samples=tracking.confirmation_min_age_windows * 960,
                confirmation_min_matches=tracking.confirmation_min_matches,
                formal_lease_samples=tracking.prediction_hold_windows * 960,
                provisional_hold_samples=tracking.max_missed_windows * 960,
            ))
        else:
            id_tracker = DirectionIdTracker(DirectionIdTrackingConfig(
                association_gate_deg=tracking.association_gate_deg,
                prediction_association_gate_deg=tracking.prediction_association_gate_deg,
                max_missed_windows=tracking.max_missed_windows,
                confirmation_min_age_windows=tracking.confirmation_min_age_windows,
                confirmation_min_matches=tracking.confirmation_min_matches,
                prediction_hold_windows=tracking.prediction_hold_windows,
            ))
        return cls(
            ProbabilityGate(),
            scanner or SrpPhatScanner(),
            kalman_filter,
            id_tracker,
        )

    def reset(self) -> None:
        self.kalman_filter.reset()
        self.id_tracker.reset()
        self.last_kalman_error = None
        self.last_id_tracking_error = None
        self._kalman_active = False
        self._id_tracking_active = False
        self._clear_voice_feedback()

    def submit_voice_feedback(
        self,
        session_id: str,
        stream_epoch: int,
        decision_sample: int,
        theta_deg: float,
    ) -> bool:
        """Thread-safe L4->L2 feedback; L2 owns all tracker mutations."""
        if (
            not session_id
            or stream_epoch < 0
            or decision_sample < 0
            or not np.isfinite(theta_deg)
            or not 0.0 <= theta_deg < 360.0
        ):
            raise ValueError("L4 voice feedback identity/angle is invalid")
        try:
            self._voice_feedback.put_nowait(_VoiceDirectionFeedback(
                session_id, stream_epoch, decision_sample, float(theta_deg)
            ))
            return True
        except queue.Full:
            self.voice_feedback_dropped += 1
            return False

    def submit_classification_feedback(
        self, session_id: str, stream_epoch: int, decision_sample: int,
        theta_deg: float, probability: float, is_voice: bool,
    ) -> bool:
        """Queue both positive and negative L4 semantic evidence for tracker V2."""
        if (not np.isfinite(probability) or not 0.0 <= probability <= 1.0
                or type(is_voice) is not bool):
            raise ValueError("L4 classification feedback is invalid")
        if (not session_id or stream_epoch < 0 or decision_sample < 0
                or not np.isfinite(theta_deg) or not 0.0 <= theta_deg < 360.0):
            raise ValueError("L4 classification feedback identity/angle is invalid")
        try:
            self._voice_feedback.put_nowait(_VoiceDirectionFeedback(
                session_id, stream_epoch, decision_sample, float(theta_deg),
                float(probability), is_voice,
            ))
            return True
        except queue.Full:
            self.voice_feedback_dropped += 1
            return False

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
        direction_kalman_enabled: bool = False,
        direction_id_tracking_enabled: bool = False,
        direction_kalman_q_scale: float = 1.0,
        direction_kalman_r_scale: float = 1.0,
    ) -> Layer2PipelineResult:
        if type(direction_kalman_enabled) is not bool or type(direction_id_tracking_enabled) is not bool:
            raise TypeError("L2 Kalman and ID tracking switches must be bool")
        if direction_kalman_enabled and not direction_id_tracking_enabled:
            raise ValueError("L2 Circular Kalman requires private ID tracking")
        for name, value in (("Q", direction_kalman_q_scale), ("R", direction_kalman_r_scale)):
            if (
                not np.isfinite(value)
                or not 0.02 <= value <= 10.0
                or (value != 0.02 and abs(value * 10.0 - round(value * 10.0)) > 1.0e-9)
            ):
                raise ValueError(
                    f"L2 Kalman {name} scale must be 0.02..10.00 in 0.1 steps (or the 0.02 minimum)"
                )
        kalman_switch_changed = direction_kalman_enabled != self._kalman_active
        id_switch_changed = direction_id_tracking_enabled != self._id_tracking_active
        if kalman_switch_changed or id_switch_changed:
            self.kalman_filter.reset()
            self.id_tracker.reset()
        self._kalman_active = direction_kalman_enabled
        self._id_tracking_active = direction_id_tracking_enabled
        if direction_id_tracking_enabled:
            self.id_tracker.prepare_stream(window.session_id, window.stream_epoch)
            self._drain_voice_feedback()
            self._retain_kalman_track_ids()
        else:
            self._clear_voice_feedback()
        natural_decision = self.gate.evaluate(
            window,
            probabilities,
            threshold=gate_threshold,
            config_revision=gate_config_revision,
        )
        decision = self._apply_confirmed_id_gate_force(
            natural_decision,
            window.decision_sample,
            direction_id_tracking_enabled and direction_kalman_enabled,
        )
        forced_existing_only = (
            natural_decision.state is ProbabilityGateState.CLOSED
            and decision.state is ProbabilityGateState.OPEN
            and decision.reason == "confirmed_id_gate_hold"
        )
        if not decision.allow_srp:
            track_ids: tuple[int, ...] = ()
            if direction_id_tracking_enabled:
                self.id_tracker.update(
                    window.session_id, window.stream_epoch, window.decision_sample, ()
                )
                track_ids = self.id_tracker.last_assignments
                track_is_new = self.id_tracker.last_assignment_is_new
                self._retain_kalman_track_ids()
                self.last_id_tracking_error = None
            if direction_kalman_enabled:
                try:
                    self.kalman_filter.update(
                        window.session_id, window.stream_epoch,
                        window.decision_sample, (), track_ids,
                        direction_kalman_q_scale, direction_kalman_r_scale,
                    )
                    self.last_kalman_error = None
                except DirectionSmoothingError as exc:
                    self.last_kalman_error = str(exc)
                    self.kalman_filter.reset()
            return Layer2PipelineResult(
                Layer2ExecutionState.BLOCKED,
                decision,
                None,
                (),
                None,
            )
        response, candidates, diagnostics = self.scanner.scan_detailed(
            window,
            geometry,
            scan_config,
            scan_config_revision,
        )
        scanner_public_candidates = tuple(candidates)
        if direction_id_tracking_enabled and isinstance(self.id_tracker, DirectionIdTrackerV2):
            observation_indices = rank_observation_indices(
                response.normalized_scores, scan_config
            )[: self.id_tracker.config.max_tracks]
            by_angle = {int(round(item.theta_deg)) % 360: item for item in candidates}
            for index in observation_indices:
                by_angle.setdefault(index, CandidateDirection(
                    window.session_id, window.stream_epoch, window.window_id,
                    window.decision_sample, window.doa_start_sample, window.doa_end_sample,
                    float(index), float(response.raw_scores[index]),
                    float(response.normalized_scores[index]),
                ))
            candidates = tuple(sorted(
                by_angle.values(), key=lambda item: (-item.normalized_score, item.theta_deg)
            ))[: self.id_tracker.config.max_tracks]
        raw_candidates = tuple(candidates)
        candidates = raw_candidates
        track_ids: tuple[int, ...] = ()
        tracking_succeeded = not direction_id_tracking_enabled
        if direction_id_tracking_enabled:
            predicted_angles = None
            if direction_kalman_enabled:
                forecast_angles = getattr(self.kalman_filter, "forecast_angles", None)
                try:
                    if callable(forecast_angles):
                        predicted_angles = forecast_angles(
                            window.session_id,
                            window.stream_epoch,
                            window.decision_sample,
                            self.id_tracker.active_track_ids,
                        )
                except DirectionSmoothingError as exc:
                    # Forecast is an association hint.  Falling back to the
                    # normal 20-degree last-measurement gate is safer than
                    # discarding otherwise healthy ID state.
                    self.last_kalman_error = str(exc)
            try:
                candidates = self.id_tracker.update(
                    window.session_id,
                    window.stream_epoch,
                    window.decision_sample,
                    candidates,
                    existing_formal_only=forced_existing_only,
                    predicted_angles=predicted_angles,
                )
                track_ids = self.id_tracker.last_assignments
                track_is_new = self.id_tracker.last_assignment_is_new
                self._retain_kalman_track_ids()
                tracking_succeeded = True
                self.last_id_tracking_error = None
            except Exception as exc:
                self.last_id_tracking_error = str(exc)
                self.id_tracker.reset()
                self.kalman_filter.reset()
                candidates = scanner_public_candidates
                track_ids = ()
        if direction_kalman_enabled and tracking_succeeded:
            try:
                candidates = self.kalman_filter.update(
                    window.session_id, window.stream_epoch, window.decision_sample,
                    candidates, track_ids,
                    direction_kalman_q_scale, direction_kalman_r_scale,
                    **({"ready_track_ids": self.id_tracker.kalman_ready_track_ids}
                       if isinstance(self.kalman_filter, CircularKalmanFilterV2)
                       and isinstance(self.id_tracker, DirectionIdTrackerV2) else {}),
                )
                if (not isinstance(self.id_tracker, DirectionIdTrackerV2)
                        and len(candidates) == 2 and circular_distance_deg(
                    candidates[0].theta_deg, candidates[1].theta_deg
                ) < scan_config.min_peak_distance_deg):
                    raise DirectionSmoothingError(
                        "the two smoothed source points violate the 45-degree circular separation"
                    )
                self.last_kalman_error = None
            except DirectionSmoothingError as exc:
                self.last_kalman_error = str(exc)
                self.kalman_filter.reset()
                candidates = candidates if forced_existing_only else raw_candidates
        if direction_kalman_enabled and tracking_succeeded:
            observed_count = len(candidates)
            candidates, track_ids = self._append_predictions(
                window, response, candidates, track_ids, scan_config.min_peak_distance_deg
            )
        else:
            observed_count = len(candidates)
        prediction_flags = (False,) * observed_count + (True,) * (len(candidates) - observed_count)
        if (direction_id_tracking_enabled
                and isinstance(self.id_tracker, DirectionIdTrackerV2)
                and tracking_succeeded):
            new_by_id = dict(zip(track_ids[:observed_count], track_is_new, strict=True))
            candidates, track_ids, prediction_flags = self.id_tracker.select_public(
                candidates, track_ids, prediction_flags, scan_config.min_peak_distance_deg
            )
            observed_count = sum(not item for item in prediction_flags)
            track_is_new = tuple(new_by_id.get(track_id, False) for track_id in track_ids)
        if not direction_id_tracking_enabled or not tracking_succeeded:
            track_is_new = (False,) * observed_count
        if direction_id_tracking_enabled and tracking_succeeded:
            self.id_tracker.record_published(window.decision_sample, candidates, track_ids)
            formal_track_ids = set(self.id_tracker.confirmed_track_ids_at(
                window.decision_sample, require_advance=False,
            ))
            kalman_ready_track_ids = set(
                getattr(self.id_tracker, "kalman_ready_track_ids", ())
            ) if direction_kalman_enabled else set()
        else:
            formal_track_ids = set()
            kalman_ready_track_ids = set()
        return Layer2PipelineResult(
            Layer2ExecutionState.PROCESSED,
            decision,
            response,
            candidates,
            diagnostics,
            tuple(track_ids) if direction_id_tracking_enabled and tracking_succeeded else (),
            prediction_flags,
            tuple(track_id in formal_track_ids for track_id in track_ids),
            tuple(track_is_new) if (direction_id_tracking_enabled
                                    and isinstance(self.id_tracker, DirectionIdTrackerV2)) else
            tuple(track_is_new) + (False,) * (len(candidates) - observed_count),
            tuple(track_id in kalman_ready_track_ids for track_id in track_ids),
        )

    def _apply_confirmed_id_gate_force(
        self,
        decision: ProbabilityGateDecision,
        decision_sample: int,
        combined_features_enabled: bool,
    ) -> ProbabilityGateDecision:
        if (
            not combined_features_enabled
            or not self.id_tracker.confirmed_track_ids_at(decision_sample)
        ):
            return decision
        if decision.state is ProbabilityGateState.OPEN:
            return decision
        if decision.state not in {ProbabilityGateState.OPEN, ProbabilityGateState.CLOSED}:
            return replace(
                decision,
                diagnostics=decision.diagnostics + (
                    "confirmed_id_gate_force_skipped_nonformal_probability=true",
                ),
            )
        return replace(
            decision,
            state=ProbabilityGateState.OPEN,
            sound_present=True,
            reason="confirmed_id_gate_hold",
            diagnostics=decision.diagnostics + (
                "confirmed_id_gate_force=true",
                f"confirmed_id_gate_force_original_state={decision.state.value}",
                "confirmed_id_gate_force_basis=live_formal_id",
            ),
        )

    def _append_predictions(
        self,
        window: DecisionWindow,
        response: SpatialResponse,
        observed: tuple[CandidateDirection, ...],
        observed_track_ids: tuple[int, ...],
        min_distance_deg: float,
    ) -> tuple[tuple[CandidateDirection, ...], tuple[int, ...]]:
        output = list(observed)
        output_ids = list(observed_track_ids)
        available_ids = tuple(
            track_id for track_id in self.id_tracker.prediction_track_ids
            if track_id not in observed_track_ids
        )
        if not available_ids:
            return tuple(output), tuple(output_ids)
        angles = self.kalman_filter.predicted_angles(available_ids)
        for track_id, theta_deg in zip(available_ids, angles, strict=True):
            if len(output) >= 3:
                break
            if any(circular_distance_deg(theta_deg, item.theta_deg) < min_distance_deg for item in output):
                continue
            score_index = int(round(theta_deg)) % 360
            output.append(CandidateDirection(
                window.session_id,
                window.stream_epoch,
                window.window_id,
                window.decision_sample,
                window.doa_start_sample,
                window.doa_end_sample,
                theta_deg,
                float(response.raw_scores[score_index]),
                float(response.normalized_scores[score_index]),
            ))
            output_ids.append(track_id)
        return tuple(output), tuple(output_ids)

    def _drain_voice_feedback(self) -> None:
        while True:
            try:
                feedback = self._voice_feedback.get_nowait()
            except queue.Empty:
                return
            apply_classification = getattr(self.id_tracker, "apply_classification_feedback", None)
            if callable(apply_classification):
                matched = apply_classification(
                    feedback.session_id, feedback.stream_epoch, feedback.decision_sample,
                    feedback.theta_deg, feedback.probability, feedback.is_voice,
                )
            elif feedback.is_voice:
                matched = self.id_tracker.apply_voice_feedback(
                    feedback.session_id, feedback.stream_epoch,
                    feedback.decision_sample, feedback.theta_deg,
                )
            else:
                matched = None
            if matched is None:
                self.voice_feedback_rejected += 1
            else:
                self.voice_feedback_applied += 1

    def _clear_voice_feedback(self) -> None:
        while True:
            try:
                self._voice_feedback.get_nowait()
            except queue.Empty:
                return

    def _retain_kalman_track_ids(self) -> None:
        retain = getattr(self.kalman_filter, "retain_track_ids", None)
        if retain is not None:
            retain(self.id_tracker.active_track_ids)
