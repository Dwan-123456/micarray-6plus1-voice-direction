from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidateSearchEvidence:
    theta_deg: float
    search_iteration: int
    search_raw: float
    search_norm: float
    pair_support: int
    frequency_support: int


@dataclass(frozen=True, slots=True)
class CandidateSearchDiagnostics:
    mode: str
    algorithm_version: str
    config_revision: int
    iterations_used: int
    stop_reason: str
    remaining_weight_ratio: float
    fallback_reason: str | None = None
    evidence: tuple[CandidateSearchEvidence, ...] = ()
    eligible_peak_count: int = 0
    candidate_limit: int = 2
    candidate_limit_applied: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"single_pass", "iterative_rank1_projection_v1"}:
            raise ValueError("unknown candidate search mode")
        if self.config_revision < 0 or self.iterations_used < 1:
            raise ValueError("candidate search revision/iteration is invalid")
        if not 0.0 <= self.remaining_weight_ratio <= 1.0:
            raise ValueError("remaining weight ratio must be in [0,1]")
        if self.eligible_peak_count < 0 or self.candidate_limit not in {2, 3}:
            raise ValueError("Layer 2 candidate diagnostics require a limit of 2 or 3")
        if self.candidate_limit_applied and self.eligible_peak_count < self.candidate_limit:
            raise ValueError("candidate limit cannot be applied below the limit")
        object.__setattr__(self, "evidence", tuple(self.evidence))


SINGLE_PASS_DIAGNOSTICS = CandidateSearchDiagnostics(
    mode="single_pass",
    algorithm_version="srp_phat_single_pass_v1",
    config_revision=0,
    iterations_used=1,
    stop_reason="single_pass",
    remaining_weight_ratio=1.0,
)
