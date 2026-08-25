from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.optimize import minimize


@dataclass(frozen=True, slots=True)
class TrackMatchFeatures:
    """Auditable evidence from globally matching two sets of segment embeddings."""

    decision_score: float
    median: float
    q25: float
    coverage_above_threshold: float
    mutual_nearest_coverage: float
    representative_cosine: float
    mad: float
    standard_deviation: float
    matched_count: int
    required_count: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LogisticCalibration:
    """A labeled-data calibration artifact for P(same speaker)."""

    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def predict(self, features: TrackMatchFeatures) -> float:
        if len(self.feature_names) != len(self.coefficients):
            raise ValueError("calibration feature and coefficient counts differ")
        values = features.as_dict()
        logit = self.intercept + sum(
            coefficient * float(values[name])
            for name, coefficient in zip(
                self.feature_names, self.coefficients, strict=True,
            )
        )
        if logit >= 0.0:
            return float(1.0 / (1.0 + np.exp(-logit)))
        exp_logit = float(np.exp(logit))
        return exp_logit / (1.0 + exp_logit)


def fit_logistic_calibration(
    samples: tuple[TrackMatchFeatures, ...],
    labels: tuple[bool, ...],
    *,
    feature_names: tuple[str, ...],
    l2_regularization: float = 1.0,
) -> LogisticCalibration:
    """Fit a reproducible logistic calibrator from labeled track pairs."""

    if len(samples) != len(labels) or len(samples) < 2:
        raise ValueError("calibration requires equally sized feature and label sets")
    if len(set(labels)) != 2:
        raise ValueError("calibration labels must contain same- and different-speaker pairs")
    rows = np.asarray([
        [float(sample.as_dict()[name]) for name in feature_names]
        for sample in samples
    ], dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[0]
        coefficients = parameters[1:]
        logits = intercept + rows @ coefficients
        loss = float(np.sum(np.logaddexp(0.0, logits) - targets * logits))
        loss += 0.5 * l2_regularization * float(coefficients @ coefficients)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        residuals = probabilities - targets
        gradient = np.concatenate((
            np.asarray([np.sum(residuals)]),
            rows.T @ residuals + l2_regularization * coefficients,
        ))
        return loss, gradient

    fitted = minimize(
        objective,
        np.zeros(len(feature_names) + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
    )
    if not fitted.success:
        raise RuntimeError(f"logistic calibration failed: {fitted.message}")
    return LogisticCalibration(
        feature_names,
        tuple(float(value) for value in fitted.x[1:]),
        float(fitted.x[0]),
    )


def hungarian_track_features(
    left: np.ndarray,
    right: np.ndarray,
    *,
    threshold: float,
    minimum_match_count: int,
    required_coverage: float,
) -> TrackMatchFeatures:
    """Globally pair segment voiceprints and summarize agreement robustly."""

    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1:] != right.shape[1:]:
        raise ValueError("segment embeddings must be two compatible matrices")
    shorter = min(len(left), len(right))
    required = max(minimum_match_count, int(np.ceil(required_coverage * shorter)))
    if shorter == 0:
        return TrackMatchFeatures(
            -1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0, required,
        )

    similarities = np.asarray(left @ right.T, dtype=np.float32)
    row_indices, column_indices = linear_sum_assignment(similarities, maximize=True)
    matched = np.asarray(similarities[row_indices, column_indices], dtype=np.float32)
    descending = np.sort(matched)[::-1]
    decision_score = float(descending[required - 1]) if shorter >= required else -1.0
    median = float(np.median(matched))
    q25 = float(np.quantile(matched, 0.25))
    mad = float(np.median(np.abs(matched - median)))

    row_best = np.argmax(similarities, axis=1)
    column_best = np.argmax(similarities, axis=0)
    mutual_count = sum(
        int(column_best[column] == row)
        for row, column in enumerate(row_best)
    )
    left_representative = np.mean(left, axis=0)
    right_representative = np.mean(right, axis=0)
    denominator = max(
        float(np.linalg.norm(left_representative) * np.linalg.norm(right_representative)),
        1e-12,
    )
    return TrackMatchFeatures(
        decision_score=decision_score,
        median=median,
        q25=q25,
        coverage_above_threshold=float(np.mean(matched >= threshold)),
        mutual_nearest_coverage=float(mutual_count / shorter),
        representative_cosine=float(
            np.dot(left_representative, right_representative) / denominator
        ),
        mad=mad,
        standard_deviation=float(np.std(matched)),
        matched_count=shorter,
        required_count=required,
    )
