from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


_SEGMENT_SAMPLES = 960
_SEGMENT_COUNT = 16
_EPSILON = 1.0e-12


@dataclass(frozen=True, slots=True)
class InputGainCompensationSettings:
    enabled: bool = True
    algorithm_version: str = "imcra_probability_rms_v1"
    target_rms_dbfs: float = -23.0
    no_compensation_probability: float = 0.30
    full_compensation_probability: float = 0.80
    peak_ceiling_dbfs: float = -3.0
    silence_floor_dbfs: float = -100.0
    time_interpolation: str = "linear_db"

    def __post_init__(self) -> None:
        values = (
            self.target_rms_dbfs,
            self.no_compensation_probability,
            self.full_compensation_probability,
            self.peak_ceiling_dbfs,
            self.silence_floor_dbfs,
        )
        if type(self.enabled) is not bool or not self.algorithm_version or not all(map(np.isfinite, values)):
            raise ValueError("invalid L4 input gain-compensation settings")
        if not 0.0 <= self.no_compensation_probability < self.full_compensation_probability <= 1.0:
            raise ValueError("gain-compensation probability breakpoints must increase within [0,1]")
        if self.peak_ceiling_dbfs > 0.0 or self.silence_floor_dbfs >= self.target_rms_dbfs:
            raise ValueError("invalid gain-compensation dBFS limits")
        if self.time_interpolation != "linear_db":
            raise ValueError("only linear_db gain interpolation is supported")


@dataclass(frozen=True, slots=True)
class SegmentGainDiagnostic:
    probability: float | None
    rms_before_dbfs: float
    full_gain_db: float
    probability_weight: float
    requested_gain_db: float
    peak_limited_gain_db: float
    applied_gain_db: float
    rms_after_dbfs: float
    peak_after_dbfs: float
    silent: bool
    peak_protection_triggered: bool


@dataclass(frozen=True, slots=True)
class InputGainCompensationDiagnostic:
    algorithm_version: str
    enabled: bool
    segments: tuple[SegmentGainDiagnostic, ...]
    max_applied_gain_db: float
    mean_applied_gain_db: float
    compensated_segment_count: int
    peak_protection_trigger_count: int


def _dbfs(amplitude: float) -> float:
    return float(20.0 * np.log10(max(float(amplitude), _EPSILON)))


def _probability_weight(probability: float | None, settings: InputGainCompensationSettings) -> float:
    if probability is None or probability <= settings.no_compensation_probability:
        return 0.0
    if probability >= settings.full_compensation_probability:
        return 1.0
    return float(
        (probability - settings.no_compensation_probability)
        / (settings.full_compensation_probability - settings.no_compensation_probability)
    )


def compensate_l4_input(
    waveform: NDArray[np.float32],
    probabilities_20ms: tuple[float | None, ...],
    settings: InputGainCompensationSettings,
) -> tuple[NDArray[np.float32], InputGainCompensationDiagnostic]:
    """Create a compensated L4-only copy without modifying the L3 waveform."""
    source = np.asarray(waveform)
    if (
        source.shape != (_SEGMENT_COUNT * _SEGMENT_SAMPLES,)
        or source.dtype != np.float32
        or not source.flags.c_contiguous
        or not np.isfinite(source).all()
    ):
        raise ValueError("gain compensation requires finite C-contiguous float32 [15360]")
    if len(probabilities_20ms) != _SEGMENT_COUNT:
        raise ValueError("gain compensation requires exactly 16 aligned IMCRA probabilities")
    for probability in probabilities_20ms:
        if probability is not None and (
            not np.isfinite(probability) or not 0.0 <= probability <= 1.0
        ):
            raise ValueError("IMCRA probabilities must be finite values in [0,1] or missing")

    chunks = source.reshape(_SEGMENT_COUNT, _SEGMENT_SAMPLES)
    rms_before = np.sqrt(np.mean(chunks.astype(np.float64) ** 2, axis=1))
    peak_before = np.max(np.abs(chunks.astype(np.float64)), axis=1)
    rms_before_dbfs = np.asarray([_dbfs(value) for value in rms_before])
    peak_before_dbfs = np.asarray([_dbfs(value) for value in peak_before])
    silent = rms_before_dbfs <= settings.silence_floor_dbfs

    full_gain = np.maximum(0.0, settings.target_rms_dbfs - rms_before_dbfs)
    full_gain[silent] = 0.0
    weights = np.asarray(
        [_probability_weight(value, settings) for value in probabilities_20ms], dtype=np.float64
    )
    if not settings.enabled:
        weights.fill(0.0)
    requested_gain = full_gain * weights
    peak_headroom = np.maximum(0.0, settings.peak_ceiling_dbfs - peak_before_dbfs)
    target_gain = np.minimum(requested_gain, peak_headroom)
    target_gain[silent] = 0.0

    output = source.copy()
    applied_mean = np.zeros(_SEGMENT_COUNT, dtype=np.float64)
    peak_triggered = np.zeros(_SEGMENT_COUNT, dtype=bool)
    previous_gain = float(target_gain[0])
    ceiling_amplitude = float(
        np.nextafter(
            np.float32(10.0 ** (settings.peak_ceiling_dbfs / 20.0)),
            np.float32(0.0),
        )
    )

    for index, chunk in enumerate(chunks):
        desired = (
            np.full(_SEGMENT_SAMPLES, target_gain[index], dtype=np.float64)
            if index == 0
            else np.linspace(previous_gain, target_gain[index], _SEGMENT_SAMPLES, dtype=np.float64)
        )
        # A previous segment may request more gain than the current segment can
        # safely accept. Cap only the positive gain at those samples; never
        # attenuate an input that already exceeds the configured ceiling.
        absolute = np.abs(chunk.astype(np.float64))
        safe_gain = np.full(_SEGMENT_SAMPLES, np.inf, dtype=np.float64)
        nonzero = absolute > _EPSILON
        safe_gain[nonzero] = 20.0 * np.log10(ceiling_amplitude / absolute[nonzero])
        safe_gain = np.maximum(0.0, safe_gain)
        envelope_db = np.minimum(desired, safe_gain)
        peak_triggered[index] = bool(
            requested_gain[index] > target_gain[index] + 1.0e-12
            or np.any(envelope_db < desired - 1.0e-12)
        )
        output[index * _SEGMENT_SAMPLES:(index + 1) * _SEGMENT_SAMPLES] = (
            chunk.astype(np.float64) * np.power(10.0, envelope_db / 20.0)
        ).astype(np.float32)
        applied_mean[index] = float(np.mean(envelope_db))
        previous_gain = float(envelope_db[-1])

    output_chunks = output.reshape(_SEGMENT_COUNT, _SEGMENT_SAMPLES)
    rms_after = np.sqrt(np.mean(output_chunks.astype(np.float64) ** 2, axis=1))
    peak_after = np.max(np.abs(output_chunks.astype(np.float64)), axis=1)
    segment_diagnostics = tuple(
        SegmentGainDiagnostic(
            probabilities_20ms[index],
            float(rms_before_dbfs[index]),
            float(full_gain[index]),
            float(weights[index]),
            float(requested_gain[index]),
            float(target_gain[index]),
            float(applied_mean[index]),
            _dbfs(rms_after[index]),
            _dbfs(peak_after[index]),
            bool(silent[index]),
            bool(peak_triggered[index]),
        )
        for index in range(_SEGMENT_COUNT)
    )
    diagnostic = InputGainCompensationDiagnostic(
        settings.algorithm_version,
        settings.enabled,
        segment_diagnostics,
        float(np.max(applied_mean)),
        float(np.mean(applied_mean)),
        int(np.count_nonzero(applied_mean > 1.0e-12)),
        int(np.count_nonzero(peak_triggered)),
    )
    return np.ascontiguousarray(output), diagnostic
