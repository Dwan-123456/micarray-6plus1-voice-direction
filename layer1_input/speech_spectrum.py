from __future__ import annotations

import numpy as np


# Byrne et al., JASA 96(4), 1994, Table II: male/female international
# long-term average speech spectra in one-third-octave bands.
_CENTERS_HZ = np.asarray(
    (
        63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
        800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
    ),
    dtype=np.float64,
)
_MALE_LEVEL_DB = np.asarray(
    (
        38.6, 43.5, 54.4, 57.7, 56.8, 58.2, 59.7, 60.0, 62.4, 62.6, 60.6,
        55.7, 53.1, 53.7, 52.3, 48.7, 48.9, 47.0, 46.0, 44.4, 43.3, 42.4,
    ),
    dtype=np.float64,
)
_FEMALE_LEVEL_DB = np.asarray(
    (
        37.0, 36.0, 37.5, 40.1, 53.4, 62.2, 60.9, 58.1, 61.7, 61.7, 60.4,
        58.0, 54.3, 52.3, 51.7, 48.8, 47.3, 46.7, 45.3, 44.6, 45.2, 44.9,
    ),
    dtype=np.float64,
)
_REFERENCE_GRID_HZ = np.arange(0.0, 8_000.0 + 5.0, 5.0, dtype=np.float64)


def _band_levels_to_density(levels_db: np.ndarray) -> np.ndarray:
    third_octave_width = _CENTERS_HZ * (
        2.0 ** (1.0 / 6.0) - 2.0 ** (-1.0 / 6.0)
    )
    # The common level offset cancels during normalization and bounds the
    # intermediate linear values.
    return np.power(10.0, 0.1 * (levels_db - np.max(levels_db))) / third_octave_width


def _interpolate_density(density: np.ndarray, frequencies_hz: np.ndarray) -> np.ndarray:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    output = np.zeros_like(frequencies)
    below = (frequencies > 0.0) & (frequencies < _CENTERS_HZ[0])
    output[below] = density[0] * frequencies[below] / _CENTERS_HZ[0]
    supported = frequencies >= _CENTERS_HZ[0]
    output[supported] = np.exp(
        np.interp(
            np.log(frequencies[supported]),
            np.log(_CENTERS_HZ),
            np.log(density),
        )
    )
    return output


def _equal_area_reference(levels_db: np.ndarray) -> np.ndarray:
    curve = _interpolate_density(_band_levels_to_density(levels_db), _REFERENCE_GRID_HZ)
    area = np.trapezoid(curve, _REFERENCE_GRID_HZ)
    if not np.isfinite(area) or area <= 0.0:
        raise RuntimeError("invalid LTASS reference area")
    return curve / area


_MALE_EQUAL_AREA = _equal_area_reference(_MALE_LEVEL_DB)
_FEMALE_EQUAL_AREA = _equal_area_reference(_FEMALE_LEVEL_DB)

_GATE_FREQUENCIES_HZ = np.asarray(
    (250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0),
    dtype=np.float64,
)
# Arithmetic mean of the capped office and clinic profiles from the 2026-08-31
# speech/noise spectral study. Keep each IMCRA-resolution bin explicit.
_GATE_WEIGHTS = np.asarray(
    (
        0.161502,
        0.180000,
        0.180000,
        0.180000,
        0.163814,
        0.088159,
        0.041165,
        0.005360,
    ),
    dtype=np.float64,
)


def speech_gate_band_weights(frequencies_hz: np.ndarray) -> np.ndarray:
    """Return research-derived 50 Hz-bin P-gate weights for office and clinic speech."""

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("Gate frequencies must be a non-empty vector")
    if not np.isfinite(frequencies).all():
        raise ValueError("Gate frequencies must be finite")
    if frequencies.shape != _GATE_FREQUENCIES_HZ.shape or not np.allclose(
        frequencies,
        _GATE_FREQUENCIES_HZ,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("Gate frequency axis must be 250-600 Hz in 50 Hz steps")
    if not np.isclose(np.sum(_GATE_WEIGHTS), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("Gate frequency weights do not sum to one")
    return _GATE_WEIGHTS.copy()


def equal_sex_ltass_weights(frequencies_hz: np.ndarray) -> np.ndarray:
    """Return 50/50 male/female LTASS weights for the supplied FFT bins."""

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("LTASS frequencies must be a non-empty vector")
    if not np.isfinite(frequencies).all() or np.any(frequencies <= 0.0):
        raise ValueError("LTASS frequencies must be positive and finite")
    if np.any(frequencies > 8_000.0):
        raise ValueError("LTASS frequencies must not exceed 8000 Hz")

    male = np.interp(frequencies, _REFERENCE_GRID_HZ, _MALE_EQUAL_AREA)
    female = np.interp(frequencies, _REFERENCE_GRID_HZ, _FEMALE_EQUAL_AREA)
    weights = 0.5 * male + 0.5 * female
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("LTASS weights have no support")
    return weights / total
