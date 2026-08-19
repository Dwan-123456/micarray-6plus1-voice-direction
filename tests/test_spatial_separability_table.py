from __future__ import annotations

import numpy as np

from common.geometry import physical_6plus1_geometry
from spatial_separability import (
    P_FREQUENCIES_HZ,
    P_FREQUENCY_BIN_INDICES,
    P_SIGNED_DELTA_DEGREES,
    P_THETA_A_MODULO_DEGREES,
    load_p_table,
    lookup_p,
    validate_p_table_context,
)


def _direct_p(theta_a_deg: float, theta_b_deg: float) -> np.ndarray:
    geometry = physical_6plus1_geometry()
    radians = np.deg2rad((theta_a_deg, theta_b_deg))
    directions = np.stack((np.cos(radians), np.sin(radians)), axis=-1)
    delays = -(directions @ geometry.positions_m.T) / geometry.speed_of_sound_mps
    steering = np.exp(-2j * np.pi * P_FREQUENCIES_HZ[None, :, None] * delays[:, None, :])
    return np.abs(np.einsum("fc,fc->f", steering[0].conj(), steering[1])) / 7.0


def test_global_p_table_contains_only_read_only_float32_probabilities():
    table = load_p_table()
    assert table.shape == (
        len(P_FREQUENCIES_HZ), len(P_THETA_A_MODULO_DEGREES), len(P_SIGNED_DELTA_DEGREES),
    )
    assert table.dtype == np.float32
    assert not table.flags.writeable
    assert np.isfinite(table).all()
    assert np.all((table >= 0.0) & (table <= 1.0))
    assert P_FREQUENCY_BIN_INDICES.tolist() == list(range(2, 171))


def test_lookup_p_matches_direct_steering_calculation_and_wraps_angles():
    for theta_a, theta_b in ((0, 120), (15, 135), (359, 1), (240, 60), (61, 310)):
        np.testing.assert_allclose(lookup_p(theta_a, theta_b), _direct_p(theta_a, theta_b), atol=2e-6)
        np.testing.assert_array_equal(lookup_p(theta_a, theta_b), lookup_p(theta_b, theta_a))


def test_table_preserves_absolute_orientation_for_the_same_separation():
    first = lookup_p(0, 120)
    rotated = lookup_p(15, 135)
    frequency_index = int(np.argmin(np.abs(P_FREQUENCIES_HZ - 5_000.0)))
    assert abs(float(first[frequency_index]) - float(rotated[frequency_index])) > 0.2


def test_table_context_matches_the_project_bf_configuration():
    validate_p_table_context(
        sample_rate=48_000,
        n_fft=1_024,
        frequency_min_hz=80.0,
        frequency_max_hz=8_000.0,
        geometry=physical_6plus1_geometry(),
    )
