from __future__ import annotations

from pathlib import Path

import numpy as np

from common.config import load_config
from common.geometry import (
    MIC_POSITIONS_M,
    PHYSICAL_GEOMETRY_VERSION,
    PHYSICAL_MIC_ANGLES_DEG,
    physical_6plus1_geometry,
)


CONFIG = Path(__file__).parents[1] / "config" / "config.yaml"


def test_led_face_geometry_uses_mic0_positive_x_and_54321_ccw_order():
    expected_angles = np.asarray(PHYSICAL_MIC_ANGLES_DEG, dtype=np.float64)
    measured_angles = np.mod(
        np.rad2deg(np.arctan2(MIC_POSITIONS_M[:6, 1], MIC_POSITIONS_M[:6, 0])),
        360.0,
    )

    np.testing.assert_allclose(measured_angles, expected_angles, atol=1.0e-6)
    np.testing.assert_allclose(np.linalg.norm(MIC_POSITIONS_M[:6], axis=1), 0.04, atol=1.0e-9)
    np.testing.assert_array_equal(MIC_POSITIONS_M[6], np.zeros(2))


def test_project_config_and_default_geometry_share_the_v2_identity():
    config = load_config(CONFIG)
    geometry = physical_6plus1_geometry()

    assert config.hardware.geometry_version == PHYSICAL_GEOMETRY_VERSION
    assert geometry.version == PHYSICAL_GEOMETRY_VERSION
    np.testing.assert_array_equal(geometry.positions_m, MIC_POSITIONS_M)
