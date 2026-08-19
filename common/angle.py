from __future__ import annotations

import math

import numpy as np


THETA_DEGREES = np.arange(360, dtype=np.float32)
THETA_DEGREES.setflags(write=False)


def normalize_theta_deg(theta_deg: float) -> float:
    value = float(theta_deg)
    if not math.isfinite(value):
        raise ValueError("theta_deg必须是有限数值")
    normalized = value % 360.0
    return 0.0 if normalized == 360.0 else normalized


def wrap_theta_deg(theta_deg: float) -> float:
    return normalize_theta_deg(theta_deg)


def circular_distance_deg(left: float, right: float) -> float:
    delta = abs(normalize_theta_deg(left) - normalize_theta_deg(right))
    return min(delta, 360.0 - delta)
