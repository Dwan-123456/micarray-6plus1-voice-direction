from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from common.geometry import MIC_POSITIONS_M, MicGeometry


P_TABLE_VERSION = "spatial_separability_p_48k_fft1024_6plus1_v1"
P_SAMPLE_RATE = 48_000
P_N_FFT = 1_024
P_FREQUENCY_MIN_HZ = 80.0
P_FREQUENCY_MAX_HZ = 8_000.0
P_SPEED_OF_SOUND_MPS = 343.0
P_GEOMETRY_VERSION = "r6plus1_mic_face_ccw_v1"

_ALL_FREQUENCIES_HZ = np.fft.rfftfreq(P_N_FFT, 1.0 / P_SAMPLE_RATE)
P_FREQUENCY_BIN_INDICES = np.flatnonzero(
    (_ALL_FREQUENCIES_HZ >= P_FREQUENCY_MIN_HZ)
    & (_ALL_FREQUENCIES_HZ <= P_FREQUENCY_MAX_HZ)
).astype(np.int16)
P_FREQUENCIES_HZ = _ALL_FREQUENCIES_HZ[P_FREQUENCY_BIN_INDICES].astype(np.float32)
P_THETA_A_MODULO_DEGREES = np.arange(60, dtype=np.int16)
P_SIGNED_DELTA_DEGREES = np.arange(-180, 180, dtype=np.int16)

for _axis in (
    P_FREQUENCY_BIN_INDICES,
    P_FREQUENCIES_HZ,
    P_THETA_A_MODULO_DEGREES,
    P_SIGNED_DELTA_DEGREES,
):
    _axis.setflags(write=False)

_EXPECTED_SHAPE = (
    P_FREQUENCIES_HZ.size,
    P_THETA_A_MODULO_DEGREES.size,
    P_SIGNED_DELTA_DEGREES.size,
)
_TABLE_PATH = Path(__file__).with_name("spatial_separability_p.npy")


@lru_cache(maxsize=1)
def load_p_table() -> np.ndarray:
    """Return the process-wide read-only table with axes [frequency, theta-A mod 60, signed delta]."""
    try:
        table = np.load(_TABLE_PATH, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"无法读取空间可分度p表: {_TABLE_PATH}") from exc
    if table.shape != _EXPECTED_SHAPE or table.dtype != np.float32:
        raise RuntimeError(
            f"空间可分度p表格式错误: expected float32 {_EXPECTED_SHAPE}, got {table.dtype} {table.shape}"
        )
    if not np.isfinite(table).all() or np.any((table < 0.0) | (table > 1.0)):
        raise RuntimeError("空间可分度p表包含非有限值或超出[0,1]")
    table.setflags(write=False)
    return table


def _nearest_degree(value: float) -> int:
    if not np.isfinite(value):
        raise ValueError("查表角度必须为有限数值")
    return int(np.floor(float(value) + 0.5))


def lookup_p(theta_a_deg: float, theta_b_deg: float) -> np.ndarray:
    """Look up p for all 169 BF bins, quantizing both input angles to the nearest degree."""
    theta_a = _nearest_degree(float(theta_a_deg) % 360.0) % 360
    theta_b = _nearest_degree(float(theta_b_deg) % 360.0) % 360
    # p is symmetric in A/B; canonical ordering makes lookup bit-identical if candidates swap order.
    theta_a, theta_b = min(theta_a, theta_b), max(theta_a, theta_b)
    signed_delta = (theta_b - theta_a + 180) % 360 - 180
    values = np.asarray(load_p_table()[:, theta_a % 60, signed_delta + 180])
    values.setflags(write=False)
    return values


def validate_p_table_context(
    *, sample_rate: int, n_fft: int, frequency_min_hz: float, frequency_max_hz: float,
    geometry: MicGeometry,
) -> None:
    """Reject use of the static table with a different STFT or physical array."""
    if (
        sample_rate != P_SAMPLE_RATE
        or n_fft != P_N_FFT
        or frequency_min_hz != P_FREQUENCY_MIN_HZ
        or frequency_max_hz != P_FREQUENCY_MAX_HZ
    ):
        raise ValueError("空间可分度p表与当前采样率、FFT或BF频带不匹配")
    if (
        geometry.version != P_GEOMETRY_VERSION
        or geometry.speed_of_sound_mps != P_SPEED_OF_SOUND_MPS
        or not np.allclose(geometry.positions_m, MIC_POSITIONS_M, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("空间可分度p表与当前7麦阵列几何、声速或版本不匹配")
