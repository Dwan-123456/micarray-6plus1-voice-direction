from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MIC_POSITIONS_M = np.asarray(
    [
        (0.040000000, 0.000000000),
        (0.020000000, 0.034641016),
        (-0.020000000, 0.034641016),
        (-0.040000000, 0.000000000),
        (-0.020000000, -0.034641016),
        (0.020000000, -0.034641016),
        (0.000000000, 0.000000000),
    ],
    dtype=np.float64,
)
MIC_POSITIONS_M.setflags(write=False)


def microphone_positions_m() -> np.ndarray:
    return MIC_POSITIONS_M


@dataclass(frozen=True, slots=True)
class MicGeometry:
    positions_m: np.ndarray
    speed_of_sound_mps: float
    version: str

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_m, dtype=np.float64)
        if positions.shape != (7, 2) or not np.isfinite(positions).all():
            raise ValueError("麦克风坐标必须是有限float64 [7,2]")
        if self.speed_of_sound_mps <= 0 or not np.isfinite(self.speed_of_sound_mps):
            raise ValueError("声速必须为有限正数")
        immutable = np.frombuffer(np.ascontiguousarray(positions).tobytes(), dtype=np.float64).reshape(7, 2)
        object.__setattr__(self, "positions_m", immutable)


def physical_6plus1_geometry(
    speed_of_sound_mps: float = 343.0,
    version: str = "r6plus1_mic_face_ccw_v1",
    ring_radius_m: float = 0.04,
) -> MicGeometry:
    if not np.isfinite(ring_radius_m) or ring_radius_m <= 0:
        raise ValueError("ring_radius_m must be finite and positive")
    scale = ring_radius_m / 0.04
    return MicGeometry(MIC_POSITIONS_M * scale, speed_of_sound_mps, version)
