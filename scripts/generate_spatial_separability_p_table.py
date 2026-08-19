from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.geometry import physical_6plus1_geometry  # noqa: E402
from spatial_separability.p_table import (  # noqa: E402
    P_FREQUENCIES_HZ,
    P_SIGNED_DELTA_DEGREES,
    P_THETA_A_MODULO_DEGREES,
)


OUTPUT = PROJECT_ROOT / "spatial_separability/spatial_separability_p.npy"


def generate() -> np.ndarray:
    geometry = physical_6plus1_geometry()
    positions = geometry.positions_m
    frequencies = P_FREQUENCIES_HZ.astype(np.float64)
    deltas = P_SIGNED_DELTA_DEGREES.astype(np.float64)
    table = np.empty(
        (frequencies.size, P_THETA_A_MODULO_DEGREES.size, deltas.size), dtype=np.float32,
    )
    for theta_a in P_THETA_A_MODULO_DEGREES:
        angle_a = np.deg2rad(float(theta_a))
        direction_a = np.asarray((np.cos(angle_a), np.sin(angle_a)))
        delay_a = -(direction_a @ positions.T) / geometry.speed_of_sound_mps
        steering_a = np.exp(-2j * np.pi * frequencies[:, None] * delay_a[None, :])

        angles_b = np.deg2rad(float(theta_a) + deltas)
        directions_b = np.stack((np.cos(angles_b), np.sin(angles_b)), axis=-1)
        delays_b = -(directions_b @ positions.T) / geometry.speed_of_sound_mps
        steering_b = np.exp(-2j * np.pi * frequencies[:, None, None] * delays_b[None, :, :])
        correlation = np.abs(np.einsum("fc,fdc->fd", steering_a.conj(), steering_b)) / positions.shape[0]
        table[:, int(theta_a), :] = correlation.astype(np.float32)
    return table


def main() -> None:
    table = generate()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".partial.npy")
    np.save(temporary, table, allow_pickle=False)
    temporary.replace(OUTPUT)
    print(f"saved {OUTPUT} shape={table.shape} dtype={table.dtype} bytes={table.nbytes}")


if __name__ == "__main__":
    main()
