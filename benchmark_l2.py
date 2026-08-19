"""Repeatable CPU benchmark for the specification-locked 360-degree scan."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import DecisionWindow, physical_6plus1_geometry
from common.config import load_config
from layer2_source_detection import DirectionScanConfig, SrpPhatScanner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=200)
    args = parser.parse_args()
    rng = np.random.default_rng(7)
    samples = rng.normal(size=(15_360, 7)).astype(np.float32)
    window = DecisionWindow("benchmark", 0, 0, 15_360, 13_440, 15_360, 0, 15_360, 48_000, samples, (0,))
    project = load_config(Path(__file__).parent / "config" / "config.yaml", environ={})
    config = DirectionScanConfig.from_project(project)
    detector = SrpPhatScanner()
    geometry = physical_6plus1_geometry(
        project.hardware.speed_of_sound_mps,
        project.hardware.geometry_version,
        project.hardware.ring_radius_m,
    )
    for _ in range(10):
        detector.scan(window, geometry, config)
    start = time.perf_counter()
    for _ in range(args.frames):
        detector.scan(window, geometry, config)
    print(f"numpy-cpu frames={args.frames} mean_ms={(time.perf_counter()-start)*1000/args.frames:.3f}")


if __name__ == "__main__":
    main()
