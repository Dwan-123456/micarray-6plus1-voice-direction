"""Repeatable CPU benchmark for rolling 360-degree NormMUSIC."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import DecisionWindow, physical_6plus1_geometry
from common.config import load_config
from layer2_source_detection import DirectionScanConfig, RollingNormMusicScanner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=200)
    args = parser.parse_args()
    rng = np.random.default_rng(7)
    context_samples = 7_680
    timeline = rng.normal(
        size=(context_samples + (args.frames + 1) * 960, 8)
    ).astype(np.float32)

    def make_window(index: int) -> DecisionWindow:
        decision_sample = context_samples + index * 960
        samples = timeline[decision_sample - context_samples : decision_sample]
        return DecisionWindow(
            "benchmark", 0, index, decision_sample,
            decision_sample - 1_920, decision_sample,
            decision_sample - context_samples, decision_sample,
            48_000, samples, (index,),
        )

    window = make_window(0)
    project = load_config(Path(__file__).parent / "config" / "config.yaml", environ={})
    config = DirectionScanConfig.from_project(project)
    detector = RollingNormMusicScanner()
    geometry = physical_6plus1_geometry(
        project.hardware.speed_of_sound_mps,
        project.hardware.geometry_version,
        project.hardware.ring_radius_m,
    )
    detector.scan(window, geometry, config)
    start = time.perf_counter()
    for index in range(args.frames):
        shifted = make_window(index + 1)
        detector.scan(shifted, geometry, config)
    print(f"numpy-cpu frames={args.frames} mean_ms={(time.perf_counter()-start)*1000/args.frames:.3f}")


if __name__ == "__main__":
    main()
