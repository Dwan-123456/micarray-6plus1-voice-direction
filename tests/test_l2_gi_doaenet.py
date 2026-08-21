from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from common.config import load_config
from common.data_types import DecisionWindow
from common.data_types import CandidateDirection
from common.geometry import physical_6plus1_geometry
from gui.dev_test_ui.settings import DevUiSettings
from layer2_source_detection import DirectionScanConfig, Layer2Pipeline
from layer2_source_detection.gi_doaenet import GiDoaEnetScanner
from layer2_source_detection.global_tracker import GlobalDirectionTracker


CONFIG = Path(__file__).parents[1] / "config" / "config.yaml"


def _window(index: int = 0) -> DecisionWindow:
    decision = 7_680 + index * 960
    audio = np.zeros((7_680, 8), dtype=np.float32)
    return DecisionWindow(
        "nn", 0, index, decision, decision - 1_920, decision,
        decision - 7_680, decision, 48_000, audio, (index,),
    )


class _FakeGiModel:
    def __call__(self, audio, coordinates):
        assert audio.shape == (1, 7, 2_560)
        assert coordinates.shape == (1, 7, 3)
        output = torch.zeros((1, 3, 360, 4), dtype=torch.float32)
        output[:, -1, 30, :] = 0.91
        output[:, -1, 210, :] = 0.82
        return output


def test_gi_adapter_preserves_common_360_degree_contract() -> None:
    config = load_config(CONFIG)
    scan = replace(
        DirectionScanConfig.from_project(config),
        scanner_backend="gi_doaenet",
        direction_threshold=0.20,
        effective_order_limit=2,
    )
    scanner = GiDoaEnetScanner(project_root=CONFIG.parents[1], device="cpu")
    scanner._torch = torch
    scanner._model = _FakeGiModel()
    scanner._device = "cpu"
    response, candidates, diagnostics = scanner.scan_detailed(
        _window(), physical_6plus1_geometry(), scan, 7
    )
    assert response.normalized_scores.shape == (360,)
    assert tuple(item.theta_deg for item in candidates) == (30.0, 210.0)
    assert diagnostics.mode == "gi_doaenet"
    assert diagnostics.config_revision == 7
    assert diagnostics.model_order.estimated_sources == 2


def test_complete_l2_backends_have_independent_association_engines() -> None:
    pipeline = Layer2Pipeline.from_project(load_config(CONFIG), project_root=CONFIG.parents[1])
    assert pipeline._trackers["frequency_normalized_music"].association_backend == "hungarian"
    assert pipeline._trackers["gi_doaenet"].association_backend == "lmb_jpda"


def _candidate(sample: int, theta: float) -> CandidateDirection:
    return CandidateDirection("jpda", 0, sample // 960, sample, sample - 1_920, sample, theta, 0.9, 0.9)


def test_lmb_jpda_is_one_to_one_and_circular_across_zero() -> None:
    tracker = GlobalDirectionTracker(association_backend="lmb_jpda")
    observed, _ = tracker.update(
        "jpda", 0, 7_680, (_candidate(7_680, 359.0), _candidate(7_680, 180.0)),
        window_id=8, doa_start_sample=5_760,
    )
    initial = {round(item.measured_theta_deg): item.track_id for item in observed}
    observed, _ = tracker.update(
        "jpda", 0, 8_640, (_candidate(8_640, 1.0), _candidate(8_640, 182.0)),
        window_id=9, doa_start_sample=6_720,
    )
    updated = {round(item.measured_theta_deg): item.track_id for item in observed}
    assert updated[1] == initial[359]
    assert updated[182] == initial[180]
    assert len(set(tracker.last_assignments)) == len(tracker.last_assignments) == 2
    assert all(value >= 0.20 for value in tracker.last_association_probabilities)


def test_ui_persists_complete_l2_backend_without_erasing_threshold(tmp_path: Path) -> None:
    settings = DevUiSettings(tmp_path)
    settings.save_direction_threshold(0.23)
    settings.save_doa_backend("gi_doaenet")
    assert settings.load_doa_backend() == "gi_doaenet"
    assert settings.load_direction_threshold(0.35) == 0.23
    settings.save_doa_backend("frequency_normalized_music")
    assert settings.load_doa_backend() == "frequency_normalized_music"
