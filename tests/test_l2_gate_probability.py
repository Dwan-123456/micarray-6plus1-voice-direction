from __future__ import annotations

from pathlib import Path

import numpy as np

from common.config import load_config
from common.data_types import DecisionWindow
from common.geometry import physical_6plus1_geometry
from layer2_source_detection import (
    DirectionScanConfig,
    Layer2ExecutionState,
    Layer2Pipeline,
    ProbabilityGate,
    ProbabilityGateState,
    SourceProbability20ms,
    SourceProbabilityState,
)


CONFIG = Path(__file__).parents[1] / "config" / "config.yaml"


def _window() -> DecisionWindow:
    return DecisionWindow(
        "gate-test", 0, 4, 7_680, 5_760, 7_680, 0, 7_680, 48_000,
        np.zeros((7_680, 8), dtype=np.float32), (1,),
    )


def _probabilities(window: DecisionWindow, previous: float, current: float):
    return (
        SourceProbability20ms(
            window.session_id, window.stream_epoch, window.doa_start_sample,
            window.doa_start_sample + 960, previous, SourceProbabilityState.READY, "ready",
        ),
        SourceProbability20ms(
            window.session_id, window.stream_epoch, window.doa_start_sample + 960,
            window.doa_end_sample, current, SourceProbabilityState.READY, "ready",
        ),
    )


def test_probability_gate_averages_two_hops_and_uses_inclusive_threshold() -> None:
    window = _window()
    gate = ProbabilityGate()
    at_threshold = gate.evaluate(
        window, _probabilities(window, 0.50, 0.70), threshold=0.60, config_revision=3
    )
    below = gate.evaluate(
        window, _probabilities(window, 0.49, 0.70), threshold=0.60, config_revision=4
    )

    assert at_threshold.probability_40ms == 0.60
    assert at_threshold.state is ProbabilityGateState.OPEN
    assert at_threshold.allow_srp is True
    assert at_threshold.config_revision == 3
    assert below.probability_40ms == 0.595
    assert below.state is ProbabilityGateState.CLOSED


def test_probability_gate_rejects_missing_warming_and_misaligned_hops() -> None:
    window = _window()
    gate = ProbabilityGate()
    missing = gate.evaluate(window, (), threshold=0.60, config_revision=0)
    warming = SourceProbability20ms(
        window.session_id, window.stream_epoch, window.doa_start_sample,
        window.doa_start_sample + 960, None, SourceProbabilityState.WARMING_UP, "warming",
    )
    second = _probabilities(window, 0.5, 0.5)[1]
    warming_decision = gate.evaluate(
        window, (warming, second), threshold=0.60, config_revision=0
    )
    misaligned = SourceProbability20ms(
        window.session_id, window.stream_epoch + 1, window.doa_start_sample,
        window.doa_start_sample + 960, 0.9, SourceProbabilityState.READY, "ready",
    )
    invalid = gate.evaluate(
        window, (misaligned, second), threshold=0.60, config_revision=0
    )

    assert missing.state is ProbabilityGateState.UNAVAILABLE
    assert warming_decision.state is ProbabilityGateState.WARMING_UP
    assert invalid.state is ProbabilityGateState.INVALID
    assert not missing.allow_srp and not warming_decision.allow_srp and not invalid.allow_srp


def test_probability_pipeline_skips_srp_when_closed_and_preserves_audio() -> None:
    class ForbiddenScanner:
        def scan_detailed(self, *args, **kwargs):
            raise AssertionError("closed probability Gate must skip SRP-PHAT")

    project = load_config(CONFIG, environ={})
    window = _window()
    before = window.samples.copy()
    pipeline = Layer2Pipeline(ProbabilityGate(), ForbiddenScanner())
    result = pipeline.process(
        window,
        _probabilities(window, 0.20, 0.30),
        physical_6plus1_geometry(),
        DirectionScanConfig.from_project(project),
        gate_threshold=0.60,
        gate_config_revision=2,
    )

    assert result.state is Layer2ExecutionState.BLOCKED
    assert result.spatial_response is None
    assert result.candidates == ()
    assert result.search_diagnostics is None
    assert np.array_equal(window.samples, before)
