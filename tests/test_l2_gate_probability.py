from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

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


def test_probability_gate_uses_only_current_hop_and_inclusive_threshold() -> None:
    window = _window()
    gate = ProbabilityGate()
    at_threshold = gate.evaluate(
        window, _probabilities(window, 0.10, 0.60), threshold=0.60, config_revision=3
    )
    below = gate.evaluate(
        window, _probabilities(window, 1.00, 0.59), threshold=0.60, config_revision=4
    )

    assert at_threshold.probability_previous_20ms == 0.10
    assert at_threshold.probability_current_20ms == 0.60
    assert at_threshold.probability_20ms == 0.60
    assert at_threshold.state is ProbabilityGateState.OPEN
    assert at_threshold.allow_srp is True
    assert at_threshold.config_revision == 3
    assert below.probability_20ms == 0.59
    assert below.state is ProbabilityGateState.CLOSED


def test_probability_gate_rejects_missing_warming_and_misaligned_hops() -> None:
    window = _window()
    gate = ProbabilityGate()
    missing = gate.evaluate(window, (), threshold=0.60, config_revision=0)
    first, second = _probabilities(window, 0.5, 0.5)
    warming = SourceProbability20ms(
        window.session_id, window.stream_epoch, window.doa_start_sample + 960,
        window.doa_end_sample, None, SourceProbabilityState.WARMING_UP, "warming",
    )
    warming_decision = gate.evaluate(
        window, (first, warming), threshold=0.60, config_revision=0
    )
    misaligned = SourceProbability20ms(
        window.session_id, window.stream_epoch + 1, window.doa_start_sample + 960,
        window.doa_end_sample, 0.9, SourceProbabilityState.READY, "ready",
    )
    invalid = gate.evaluate(
        window, (first, misaligned), threshold=0.60, config_revision=0
    )

    assert missing.state is ProbabilityGateState.UNAVAILABLE
    assert warming_decision.state is ProbabilityGateState.WARMING_UP
    assert invalid.state is ProbabilityGateState.INVALID
    assert not missing.allow_srp and not warming_decision.allow_srp and not invalid.allow_srp


def test_probability_gate_ignores_previous_hop_state_when_current_is_ready() -> None:
    window = _window()
    previous = SourceProbability20ms(
        window.session_id, window.stream_epoch, window.doa_start_sample,
        window.doa_start_sample + 960, None, SourceProbabilityState.WARMING_UP, "warming",
    )
    current = _probabilities(window, 0.0, 0.80)[1]

    decision = ProbabilityGate().evaluate(
        window, (previous, current), threshold=0.60, config_revision=0
    )

    assert decision.state is ProbabilityGateState.OPEN
    assert decision.probability_previous_20ms is None
    assert decision.probability_20ms == 0.80


@pytest.mark.parametrize("whitening_enabled", [False, True])
@pytest.mark.parametrize("gate_state", ["closed", "warming_up"])
def test_probability_pipeline_skips_music_before_gate_opens(
    whitening_enabled: bool,
    gate_state: str,
) -> None:
    class ForbiddenScanner:
        def scan_detailed(self, *args, **kwargs):
            raise AssertionError("non-open probability Gate must skip MUSIC")

    project = load_config(CONFIG, environ={})
    window = _window()
    before = window.samples.copy()
    pipeline = Layer2Pipeline(ProbabilityGate(), ForbiddenScanner())
    scan_config = replace(
        DirectionScanConfig.from_project(project),
        noise_whitening_enabled=whitening_enabled,
    )
    probabilities = _probabilities(window, 0.20, 0.30)
    if gate_state == "warming_up":
        probabilities = (
            probabilities[0],
            SourceProbability20ms(
                window.session_id,
                window.stream_epoch,
                window.doa_start_sample + 960,
                window.doa_end_sample,
                None,
                SourceProbabilityState.WARMING_UP,
                "warming",
            ),
        )
    result = pipeline.process(
        window,
        probabilities,
        physical_6plus1_geometry(),
        scan_config,
        gate_threshold=0.60,
        gate_config_revision=2,
    )

    assert result.state is Layer2ExecutionState.BLOCKED
    expected_gate_state = (
        ProbabilityGateState.CLOSED
        if gate_state == "closed"
        else ProbabilityGateState.WARMING_UP
    )
    assert result.gate_decision.state is expected_gate_state
    assert result.gate_decision.allow_srp is False
    assert result.spatial_response is None
    assert result.candidates == ()
    assert result.search_diagnostics is None
    assert result.id_tracking_ms is not None
    assert result.id_tracking_ms >= 0.0
    assert np.array_equal(window.samples, before)
