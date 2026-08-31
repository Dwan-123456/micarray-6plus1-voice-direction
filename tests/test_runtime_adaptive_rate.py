import threading
from types import SimpleNamespace

from app.adaptive_rate import AdaptiveRateController
from app.runtime import ApplicationRuntime
from gui.dev_test_ui.contracts import L2DevUiSnapshot
from layer2_source_detection.probability_gate import ProbabilityGateDecision, ProbabilityGateState


def test_adaptive_rate_steps_down_and_keeps_base_output_clock() -> None:
    controller = AdaptiveRateController(maximum_period_ms=100)
    assert controller.should_compute()
    controller.observe_compute(queue_wait_ms=21.0, stage_ms={"music": 4.0})
    assert controller.period_ms == 40
    assert not controller.should_compute()
    assert controller.should_compute()

    controller.observe_compute(queue_wait_ms=0.0, stage_ms={"music": 24.0})
    assert controller.period_ms == 60
    assert [controller.should_compute() for _ in range(3)] == [False, False, True]


def test_adaptive_rate_recovers_one_step_after_stable_healthy_time() -> None:
    controller = AdaptiveRateController(maximum_period_ms=100, recovery_stable_ms=80)
    assert controller.should_compute()
    controller.observe_compute(queue_wait_ms=0.0, stage_ms={"l2_total": 21.0})
    assert controller.period_ms == 40
    controller.observe_compute(queue_wait_ms=0.0, stage_ms={"l2_total": 5.0})
    controller.observe_compute(queue_wait_ms=0.0, stage_ms={"l2_total": 5.0})
    assert controller.period_ms == 20
    assert controller.snapshot.last_overload_reason is None


def test_adaptive_rate_fault_forces_one_step_down() -> None:
    controller = AdaptiveRateController(maximum_period_ms=60)
    controller.force_overload("music_fault")
    assert controller.period_ms == 40
    assert controller.snapshot.last_overload_reason == "music_fault"
    controller.force_overload("id_fault")
    assert controller.period_ms == 60


def test_reused_l2_output_is_retimed_and_marked() -> None:
    gate = ProbabilityGateDecision(
        "session", 0, 1, 1_920, "current_20ms_v1", ProbabilityGateState.CLOSED,
        0.2, 0.3, 0.3, 0.6, 0, False, "below_threshold",
    )
    previous = L2DevUiSnapshot(
        "session", 0, 1, 1_920, None, (), gate, 0.6, 0, 0.35, True, 0,
        None, (), (), 1.0, "below_threshold",
    )
    window = SimpleNamespace(
        session_id="session", stream_epoch=0, window_id=2, decision_sample=2_880,
        doa_start_sample=960, doa_end_sample=2_880,
    )
    reused = ApplicationRuntime._reuse_l2_snapshot(
        previous, window, period_ms=40, queue_wait_ms=22.0,
    )
    assert (reused.window_id, reused.decision_sample) == (2, 2_880)
    assert (reused.gate_decision.window_id, reused.gate_decision.decision_sample) == (2, 2_880)
    assert reused.reused_output and reused.processing_period_ms == 40
    assert reused.queue_wait_ms == 22.0
    assert reused.missing_reason == "adaptive_reuse_40ms"


def test_reused_l2_output_uses_current_window_gate_decision() -> None:
    previous_gate = ProbabilityGateDecision(
        "session", 0, 1, 1_920, "current_20ms_v1", ProbabilityGateState.OPEN,
        0.9, 0.9, 0.9, 0.6, 0, True, "above_threshold",
    )
    previous = L2DevUiSnapshot(
        "session", 0, 1, 1_920, None, (), previous_gate, 0.6, 0, 0.35, True, 0,
        None, (), (), 1.0, None,
    )
    window = SimpleNamespace(
        session_id="session", stream_epoch=0, window_id=2, decision_sample=2_880,
        doa_start_sample=960, doa_end_sample=2_880,
    )
    current_gate = ProbabilityGateDecision(
        "session", 0, 2, 2_880, "current_20ms_v1", ProbabilityGateState.OPEN,
        0.7, 0.7, 0.7, 0.6, 0, True, "above_threshold",
    )

    reused = ApplicationRuntime._reuse_l2_snapshot(
        previous,
        window,
        gate_decision=current_gate,
        period_ms=40,
        queue_wait_ms=0.0,
    )

    assert reused.gate_decision.probability_20ms == 0.7
    assert reused.gate_decision.state is ProbabilityGateState.OPEN
    assert "adaptive_reuse_previous_output" in reused.gate_decision.reason


def test_inactive_pre_denoise_tail_is_not_republished_at_shutdown() -> None:
    runtime = object.__new__(ApplicationRuntime)
    runtime.pre_denoiser = SimpleNamespace(
        flush=lambda: (SimpleNamespace(raw="raw", denoised="denoised"),)
    )
    runtime._pre_denoise_latency_active = False
    runtime._pre_denoise_enabled = False
    runtime._pre_denoise_lock = threading.Lock()

    assert runtime._flush_pre_denoiser_tail() == ()

    runtime._pre_denoise_latency_active = True
    assert runtime._flush_pre_denoiser_tail() == ("raw",)


def test_test_ui_has_no_l3_to_l6_reserved_panels() -> None:
    source = __import__("pathlib").Path("gui/dev_test_ui/app.py").read_text(encoding="utf-8")
    assert "ReservedPanel" not in source
    assert "L3 · Reserved" not in source
    assert "class L2ControlPanel" in source
    assert "class L2PolarPanel" in source
    assert "class SquarePolarHost" in source
    assert "left.setFixedWidth(820)" in source
    assert "QSizePolicy.Policy.Ignored" in source
    assert "DPD rank-1" not in source
    assert "IMCRA Whitening" not in source
    assert "grid.addLayout(footer_row, 1, 0, 1, 2)" in source
