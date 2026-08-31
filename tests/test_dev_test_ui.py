from __future__ import annotations

from pathlib import Path
from time import monotonic

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QHeaderView

from app.runtime import ApplicationRuntime
from common.config import load_config
from gui.dev_test_ui.app import MainWindow
from gui.dev_test_ui.contracts import L2DevUiSnapshot
from layer2_source_detection import ProbabilityGateDecision, ProbabilityGateState
from source_counting import SourceCountSnapshot


CONFIG = Path(__file__).parents[1] / "config" / "config.yaml"
APPLICATION = QApplication.instance() or QApplication([])


def _window() -> MainWindow:
    runtime = ApplicationRuntime(load_config(CONFIG), project_root=CONFIG.parent.parent)
    window = MainWindow(runtime)
    window.timer.stop()
    return window


def _close(window: MainWindow) -> None:
    window.close()
    APPLICATION.processEvents()


def _footer_widgets(window: MainWindow) -> tuple[object, ...]:
    footer = window.centralWidget().layout().itemAtPosition(1, 0).layout()
    assert footer is not None
    return tuple(footer.itemAt(index).widget() for index in range(footer.count()))


def _l2_with_count(source_count: SourceCountSnapshot) -> L2DevUiSnapshot:
    gate = ProbabilityGateDecision(
        source_count.session_id,
        source_count.stream_epoch,
        source_count.window_id,
        source_count.decision_sample,
        "test_gate",
        ProbabilityGateState.CLOSED,
        0.1,
        0.1,
        0.1,
        0.8,
        0,
        False,
        "below_threshold",
    )
    return L2DevUiSnapshot(
        source_count.session_id,
        source_count.stream_epoch,
        source_count.window_id,
        source_count.decision_sample,
        None,
        (),
        gate,
        0.8,
        0,
        0.35,
        True,
        0,
        None,
        (),
        (),
        monotonic(),
        "below_threshold",
        source_count_snapshot=source_count,
    )


@pytest.mark.parametrize("count", (0, 1, 2))
def test_test_ui_right_overlay_displays_only_latest_source_count(count: int) -> None:
    window = _window()
    try:
        source_count = SourceCountSnapshot("session", 0, 1, 7_680, count, monotonic())
        window.runtime.latest_l2_dev_ui.put_nowait(_l2_with_count(source_count))
        window.refresh()

        assert window.source_count_label.text() == f"突出声源数：{count}"
        assert window.source_controls.parent() is window.right_host
        assert window.source_count_label.parent() is window.source_controls
    finally:
        _close(window)


def test_test_ui_performance_footer_contains_no_source_count_controls() -> None:
    window = _window()
    try:
        assert _footer_widgets(window) == (window.performance_toggle, window.footer)
        footer_widgets = _footer_widgets(window)
        assert window.source_controls not in footer_widgets
        assert window.source_controls.source_count_toggle not in footer_widgets
        assert window.source_controls.follow_order_toggle not in footer_widgets
        assert window.source_count_label not in footer_widgets
    finally:
        _close(window)


def test_l1_meters_use_minus_60_dbfs_floor() -> None:
    window = _window()
    try:
        assert all(bar.minimum() == -60 for bar in window.l1.bars)
        assert all(bar.value() == -60 for bar in window.l1.bars)
        assert all(label.text() == "-60.0 dB" for label in window.l1.values)
    finally:
        _close(window)


def test_track_table_uses_stable_fixed_columns_for_coasting_state() -> None:
    window = _window()
    try:
        table = window.l2.table
        header = table.horizontalHeader()
        for column, width in enumerate(table.COLUMN_WIDTHS):
            assert header.sectionResizeMode(column) == QHeaderView.ResizeMode.Fixed
            assert table.columnWidth(column) == width
        assert header.sectionResizeMode(table.columnCount() - 1) == QHeaderView.ResizeMode.Stretch
    finally:
        _close(window)


def test_test_ui_source_count_and_music_order_switches_are_interlocked() -> None:
    window = _window()
    controls = window.source_controls
    try:
        assert window.runtime.source_counting_enabled
        assert controls.source_count_toggle.isChecked()
        assert controls.follow_order_toggle.isEnabled()
        assert controls.follow_order_toggle.isChecked()
        assert window.runtime.music_order_follows_source_count
        assert window.runtime.current_music_effective_order == 1
        assert controls.music_order_label.text() == "当前阶数：1"

        controls.source_count_toggle.setChecked(False)
        window.refresh()
        assert not window.runtime.source_counting_enabled
        assert not window.runtime.music_order_follows_source_count
        assert not controls.source_count_toggle.isChecked()
        assert not controls.follow_order_toggle.isChecked()
        assert not controls.follow_order_toggle.isEnabled()
        assert window.runtime.current_music_effective_order == 2
        assert controls.music_order_label.text() == "当前阶数：2"
        assert window.source_count_label.text() == "突出声源数：—"

        controls.source_count_toggle.setChecked(True)
        assert window.runtime.source_counting_enabled
        assert controls.source_count_toggle.isChecked()
        assert controls.follow_order_toggle.isEnabled()
        assert not controls.follow_order_toggle.isChecked()

        controls.follow_order_toggle.setChecked(True)
        controls.follow_order_toggle.setChecked(False)
        assert not window.runtime.music_order_follows_source_count
        assert window.runtime.current_music_effective_order == 2
        assert controls.music_order_label.text() == "当前阶数：2"
    finally:
        _close(window)


@pytest.mark.parametrize(
    ("host_width", "host_height"),
    ((522, 716), (1_076, 1_028), (1_716, 1_388)),
)
def test_test_ui_overlay_and_switches_do_not_resize_square_polar_panel(
    host_width: int,
    host_height: int,
) -> None:
    window = _window()
    host = window.right_host
    controls = window.source_controls
    try:
        window.show()
        APPLICATION.processEvents()
        host.setFixedSize(host_width, host_height)
        APPLICATION.processEvents()

        assert (host.width(), host.height()) == (host_width, host_height)

        side = min(host_width, host_height)
        expected_polar = ((host_width - side) // 2, 0, side, side)
        assert window.l2_polar.geometry().getRect() == expected_polar
        original_polar = window.l2_polar.geometry().getRect()

        assert controls.x() == max(0, host_width - controls.width() - 10)
        assert controls.y() == max(0, host_height - controls.height() - 10)

        controls.follow_order_toggle.setChecked(True)
        controls.source_count_toggle.setChecked(False)
        controls.source_count_toggle.setChecked(True)
        controls.follow_order_toggle.setChecked(True)
        APPLICATION.processEvents()

        assert window.l2_polar.geometry().getRect() == original_polar
        assert window.l2_polar.width() == window.l2_polar.height() == side
    finally:
        _close(window)


def test_test_ui_hides_warming_or_stale_source_count() -> None:
    window = _window()
    try:
        warming = SourceCountSnapshot("session", 0, 1, 7_680, None, monotonic())
        window.runtime.latest_l2_dev_ui.put_nowait(_l2_with_count(warming))
        window.refresh()
        assert window.source_count_label.text() == "突出声源数：—"

        stale = SourceCountSnapshot("session", 0, 2, 8_640, 2, monotonic() - 60.0)
        window.runtime.latest_l2_dev_ui.put_nowait(_l2_with_count(stale))
        window.refresh()
        assert window.source_count_label.text() == "突出声源数：—"
    finally:
        _close(window)


def test_test_ui_immediately_hides_atomic_count_fault_or_restart() -> None:
    window = _window()
    try:
        published = monotonic()
        source_count = SourceCountSnapshot("old-session", 0, 1, 7_680, 2, published)
        window.runtime.latest_l2_dev_ui.put_nowait(_l2_with_count(source_count))
        window.refresh()
        assert window.source_count_label.text() == "突出声源数：2"

        fault = SourceCountSnapshot("old-session", 0, 2, 8_640, None, monotonic())
        window.runtime.latest_l2_dev_ui.put_nowait(_l2_with_count(fault))
        window.refresh()
        assert window.source_count_label.text() == "突出声源数：—"

        window.runtime.latest_l2_dev_ui.put_nowait(_l2_with_count(source_count))
        window.runtime._started_at = published + 0.001
        window.refresh()
        assert window.source_count_label.text() == "突出声源数：—"
    finally:
        _close(window)


def test_test_ui_ignores_racing_independent_count_mailbox() -> None:
    window = _window()
    try:
        atomic = SourceCountSnapshot("session", 0, 1, 7_680, 1, monotonic())
        racing = SourceCountSnapshot("session", 0, 2, 8_640, 2, monotonic())
        window.runtime.latest_l2_dev_ui.put_nowait(_l2_with_count(atomic))
        window.runtime.latest_source_count.put_nowait(racing)

        window.refresh()

        assert window.source_count_label.text() == "突出声源数：1"
    finally:
        _close(window)
