from __future__ import annotations

from pathlib import Path
from time import monotonic

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from app.runtime import ApplicationRuntime
from common.config import load_config
from gui.dev_test_ui.app import MainWindow
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


@pytest.mark.parametrize("count", (0, 1, 2))
def test_test_ui_right_overlay_displays_only_latest_source_count(count: int) -> None:
    window = _window()
    try:
        window.runtime.latest_source_count.put_nowait(
            SourceCountSnapshot("session", 0, 1, 7_680, count, monotonic())
        )
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


def test_test_ui_source_count_and_music_order_switches_are_interlocked() -> None:
    window = _window()
    controls = window.source_controls
    try:
        assert window.runtime.source_counting_enabled
        assert controls.source_count_toggle.isChecked()
        assert controls.follow_order_toggle.isEnabled()
        assert not controls.follow_order_toggle.isChecked()
        assert window.runtime.current_music_effective_order == 2

        controls.follow_order_toggle.setChecked(True)
        assert window.runtime.music_order_follows_source_count
        assert window.runtime.current_music_effective_order is None
        assert controls.music_order_label.text() == "当前阶数：—"

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
        window.runtime.latest_source_count.put_nowait(
            SourceCountSnapshot("session", 0, 1, 7_680, None, monotonic())
        )
        window.refresh()
        assert window.source_count_label.text() == "突出声源数：—"

        window.runtime.latest_source_count.put_nowait(
            SourceCountSnapshot("session", 0, 2, 8_640, 2, monotonic() - 60.0)
        )
        window.refresh()
        assert window.source_count_label.text() == "突出声源数：—"
    finally:
        _close(window)


def test_test_ui_immediately_hides_count_after_count_fault_or_restart() -> None:
    window = _window()
    try:
        published = monotonic()
        window.runtime.latest_source_count.put_nowait(
            SourceCountSnapshot("old-session", 0, 1, 7_680, 2, published)
        )
        window.refresh()
        assert window.source_count_label.text() == "突出声源数：2"

        window.runtime.source_count_last_error = "count-only"
        window.refresh()
        assert window.source_count_label.text() == "突出声源数：—"

        window.runtime.source_count_last_error = None
        window.runtime._started_at = published + 0.001
        window.refresh()
        assert window.source_count_label.text() == "突出声源数：—"
    finally:
        _close(window)
