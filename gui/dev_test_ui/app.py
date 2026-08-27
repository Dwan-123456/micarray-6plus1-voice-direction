from __future__ import annotations

import queue
import sys
from pathlib import Path

from app.runtime import ApplicationRuntime
from common.config import load_config
try:
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout,
        QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QSlider,
        QSizePolicy, QVBoxLayout, QWidget,
    )
except ImportError:  # pragma: no cover
    QApplication = None


if QApplication is not None:
    from .srp_panel import DirectionTrackTable, MusicPanelSnapshot, MusicPolarPanel, sync_track_colours

    class L1Panel(QGroupBox):
        CHANNELS = ("MIC0", "MIC1", "MIC2", "MIC3", "MIC4", "MIC5", "Center", "HardwareMix")

        def __init__(self, runtime: ApplicationRuntime) -> None:
            super().__init__("L1 · Input / IMCRA")
            self.runtime = runtime
            outer = QVBoxLayout(self)
            self.status = QLabel("STOPPED | capture closed")
            outer.addWidget(self.status)
            controls = QHBoxLayout()
            self.start_button = QPushButton("启动采集")
            self.stop_button = QPushButton("停止采集")
            self.light_on = QPushButton("灯光开")
            self.light_off = QPushButton("灯光关")
            self.denoise = QCheckBox("IMCRA预降噪")
            self.denoise.setChecked(runtime.l1_pre_denoise_enabled)
            for widget in (self.start_button, self.stop_button, self.light_on, self.light_off, self.denoise):
                controls.addWidget(widget)
            outer.addLayout(controls)
            self.imcra = QLabel("IMCRA: WAITING")
            outer.addWidget(self.imcra)
            meters = QHBoxLayout()
            self.bars: list[QProgressBar] = []
            self.values: list[QLabel] = []
            for name in self.CHANNELS:
                column = QVBoxLayout()
                bar = QProgressBar()
                bar.setOrientation(Qt.Orientation.Vertical)
                bar.setRange(-90, 0)
                bar.setValue(-90)
                value = QLabel("-120.0 dB")
                column.addWidget(bar, 1)
                column.addWidget(QLabel(name), 0, Qt.AlignmentFlag.AlignHCenter)
                column.addWidget(value, 0, Qt.AlignmentFlag.AlignHCenter)
                meters.addLayout(column)
                self.bars.append(bar)
                self.values.append(value)
            outer.addLayout(meters, 1)
            self.start_button.clicked.connect(self._start)
            self.stop_button.clicked.connect(runtime.stop)
            self.light_on.clicked.connect(lambda: self._light(True))
            self.light_off.clicked.connect(lambda: self._light(False))
            self.denoise.toggled.connect(runtime.set_l1_pre_denoise_enabled)

        def _start(self) -> None:
            try:
                self.runtime.start()
            except Exception as exc:
                QMessageBox.critical(self, "采集启动失败", str(exc))

        def _light(self, enabled: bool) -> None:
            try:
                self.runtime.set_light(enabled)
            except Exception as exc:
                QMessageBox.warning(self, "灯光控制失败", str(exc))

        def update_snapshot(self, item: object) -> None:
            self.status.setText(
                f"RUNNING | session {item.session_id[:8]} | epoch {item.stream_epoch:03d} | "
                f"sample {item.end_sample:010d} | seq {item.sequence_id:08d}"
            )
            hop = item.imcra_hop
            probability = None if hop is None else hop.array_source_probability_20ms
            per_mic = "—" if hop is None else ", ".join(f"{float(value):.2f}" for value in hop.source_probability_per_mic)
            self.imcra.setText(
                f"IMCRA: {'WAITING' if hop is None else hop.state.upper()} | "
                f"P1[mic0..6] {per_mic} | P2[array median] "
                f"{'—' if probability is None else f'{probability:.3f}'} | "
                f"预降噪 {'ON' if item.pre_denoise_enabled else 'OFF'} "
                f"{item.pre_denoise_mean_gain_db:.1f} dB"
            )
            for index, value in enumerate(item.rms_dbfs):
                self.bars[index].setValue(max(-90, min(0, int(round(float(value))))))
                self.values[index].setText(f"{float(value):.1f} dB")


    class L2ControlPanel(QGroupBox):
        def __init__(self, runtime: ApplicationRuntime) -> None:
            super().__init__("L2 · MUSIC / ID Tracking Controls")
            self.runtime = runtime
            outer = QVBoxLayout(self)
            self.status = QLabel("STOPPED | no L2 result")
            outer.addWidget(self.status)
            controls = QGridLayout()
            self.gate_label = QLabel(f"Gate threshold {runtime.gate_probability_threshold:.2f}")
            self.gate = QSlider(Qt.Orientation.Horizontal)
            self.gate.setRange(0, 100)
            self.gate.setValue(round(runtime.gate_probability_threshold * 100))
            controls.addWidget(self.gate_label, 0, 0)
            controls.addWidget(self.gate, 0, 1, 1, 3)
            self.id_tracking = QCheckBox("ID Tracking / Prediction")
            self.id_tracking.setChecked(True)
            controls.addWidget(self.id_tracking, 1, 0, 1, 4)
            controls.addWidget(QLabel("MUSIC阶数"), 2, 0)
            self.order = QComboBox()
            self.order.addItems(("1", "2", "3"))
            self.order.setCurrentText(str(runtime.music_effective_order_limit))
            controls.addWidget(self.order, 2, 1)
            self.threshold_label = QLabel(f"Candidate threshold {runtime.direction_threshold:.2f}")
            self.threshold = QSlider(Qt.Orientation.Horizontal)
            self.threshold.setRange(0, 100)
            self.threshold.setValue(round(runtime.direction_threshold * 100))
            controls.addWidget(self.threshold_label, 2, 2)
            controls.addWidget(self.threshold, 2, 3)
            controls.setColumnStretch(1, 1)
            controls.setColumnStretch(3, 2)
            outer.addLayout(controls)
            self.table = DirectionTrackTable()
            outer.addWidget(self.table, 1)
            self.gate.valueChanged.connect(self._gate_changed)
            self.threshold.valueChanged.connect(self._threshold_changed)
            self.id_tracking.toggled.connect(runtime.set_direction_id_tracking_enabled)
            self.order.currentTextChanged.connect(lambda value: runtime.set_music_effective_order_limit(int(value)))

        def _gate_changed(self, value: int) -> None:
            applied = self.runtime.set_gate_probability_threshold(value / 100)
            self.gate_label.setText(f"Gate threshold {applied:.2f}")

        def _threshold_changed(self, value: int) -> None:
            applied = self.runtime.set_direction_threshold(value / 100)
            self.threshold_label.setText(f"Candidate threshold {applied:.2f}")

        def update_snapshot(self, item: object) -> None:
            gate = item.gate_decision
            self.status.setText(
                f"LIVE | window {item.window_id:08d} | P "
                f"{'—' if gate.probability_20ms is None else f'{gate.probability_20ms:.3f}'} | "
                f"Gate {gate.state.value.upper()} | output {len(item.directions)} | "
                f"{'REUSE' if item.reused_output else 'COMPUTE'} @{item.processing_period_ms} ms"
            )
            panel = None if item.spatial_response is None else MusicPanelSnapshot(
                item.spatial_response, item.directions, item.active_tracks, item.published_monotonic,
                effective_order=(None if item.spatial_response.model_order is None else
                                 item.spatial_response.model_order.estimated_sources),
                raw_peaks=item.candidates,
                direction_id_tracking_enabled=item.direction_id_tracking_enabled,
            )
            self.table.set_snapshot(panel)


    class L2PolarPanel(QGroupBox):
        def __init__(self, runtime: ApplicationRuntime) -> None:
            super().__init__("L2 · 360° MUSIC DOA")
            layout = QVBoxLayout(self)
            self.polar = MusicPolarPanel(runtime.config.dev_test_ui.stale_after_ms)
            layout.addWidget(self.polar, 1)

        def update_snapshot(self, item: object) -> None:
            sync_track_colours((item.session_id, item.stream_epoch), item.active_tracks)
            if item.spatial_response is None:
                self.polar.set_gate_closed_tracks(item.active_tracks, window_id=item.window_id, live=True)
                return
            panel = MusicPanelSnapshot(
                item.spatial_response, item.directions, item.active_tracks, item.published_monotonic,
                effective_order=(None if item.spatial_response.model_order is None else
                                 item.spatial_response.model_order.estimated_sources),
                raw_peaks=item.candidates,
                direction_id_tracking_enabled=item.direction_id_tracking_enabled,
            )
            self.polar.set_snapshot(panel, live=True)


    class SquarePolarHost(QWidget):
        """Keep a stable square angle panel independent of changing text hints."""

        _LAYOUT_JITTER_PX = 3

        def __init__(self, panel: L2PolarPanel) -> None:
            super().__init__()
            self.panel = panel
            self.panel.setParent(self)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            self.panel.setMinimumSize(1, 1)
            self.panel.show()

        def resizeEvent(self, event) -> None:  # noqa: N802
            side = max(1, min(self.width(), self.height()))
            x = max(0, (self.width() - side) // 2)
            geometry = self.panel.geometry()
            # Windows/Qt can renegotiate the maximized client area by one or
            # two pixels when focus moves between applications.  Treat that as
            # decoration jitter, otherwise the angle square visibly "breathes"
            # even though the user did not resize the window.  Real resizes
            # still take effect immediately once either dimension changes by
            # more than this small tolerance.
            if (
                geometry.width() > 1
                and abs(side - geometry.width()) <= self._LAYOUT_JITTER_PX
                and abs(side - geometry.height()) <= self._LAYOUT_JITTER_PX
            ):
                super().resizeEvent(event)
                return
            if (geometry.x(), geometry.y(), geometry.width(), geometry.height()) != (
                x, 0, side, side,
            ):
                self.panel.setGeometry(x, 0, side, side)
            super().resizeEvent(event)


    class MainWindow(QMainWindow):
        def __init__(self, runtime: ApplicationRuntime) -> None:
            super().__init__()
            self.runtime = runtime
            self.setWindowTitle("6+1 Microphone Array — v1.4 L1/L2 Development Test UI")
            central = QWidget()
            self.setCentralWidget(central)
            grid = QGridLayout(central)
            self.l1 = L1Panel(runtime)
            self.l2 = L2ControlPanel(runtime)
            self.l2_polar = L2PolarPanel(runtime)
            left = QWidget()
            left_layout = QVBoxLayout(left)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.addWidget(self.l1, 3)
            left_layout.addWidget(self.l2, 2)
            left.setFixedWidth(820)
            grid.addWidget(left, 0, 0)
            right = SquarePolarHost(self.l2_polar)
            grid.addWidget(right, 0, 1)
            grid.setRowStretch(0, 1)
            grid.setColumnStretch(0, 2)
            grid.setColumnStretch(1, 3)
            self.footer = QLabel("v1.4.2 | L1/L2 only")
            self.footer.setMinimumWidth(0)
            self.footer.setFixedHeight(28)
            self.footer.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            footer_row = QHBoxLayout()
            self.performance_toggle = QCheckBox("性能监控")
            self.performance_toggle.setFixedHeight(28)
            self.performance_toggle.setChecked(runtime.performance_monitor_enabled)
            self.performance_toggle.toggled.connect(runtime.set_performance_monitor_enabled)
            footer_row.addWidget(self.performance_toggle)
            footer_row.addWidget(self.footer, 1)
            grid.addLayout(footer_row, 1, 0, 1, 2)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(20)
            self._last_performance_second = -1
            self.resize(1920, 1080)

        @staticmethod
        def _latest(mailbox: queue.Queue[object]) -> object | None:
            value = None
            while True:
                try:
                    value = mailbox.get_nowait()
                except queue.Empty:
                    return value

        def refresh(self) -> None:
            if (item := self._latest(self.runtime.latest_l1)) is not None:
                self.l1.update_snapshot(item)
            if (item := self._latest(self.runtime.latest_l2_dev_ui)) is not None:
                self.l2.update_snapshot(item)
                self.l2_polar.update_snapshot(item)
            second = int(__import__("time").monotonic())
            if second != self._last_performance_second:
                self._last_performance_second = second
                perf = self.runtime.performance_snapshot
                status = self.runtime.processing_status
                self.footer.setText(
                    f"上一秒平均 | IMCRA {perf['imcra_ms']:.2f} ms | P {perf['probability_ms']:.3f} ms | "
                    f"MUSIC {perf['music_ms']:.2f} ms | ID {perf['id_tracking_ms']:.2f} ms | "
                    f"总计 {perf['total_ms']:.2f} ms | 输出 {perf['frames_per_second']} fps | "
                    f"实算 {perf['compute_frames_per_second']} fps | L2周期 {perf['adaptive_period_ms']} ms | "
                    f"排队 {perf['queue_wait_ms']:.2f} ms | "
                    f"故障 {perf['faults_per_second']} | "
                    f"drop {status['processing_drops']} | {status['last_error'] or 'OK'}"
                )

        def closeEvent(self, event) -> None:  # noqa: N802
            self.runtime.close()
            event.accept()


def main(argv: list[str] | None = None) -> int:
    if QApplication is None:
        raise RuntimeError("PySide6 is required for Development Test UI")
    arguments = sys.argv[1:] if argv is None else argv
    if arguments:
        raise ValueError("v1.4 Test UI accepts live microphone input only")
    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setStyleSheet(
        "QMainWindow,QWidget{background:#edf2f7;color:#101820;font-size:14px}"
        "QGroupBox{font-weight:bold;border:1px solid #aab8c6;margin-top:10px;padding-top:10px}"
        "QPushButton,QComboBox{min-height:30px;padding:3px 10px}"
        "QProgressBar::chunk{background:#2399ad}"
    )
    config_path = Path("config/config.yaml").resolve()
    runtime = ApplicationRuntime(load_config(config_path), project_root=config_path.parent.parent)
    window = MainWindow(runtime)
    if runtime.config.dev_test_ui.start_fullscreen:
        window.showFullScreen()
    else:
        window.showMaximized()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
