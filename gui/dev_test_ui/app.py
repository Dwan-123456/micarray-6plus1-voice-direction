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
        QVBoxLayout, QWidget,
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


    class L2Panel(QGroupBox):
        def __init__(self, runtime: ApplicationRuntime) -> None:
            super().__init__("L2 · MUSIC DOA / ID Tracking")
            self.runtime = runtime
            outer = QVBoxLayout(self)
            self.status = QLabel("STOPPED | no L2 result")
            outer.addWidget(self.status)
            body = QHBoxLayout()
            self.polar = MusicPolarPanel(runtime.config.dev_test_ui.stale_after_ms)
            body.addWidget(self.polar, 3)
            controls = QVBoxLayout()
            self.gate_label = QLabel(f"Gate threshold {runtime.gate_probability_threshold:.2f}")
            self.gate = QSlider(Qt.Orientation.Horizontal)
            self.gate.setRange(0, 100)
            self.gate.setValue(round(runtime.gate_probability_threshold * 100))
            controls.addWidget(self.gate_label)
            controls.addWidget(self.gate)
            self.id_tracking = QCheckBox("ID Tracking / Prediction")
            self.id_tracking.setChecked(True)
            self.dpd = QCheckBox("DPD rank-1")
            self.dpd.setChecked(runtime.music_dpd_rank1_enabled)
            self.whitening = QCheckBox("IMCRA Whitening")
            self.whitening.setChecked(runtime.music_noise_whitening_enabled)
            for widget in (self.id_tracking, self.dpd, self.whitening):
                controls.addWidget(widget)
            order_row = QHBoxLayout()
            order_row.addWidget(QLabel("MUSIC阶数"))
            self.order = QComboBox()
            self.order.addItems(("1", "2", "3"))
            self.order.setCurrentText(str(runtime.music_effective_order_limit))
            order_row.addWidget(self.order)
            controls.addLayout(order_row)
            self.threshold_label = QLabel(f"Candidate threshold {runtime.direction_threshold:.2f}")
            self.threshold = QSlider(Qt.Orientation.Horizontal)
            self.threshold.setRange(0, 100)
            self.threshold.setValue(round(runtime.direction_threshold * 100))
            controls.addWidget(self.threshold_label)
            controls.addWidget(self.threshold)
            self.table = DirectionTrackTable()
            controls.addWidget(self.table)
            body.addLayout(controls, 2)
            outer.addLayout(body, 1)
            self.gate.valueChanged.connect(self._gate_changed)
            self.threshold.valueChanged.connect(self._threshold_changed)
            self.id_tracking.toggled.connect(runtime.set_direction_id_tracking_enabled)
            self.dpd.toggled.connect(runtime.set_music_dpd_rank1_enabled)
            self.whitening.toggled.connect(runtime.set_music_noise_whitening_enabled)
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
            sync_track_colours((item.session_id, item.stream_epoch), item.active_tracks)
            if item.spatial_response is None:
                self.polar.set_gate_closed_tracks(item.active_tracks, window_id=item.window_id, live=True)
                self.table.set_snapshot(None)
                return
            panel = MusicPanelSnapshot(
                item.spatial_response, item.directions, item.active_tracks, item.published_monotonic,
                effective_order=(None if item.spatial_response.model_order is None else
                                 item.spatial_response.model_order.estimated_sources),
                raw_peaks=item.candidates,
                direction_id_tracking_enabled=item.direction_id_tracking_enabled,
            )
            self.polar.set_snapshot(panel, live=True)
            self.table.set_snapshot(panel)


    class MainWindow(QMainWindow):
        def __init__(self, runtime: ApplicationRuntime) -> None:
            super().__init__()
            self.runtime = runtime
            self.setWindowTitle("6+1 Microphone Array — v1.4 L1/L2 Development Test UI")
            central = QWidget()
            self.setCentralWidget(central)
            grid = QGridLayout(central)
            self.l1 = L1Panel(runtime)
            self.l2 = L2Panel(runtime)
            grid.addWidget(self.l1, 0, 0, 1, 2)
            grid.addWidget(self.l2, 0, 2, 1, 2)
            grid.setRowStretch(0, 1)
            self.footer = QLabel("v1.4.1 | L1/L2 only")
            footer_row = QHBoxLayout()
            self.performance_toggle = QCheckBox("性能监控")
            self.performance_toggle.setChecked(runtime.performance_monitor_enabled)
            self.performance_toggle.toggled.connect(runtime.set_performance_monitor_enabled)
            footer_row.addWidget(self.performance_toggle)
            footer_row.addWidget(self.footer, 1)
            grid.addLayout(footer_row, 1, 0, 1, 4)
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
