from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from common.config import load_config

from .backend import CENTER_CHANNEL_INDEX, CHANNEL_NAMES, L1SpectrumFrame
from .host import L1SpectrumHost


class SpectrumPlot(QWidget):
    """Small QPainter spectrum plot suitable for a real 50 Hz UI refresh."""

    def __init__(self, *, bars: bool, empty_text: str) -> None:
        super().__init__()
        self._bars = bars
        self._empty_text = empty_text
        self._frequencies: np.ndarray | None = None
        self._levels: np.ndarray | None = None
        self._caption = ""
        self.setMinimumSize(280, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_spectrum(self, frequencies: np.ndarray, levels: np.ndarray, caption: str = "") -> None:
        self._frequencies = np.asarray(frequencies, dtype=np.float32).copy()
        self._levels = np.asarray(levels, dtype=np.float32).copy()
        self._caption = caption
        self.update()

    def clear(self, text: str | None = None) -> None:
        self._frequencies = self._levels = None
        if text is not None:
            self._empty_text = text
        self._caption = ""
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, not self._bars)
        painter.fillRect(self.rect(), QColor("#101821"))
        plot = QRectF(58, 24, max(1, self.width() - 78), max(1, self.height() - 62))
        painter.setPen(QPen(QColor("#3d5064"), 1))
        for db in (-120, -90, -60, -30, 0):
            y = plot.bottom() - (db + 120.0) / 120.0 * plot.height()
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor("#8da2b8"))
            painter.drawText(QRectF(4, y - 9, 48, 18), Qt.AlignmentFlag.AlignRight, str(db))
            painter.setPen(QPen(QColor("#3d5064"), 1))
        for hz in (0, 2_000, 4_000, 6_000, 8_000, 10_000):
            x = plot.left() + hz / 10_000.0 * plot.width()
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.setPen(QColor("#8da2b8"))
            painter.drawText(QRectF(x - 28, plot.bottom() + 5, 56, 20), Qt.AlignmentFlag.AlignCenter, f"{hz // 1000}k")
            painter.setPen(QPen(QColor("#3d5064"), 1))
        painter.setPen(QColor("#8da2b8"))
        painter.drawText(QRectF(4, 1, self.width() - 8, 20), Qt.AlignmentFlag.AlignLeft, "响度 (dBFS)")
        if self._caption:
            painter.drawText(QRectF(4, 1, self.width() - 8, 20), Qt.AlignmentFlag.AlignRight, self._caption)

        if self._frequencies is None or self._levels is None:
            painter.setPen(QColor("#fbbf24"))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, self._empty_text)
            return
        mask = (self._frequencies >= 0.0) & (self._frequencies <= 10_000.0)
        frequencies = self._frequencies[mask]
        levels = np.clip(self._levels[mask], -120.0, 0.0)
        if not frequencies.size:
            return
        x = plot.left() + frequencies.astype(np.float64) / 10_000.0 * plot.width()
        y = plot.bottom() - (levels.astype(np.float64) + 120.0) / 120.0 * plot.height()
        painter.setClipRect(plot)
        if self._bars:
            width = max(1.0, plot.width() / max(1, frequencies.size))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#2fb6d0"))
            for x_value, y_value in zip(x, y, strict=True):
                painter.drawRect(QRectF(x_value - width * 0.45, y_value, width * 0.9, plot.bottom() - y_value))
        else:
            path = QPainterPath(QPointF(float(x[0]), float(y[0])))
            for x_value, y_value in zip(x[1:], y[1:], strict=True):
                path.lineTo(float(x_value), float(y_value))
            painter.setPen(QPen(QColor("#47c7ff"), 2))
            painter.drawPath(path)


class L1SpectrumWindow(QMainWindow):
    def __init__(self, config, *, host: L1SpectrumHost | None = None, auto_start: bool = True) -> None:
        super().__init__()
        self.config = config
        self.host = host or L1SpectrumHost(config)
        self._frame: L1SpectrumFrame | None = None
        self._selected_channel = CENTER_CHANNEL_INDEX
        self.setWindowTitle("6+1 Microphone Array — L1 Spectrum UI")
        self.setMinimumSize(1200, 720)
        self._build_ui()
        self.host.frame_ready.connect(self._on_frame)
        self.host.state_changed.connect(self.status_bar.setText)
        self.host.light_state_changed.connect(self._on_light_state)
        self.host.error.connect(self._on_error)
        if auto_start:
            QTimer.singleShot(0, self.host.start)

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.status_bar = QLabel("STARTING | L1 microphone + IMCRA only")
        self.status_bar.setFixedHeight(30)
        self.status_bar.setStyleSheet("background:#17212b;color:#dce7f2;padding-left:10px;font-family:Consolas")
        outer.addWidget(self.status_bar)
        grid = QGridLayout()
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setSpacing(2)
        for index in (0, 1):
            grid.setRowStretch(index, 1)
            grid.setColumnStretch(index, 1)
        grid.addWidget(self._l1_panel(), 0, 0)
        grid.addWidget(self._current_spectrum_panel(), 0, 1)
        grid.addWidget(self._imcra_panel(), 1, 0)
        grid.addWidget(self._snapshot_panel(), 1, 1)
        outer.addLayout(grid, 1)
        self.setCentralWidget(root)
        self.setStyleSheet(
            "QMainWindow,QWidget{background:#eef2f6;color:#0f172a;font-size:14px;}"
            "QGroupBox{font-size:16px;font-weight:700;border:1px solid #cbd5e1;margin-top:8px;padding-top:8px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 4px;}"
            "QPushButton{background:#f8fafc;border:1px solid #9aa8b7;border-radius:3px;padding:6px 12px;}"
            "QPushButton:hover{background:#dbeafe;} QPushButton:checked{background:#149db5;color:white;font-weight:700;}"
            "QProgressBar{border:1px solid #9aa8b7;background:#f8fafc;}"
            "QProgressBar::chunk{background:#2c93a6;}"
        )

    def _l1_panel(self) -> QGroupBox:
        box = QGroupBox("L1 · Input / IMCRA")
        layout = QVBoxLayout(box)
        self.l1_header = QLabel("等待首个20 ms音频块")
        self.l1_header.setStyleSheet("font-family:Consolas")
        layout.addWidget(self.l1_header)
        controls = QHBoxLayout()
        start = QPushButton("连接麦克风")
        stop = QPushButton("停止采集")
        self.pre_denoise = QPushButton("IMCRA预降噪")
        self.pre_denoise.setCheckable(True)
        self.pre_denoise.setChecked(self.host.pre_denoise_enabled)
        start.clicked.connect(self.host.start)
        stop.clicked.connect(lambda: self.host.stop())
        self.pre_denoise.toggled.connect(self.host.set_pre_denoise_enabled)
        self.light_on = QPushButton("灯光开")
        self.light_off = QPushButton("灯光关")
        self.light_status = QLabel(f"灯光: {self.host.light_state.upper()}")
        self.light_on.clicked.connect(lambda: self.host.set_light(True))
        self.light_off.clicked.connect(lambda: self.host.set_light(False))
        controls.addWidget(start)
        controls.addWidget(stop)
        controls.addWidget(self.pre_denoise)
        controls.addWidget(self.light_on)
        controls.addWidget(self.light_off)
        controls.addWidget(self.light_status)
        controls.addStretch()
        layout.addLayout(controls)
        selectors = QHBoxLayout()
        selectors.addWidget(QLabel("观察通道"))
        self.channel_group = QButtonGroup(self)
        self.channel_group.setExclusive(True)
        self.channel_buttons: list[QPushButton] = []
        for index, name in enumerate(CHANNEL_NAMES):
            button = QPushButton(name)
            button.setCheckable(True)
            button.setChecked(index == CENTER_CHANNEL_INDEX)
            self.channel_group.addButton(button, index)
            self.channel_buttons.append(button)
            selectors.addWidget(button)
        self.channel_group.idClicked.connect(self._select_channel)
        selectors.addStretch()
        layout.addLayout(selectors)
        self.imcra_summary = QLabel("IMCRA: WAITING")
        self.imcra_summary.setStyleSheet("font-family:Consolas")
        layout.addWidget(self.imcra_summary)
        meters = QHBoxLayout()
        self.meter_bars: list[QProgressBar] = []
        self.meter_labels: list[QLabel] = []
        for name in CHANNEL_NAMES:
            column = QVBoxLayout()
            bar = QProgressBar()
            bar.setOrientation(Qt.Orientation.Vertical)
            bar.setRange(-90, 0)
            bar.setValue(-90)
            bar.setTextVisible(False)
            label = QLabel(f"{name}\n-120.0 dB")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(bar, 1)
            column.addWidget(label)
            meters.addLayout(column, 1)
            self.meter_bars.append(bar)
            self.meter_labels.append(label)
        layout.addLayout(meters, 1)
        return box

    def _current_spectrum_panel(self) -> QGroupBox:
        box = QGroupBox("当前20 ms输入频谱 · 0–10 kHz")
        layout = QVBoxLayout(box)
        row = QHBoxLayout()
        self.current_info = QLabel("Center | 等待数据")
        capture = QPushButton("抓拍到右下")
        capture.clicked.connect(self._capture_snapshot)
        row.addWidget(self.current_info)
        row.addStretch()
        row.addWidget(capture)
        layout.addLayout(row)
        self.current_plot = SpectrumPlot(bars=True, empty_text="等待L1音频")
        layout.addWidget(self.current_plot, 1)
        return box

    def _imcra_panel(self) -> QGroupBox:
        box = QGroupBox("IMCRA当前噪声估计")
        layout = QVBoxLayout(box)
        self.noise_info = QLabel("Center | state WAITING | noise — | signal — | SNR — | SPP —")
        self.noise_info.setStyleSheet("font-family:Consolas")
        layout.addWidget(self.noise_info)
        self.noise_plot = SpectrumPlot(bars=False, empty_text="等待IMCRA噪声估计")
        layout.addWidget(self.noise_plot, 1)
        return box

    def _snapshot_panel(self) -> QGroupBox:
        box = QGroupBox("频谱抓拍")
        layout = QVBoxLayout(box)
        self.snapshot_info = QLabel("尚未抓拍")
        self.snapshot_info.setStyleSheet("font-family:Consolas")
        layout.addWidget(self.snapshot_info)
        self.snapshot_plot = SpectrumPlot(bars=True, empty_text="点击右上“抓拍到右下”")
        layout.addWidget(self.snapshot_plot, 1)
        return box

    def _select_channel(self, channel: int) -> None:
        self._selected_channel = channel
        self._render_selected()

    def _on_frame(self, frame: L1SpectrumFrame) -> None:
        self._frame = frame
        meter = frame.meter
        self.l1_header.setText(
            f"RUNNING | session {frame.session_id[:8]} | epoch {frame.stream_epoch} | "
            f"sample {frame.end_sample:010d} | seq {frame.sequence_id:08d}"
        )
        for index, name in enumerate(CHANNEL_NAMES):
            value = float(meter.rms_dbfs[index])
            self.meter_bars[index].setValue(max(-90, min(0, round(value))))
            self.meter_labels[index].setText(f"{name}\n{value:.1f} dB")
        hop = meter.imcra_hop
        state = hop.state.upper() if hop is not None else "WAITING"
        self.imcra_summary.setText(f"IMCRA: {state} | 每20 ms更新 | 物理7路独立噪声PSD")
        self._render_selected()

    def _render_selected(self) -> None:
        frame = self._frame
        name = CHANNEL_NAMES[self._selected_channel]
        if frame is None:
            self.current_info.setText(f"{name} | 等待数据")
            return
        levels = frame.channel_levels_dbfs[self._selected_channel]
        peak_index = int(np.argmax(levels))
        self.current_info.setText(
            f"{name} | RMS {frame.meter.rms_dbfs[self._selected_channel]:.1f} dBFS | "
            f"峰值频率 {frame.frequencies_hz[peak_index]:.0f} Hz | 20 ms / 2048 FFT"
        )
        self.current_plot.set_spectrum(frame.frequencies_hz, levels, name)
        hop = frame.meter.imcra_hop
        if self._selected_channel >= 7 or hop is None or frame.noise_levels_dbfs is None:
            reason = "Mix是硬件混音，IMCRA没有独立噪声PSD" if self._selected_channel >= 7 else "等待IMCRA"
            self.noise_info.setText(f"{name} | {reason}")
            self.noise_plot.clear(reason)
            return
        features = hop.noise_features[self._selected_channel]
        self.noise_info.setText(
            f"{name} | state {hop.state.upper()} | noise {features[0]:.1f} dB | "
            f"signal {features[1]:.1f} dB | SNR {features[2]:.1f} dB | SPP {features[3]:.3f}"
        )
        self.noise_plot.set_spectrum(
            frame.noise_frequencies_hz,
            frame.noise_levels_dbfs[self._selected_channel],
            f"{name} · {hop.state.upper()}",
        )

    def _capture_snapshot(self) -> None:
        if self._frame is None:
            return
        frame = self._frame
        name = CHANNEL_NAMES[self._selected_channel]
        self.snapshot_info.setText(
            f"{name} | session {frame.session_id[:8]} | epoch {frame.stream_epoch} | "
            f"sample {frame.end_sample:010d} | seq {frame.sequence_id:08d}"
        )
        self.snapshot_plot.set_spectrum(
            frame.frequencies_hz,
            frame.channel_levels_dbfs[self._selected_channel],
            name,
        )

    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "L1 Spectrum UI", message)

    def _on_light_state(self, state: str) -> None:
        self.light_status.setText(f"灯光: {state.upper()}")
        pending = state == "pending"
        self.light_on.setEnabled(not pending)
        self.light_off.setEnabled(not pending)

    def closeEvent(self, event) -> None:
        if self.host.stop(timeout=5.0):
            close = getattr(self.host, "close", None)
            if callable(close):
                close()
            event.accept()
        else:
            event.ignore()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone L1 microphone spectrum UI")
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--config", type=Path, default=project_root / "config" / "config.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    application = QApplication.instance() or QApplication(sys.argv[:1])
    window = L1SpectrumWindow(load_config(args.config))
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
