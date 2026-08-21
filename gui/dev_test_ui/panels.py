from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QSlider,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .srp_panel import track_colour_hex


class UnavailableCanvas(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.title, self.reason = title, "UNAVAILABLE"

    def set_reason(self, reason: str) -> None:
        if self.reason == reason:
            return
        self.reason = reason
        self.setToolTip(reason)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#10161d"))
        painter.setPen(QColor("#dbe7f2"))
        painter.drawText(16, 24, self.title)
        painter.setPen(QColor("#efad4b"))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.reason)


class SrpThresholdControl(QWidget):
    threshold_changed = Signal(float)

    def __init__(self, value: float, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        self.label = QLabel("Candidate threshold")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(5)
        self.slider.setTracking(True)
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(44)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._changed)
        self.set_value(value)

    @property
    def value(self) -> float:
        return self.slider.value() / 100.0

    def set_value(self, value: float) -> None:
        slider_value = round(float(value) * 100)
        if not 0 <= slider_value <= 100:
            raise ValueError("threshold必须位于[0,1]")
        self.slider.setValue(slider_value)
        self.value_label.setText(f"{slider_value / 100:.2f}")

    def _changed(self, slider_value: int) -> None:
        threshold = slider_value / 100.0
        self.value_label.setText(f"{threshold:.2f}")
        self.threshold_changed.emit(threshold)


class MusicOrderLimitControl(QWidget):
    order_changed = Signal(int)

    def __init__(self, value: int, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        self.label = QLabel("MUSIC阶数")
        self.label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.combo = QComboBox()
        self.combo.addItems(("1", "2", "3"))
        self.combo.setFixedWidth(64)
        self.combo.setToolTip("实际MUSIC阶数 = min(MDL诊断阶数, 手动上限)")
        layout.addWidget(self.label)
        layout.addWidget(self.combo)
        self.setMaximumWidth(185)
        self.set_value(value)
        self.combo.currentIndexChanged.connect(
            lambda _index: self.order_changed.emit(self.value)
        )

    @property
    def value(self) -> int:
        return int(self.combo.currentText())

    def set_value(self, value: int) -> None:
        if type(value) is not int or value not in {1, 2, 3}:
            raise ValueError("MUSIC order limit must be 1, 2, or 3")
        self.combo.setCurrentText(str(value))


class _RuntimeSwitchControl(QPushButton):
    enabled_changed = Signal(bool)
    label = "L2 switch"

    def __init__(self, enabled: bool, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setMinimumHeight(34)
        self.toggled.connect(self._changed)
        self.set_enabled(enabled)

    def set_enabled(self, enabled: bool, *, pending: bool = False) -> None:
        self.setChecked(bool(enabled))
        self.setText(self.label)
        color = "#9a6b00" if pending else ("#16794b" if enabled else "#5b6570")
        self.setStyleSheet(f"QPushButton {{ background:{color}; color:white; font-weight:600; }}")

    def _changed(self, enabled: bool) -> None:
        self.set_enabled(bool(enabled), pending=True)
        self.enabled_changed.emit(bool(enabled))


class DirectionKalmanControl(_RuntimeSwitchControl):
    label = "Kalman"


class DirectionIdTrackingControl(_RuntimeSwitchControl):
    label = "ID Tracking"


class MusicDpdRank1Control(_RuntimeSwitchControl):
    label = "DPD"


class MusicNoiseWhiteningControl(_RuntimeSwitchControl):
    label = "Whitening"


class ContinuousTrackGainControl(_RuntimeSwitchControl):
    label = "连续轨响度补偿"


class KalmanNoiseScaleControl(QWidget):
    apply_requested = Signal(float)

    def __init__(self, name: str, value: float, parent: QWidget | None = None):
        super().__init__(parent)
        self.name = name
        self._applied = round(float(value), 2)
        self._staged = self._applied
        self._dirty = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(145)
        self.minus_button = QPushButton("−")
        self.plus_button = QPushButton("+")
        self.apply_button = QPushButton("应用")
        for button in (self.minus_button, self.plus_button, self.apply_button):
            button.setMinimumHeight(30)
        self.minus_button.clicked.connect(lambda: self._adjust(-0.1))
        self.plus_button.clicked.connect(lambda: self._adjust(0.1))
        self.apply_button.clicked.connect(lambda: self.apply_requested.emit(self._staged))
        layout.addWidget(self.value_label, 1)
        layout.addWidget(self.minus_button)
        layout.addWidget(self.plus_button)
        layout.addWidget(self.apply_button)
        self._render(False)

    @property
    def staged_value(self) -> float:
        return self._staged

    def _adjust(self, delta: float) -> None:
        if delta > 0.0 and self._staged < 0.1:
            self._staged = 0.1
        elif delta < 0.0 and self._staged <= 0.1:
            self._staged = 0.02
        else:
            self._staged = round(min(10.0, max(0.02, self._staged + delta)), 2)
        self._dirty = self._staged != self._applied
        self._render(False)

    def commit(self, value: float, *, pending: bool = False) -> None:
        self._applied = self._staged = round(float(value), 2)
        self._dirty = False
        self._render(pending)

    def set_applied_value(self, value: float, *, pending: bool = False) -> None:
        value = round(float(value), 2)
        self._applied = value
        if not self._dirty:
            self._staged = value
        self._render(pending)

    def _render(self, pending: bool) -> None:
        if self._dirty:
            suffix = " · 待应用"
        elif pending:
            suffix = " · 下一窗口"
        else:
            suffix = ""
        self.value_label.setText(f"Kalman {self.name}: {self._staged:.2f}{suffix}")
        self.minus_button.setEnabled(self._staged > 0.02)
        self.plus_button.setEnabled(self._staged < 10.0)
        self.apply_button.setEnabled(self._dirty)


class GateProbabilityThresholdControl(SrpThresholdControl):
    """Independent runtime threshold for the Layer 2 probability Gate."""

    def __init__(self, value: float, parent: QWidget | None = None):
        super().__init__(value, parent)
        self.label.setText("L2 Gate probability")

    def set_value(self, value: float, *, pending: bool = False) -> None:
        super().set_value(value)
        self.label.setText("L2 Gate probability" + (" · next window" if pending else ""))


class ProbabilityGateReadout(QLabel):
    """Display the Gate's 40 ms mean probability as one P value."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setStyleSheet(
            "QLabel { background:#202a34; color:#dce7f2; padding:0 8px; "
            "font-family:Consolas; font-weight:600; }"
        )
        self.set_unavailable("WAITING")

    def set_decision(self, decision) -> None:
        def value(item) -> str:
            return "—" if item is None else f"{item:.2f}"

        self.setText(
            f"P {value(decision.probability_40ms)} >= {decision.threshold:.2f} | "
            f"Gate {decision.state.value.upper()} | rev {decision.config_revision}"
        )
        self.setToolTip(
            f"40 ms mean of two consecutive 20 ms probabilities | "
            f"window={decision.window_id} | backend={decision.backend} | reason={decision.reason}"
        )

    def set_unavailable(self, state: str = "UNAVAILABLE") -> None:
        self.setText(f"P — | Gate {state}")
        self.setToolTip("No window-aligned probability Gate result")


class BeamformPanel(QGroupBox):
    track_play_requested = Signal(int)
    track_stop_requested = Signal()
    mode_change_requested = Signal(str)
    downstream_processing_changed = Signal(bool)
    gain_compensation_changed = Signal(bool)

    def __init__(
        self, config, gain_compensation_enabled: bool = True,
        parent: QWidget | None = None,
    ):
        super().__init__("L3 · Directional Audio Preview", parent)
        layout = QVBoxLayout(self)
        self._preprocessing_version = config.feature.preprocessing_version
        self._minimum_listening_track_seconds = float(
            config.dev_test_ui.minimum_listening_track_seconds
        )
        self._latest_previews = {}
        self._selected_key: tuple[int, float] | None = None
        self._frozen_preview = None
        preview_controls = QHBoxLayout()
        self.preview_summary = QLabel("L3 formal preview: waiting for an open Gate and candidate")
        self.preview_summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.mode_switch = QPushButton("BF：优化算法")
        self._processing_mode = "optimized"
        self.mode_switch.setToolTip(
            "依次切换：优化算法 → DS基线 → 恒定波束宽度30°；只影响后续L3窗口"
        )
        self.mode_switch.clicked.connect(self._cycle_mode)
        self.downstream_switch = QPushButton()
        self.downstream_switch.setCheckable(True)
        self.downstream_switch.setChecked(True)
        self.downstream_switch.setToolTip(
            "关闭后L2继续运行；L3不再接收L2窗口，L4同步停止。跳过结果属于正常状态，不报错。"
        )
        self.downstream_switch.toggled.connect(self._toggle_downstream_processing)
        self.set_downstream_processing_enabled(True)
        self.gain_compensation = ContinuousTrackGainControl(gain_compensation_enabled)
        self.gain_compensation.setToolTip(
            "控制正式连续方向音轨的响度补偿；从下一完整20 ms音频块生效。"
        )
        self.gain_compensation.enabled_changed.connect(
            self.gain_compensation_changed.emit
        )
        preview_controls.addWidget(self.preview_summary, 1)
        preview_controls.addWidget(self.mode_switch)
        preview_controls.addWidget(self.downstream_switch)
        preview_controls.addWidget(self.gain_compensation)
        layout.addLayout(preview_controls)
        self.help = QLabel("仅按L2权威ID缓存方向音频；Kalman只平滑角度，不控制ID存在。")
        layout.addWidget(self.help)
        self.track_scroll = QScrollArea()
        self.track_scroll.setWidgetResizable(True)
        self.track_container = QWidget()
        self.track_layout = QVBoxLayout(self.track_container)
        self.track_layout.setContentsMargins(4, 4, 4, 4)
        self.track_layout.addStretch(1)
        self.track_scroll.setWidget(self.track_container)
        layout.addWidget(self.track_scroll, 1)
        self._track_rows: dict[int, AudioTrackRow] = {}
        self._track_snapshots = {}
        self._track_stream: tuple[str, int] | None = None
        self._playing_track_id: int | None = None

    def _cycle_mode(self) -> None:
        modes = (
            "optimized", "ds_baseline", "loaded_mvdr_baseline", "subband_robust_baseline",
        )
        mode = modes[(modes.index(self._processing_mode) + 1) % len(modes)]
        self.set_processing_mode(mode)
        self.mode_change_requested.emit(mode)

    def set_processing_mode(self, mode: str) -> None:
        if mode not in {
            "optimized", "ds_baseline", "loaded_mvdr_baseline", "subband_robust_baseline",
        }:
            raise ValueError(f"unknown L3 processing mode: {mode}")
        self._processing_mode = mode
        labels = {
            "optimized": "BF：优化算法",
            "ds_baseline": "BF：DS基线",
            "loaded_mvdr_baseline": "BF：Loaded MVDR基线",
            "subband_robust_baseline": "BF：五频段鲁棒对照",
        }
        colors = {
            "optimized": "",
            "ds_baseline": "background:#9a6b00;color:white",
            "loaded_mvdr_baseline": "background:#6f4a8e;color:white",
            "subband_robust_baseline": "background:#285f9a;color:white",
        }
        self.mode_switch.setText(labels[mode])
        self.mode_switch.setStyleSheet(colors[mode])

    def reset_for_mode_change(self, mode: str) -> None:
        self.set_processing_mode(mode)
        self._latest_previews.clear()
        self._selected_key = None
        self._frozen_preview = None
        self.preview_summary.setText("L3 mode changed; waiting for the next completed window")
        self.preview_summary.setToolTip("")
        self._playing_track_id = None
        self.clear_tracks()

    def clear_tracks(self) -> None:
        """Clear rows only at an explicit Test-UI session boundary."""
        for row in self._track_rows.values():
            self.track_layout.removeWidget(row)
            row.deleteLater()
        self._track_rows.clear()
        self._track_snapshots.clear()
        self._track_stream = None
        self._playing_track_id = None

    @staticmethod
    def _preview_key(preview) -> tuple[int, float]:
        return preview.window_id, round(float(preview.theta_deg), 4)

    def set_previews(self, previews, *, missing_reason: str | None = None) -> None:
        previews = tuple(previews)
        self._latest_previews = {self._preview_key(item): item for item in previews}
        chosen = None
        if self._selected_key is not None:
            chosen = self._latest_previews.get(self._selected_key)
            if chosen is not None:
                self._frozen_preview = chosen
            else:
                chosen = self._frozen_preview
        elif previews:
            chosen = previews[0]
            self._frozen_preview = chosen
        if chosen is None:
            self._frozen_preview = None
            self.preview_summary.setText(f"L3 formal preview: {missing_reason or 'NO CANDIDATE'}")
            self.preview_summary.setToolTip(missing_reason or "")
            return
        self._render_preview(chosen, frozen=self._selected_key is not None)

    def select_preview(self, theta_deg: float, window_id: int) -> bool:
        key = (int(window_id), round(float(theta_deg), 4))
        preview = self._latest_previews.get(key)
        if preview is None:
            return False
        self._selected_key = key
        self._frozen_preview = preview
        self._render_preview(preview, frozen=True)
        return True

    def _render_preview(self, preview, *, frozen: bool) -> None:
        fallback = "none" if preview.fallback_reason is None else preview.fallback_reason
        mode = "FROZEN" if frozen else "LATEST"
        self.preview_summary.setText(
            f"L3 {mode} | window {preview.window_id:08d} | θ={preview.theta_deg:.1f}° | "
            f"{preview.runtime_backend} | fallback: {fallback} | {self._preprocessing_version}"
        )
        self.preview_summary.setToolTip(" | ".join(preview.diagnostics))

    def _toggle_downstream_processing(self, enabled: bool) -> None:
        self.set_downstream_processing_enabled(enabled)
        self.downstream_processing_changed.emit(enabled)

    def set_downstream_processing_enabled(self, enabled: bool) -> None:
        self.downstream_switch.setChecked(bool(enabled))
        self.mode_switch.setEnabled(bool(enabled))
        if enabled:
            self.downstream_switch.setText("L3/L4：运行中")
            self.downstream_switch.setStyleSheet("background:#36875f;color:white")
        else:
            self.downstream_switch.setText("L3/L4：已停止")
            self.downstream_switch.setStyleSheet("background:#8a4b3b;color:white")

    def set_gain_compensation_enabled(self, enabled: bool) -> None:
        self.gain_compensation.set_enabled(bool(enabled))

    def set_unavailable(self, reason: str) -> None:
        self.help.setText(reason)

    def set_tracks(self, tracks) -> None:
        all_tracks = tuple(tracks)
        streams = {
            (item.session_id, item.stream_epoch) for item in all_tracks
        }
        if len(streams) > 1:
            raise ValueError("L3 listening rows must belong to one capture stream")
        if streams:
            incoming_stream = next(iter(streams))
            if self._track_stream is not None and incoming_stream != self._track_stream:
                self.clear_tracks()
            self._track_stream = incoming_stream
        incoming_ids = {item.track_id for item in all_tracks}
        # A Center Mic row marks a complete authoritative tracker snapshot.
        # Directional IDs omitted from such a snapshot were explicitly
        # filtered together with their cache files, so their previously drawn
        # waveform rows must not remain as unplayable UI ghosts.  Empty/error
        # projections still retain the last good rows.
        if 0 in incoming_ids:
            removed_ids = set(self._track_rows) - incoming_ids
            if self._playing_track_id in removed_ids:
                self._playing_track_id = None
                self.track_stop_requested.emit()
            for track_id in removed_ids:
                self._track_snapshots.pop(track_id, None)
                row = self._track_rows.pop(track_id)
                self.track_layout.removeWidget(row)
                row.deleteLater()
        tracks = tuple(
            item
            for item in all_tracks
            if item.track_id == 0
            or item.duration_seconds >= self._minimum_listening_track_seconds
        )
        hidden_count = sum(
            item.track_id != 0
            and item.duration_seconds < self._minimum_listening_track_seconds
            for item in all_tracks
        )
        for track in tracks:
            self._track_snapshots[track.track_id] = track
            row = self._track_rows.get(track.track_id)
            if row is None:
                row = AudioTrackRow(track.track_id)
                row.toggle_requested.connect(self._toggle_track)
                self._track_rows[track.track_id] = row
                self.track_layout.insertWidget(self.track_layout.count() - 1, row)
            row.set_snapshot(track, playing=track.track_id == self._playing_track_id)
        ordered_ids = sorted(
            self._track_rows,
            key=lambda track_id: (
                track_id != 0,
                -self._track_snapshots[track_id].duration_seconds,
                track_id,
            ),
        )
        for index, track_id in enumerate(ordered_ids):
            row = self._track_rows[track_id]
            self.track_layout.removeWidget(row)
            self.track_layout.insertWidget(index, row)
        if tracks or self._track_rows:
            self.help.setText(
                "首行为Center Mic原始输入对照；其余仅显示L2 confirmed/coasting权威ID，并按缓存时长从长到短排列。"
                "权威ID音频：ACTIVE实时追加，"
                "COASTING等待恢复，ENDED停止追加；"
                f"仅显示≥{self._minimum_listening_track_seconds:.1f}s音频，"
                "已显示ID保留至新会话或关闭窗口。"
            )
        elif hidden_count:
            self.help.setText(
                f"正在缓存 {hidden_count} 个试听ID；累计达到"
                f"{self._minimum_listening_track_seconds:.1f}s后显示。"
            )
        else:
            self.help.setText("等待L2 confirmed权威ID；UI不会按角度创建、合并或修补ID。")

    def _toggle_track(self, track_id: int) -> None:
        if self._playing_track_id == track_id:
            self._playing_track_id = None
            self.track_stop_requested.emit()
        else:
            self._playing_track_id = track_id
            self.track_play_requested.emit(track_id)
        for item_id, row in self._track_rows.items():
            row.set_playing(item_id == self._playing_track_id)

    def sync_track_playback_stopped(self) -> None:
        """Apply player EOF/device failure without guessing from button state."""
        if self._playing_track_id is None:
            return
        self._playing_track_id = None
        for row in self._track_rows.values():
            row.set_playing(False)

    def set_track_playback_progress(self, track_id: int, progress: float) -> None:
        """Draw the player's real sample position only on its loaded track."""
        for item_id, row in self._track_rows.items():
            row.set_playback_progress(progress if item_id == int(track_id) else None)

    def clear_track_playback_progress(self) -> None:
        for row in self._track_rows.values():
            row.set_playback_progress(None)


class AudioWaveformThumbnail(QWidget):
    """Compact fixed-scale dBFS envelope; no audio-file rescans on UI refresh."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._envelope: tuple[float, ...] = ()
        self._playback_progress: float | None = None
        self.setMinimumWidth(180)
        self.setFixedHeight(42)

    def set_envelope(self, envelope) -> None:
        value = tuple(float(item) for item in envelope)
        if value != self._envelope:
            self._envelope = value
            self.update()

    def set_playback_progress(self, progress: float | None) -> None:
        value = None if progress is None else float(np.clip(progress, 0.0, 1.0))
        if value != self._playback_progress:
            self._playback_progress = value
            self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#10161d"))
        middle = self.height() // 2
        painter.setPen(QPen(QColor("#526170"), 1))
        painter.drawLine(0, middle, self.width(), middle)
        if self._envelope and self.width() > 0:
            values = np.asarray(self._envelope, dtype=np.float32)
            painter.setPen(QPen(QColor("#39b6d4"), 1))
            half_height = max(1, middle - 3)
            for x in range(self.width()):
                start = x * len(values) // self.width()
                stop = max(start + 1, (x + 1) * len(values) // self.width())
                peak = float(np.max(values[start:stop], initial=0.0))
                dbfs = max(-60.0, 20.0 * np.log10(max(peak, 1.0e-6)))
                height = round(half_height * (dbfs + 60.0) / 60.0)
                painter.drawLine(x, middle - height, x, middle + height)
        if self._playback_progress is not None and self.width() > 0:
            x = round(self._playback_progress * max(0, self.width() - 1))
            painter.setPen(QPen(QColor("#ffb000"), 2))
            painter.drawLine(x, 0, x, self.height() - 1)


class AudioTrackRow(QWidget):
    toggle_requested = Signal(int)

    def __init__(self, track_id: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.track_id = int(track_id)
        self._rendered_text = ""
        self._rendered_state: str | None = None
        self._rendered_playing: bool | None = None
        self.setFixedHeight(58)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        self.play = QPushButton("▶")
        self.play.setFixedWidth(44)
        self.play.clicked.connect(lambda: self.toggle_requested.emit(self.track_id))
        self.label = QLabel()
        self.label.setStyleSheet("font-family:Consolas")
        self.label.setFixedWidth(190)
        self.duration = QLabel()
        self.duration.setStyleSheet("font-family:Consolas")
        self.duration.setFixedWidth(90)
        self.waveform = AudioWaveformThumbnail()
        layout.addWidget(self.play)
        layout.addWidget(self.label)
        layout.addWidget(self.duration)
        layout.addWidget(self.waveform, 1)

    def set_snapshot(self, snapshot, *, playing: bool) -> None:
        if snapshot.track_id == 0:
            text = "Center Mic 对照"
            label_style = "font-family:Consolas"
        else:
            text = f"{snapshot.track_id}  {snapshot.theta_deg:.1f}°"
            label_style = f"font-family:Consolas;color:{track_colour_hex(snapshot.track_id)}"
        if text != self._rendered_text:
            self.label.setText(text)
            self._rendered_text = text
        if self.label.styleSheet() != label_style:
            self.label.setStyleSheet(label_style)
        self.duration.setText(f"{snapshot.duration_seconds:5.1f} s")
        self.waveform.set_envelope(snapshot.waveform_envelope)
        self.set_playing(playing)

    def set_playing(self, playing: bool) -> None:
        playing = bool(playing)
        if playing != self._rendered_playing:
            self.play.setText("Ⅱ" if playing else "▶")
            self._rendered_playing = playing

    def set_playback_progress(self, progress: float | None) -> None:
        self.waveform.set_playback_progress(progress)


class VoiceProbabilityPolar(QWidget):
    """L4 candidate probabilities on the same 0°/90°/180°/270° geometry as L2."""

    candidate_selected = Signal(float, int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._detections = ()
        self._threshold = 0.7
        self._window_id = -1
        self._reason = "UNAVAILABLE"
        self.setMinimumSize(260, 220)

    def set_data(self, detections, threshold: float) -> None:
        self._detections = tuple(detections)
        self._threshold = float(threshold)
        self._window_id = -1 if not self._detections else self._detections[0].window_id
        self._reason = ""
        self.update()

    def set_unavailable(self, reason: str) -> None:
        self._detections = ()
        self._window_id = -1
        self._reason = "" if (reason or "").strip().upper() == "NO CANDIDATE" else (reason or "UNAVAILABLE")
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#10161d"))
        cx, cy = self.width() / 2.0, self.height() / 2.0
        # Leave enough room for the four angle labels while making the polar
        # scale slightly larger than before.
        label_margin = 34.0
        available_radius = max(20.0, min(cx, cy) - label_margin)
        radius = min(min(self.width(), self.height()) * 0.40, available_radius)
        ring_color = QColor("#53677a")
        ring_width = 2.5
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(ring_color, ring_width))
        painter.drawEllipse(round(cx - radius), round(cy - radius), round(radius * 2), round(radius * 2))

        # The adjustable dashed circle is the UI probability threshold:
        # threshold 0 is at the center and threshold 1 reaches the outer ring.
        threshold_radius = radius * min(1.0, max(0.0, self._threshold))
        if threshold_radius > 0.0:
            threshold_pen = QPen(ring_color, ring_width)
            threshold_pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(threshold_pen)
            painter.drawEllipse(
                round(cx - threshold_radius), round(cy - threshold_radius),
                round(threshold_radius * 2), round(threshold_radius * 2),
            )
        painter.setPen(QColor("#6f859b"))
        for angle, label in ((0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")):
            rad = np.deg2rad(angle)
            x, y = cx + np.cos(rad) * (radius + 22), cy - np.sin(rad) * (radius + 22)
            painter.drawText(round(x - 18), round(y - 9), 36, 18, Qt.AlignmentFlag.AlignCenter, label)
        if not self._detections:
            painter.setPen(QColor("#efad4b"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._reason)
            return
        for item in self._detections:
            rad = np.deg2rad(item.theta_deg)
            # Probability 0 stays near the center; 1 approaches the outer ring.
            point_radius = radius * (0.08 + 0.86 * float(item.probability))
            x, y = cx + np.cos(rad) * point_radius, cy - np.sin(rad) * point_radius
            color = QColor("#e5484d" if item.probability >= self._threshold else "#8796a5")
            painter.setBrush(color)
            painter.setPen(QPen(color, 1))
            painter.drawEllipse(round(x - 7), round(y - 7), 14, 14)


class CnnPanel(QGroupBox):
    selection_requested = Signal(float, int)

    def __init__(self, configured_threshold: float, parent: QWidget | None = None):
        super().__init__("L4 · CNN Voice Direction", parent)
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.summary = QLabel(f"Voice directions: — | Runtime threshold: {configured_threshold:.2f}")
        self.threshold = QSlider(Qt.Orientation.Horizontal)
        self.threshold.setRange(0, 100)
        self.threshold.setValue(round(configured_threshold * 100))
        self.threshold.setEnabled(True)
        self.threshold_value = QLabel(f"UI threshold: {configured_threshold:.2f}")
        self.threshold.valueChanged.connect(self._threshold_changed)
        top.addWidget(self.summary)
        top.addWidget(self.threshold, 1)
        top.addWidget(self.threshold_value)
        layout.addLayout(top)
        self.polar = VoiceProbabilityPolar()
        layout.addWidget(self.polar, 1)
        self._result = None

    def _threshold_changed(self, value: int) -> None:
        self.threshold_value.setText(f"UI threshold: {value / 100:.2f}")
        self._render()

    def set_result(self, result) -> None:
        self._result = result
        self._render()

    def _render(self) -> None:
        if self._result is None:
            return
        threshold = self.threshold.value() / 100.0
        detections = self._result.detections
        voice_count = 0
        for item in detections:
            is_voice = item.probability >= threshold
            voice_count += int(is_voice)
        self.polar.set_data(detections, threshold)
        self.summary.setText(
            f"Voice directions: {voice_count} | {self._result.primary_model_id} | "
            f"Runtime threshold: {self._result.threshold:.2f}"
        )

    def set_selection(self, theta: float, window_id: int) -> None:
        self.polar.setToolTip(f"Selected {theta:.0f}° · window {window_id}")

    def set_unavailable(self, reason: str) -> None:
        self._result = None
        self.polar.set_unavailable(reason)
        self.summary.setText(
            "L4 unavailable"
            if (reason or "").strip().upper() == "NO CANDIDATE"
            else f"L4 unavailable: {reason}"
        )
