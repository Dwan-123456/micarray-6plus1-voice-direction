from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from .adapter import PublicApiAdapter
from .controller import LogUiController, page
from .models import SessionReadModel, StageState, WindowKey
from .statistics import StatisticsEngine

try:
    from PySide6.QtCore import QObject, QPointF, Qt, QUrl, Signal
    from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QComboBox,
        QSplitter,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # pragma: no cover - importable headless core
    QApplication = None
    QMainWindow = object


_STAGE_COLOURS = {
    StageState.COMPLETED: "#218c74",
    StageState.SKIPPED: "#6c757d",
    StageState.DROPPED: "#d35400",
    StageState.TIMED_OUT: "#8e44ad",
    StageState.FAILED: "#c0392b",
    StageState.CANCELLED: "#7f8c8d",
    StageState.UNKNOWN: "#34495e",
}


if QApplication is not None:

    class _LoadBridge(QObject):
        completed = Signal(object, object)


    class MusicSpectrum(QWidget):
        def __init__(self, parent: QWidget | None = None):
            super().__init__(parent)
            self._scores: tuple[float, ...] | None = None
            self.setMinimumHeight(220)

        def set_scores(self, scores: tuple[float, ...] | None) -> None:
            self._scores = scores
            self.update()

        def paintEvent(self, _event) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("#111820"))
            painter.setPen(QColor("#dce7f2"))
            if self._scores is None:
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "MUSIC 360°：N/A / 未记录")
                return
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            radius = max(20.0, min(self.width(), self.height()) / 2.0 - 28.0)
            center = QPointF(self.width() / 2.0, self.height() / 2.0)
            painter.setPen(QPen(QColor("#536273"), 1.4))
            painter.drawEllipse(center, radius, radius)
            maximum = max(max(self._scores), 1.0e-12)
            points = []
            for theta, score in enumerate(self._scores):
                angle = theta * 3.141592653589793 / 180.0
                length = radius * (0.05 + 0.95 * max(0.0, score) / maximum)
                points.append(QPointF(center.x() + length * __import__("math").cos(angle), center.y() - length * __import__("math").sin(angle)))
            points.append(points[0])
            painter.setPen(QPen(QColor("#42b8ff"), 2.0))
            painter.drawPolyline(QPolygonF(points))


    class RecordsPage(QWidget):
        selected = Signal(str)

        def __init__(self, parent: QWidget | None = None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            self.notice = QLabel("只读离线回看；Log UI 不启动或控制 Runtime，也不直接打开 Catalog。")
            self.notice.setWordWrap(True)
            layout.addWidget(self.notice)
            self.table = QTableWidget(0, 13)
            self.table.setHorizontalHeaderLabels((
                "session", "状态", "开始", "结束", "时长", "项目版本", "算法版本", "schema", "模式",
                "配置hash", "校准hash", "完整性", "capability",
            ))
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.cellDoubleClicked.connect(lambda row, _column: self.selected.emit(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)))
            layout.addWidget(self.table)

        def set_sessions(self, sessions) -> None:
            self.table.setRowCount(len(sessions))
            for row, session in enumerate(sessions):
                values = (
                    session.session_id[:12], session.status, session.started_at or "N/A", session.ended_at or "N/A",
                    _number(session.duration_seconds, "{:.3f}s"), session.project_version or "N/A",
                    session.algorithm_version or "N/A", session.schema_version, session.mode or "N/A",
                    _short(session.config_hash), _short(session.calibration_hash), session.data_integrity or "N/A",
                    ", ".join(session.capabilities.labels()) or "N/A",
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if column == 0:
                        item.setData(Qt.ItemDataRole.UserRole, session.session_id)
                    if session.warning:
                        item.setToolTip(session.warning)
                        item.setBackground(QColor("#fff3cd"))
                    self.table.setItem(row, column, item)


    class OverviewPage(QWidget):
        def __init__(self, parent: QWidget | None = None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            self.header = QLabel("尚未选择 session")
            self.header.setWordWrap(True)
            layout.addWidget(self.header)
            self.table = QTableWidget(0, 13)
            self.table.setHorizontalHeaderLabels((
                "阶段", "适用n", "缺失", "完成", "跳过", "丢弃", "超时", "失败", "取消",
                "实际完成Hz", "compute p50/p95/p99", "wait p50/p95/p99", "age p50/p95/p99",
            ))
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            layout.addWidget(self.table)

        def set_session(self, session: SessionReadModel) -> None:
            stats = StatisticsEngine().calculate(session)
            descriptor = session.descriptor
            warning = f" | {descriptor.warning}" if descriptor.warning else ""
            self.header.setText(
                f"session {descriptor.session_id} | windows {stats.total_windows} | epochs {stats.epochs} | "
                f"direction IDs {_na(stats.track_count)} | anomalies {stats.anomaly_count} | {stats.availability.value}{warning}"
            )
            self.table.setRowCount(len(stats.stage))
            for row, (name, item) in enumerate(stats.stage.items()):
                values = (
                    name.upper(), str(item.applicable), str(item.missing),
                    str(item.counts.get(StageState.COMPLETED, 0)), str(item.counts.get(StageState.SKIPPED, 0)),
                    str(item.counts.get(StageState.DROPPED, 0)), str(item.counts.get(StageState.TIMED_OUT, 0)),
                    str(item.counts.get(StageState.FAILED, 0)), str(item.counts.get(StageState.CANCELLED, 0)),
                    _number(item.completed_hz, "{:.2f}"), _percentiles(item.compute),
                    _percentiles(item.queue_wait), _percentiles(item.end_to_end),
                )
                for column, value in enumerate(values):
                    self.table.setItem(row, column, QTableWidgetItem(value))


    class TimelinePage(QWidget):
        window_selected = Signal(object)
        PAGE_SIZE = 500

        def __init__(self, parent: QWidget | None = None):
            super().__init__(parent)
            self._session: SessionReadModel | None = None
            self._offset = 0
            self._visible_keys: list[WindowKey | None] = []
            layout = QVBoxLayout(self)
            controls = QHBoxLayout()
            self.previous = QPushButton("上一页")
            self.next = QPushButton("下一页")
            self.position = QLabel("0 / 0")
            controls.addWidget(self.previous)
            controls.addWidget(self.next)
            controls.addWidget(self.position)
            controls.addStretch(1)
            layout.addLayout(controls)
            self.table = QTableWidget(0, 10)
            self.table.setHorizontalHeaderLabels(("epoch", "window", "sample", "L1", "Gate", "L2", "L3", "L5", "commit", "reason"))
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.cellDoubleClicked.connect(self._select)
            layout.addWidget(self.table)
            self.previous.clicked.connect(lambda: self._move(-self.PAGE_SIZE))
            self.next.clicked.connect(lambda: self._move(self.PAGE_SIZE))

        def set_session(self, session: SessionReadModel) -> None:
            self._session, self._offset = session, 0
            self._render()

        def _move(self, amount: int) -> None:
            if self._session is None:
                return
            self._offset = max(0, min(max(0, len(self._session.windows) - 1), self._offset + amount))
            self._render()

        def _render(self) -> None:
            windows = () if self._session is None else self._session.windows
            current = page(windows, self._offset, self.PAGE_SIZE)
            self.position.setText(f"{current.offset + (1 if current.items else 0)}–{current.offset + len(current.items)} / {current.total}")
            self.previous.setEnabled(current.offset > 0)
            self.next.setEnabled(current.offset + len(current.items) < current.total)
            display: list[object] = []
            previous_epoch = None
            for window in current.items:
                if previous_epoch != window.key.stream_epoch:
                    display.append(("epoch", window.key.stream_epoch))
                    previous_epoch = window.key.stream_epoch
                display.append(window)
            self._visible_keys = []
            self.table.setRowCount(len(display))
            for row, value in enumerate(display):
                if isinstance(value, tuple):
                    epoch = value[1]
                    item = QTableWidgetItem(f"EPOCH {epoch} — 时间线在此断开")
                    item.setBackground(QColor("#1f618d"))
                    item.setForeground(QColor("white"))
                    self.table.setItem(row, 0, item)
                    self.table.setSpan(row, 0, 1, self.table.columnCount())
                    self._visible_keys.append(None)
                    continue
                window = value
                self._visible_keys.append(window.key)
                values = [str(window.key.stream_epoch), str(window.key.window_id), str(window.key.decision_sample)]
                values.extend(window.stages[name].state.value.upper() for name in ("l1", "gate", "l2", "l3", "l5", "commit"))
                values.append(window.terminal_reason or "")
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setData(Qt.ItemDataRole.UserRole, window.key)
                    if 3 <= column <= 8:
                        item.setBackground(QColor(_STAGE_COLOURS[window.stages[("l1", "gate", "l2", "l3", "l5", "commit")[column - 3]].state]))
                        item.setForeground(QColor("white"))
                    self.table.setItem(row, column, item)

        def _select(self, row: int, _column: int) -> None:
            if 0 <= row < len(self._visible_keys) and self._visible_keys[row] is not None:
                self.window_selected.emit(self._visible_keys[row])


    class WindowDetailPage(QWidget):
        def __init__(self, parent: QWidget | None = None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            self.header = QLabel("从 Pipeline 时间线选择窗口")
            self.header.setWordWrap(True)
            layout.addWidget(self.header)
            splitter = QSplitter(Qt.Orientation.Vertical)
            self.spectrum = MusicSpectrum()
            self.public_json = QTextEdit()
            self.public_json.setReadOnly(True)
            splitter.addWidget(self.spectrum)
            splitter.addWidget(self.public_json)
            layout.addWidget(splitter)

        def set_window(self, session: SessionReadModel, key: WindowKey) -> None:
            window = session.window(key)
            if window is None:
                return
            statuses = " | ".join(f"{name.upper()}={item.state.value}" for name, item in window.stages.items())
            self.header.setText(
                f"WindowKey=({key.session_id}, {key.stream_epoch}, {key.window_id}, {key.decision_sample})\n{statuses}"
            )
            self.spectrum.set_scores(window.normalized_scores)
            self.public_json.setPlainText(json.dumps(window.raw_public, ensure_ascii=False, indent=2, default=str))


    class TracksAnomaliesPage(QWidget):
        window_selected = Signal(object)

        def __init__(self, adapter: PublicApiAdapter, parent: QWidget | None = None):
            super().__init__(parent)
            self.adapter = adapter
            self._session: SessionReadModel | None = None
            self._track_keys: list[tuple[str, int, int]] = []
            self._all_anomalies = ()
            self._visible_anomalies = []
            self.audio_output = QAudioOutput(self)
            self.player = QMediaPlayer(self)
            self.player.setAudioOutput(self.audio_output)
            layout = QVBoxLayout(self)
            self.notice = QLabel("track_id 是方向轨 ID，不表示人物身份。音频仅在点击播放后经公开校验接口按需请求。")
            self.notice.setWordWrap(True)
            layout.addWidget(self.notice)
            splitter = QSplitter(Qt.Orientation.Vertical)
            tracks_widget = QWidget()
            tracks_layout = QVBoxLayout(tracks_widget)
            self.track_table = QTableWidget(0, 11)
            self.track_table.setHorizontalHeaderLabels((
                "epoch", "track", "首sample", "末sample", "寿命ms", "首角", "末角", "连续展开", "状态", "L5最新", "资产",
            ))
            self.track_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            tracks_layout.addWidget(self.track_table)
            self.play = QPushButton("播放所选方向的首个已校验增强资产")
            self.play.clicked.connect(self._play_selected)
            tracks_layout.addWidget(self.play)
            anomaly_controls = QHBoxLayout()
            anomaly_controls.addWidget(QLabel("异常筛选："))
            self.anomaly_filter = QComboBox()
            self.anomaly_filter.currentTextChanged.connect(self._render_anomalies)
            anomaly_controls.addWidget(self.anomaly_filter)
            anomaly_controls.addStretch(1)
            tracks_layout.addLayout(anomaly_controls)
            self.anomaly_table = QTableWidget(0, 5)
            self.anomaly_table.setHorizontalHeaderLabels(("类别", "epoch", "window", "sample", "说明"))
            self.anomaly_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            self.anomaly_table.horizontalHeader().setStretchLastSection(True)
            self.anomaly_table.cellDoubleClicked.connect(self._select_anomaly)
            splitter.addWidget(tracks_widget)
            splitter.addWidget(self.anomaly_table)
            layout.addWidget(splitter)

        def set_session(self, session: SessionReadModel) -> None:
            self._session = session
            grouped = defaultdict(list)
            for item in session.tracks:
                grouped[(item.session_id, item.stream_epoch, item.track_id)].append(item)
            self._track_keys = list(sorted(grouped))
            self.track_table.setRowCount(len(self._track_keys))
            sample_rate = session.descriptor.sample_rate
            for row, key in enumerate(self._track_keys):
                observations = grouped[key]
                first, last = observations[0], observations[-1]
                duration = None if sample_rate is None else (last.decision_sample - first.decision_sample + 960) * 1000 / sample_rate
                probability = next((item.l5_probability for item in reversed(observations) if item.l5_probability is not None), None)
                unwrapped = _unwrap([item.theta_deg for item in observations if item.theta_deg is not None])
                trajectory = "N/A" if not unwrapped else f"{unwrapped[0]:.1f}→{unwrapped[-1]:.1f}°"
                values = (
                    str(key[1]), str(key[2]), str(first.decision_sample), str(last.decision_sample), _number(duration, "{:.1f}"),
                    _number(first.theta_deg, "{:.1f}°"), _number(last.theta_deg, "{:.1f}°"), trajectory,
                    last.state or "N/A", _number(probability, "{:.3f}"), "按需",
                )
                for column, value in enumerate(values):
                    self.track_table.setItem(row, column, QTableWidgetItem(value))
            self._all_anomalies = session.anomalies
            categories = ["全部", *sorted({item.category for item in session.anomalies})]
            self.anomaly_filter.blockSignals(True)
            self.anomaly_filter.clear()
            self.anomaly_filter.addItems(categories)
            self.anomaly_filter.blockSignals(False)
            self._render_anomalies("全部")
            self.play.setEnabled(bool(self._track_keys) and self.adapter.capabilities.track_audio)

        def _render_anomalies(self, category: str) -> None:
            self._visible_anomalies = [
                item for item in self._all_anomalies if category in {"", "全部", item.category}
            ]
            self.anomaly_table.setRowCount(len(self._visible_anomalies))
            for row, anomaly in enumerate(self._visible_anomalies):
                key = anomaly.key
                values = (
                    anomaly.category, "N/A" if key is None else str(key.stream_epoch),
                    "N/A" if key is None else str(key.window_id), "N/A" if key is None else str(key.decision_sample), anomaly.message,
                )
                for column, value in enumerate(values):
                    self.anomaly_table.setItem(row, column, QTableWidgetItem(value))

        def _select_anomaly(self, row: int, _column: int) -> None:
            if 0 <= row < len(self._visible_anomalies):
                key = self._visible_anomalies[row].key
                if key is not None:
                    self.window_selected.emit(key)

        def _play_selected(self) -> None:
            row = self.track_table.currentRow()
            if row < 0 or row >= len(self._track_keys):
                QMessageBox.information(self, "只读回放", "请先选择方向轨。")
                return
            session_id, epoch, track_id = self._track_keys[row]
            try:
                assets = self.adapter.track_assets(session_id, epoch, track_id)
                asset = next((item for item in assets if isinstance(item.get("absolute_path"), str)), None)
                if asset is None:
                    raise FileNotFoundError("接口未返回可播放的已校验资产")
                path = Path(str(asset["absolute_path"]))
                self.player.setSource(QUrl.fromLocalFile(str(path)))
                self.player.play()
            except (OSError, ValueError, NotImplementedError) as exc:
                QMessageBox.warning(self, "只读回放不可用", str(exc))


    class PipelineLogWindow(QMainWindow):
        def __init__(self, provider: object, parent: QWidget | None = None):
            super().__init__(parent)
            self.setWindowTitle("Pipeline Log UI — Read Only")
            self.resize(1500, 920)
            self.adapter = PublicApiAdapter(provider)
            self.controller = LogUiController(self.adapter)
            self.current_session: SessionReadModel | None = None
            self.bridge = _LoadBridge(self)
            self.bridge.completed.connect(self._loaded)
            self.tabs = QTabWidget()
            self.records = RecordsPage()
            self.overview = OverviewPage()
            self.timeline = TimelinePage()
            self.detail = WindowDetailPage()
            self.tracks = TracksAnomaliesPage(self.adapter)
            self.tabs.addTab(self.records, "记录列表")
            self.tabs.addTab(self.overview, "会话总览")
            self.tabs.addTab(self.timeline, "Pipeline 时间线")
            self.tabs.addTab(self.detail, "单窗详情")
            self.tabs.addTab(self.tracks, "ID 与异常")
            self.setCentralWidget(self.tabs)
            self.records.selected.connect(self.open_session)
            self.timeline.window_selected.connect(self.open_window)
            self.tracks.window_selected.connect(self.open_window)
            self.records.set_sessions(self.controller.sessions(limit=500).items)
            if not self.adapter.capabilities.offline_review:
                self.records.notice.setText("Unavailable：宿主未提供完整公开只读 session/decision 查询能力。")

        def open_session(self, session_id: str) -> None:
            self.statusBar().showMessage(f"正在只读加载 {session_id}…")
            self.controller.load_session(session_id, lambda result, error: self.bridge.completed.emit(result, error))

        def _loaded(self, session: SessionReadModel | None, error: BaseException | None) -> None:
            if error is not None:
                self.statusBar().showMessage(f"加载失败：{error}")
                return
            if session is None:
                return
            self.current_session = session
            self.overview.set_session(session)
            self.timeline.set_session(session)
            self.tracks.set_session(session)
            self.tabs.setCurrentWidget(self.overview)
            self.statusBar().showMessage(f"只读加载完成：{len(session.windows)} windows")

        def open_window(self, key: WindowKey) -> None:
            if self.current_session is None:
                return
            self.detail.set_window(self.current_session, key)
            self.tabs.setCurrentWidget(self.detail)

        def closeEvent(self, event) -> None:  # noqa: N802
            self.controller.close()
            self.player_stop()
            super().closeEvent(event)

        def player_stop(self) -> None:
            self.tracks.player.stop()


    def launch_log_ui(provider: object) -> int:
        """Launch with an injected public provider; never accepts a data path."""
        application = QApplication.instance() or QApplication([])
        window = PipelineLogWindow(provider)
        window.show()
        return application.exec()

else:

    class PipelineLogWindow:  # pragma: no cover
        def __init__(self, *_args, **_kwargs):
            raise ImportError("安装PySide6后才能使用Pipeline Log UI")

    def launch_log_ui(_provider: object) -> int:  # pragma: no cover
        raise ImportError("安装PySide6后才能使用Pipeline Log UI")


def _short(value: str | None) -> str:
    return "N/A" if value is None else value[:12]


def _na(value: object | None) -> str:
    return "N/A" if value is None else str(value)


def _number(value: float | None, pattern: str) -> str:
    return "N/A" if value is None else pattern.format(value)


def _percentiles(value) -> str:
    if value.n == 0:
        return f"N/A (n=0, missing={value.missing})"
    return f"{value.p50:.2f}/{value.p95:.2f}/{value.p99:.2f} (n={value.n}, missing={value.missing})"


def _unwrap(values: list[float]) -> list[float]:
    if not values:
        return []
    output = [float(values[0])]
    for value in values[1:]:
        delta = (float(value) - (output[-1] % 360.0) + 180.0) % 360.0 - 180.0
        output.append(output[-1] + delta)
    return output


__all__ = ["PipelineLogWindow", "launch_log_ui"]
