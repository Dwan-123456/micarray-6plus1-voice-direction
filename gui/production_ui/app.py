from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

try:
    from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFrame,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QInputDialog,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Audio Data Manager需要安装项目ui依赖（PySide6）") from exc

from data_management.service import DataManagerService
from data_management.contracts import Annotation
from data_management.corpus_naming import build_corpus_display_name
from data_management.wizard import WizardInput, validate_wizard
from gui.production_ui.capture_host import CaptureHost
from gui.production_ui.channel_player import NativeChannelPlayer


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)


class _Job(QRunnable):
    def __init__(self, fn: Callable[[], Any]):
        super().__init__()
        self.setAutoDelete(False)
        self.fn = fn
        self.signals = _Signals()

    @Slot()
    def run(self):
        try:
            self.signals.done.emit(self.fn())
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class DataTable(QTableWidget):
    def __init__(self, columns: list[tuple[str, str]], empty_message: str = "暂无数据"):
        super().__init__(0, len(columns))
        self.columns = columns
        self.empty_message = empty_message
        self.setHorizontalHeaderLabels([x[1] for x in columns])
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.horizontalHeader().setStretchLastSection(True)

    def load(self, rows: list[dict[str, Any]]) -> None:
        self.clearSpans()
        if not rows:
            self.setRowCount(1)
            item = QTableWidgetItem(self.empty_message)
            item.setTextAlignment(Qt.AlignCenter)
            item.setFlags(Qt.ItemIsEnabled)
            self.setItem(0, 0, item)
            self.setSpan(0, 0, 1, len(self.columns))
            return
        self.setRowCount(len(rows))
        for row, item in enumerate(rows):
            for col, (key, _) in enumerate(self.columns):
                raw = item.get(key, "")
                cell = QTableWidgetItem(self._display_value(key, raw))
                cell.setData(Qt.UserRole, raw)
                cell.setToolTip(str(raw))
                self.setItem(row, col, cell)

    @staticmethod
    def _display_value(key: str, value: Any) -> str:
        if value is None or value == "":
            return "—"
        translations = {
            "off": "关闭",
            "manual": "手动录音",
            "continuous": "连续录音",
            "event": "人声事件录音",
            "complete": "已完成",
            "incomplete": "未完整结束",
            "corrupt": "文件异常",
            "open": "进行中",
            "pending": "待检查",
            "passed": "检查通过",
            "failed": "检查未通过",
            "quarantine": "待处理区",
            "annotated": "已标注",
            "versioned": "已纳入版本",
            "dedicated": "专门录制",
            "promoted_runtime": "从运行录音提取",
            "imported": "外部导入",
            "train": "训练集",
            "validation": "验证集",
            "test": "测试集",
            "calibration": "校准集",
            "unset": "尚未分配",
        }
        if str(value) in translations:
            return translations[str(value)]
        if key in {"id", "dataset_id"} and len(str(value)) > 14:
            return f"{str(value)[:8]}…{str(value)[-4:]}"
        if key in {"started_at", "ended_at"}:
            try:
                return (
                    datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M:%S")
                )
            except ValueError:
                pass
        if key == "duration_samples":
            return f"{int(value) / 48_000:.1f} 秒"
        if key == "duration_seconds":
            return f"{float(value):.2f} 秒"
        if key in {"first_theta_deg", "last_theta_deg", "angle_change_deg"}:
            return f"{float(value):.1f}°"
        if key in {"latest_l5_probability", "mean_l5_probability"}:
            return "—" if value in {None, ""} else f"{float(value):.3f}"
        return str(value)

    def selected_id(self) -> str | None:
        rows = self.selectionModel().selectedRows()
        if not rows:
            return None
        value = self.item(rows[0].row(), 0).data(Qt.UserRole)
        return None if value in {None, ""} else str(value)

    def select_id(self, value: str) -> bool:
        for row in range(self.rowCount()):
            item = self.item(row, 0)
            if item is not None and str(item.data(Qt.UserRole)) == value:
                self.selectRow(row)
                return True
        return False


class ImportMetadataDialog(QDialog):
    """Self-contained import metadata gate; no hidden dependency on another page."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("导入外部WAV：填写样本信息")
        self.resize(620, 620)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "导入文件必须是48 kHz、7通道或8通道PCM16 WAV。请补全来源、环境和授权信息；系统随后会自动检查质量。"
        )
        intro.setWordWrap(True)
        intro.setObjectName("pageIntro")
        layout.addWidget(intro)
        form = QFormLayout()
        self.fields: dict[str, QLineEdit] = {}
        for key, label, placeholder in (
            ("dataset", "数据集编号 *", "例如：office-voice-v1"),
            ("room", "房间编号 *", "例如：meeting-room-a"),
            ("environment", "环境说明 *", "例如：安静、空调"),
            ("pose", "阵列摆放编号 *", "例如：桌面中央-01"),
            ("source_count", "声源数量", "1"),
            ("theta", "真实角度（可留空）", "多个角度用逗号分隔"),
            ("distance", "距离（米，可留空）", "例如：1.0"),
        ):
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            if key == "source_count":
                edit.setText("1")
            self.fields[key] = edit
            form.addRow(label, edit)
        self.allowed_uses = QComboBox()
        self.allowed_uses.addItem("仅用于内部研究", ("research",))
        self.allowed_uses.addItem("用于研究和模型训练", ("research", "ml_training"))
        self.allowed_uses.addItem("仅用于质量检查", ("quality_assurance",))
        form.addRow("允许用途 *", self.allowed_uses)
        self.consent = QComboBox()
        self.consent.addItem("尚未获得授权（不能导入）", "invalid")
        self.consent.addItem("已获得参与者授权", "granted")
        self.consent.addItem("不涉及个人声音/无需授权", "not_applicable")
        form.addRow("录音使用授权 *", self.consent)
        layout.addLayout(form)
        self.error_label = QLabel("")
        self.error_label.setObjectName("warningText")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        buttons.button(QDialogButtonBox.Ok).setText("确认并开始导入")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _csv_float(value: str) -> tuple[float, ...]:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())

    def value(self) -> WizardInput:
        source_count_text = self.fields["source_count"].text().strip()
        if not source_count_text:
            raise ValueError("请填写声源数量")
        return WizardInput(
            self.fields["dataset"].text().strip(),
            self.fields["room"].text().strip(),
            self.fields["environment"].text().strip(),
            self.fields["pose"].text().strip(),
            int(source_count_text),
            self.consent.currentData(),
            tuple(self.allowed_uses.currentData()),
            self._csv_float(self.fields["theta"].text()),
            self._csv_float(self.fields["distance"].text()),
            recording_name=self.fields["dataset"].text().strip() or "导入录音",
        )

    def _validate(self):
        try:
            errors = validate_wizard(self.value())
        except ValueError as exc:
            errors = [str(exc)]
        if errors:
            self.error_label.setText("请修正以下问题：\n" + "\n".join(f"• {item}" for item in errors))
            self.error_label.show()
            return
        self.accept()


class AudioDataManager(QMainWindow):
    recording_command = Signal(str)

    def __init__(self, data_root: str | Path = "data"):
        super().__init__()
        self.service = DataManagerService(data_root)
        self.pool = QThreadPool.globalInstance()
        self._jobs: set[_Job] = set()
        self.capture_connected = False
        self._pending_wizard_input: WizardInput | None = None
        self._closing = False
        self._latest_dashboard: dict[str, Any] = {}
        self.capture_host = CaptureHost(Path(__file__).resolve().parents[2], self.service)
        self.capture_host.connected_changed.connect(self._capture_connection_changed)
        self.capture_host.runtime_status.connect(self._apply_runtime_status)
        self.capture_host.wizard_status.connect(self._apply_wizard_status)
        self.capture_host.error.connect(self._capture_error)
        self.channel_player = NativeChannelPlayer()
        self.recording_command.connect(self.capture_host.handle_command)
        self.setWindowTitle("麦克风阵列录音与数据管理")
        # This is only a restored-window fallback. The desktop entry opens
        # maximized inside the current screen's available work area.
        self.resize(1200, 760)
        self._build()
        self.refresh_all()

    def _build(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        top = QVBoxLayout()
        status_row = QHBoxLayout()
        self.rec_badge = QLabel("● 录音已关闭")
        self.rec_badge.setObjectName("recBadge")
        self.session_label = QLabel("当前会话：—")
        self.duration_label = QLabel("时长：00:00:00")
        self.disk_label = QLabel("磁盘余量：—")
        self.capacity_label = QLabel("预计可录：—")
        for widget in (self.rec_badge, self.session_label, self.duration_label, self.disk_label, self.capacity_label):
            status_row.addWidget(widget)
        status_row.addStretch()
        self.connection_badge = QLabel("● 采集源未连接")
        self.connection_badge.setObjectName("warningText")
        status_row.addWidget(self.connection_badge)
        top.addLayout(status_row)

        controls_row = QHBoxLayout()
        controls_row.addStretch()
        self.connect_button = QPushButton("连接麦克风")
        self.connect_button.clicked.connect(self._toggle_capture)
        self.connect_button.setToolTip("连接配置文件中指定的8通道麦克风设备")
        controls_row.addWidget(self.connect_button)
        self.mode_select = QComboBox()
        for label, value in (
            ("关闭", "off"),
            ("手动录音", "manual"),
            ("连续录音", "continuous"),
        ):
            self.mode_select.addItem(label, value)
        self.mode_select.currentIndexChanged.connect(lambda _: self._mode_changed(self.mode_select.currentData()))
        self.mode_select.setToolTip(
            "选择实际运行时的录音方式。初次使用建议选择“手动录音”。\n"
            "独立桌面版未运行人声检测算法，因此不提供事件触发录音。"
        )
        self.mode_select.setEnabled(False)
        controls_row.addWidget(self.mode_select)
        self.recording_buttons: dict[str, QPushButton] = {}
        for label, command, tip in (
            ("开始录音", "record", "手动录音模式下开始保存"),
            ("暂停", "pause", "暂停保存，但实时算法仍继续运行"),
            ("结束当前录音", "stop", "停止保存当前片段；麦克风保持连接，完整会话会在断开时安全封存"),
        ):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.setEnabled(False)
            button.clicked.connect(lambda checked=False, value=command: self._recording_action(value))
            self.recording_buttons[command] = button
            controls_row.addWidget(button)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.refresh_all)
        controls_row.addWidget(refresh)
        help_button = QPushButton("使用说明")
        help_button.clicked.connect(self._show_quick_help)
        controls_row.addWidget(help_button)
        top.addLayout(controls_row)
        layout.addLayout(top)
        self.tabs = QTabWidget()
        self.tabs.tabBar().setExpanding(True)
        self.tabs.tabBar().setUsesScrollButtons(True)
        layout.addWidget(self.tabs)
        self.setCentralWidget(central)
        self._home_tab()
        self._runtime_tab()
        self._corpus_tab()
        self._wizard_tab()
        self._qa_tab()
        self._storage_tab()
        self.statusBar().showMessage("数据只保存在本机，不会自动上传。首次使用请从“操作首页”开始。")
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #071a33;
                color: #f8fafc;
                font-size: 15px;
            }
            QLabel {
                background: transparent;
                color: #f8fafc;
            }
            QLabel#pageTitle {
                color: #ffffff;
                font-size: 24px;
                font-weight: 800;
                padding: 2px 0 4px 0;
            }
            QLabel#pageIntro {
                color: #c9def4;
                font-size: 14px;
                padding-bottom: 8px;
            }
            QLabel#warningText {
                color: #fde68a;
                background-color: #3b2f16;
                border: 1px solid #d97706;
                border-radius: 6px;
                padding: 10px;
            }
            QFrame#taskCard {
                background-color: #0d2949;
                border: 1px solid #466b91;
                border-radius: 10px;
                padding: 8px;
            }
            QTabWidget::pane {
                border: 1px solid #3b5f87;
                background-color: #0a2240;
                top: -1px;
            }
            QTabBar::tab {
                background-color: #173657;
                color: #dbeafe;
                border: 1px solid #34597f;
                border-bottom: none;
                padding: 12px 10px;
                min-width: 90px;
                font-weight: 600;
            }
            QTabBar::tab:hover {
                background-color: #24517c;
                color: #ffffff;
            }
            QTabBar::tab:selected {
                background-color: #0e7490;
                color: #ffffff;
                border-color: #22d3ee;
            }
            QLineEdit, QComboBox, QSpinBox, QTextEdit {
                background-color: #0d2949;
                color: #ffffff;
                selection-background-color: #0891b2;
                selection-color: #ffffff;
                border: 1px solid #5279a4;
                border-radius: 5px;
                padding: 8px;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QTextEdit:focus {
                border: 2px solid #38bdf8;
            }
            QLineEdit::placeholder {
                color: #9fb8d3;
            }
            QComboBox QAbstractItemView {
                background-color: #102f52;
                color: #ffffff;
                selection-background-color: #0e7490;
                selection-color: #ffffff;
                border: 1px solid #60a5fa;
                outline: 0;
            }
            QPushButton {
                background-color: #0e7490;
                color: #ffffff;
                border: 1px solid #22d3ee;
                border-radius: 6px;
                padding: 9px 17px;
                font-weight: 700;
            }
            QPushButton:hover {
                background-color: #0891b2;
                border-color: #67e8f9;
            }
            QPushButton:pressed {
                background-color: #155e75;
            }
            QPushButton:disabled {
                background-color: #334155;
                color: #94a3b8;
                border-color: #475569;
            }
            QPushButton#destructiveButton {
                background-color: #7f1d1d;
                border-color: #f87171;
            }
            QPushButton#destructiveButton:hover {
                background-color: #991b1b;
            }
            QPushButton#warningButton {
                background-color: #92400e;
                border-color: #fbbf24;
            }
            QPushButton#warningButton:hover {
                background-color: #b45309;
            }
            QTableWidget, QTableView {
                background-color: #081c35;
                alternate-background-color: #0d2949;
                color: #f8fafc;
                gridline-color: #365b80;
                border: 1px solid #466b91;
                selection-background-color: #0369a1;
                selection-color: #ffffff;
            }
            QTableWidget::item, QTableView::item {
                color: #f8fafc;
                padding: 7px;
                border-bottom: 1px solid #294c70;
            }
            QTableWidget::item:selected, QTableView::item:selected {
                background-color: #0369a1;
                color: #ffffff;
            }
            QTableWidget#corpusTable {
                outline: none;
                selection-background-color: #7dd3fc;
                selection-color: #082f49;
            }
            QTableWidget#corpusTable::item:selected {
                background-color: #7dd3fc;
                color: #082f49;
                border: 2px solid #38bdf8;
                padding: 5px 7px;
            }
            QHeaderView::section {
                background-color: #1b3d61;
                color: #ffffff;
                border: 1px solid #466b91;
                padding: 9px;
                font-weight: 700;
            }
            QTableCornerButton::section {
                background-color: #1b3d61;
                border: 1px solid #466b91;
            }
            QGroupBox {
                background-color: #0a2240;
                color: #ffffff;
                border: 1px solid #466b91;
                border-radius: 7px;
                margin-top: 16px;
                padding: 15px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 6px;
                color: #7dd3fc;
                background-color: #0a2240;
            }
            QStatusBar {
                background-color: #06162b;
                color: #cde7ff;
                border-top: 1px solid #365b80;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background-color: #0b2442;
                border: none;
            }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background-color: #4776a6;
                border-radius: 5px;
                min-height: 24px;
                min-width: 24px;
            }
            QToolTip {
                background-color: #173657;
                color: #ffffff;
                border: 1px solid #67e8f9;
                padding: 5px;
            }
            QMessageBox {
                background-color: #0a2240;
            }
            QMessageBox QLabel {
                color: #ffffff;
            }
            #recBadge {
                font-weight: 800;
                color: #cbd5e1;
                background-color: #0b2442;
                padding: 8px 15px;
                border: 1px solid #5279a4;
                border-radius: 13px;
            }
            """
        )

    @staticmethod
    def _page_heading(title: str, intro: str) -> tuple[QLabel, QLabel]:
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        description = QLabel(intro)
        description.setObjectName("pageIntro")
        description.setWordWrap(True)
        return heading, description

    def _task_card(self, title: str, description: str, button_text: str, route: str) -> QFrame:
        card = QFrame()
        card.setObjectName("taskCard")
        box = QVBoxLayout(card)
        heading = QLabel(title)
        heading.setStyleSheet("font-size:18px;font-weight:800;color:#7dd3fc")
        text = QLabel(description)
        text.setWordWrap(True)
        text.setStyleSheet("color:#dbeafe")
        button = QPushButton(button_text)
        button.clicked.connect(lambda: self._go(route))
        box.addWidget(heading)
        box.addWidget(text)
        box.addStretch()
        box.addWidget(button)
        return card

    def _home_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        title, intro = self._page_heading(
            "操作首页",
            "按目的选择下一步。日常录音、建立正式测试语料和检查已有数据是三条相互独立的流程。",
        )
        layout.addWidget(title)
        layout.addWidget(intro)
        self.capture_notice = QLabel(
            "当前未连接麦克风采集主程序：可以浏览、导入、质检和维护数据；实际录音按钮暂不可用。"
        )
        self.capture_notice.setObjectName("warningText")
        self.capture_notice.setWordWrap(True)
        layout.addWidget(self.capture_notice)
        cards = QGridLayout()
        cards.addWidget(
            self._task_card(
                "录制日常运行音频",
                "用于保存系统正常运行时的声音，方便回看、复现问题和提取候选片段。没有人工真值。",
                "查看运行录音",
                "runtime",
            ),
            0,
            0,
        )
        cards.addWidget(
            self._task_card(
                "采集带角度真值的测试样本",
                "按向导填写环境和声源信息，边录边保存原始8通道音频与热力图；质量检查可稍后手动执行。",
                "开始测试录制向导",
                "wizard",
            ),
            0,
            1,
        )
        cards.addWidget(
            self._task_card(
                "导入或整理已有音频",
                "查看专门录制、外部导入或从运行录音提取的样本；可导入、导出和进入质量管理。",
                "打开测试语料库",
                "corpus",
            ),
            1,
            0,
        )
        cards.addWidget(
            self._task_card(
                "检查并发布数据集版本",
                "运行音频质量检查、检查训练/验证/测试泄漏，并在确认后生成不可修改的数据集版本。",
                "打开质量管理",
                "quality",
            ),
            1,
            1,
        )
        layout.addLayout(cards)
        summary_box = QGroupBox("当前数据概况")
        summary_layout = QVBoxLayout(summary_box)
        self.home_summary = QLabel("正在读取数据…")
        self.home_summary.setWordWrap(True)
        summary_layout.addWidget(self.home_summary)
        layout.addWidget(summary_box)
        self.tabs.addTab(page, "操作首页")
        self.pages = {"home": page}

    def _go(self, route: str) -> None:
        page = self.pages.get(route)
        if page is not None:
            self.tabs.setCurrentWidget(page)

    def _runtime_tab(self):
        page = QWidget()
        self.pages["runtime"] = page
        box = QVBoxLayout(page)
        title, intro = self._page_heading(
            "运行录音",
            "这里保存系统日常运行时的录音，用于回听和故障复现。它们不能直接作为训练集或正式测试集。",
        )
        box.addWidget(title)
        box.addWidget(intro)
        filters = QHBoxLayout()
        self.runtime_search = QLineEdit()
        self.runtime_search.setPlaceholderText("输入完整或部分会话编号进行搜索")
        self.runtime_search.textChanged.connect(self._filter_runtime)
        filters.addWidget(self.runtime_search)
        self.runtime_table = DataTable(
            [
                ("id", "会话编号"),
                ("started_at", "开始时间"),
                ("ended_at", "结束时间"),
                ("mode", "模式"),
                ("status", "状态"),
                ("path", "保存位置"),
            ],
            "还没有运行录音。连接采集源后，选择“手动录音”并点击“开始录音”。",
        )
        box.addLayout(filters)
        box.addWidget(self.runtime_table)
        self.runtime_table.itemSelectionChanged.connect(self._runtime_session_selected)

        detail_box = QGroupBox("方向ID与录音试听")
        detail_layout = QVBoxLayout(detail_box)
        self.runtime_direction_status = QLabel("请选择一条运行录音。")
        self.runtime_direction_status.setObjectName("pageIntro")
        detail_layout.addWidget(self.runtime_direction_status)
        self.runtime_track_table = DataTable(
            [
                ("track_key", "内部轨迹键"),
                ("stream_epoch", "Epoch"),
                ("track_id", "方向ID"),
                ("first_sample", "首Sample"),
                ("last_sample", "末Sample"),
                ("duration_seconds", "持续时间"),
                ("first_theta_deg", "起始角"),
                ("last_theta_deg", "结束角"),
                ("angle_change_deg", "角度变化"),
                ("state", "状态"),
                ("latest_l5_probability", "最新L5概率"),
            ],
            "该运行录音没有算法方向ID。",
        )
        self.runtime_track_table.setColumnHidden(0, True)
        detail_layout.addWidget(self.runtime_track_table)
        listen_actions = QHBoxLayout()
        listen_track = QPushButton("试听所选方向ID")
        listen_track.clicked.connect(self._listen_runtime_track)
        listen_center = QPushButton("试听Center参考")
        listen_center.clicked.connect(self._listen_runtime_center)
        self.runtime_audio_kind = QComboBox()
        self.runtime_audio_kind.addItem("Native原始8通道", ("native_8ch", 8))
        self.runtime_audio_kind.addItem("Logical逻辑8通道", ("logical_8ch", 8))
        self.runtime_audio_kind.addItem("Physical物理7通道", ("physical_7ch", 7))
        self.runtime_audio_kind.currentIndexChanged.connect(self._update_runtime_audio_channels)
        self.runtime_audio_channel = QComboBox()
        self._update_runtime_audio_channels()
        listen_raw = QPushButton("试听所选原始通道")
        listen_raw.clicked.connect(self._listen_runtime_raw_channel)
        stop_listen = QPushButton("停止试听")
        stop_listen.clicked.connect(self.channel_player.stop)
        for widget in (
            listen_track,
            listen_center,
            self.runtime_audio_kind,
            self.runtime_audio_channel,
            listen_raw,
            stop_listen,
        ):
            listen_actions.addWidget(widget)
        listen_actions.addStretch()
        detail_layout.addLayout(listen_actions)
        box.addWidget(detail_box)
        actions = QHBoxLayout()
        promote = QPushButton("提取片段到测试语料")
        promote.setToolTip("复制所选运行录音中的一段，原始录音不会被修改")
        promote.setEnabled(False)
        promote.setToolTip("需要波形范围选择器；当前版本暂不可用。可先导出录音，或在测试语料库导入WAV。")
        actions.addWidget(promote)
        trash = QPushButton("移到回收站")
        trash.setObjectName("destructiveButton")
        trash.clicked.connect(self._trash_session)
        actions.addWidget(trash)
        export_button = QPushButton("导出所选录音")
        export_button.clicked.connect(self._export_session)
        actions.addWidget(export_button)
        actions.addStretch()
        box.addLayout(actions)
        self.runtime_selection_help = QLabel("提示：先在表格中选中一行，再执行下方操作。删除只会进入可恢复的回收站。")
        self.runtime_selection_help.setObjectName("pageIntro")
        box.addWidget(self.runtime_selection_help)
        self.tabs.addTab(page, "运行录音")

    def _corpus_tab(self):
        page = QWidget()
        self.pages["corpus"] = page
        box = QVBoxLayout(page)
        title, intro = self._page_heading(
            "测试语料库",
            "正式训练和评测使用的音频样本。每条样本都有来源、授权、质量状态和数据集归属。",
        )
        box.addWidget(title)
        box.addWidget(intro)
        self.corpus_table = DataTable(
            [
                ("id", "内部编号"),
                ("display_name", "音频名称"),
            ],
            "还没有录音，请先使用“测试录制向导”录制一条数据。",
        )
        self.corpus_table.setObjectName("corpusTable")
        self.corpus_table.setColumnHidden(0, True)
        box.addWidget(self.corpus_table)
        actions = QHBoxLayout()
        self.listen_channel = QComboBox()
        for index in range(8):
            self.listen_channel.addItem(f"通道 {index + 1}", index)
        listen_button = QPushButton("试听所选通道")
        listen_button.clicked.connect(self._listen_selected_channel)
        stop_listen_button = QPushButton("停止试听")
        stop_listen_button.clicked.connect(self.channel_player.stop)
        simulate_button = QPushButton("用所选样本进行模拟测试")
        simulate_button.clicked.connect(self._simulate_selected_recording)
        simulate_button.setToolTip("自动打开Test UI并模拟原始8通道音频输入；模拟时不读取热力图")
        rename_button = QPushButton("修改所选名称")
        rename_button.clicked.connect(self._rename_selected_recording)
        rename_button.setToolTip("只修改列表名称和标签记录，不改变录音文件或内部编号")
        trash_button = QPushButton("移到回收站")
        trash_button.setObjectName("destructiveButton")
        trash_button.clicked.connect(self._trash_recording)
        actions.addWidget(self.listen_channel)
        actions.addWidget(listen_button)
        actions.addWidget(stop_listen_button)
        actions.addWidget(simulate_button)
        actions.addWidget(rename_button)
        actions.addWidget(trash_button)
        actions.addStretch()
        box.addLayout(actions)
        self.tabs.addTab(page, "测试语料库")

    def _wizard_tab(self):
        page = QWidget()
        self.pages["wizard"] = page
        page_layout = QVBoxLayout(page)
        title, intro = self._page_heading(
            "测试录制向导",
            "填写基本信息后点击开始，程序会自动连接麦克风并立即录制。可随时暂停、继续或结束，结束后自动统计时长、保存并检查质量。",
        )
        page_layout.addWidget(title)
        page_layout.addWidget(intro)
        outer = QHBoxLayout()
        page_layout.addLayout(outer)
        formbox = QGroupBox("录制信息")
        form = QFormLayout(formbox)
        self.wizard_fields = {}
        environment = QLineEdit()
        environment.setPlaceholderText("例如：诊室、会议室、室外走廊")
        self.wizard_fields["environment"] = environment
        form.addRow("环境 *", environment)

        self.wizard_source_count = QSpinBox()
        self.wizard_source_count.setRange(0, 20)
        self.wizard_source_count.setValue(1)
        self.wizard_source_count.setToolTip("只录制环境噪音时可以填写0")
        form.addRow("声源数量 *", self.wizard_source_count)

        self.wizard_sources_widget = QWidget()
        self.wizard_sources_layout = QGridLayout(self.wizard_sources_widget)
        self.wizard_sources_layout.setContentsMargins(0, 0, 0, 0)
        self.wizard_sources_layout.addWidget(QLabel("声源"), 0, 0)
        self.wizard_sources_layout.addWidget(QLabel("类型"), 0, 1)
        self.wizard_sources_layout.addWidget(QLabel("移动方式"), 0, 2)
        self.wizard_source_rows: list[tuple[QLabel, QLineEdit, QLineEdit]] = []
        form.addRow("各声源信息 *", self.wizard_sources_widget)
        self.wizard_source_count.valueChanged.connect(self._sync_wizard_source_rows)
        self._sync_wizard_source_rows(self.wizard_source_count.value())

        noise_source = QLineEdit()
        noise_source.setPlaceholderText("例如：空调、风扇、走廊人声；没有请填写“无”")
        self.wizard_fields["noise_source"] = noise_source
        form.addRow("噪音来源 *", noise_source)
        self.wizard_start = QPushButton("开始录制")
        self.wizard_start.clicked.connect(self._start_wizard)
        self.wizard_start.setToolTip("如麦克风尚未连接，程序会先自动连接再开始录制")
        self.wizard_pause = QPushButton("暂停录制")
        self.wizard_pause.clicked.connect(self._pause_or_resume_wizard)
        self.wizard_pause.setEnabled(False)
        self.wizard_stop = QPushButton("结束并保存")
        self.wizard_stop.clicked.connect(self._finish_wizard_recording)
        self.wizard_stop.setEnabled(False)
        form.addRow(self.wizard_start)
        form.addRow(self.wizard_pause)
        form.addRow(self.wizard_stop)
        self.wizard_status = QLabel("当前阶段：等待填写录制信息")
        self.wizard_status.setObjectName("warningText")
        self.wizard_status.setWordWrap(True)
        form.addRow(self.wizard_status)
        self.wizard_direction_id_status = QLabel("方向ID：无算法方向ID（专用录音仅采集L1输入）")
        self.wizard_direction_id_status.setObjectName("pageIntro")
        self.wizard_direction_id_status.setWordWrap(True)
        form.addRow(self.wizard_direction_id_status)
        outer.addWidget(formbox, 1)
        steps_box = QGroupBox("录制流程与当前操作")
        steps_layout = QVBoxLayout(steps_box)
        steps = QLabel(
            "① 填写左侧录制信息\n\n"
            "② 点击“开始录制”，程序自动连接麦克风并立即开始采集\n\n"
            "③ 录制期间可以暂停和继续，暂停部分不会写入数据集\n\n"
            "④ 点击“结束并保存”，程序自动统计实际录制时长\n\n"
            "⑤ 系统边录边保存麦克风传入电脑的原始8通道音频和热力图\n\n"
            "保存完成后录音会直接进入测试语料库；需要时可在“质量与标注”页面手动检查。"
        )
        steps.setWordWrap(True)
        steps_layout.addWidget(steps)
        steps_layout.addStretch()
        outer.addWidget(steps_box, 1)
        self.tabs.addTab(page, "测试录制向导")

    def _sync_wizard_source_rows(self, source_count: int) -> None:
        while len(self.wizard_source_rows) > source_count:
            row = self.wizard_source_rows.pop()
            for widget in row:
                self.wizard_sources_layout.removeWidget(widget)
                widget.deleteLater()
        while len(self.wizard_source_rows) < source_count:
            source_index = len(self.wizard_source_rows) + 1
            label = QLabel(f"声源 {source_index}")
            source_type = QLineEdit()
            source_type.setPlaceholderText("例如：人声、音箱、设备")
            movement = QLineEdit()
            movement.setPlaceholderText("例如：静止、走动、环绕移动")
            self.wizard_sources_layout.addWidget(label, source_index, 0)
            self.wizard_sources_layout.addWidget(source_type, source_index, 1)
            self.wizard_sources_layout.addWidget(movement, source_index, 2)
            self.wizard_source_rows.append((label, source_type, movement))

    def _qa_tab(self):
        page = QWidget()
        self.pages["quality"] = page
        layout = QVBoxLayout(page)
        title, intro = self._page_heading(
            "质量与标注",
            "先选择测试样本，再检查音频质量或添加人工标注。数据准备完成后，最后再执行数据集分组和锁定。",
        )
        layout.addWidget(title)
        layout.addWidget(intro)
        sections = QTabWidget()
        layout.addWidget(sections)

        quality_page = QWidget()
        quality_layout = QVBoxLayout(quality_page)
        quality_layout.addWidget(QLabel("样本质量检查会核对文件哈希、通道电平、削波、静音、直流偏置和通道相关性。"))
        self.current_sample_label = QLabel("当前样本：尚未选择。请先到“测试语料库”选中一条样本。")
        self.current_sample_label.setObjectName("warningText")
        quality_layout.addWidget(self.current_sample_label)
        actions = QHBoxLayout()
        run = QPushButton("检查当前选中的测试样本")
        run.clicked.connect(self._run_qa)
        leak = QPushButton("检查数据用途是否相互泄漏")
        leak.clicked.connect(self._run_leakage)
        lock = QPushButton("预览并发布数据集版本…")
        lock.setObjectName("warningButton")
        lock.setToolTip("这是不可逆的版本化操作。锁定后不能原地修改标注或删除样本。")
        lock.clicked.connect(self._lock_dataset)
        actions.addWidget(run)
        actions.addWidget(leak)
        actions.addWidget(lock)
        actions.addStretch()
        quality_layout.addLayout(actions)
        self.qa_output = QTextEdit()
        self.qa_output.setReadOnly(True)
        self.qa_output.setPlaceholderText("质量检查结果会显示在这里。")
        quality_layout.addWidget(self.qa_output)
        sections.addTab(quality_page, "质量检查与数据集版本")

        annotation_page = QWidget()
        annotation_layout = QVBoxLayout(annotation_page)
        annotation_layout.addWidget(
            QLabel("先在“测试语料库”中选中一个样本，再点击“为所选样本添加标注”；样本编号会自动带入。")
        )
        annotation_form = QFormLayout()
        self.annotation_fields: list[QLineEdit] = []
        for label, placeholder in (
            ("样本编号", "从测试语料库自动带入"),
            ("开始时间（秒）", "从0开始，例如：0.00"),
            ("结束时间（秒）", "必须大于开始时间，例如：1.25"),
        ):
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            self.annotation_fields.append(edit)
            annotation_form.addRow(label, edit)
        self.annotation_type = QComboBox()
        for label, value in (
            ("语音活动", "voice_activity"),
            ("声源方向", "source_direction"),
            ("噪声事件", "noise_event"),
            ("排除区间", "exclusion"),
        ):
            self.annotation_type.addItem(label, value)
        annotation_form.addRow("标注类型", self.annotation_type)
        for label, placeholder in (
            ("标注内容", "例如：有人声、无人声、无法确定"),
            ("物理方向（度，可留空）", "0到359"),
            ("置信度", "0到1，例如：0.9"),
            ("标注员", "匿名编号，例如：标注员-a"),
            ("新版本号", "例如：v0001；已有版本不能覆盖"),
        ):
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            self.annotation_fields.append(edit)
            annotation_form.addRow(label, edit)
        annotation_layout.addLayout(annotation_form)
        save = QPushButton("保存这个新标注版本")
        save.clicked.connect(self._save_annotation)
        annotation_layout.addWidget(save)
        annotation_layout.addStretch()
        sections.addTab(annotation_page, "人工标注")

        summary_page = QWidget()
        summary_layout = QVBoxLayout(summary_page)
        summary_layout.addWidget(QLabel("这里汇总正式数据的数量、总时长、质量状态和训练/验证/测试用途分布。"))
        self.stats_cards = QLabel("正在读取统计信息…")
        self.stats_cards.setAlignment(Qt.AlignTop)
        self.stats_cards.setTextInteractionFlags(Qt.TextSelectableByMouse)
        summary_layout.addWidget(self.stats_cards)
        summary_layout.addStretch()
        sections.addTab(summary_page, "数据概况")
        self.quality_sections = sections
        self.tabs.addTab(page, "质量与标注")

    def _storage_tab(self):
        page = QWidget()
        self.pages["maintenance"] = page
        layout = QVBoxLayout(page)
        title, intro = self._page_heading(
            "系统维护",
            "这里是低频使用的高级功能：查看磁盘和恢复状态、重建查询索引，以及创建不可变实验快照。日常录音无需操作本页。",
        )
        layout.addWidget(title)
        layout.addWidget(intro)
        sections = QTabWidget()
        layout.addWidget(sections)

        storage_page = QWidget()
        storage_layout = QVBoxLayout(storage_page)
        self.storage_output = QTextEdit()
        self.storage_output.setReadOnly(True)
        self.storage_output.setPlaceholderText("磁盘用量、恢复状态和回收站记录会显示在这里。")
        storage_layout.addWidget(self.storage_output)
        buttons = QHBoxLayout()
        rebuild = QPushButton("从资产清单重建查询索引")
        rebuild.setToolTip("资产文件不会改变，只重新生成用于快速查询的本地索引。")
        rebuild.clicked.connect(lambda: self._job(self.service.rebuild_catalog, self._show_storage))
        restore = QPushButton("从回收站恢复…")
        restore.clicked.connect(self._restore_trash)
        recovery = QPushButton("扫描未完整文件")
        recovery.clicked.connect(lambda: self._job(self.service.recovery_status, self._show_storage))
        quarantine = QPushButton("隔离未完整文件")
        quarantine.setToolTip("把意外中断留下的.partial文件移动到待处理区，不删除数据。")
        quarantine.clicked.connect(lambda: self._job(self.service.quarantine_partials, self._show_storage))
        buttons.addWidget(rebuild)
        buttons.addWidget(restore)
        buttons.addWidget(recovery)
        buttons.addWidget(quarantine)
        buttons.addStretch()
        storage_layout.addLayout(buttons)
        sections.addTab(storage_page, "存储与恢复")

        experiments_page = QWidget()
        experiments_layout = QVBoxLayout(experiments_page)
        experiments_layout.addWidget(
            QLabel("实验快照用于固定某次实验使用的数据集版本、配置、模型和样本。创建后相关版本不可原地修改。")
        )
        form = QFormLayout()
        self.experiment_name = QLineEdit()
        self.experiment_name.setPlaceholderText("例如：MVDR基线实验-01")
        self.experiment_dataset = QLineEdit()
        self.experiment_dataset.setPlaceholderText("已锁定的数据集编号")
        self.experiment_dataset_version = QLineEdit("0.1.0")
        self.experiment_config_hash = QLineEdit()
        self.experiment_config_hash.setPlaceholderText("本次运行的配置哈希")
        self.experiment_model_version = QLineEdit()
        self.experiment_model_version.setPlaceholderText("例如：voice-cnn-v1")
        self.experiment_recordings = QLineEdit()
        self.experiment_recordings.setPlaceholderText("多个样本编号用逗号分隔")
        self.experiment_notes = QTextEdit()
        self.experiment_notes.setPlaceholderText("实验目的、环境和备注")
        for label, widget in (
            ("实验名称 *", self.experiment_name),
            ("数据集编号 *", self.experiment_dataset),
            ("数据集版本 *", self.experiment_dataset_version),
            ("配置哈希 *", self.experiment_config_hash),
            ("模型版本 *", self.experiment_model_version),
            ("使用的样本编号 *", self.experiment_recordings),
            ("备注", self.experiment_notes),
        ):
            form.addRow(label, widget)
        experiments_layout.addLayout(form)
        create = QPushButton("创建不可变实验快照")
        create.clicked.connect(self._create_experiment)
        experiments_layout.addWidget(create)
        self.experiments_output = QTextEdit()
        self.experiments_output.setReadOnly(True)
        self.experiments_output.setPlaceholderText("已有实验快照会显示在这里。")
        experiments_layout.addWidget(self.experiments_output)
        sections.addTab(experiments_page, "实验快照")
        self.maintenance_sections = sections
        self.tabs.addTab(page, "系统维护")

    def _job(self, fn: Callable[[], Any], done: Callable[[Any], None]) -> None:
        if self._closing:
            return
        job = _Job(fn)
        self._jobs.add(job)

        def completed(value: Any) -> None:
            try:
                if not self._closing:
                    done(value)
            finally:
                self._jobs.discard(job)

        def failed(text: str) -> None:
            try:
                if not self._closing:
                    QMessageBox.critical(self, "操作失败", text)
            finally:
                self._jobs.discard(job)

        job.signals.done.connect(completed)
        job.signals.failed.connect(failed)
        self.pool.start(job)

    def refresh_all(self):
        self._job(self.service.dashboard, self._show_dashboard)
        self._job(self.service.runtime_sessions, self._load_sessions)
        self.refresh_recordings()
        self._job(
            self.service.experiments,
            lambda rows: self.experiments_output.setText(json.dumps(rows, ensure_ascii=False, indent=2)),
        )

    def refresh_recordings(self):
        self._job(self.service.recordings, self.corpus_table.load)

    def _runtime_session_selected(self) -> None:
        session_id = self.runtime_table.selected_id()
        if not session_id:
            self.runtime_track_table.load([])
            self.runtime_direction_status.setText("请选择一条运行录音。")
            return
        self.runtime_direction_status.setText("正在读取方向ID时间线……")
        self._job(
            lambda: self.service.runtime_session_tracks(session_id),
            lambda rows: self._load_runtime_tracks(session_id, rows),
        )

    def _load_runtime_tracks(self, session_id: str, rows: list[dict[str, Any]]) -> None:
        if self.runtime_table.selected_id() != session_id:
            return
        display_rows = [
            {**row, "track_key": f"{row['stream_epoch']}:{row['track_id']}"}
            for row in rows
        ]
        self._runtime_track_rows = display_rows
        self.runtime_track_table.load(display_rows)
        self.runtime_direction_status.setText(
            f"方向ID数量：{len(rows)}"
            if rows
            else "无算法方向ID：该录音是旧v3、L1-only，或录制期间没有形成公共方向轨迹。"
        )

    def _selected_runtime_track(self) -> tuple[str, int, int] | None:
        session_id = self.runtime_table.selected_id()
        track_key = self.runtime_track_table.selected_id()
        if not session_id or not track_key or ":" not in track_key:
            return None
        epoch_text, track_text = track_key.split(":", 1)
        return session_id, int(epoch_text), int(track_text)

    def _listen_runtime_track(self) -> None:
        selected = self._selected_runtime_track()
        if selected is None:
            return self._info("请先选择运行录音和其中一个方向ID。")
        session_id, epoch, track_id = selected
        self.channel_player.stop()
        self._job(
            lambda: self.service.track_audio_assets(session_id, epoch, track_id),
            lambda assets: self._play_runtime_track(track_id, assets),
        )

    def _play_runtime_track(self, track_id: int, assets: list[dict[str, Any]]) -> None:
        try:
            self.channel_player.play_track_assets(assets)
        except (OSError, ValueError) as exc:
            return QMessageBox.warning(self, "无法试听方向ID", str(exc))
        self.statusBar().showMessage(f"正在试听方向ID {track_id} 的连续增强音频", 6000)

    def _listen_runtime_center(self) -> None:
        session_id = self.runtime_table.selected_id()
        if not session_id:
            return self._info("请先选择一条运行录音。")
        self._listen_runtime_assets(session_id, "logical_8ch", 6, "Center参考")

    def _update_runtime_audio_channels(self, _index: int | None = None) -> None:
        _kind, channel_count = self.runtime_audio_kind.currentData()
        self.runtime_audio_channel.clear()
        for index in range(int(channel_count)):
            self.runtime_audio_channel.addItem(f"通道 {index + 1}", index)

    def _listen_runtime_raw_channel(self) -> None:
        session_id = self.runtime_table.selected_id()
        if not session_id:
            return self._info("请先选择一条运行录音。")
        kind, _channel_count = self.runtime_audio_kind.currentData()
        channel = int(self.runtime_audio_channel.currentData())
        self._listen_runtime_assets(session_id, str(kind), channel, self.runtime_audio_kind.currentText())

    def _listen_runtime_assets(self, session_id: str, kind: str, channel: int, label: str) -> None:
        self.channel_player.stop()
        self._job(
            lambda: self.service.session_audio_assets(session_id, kind),
            lambda assets: self._play_runtime_assets(label, channel, assets),
        )

    def _play_runtime_assets(self, label: str, channel: int, assets: list[dict[str, Any]]) -> None:
        try:
            paths = [item["absolute_path"] for item in assets]
            channel_count = int(assets[0]["channel_count"]) if assets else 0
            self.channel_player.play_files(paths, channel=channel, channel_count=channel_count)
        except (OSError, ValueError, KeyError) as exc:
            return QMessageBox.warning(self, "无法试听运行录音", str(exc))
        self.statusBar().showMessage(f"正在试听 {label} · 通道 {channel + 1}", 6000)

    def _load_sessions(self, rows):
        self._runtime_rows = rows
        self._filter_runtime()

    def _filter_runtime(self):
        query = self.runtime_search.text().strip().lower()
        self.runtime_table.load([x for x in getattr(self, "_runtime_rows", []) if query in x["id"].lower()])

    def _show_dashboard(self, data):
        self._latest_dashboard = data
        gb = 1024**3
        free_gb = data["free_bytes"] / gb
        self.disk_label.setText(f"磁盘余量：{free_gb:.1f} GB")
        # Native 8ch PCM16 + physical 7ch PCM16 + physical 7ch float32.
        hours = data["free_bytes"] / ((8 * 2 + 7 * 2 + 7 * 4) * 48_000) / 3600
        self.capacity_label.setText(f"预计可录：约 {hours:.1f} 小时")
        s = data["statistics"]
        status_names = {
            "pending": "待检查",
            "passed": "检查通过",
            "failed": "检查未通过",
            "quarantine": "待处理区",
            "annotated": "已标注",
            "versioned": "已纳入版本",
        }
        split_names = {
            "train": "训练集",
            "validation": "验证集",
            "test": "测试集",
            "calibration": "校准集",
            "unset": "尚未分配",
        }
        by_status = (
            "，".join(f"{status_names.get(key, key)} {value}" for key, value in s["by_status"].items()) or "暂无"
        )
        by_split = "，".join(f"{split_names.get(key, key)} {value}" for key, value in s["by_split"].items()) or "暂无"
        self.home_summary.setText(
            f"运行录音：{data['sessions']} 个　|　测试样本：{data['recordings']} 个　|　测试语料总时长：{s['duration_hours']:.2f} 小时\n"
            f"质量状态：{by_status}\n数据用途：{by_split}\n最后刷新：{datetime.now().strftime('%H:%M:%S')}"
        )
        self.stats_cards.setText(
            f"运行录音数量：{data['sessions']}\n测试样本数量：{data['recordings']}\n测试语料总时长：{s['duration_hours']:.2f} 小时\n\n"
            f"质量状态：{by_status}\n\n数据用途：{by_split}"
        )
        self.storage_output.setText(
            f"磁盘余量：{data['free_bytes'] / gb:.1f} GB\n正式数据占用：{data['storage_bytes'] / gb:.2f} GB\n"
            f"运行录音：{data['sessions']} 个\n测试样本：{data['recordings']} 个\n\n"
            "如需检查意外中断文件，请点击“扫描未完整文件”。"
        )

    def _show_quick_help(self):
        QMessageBox.information(
            self,
            "快速使用说明",
            "第一次使用建议按下面顺序：\n\n"
            "1. 如果要日常录音：点击顶部“连接麦克风”，再选择“手动录音”。\n"
            "2. 如果要制作正式测试样本：进入“测试录制向导”，填写样本信息与授权，按阶段提示操作。\n"
            "3. 如果已经有WAV文件：进入“测试语料库”，点击“导入外部WAV”。\n"
            "4. 样本准备好后：在“质量与标注”运行检查、添加标注，最后发布并锁定数据集版本。\n\n"
            "“系统维护”仅用于恢复文件、重建索引和创建实验快照。",
        )

    def set_capture_connected(self, connected: bool, message: str | None = None) -> None:
        """Called by the production capture host when authoritative ingest is available."""
        self.capture_connected = connected
        self.connect_button.setText("断开麦克风" if connected else "连接麦克风")
        self.connect_button.setEnabled(True)
        self.mode_select.setEnabled(connected)
        self.wizard_start.setEnabled(self.service.wizard.phase.value in {"idle", "complete", "error"})
        if connected and self._pending_wizard_input is not None:
            pending = self._pending_wizard_input
            self._pending_wizard_input = None
            self._begin_wizard_recording(pending)
        self.connection_badge.setText("● 采集源已连接" if connected else "● 采集源未连接")
        self.connection_badge.setStyleSheet(f"color:{'#86efac' if connected else '#fde68a'}")
        self.capture_notice.setText(
            message
            or (
                "麦克风采集源已连接：可以进行运行录音和正式测试录制。"
                if connected
                else "当前未连接麦克风采集主程序：可以浏览、导入、质检和维护数据；实际录音按钮暂不可用。"
            )
        )
        self._update_recording_buttons("idle")

    def _toggle_capture(self):
        self.connect_button.setEnabled(False)
        if self.capture_host.connected:
            self.connection_badge.setText("● 正在断开…")
            self.capture_host.stop()
        else:
            self.connection_badge.setText("● 正在连接…")
            self.capture_notice.setText("正在打开配置的8通道麦克风，请稍候…")
            self.capture_host.start()

    def _capture_connection_changed(self, connected: bool, message: str):
        self.set_capture_connected(connected, message)

    @Slot(object)
    def _apply_runtime_status(self, item: dict[str, Any]) -> None:
        """Apply capture-thread status from the Qt GUI thread."""
        self.update_recording_status(**item)

    def _capture_error(self, message: str):
        self._pending_wizard_input = None
        self.connect_button.setEnabled(True)
        self.capture_notice.setText(f"连接或采集失败：{message}\n请检查麦克风USB连接、设备名称和主机接口配置。")
        self.capture_notice.setObjectName("warningText")
        self.statusBar().showMessage(f"采集错误：{message}", 10000)

    def _apply_wizard_status(self, status):
        self.wizard_status.setText(f"当前状态：{status.message}")
        recording = status.phase.value == "recording"
        paused = status.phase.value == "paused"
        self.wizard_start.setEnabled(status.phase.value in {"idle", "complete", "error"})
        self.wizard_pause.setEnabled(recording or paused)
        self.wizard_pause.setText("继续录制" if paused else "暂停录制")
        self.wizard_stop.setEnabled(recording or paused)

    def _update_recording_buttons(self, state: str) -> None:
        manual = self.mode_select.currentData() == "manual"
        connected = self.capture_connected
        self.recording_buttons["record"].setEnabled(connected and manual and state in {"idle", "paused"})
        self.recording_buttons["record"].setText("继续录音" if state == "paused" else "开始录音")
        self.recording_buttons["pause"].setEnabled(connected and manual and state == "recording")
        self.recording_buttons["stop"].setEnabled(connected and state in {"recording", "paused", "continuous", "event"})

    def _prepare_annotation(self):
        recording_id = self.corpus_table.selected_id()
        if not recording_id:
            return self._info("请先在测试语料库的表格中选中一个样本。")
        self.annotation_fields[0].setText(recording_id)
        self.current_sample_label.setText(f"当前样本：{recording_id}")
        self._go("quality")
        self.quality_sections.setCurrentIndex(1)

    def _open_selected_qa(self):
        recording_id = self.corpus_table.selected_id()
        if not recording_id:
            return self._info("请先在测试语料库的表格中选中一个样本。")
        self.annotation_fields[0].setText(recording_id)
        self.current_sample_label.setText(f"当前样本：{recording_id}")
        self._go("quality")
        self.quality_sections.setCurrentIndex(0)
        self._run_qa()

    def _show_storage(self, data):
        if isinstance(data, dict) and "partial_files" in data:
            partials = data.get("partial_files", [])
            trash = data.get("trash_operations", [])
            recoverable = data.get("recoverable_sessions", [])
            text = (
                f"扫描完成\n\n未完整保存文件：{len(partials)} 个\n需要检查的运行录音：{len(recoverable)} 个\n"
                f"回收站可恢复操作：{len(trash)} 个"
            )
            if partials:
                text += "\n\n未完整文件：\n" + "\n".join(partials)
        elif isinstance(data, dict):
            text = (
                f"查询索引重建完成。\n运行录音：{data.get('sessions', 0)} 个\n测试样本：{data.get('recordings', 0)} 个"
            )
        elif isinstance(data, list):
            text = f"已将 {len(data)} 个未完整文件移动到待处理区。\n" + "\n".join(data)
        else:
            text = str(data)
        self.storage_output.setText(text)
        self.statusBar().showMessage("维护操作完成", 5000)

    def _wizard_input(self) -> WizardInput:
        environment = self.wizard_fields["environment"].text().strip()
        if not environment:
            raise ValueError("请填写环境")
        source_count = self.wizard_source_count.value()
        source_categories = tuple(row[1].text().strip() for row in self.wizard_source_rows)
        source_movements = tuple(row[2].text().strip() for row in self.wizard_source_rows)
        for index, (source_type, movement) in enumerate(
            zip(source_categories, source_movements, strict=True), 1
        ):
            if not source_type:
                raise ValueError(f"请填写声源 {index} 的类型")
            if not movement:
                raise ValueError(f"请填写声源 {index} 的移动方式")
        noise_source = self.wizard_fields["noise_source"].text().strip()
        if not noise_source:
            raise ValueError("请填写噪音来源；没有噪音时请填写“无”")
        recording_name = build_corpus_display_name(
            environment,
            datetime.now().astimezone(),
            source_count,
            source_categories,
            source_movements,
            noise_source,
        )
        return WizardInput(
            dataset_id="test-recordings",
            room_id="unspecified",
            environment_id=environment,
            array_pose_id="r6plus1-default",
            source_count=source_count,
            consent_status="not_applicable",
            allowed_uses=("internal_research",),
            recording_name=recording_name,
            source_categories=source_categories,
            source_movements=source_movements,
            noise_source=noise_source,
        )

    def _validate_wizard(self):
        try:
            data = self._wizard_input()
            errors = validate_wizard(data)
            QMessageBox.warning(self, "无法开始", "\n".join(errors)) if errors else QMessageBox.information(
                self, "门禁通过", "元数据与权利检查通过，可以连接正式采集源并进行通道健康检查。"
            )
        except ValueError as exc:
            QMessageBox.warning(self, "字段格式错误", str(exc))

    def _start_wizard(self):
        try:
            wizard_input = self._wizard_input()
            errors = validate_wizard(wizard_input)
            if errors:
                return QMessageBox.warning(self, "无法开始", "\n".join(errors))
        except ValueError as exc:
            return QMessageBox.warning(self, "无法开始", str(exc))
        self.wizard_start.setEnabled(False)
        if not self.capture_host.connected:
            self._pending_wizard_input = wizard_input
            self.wizard_status.setText("当前状态：正在自动连接麦克风，连接成功后立即开始录制……")
            self.capture_host.start()
            return
        self._begin_wizard_recording(wizard_input)

    def _begin_wizard_recording(self, wizard_input: WizardInput):
        try:
            status = self.service.wizard.begin(wizard_input)
        except RuntimeError as exc:
            self.wizard_start.setEnabled(True)
            return QMessageBox.warning(self, "无法开始", str(exc))
        self._apply_wizard_status(status)

    def _pause_or_resume_wizard(self):
        try:
            status = (
                self.service.wizard.resume()
                if self.service.wizard.phase.value == "paused"
                else self.service.wizard.pause()
            )
        except RuntimeError as exc:
            return QMessageBox.warning(self, "无法切换录制状态", str(exc))
        self._apply_wizard_status(status)

    def _finish_wizard_recording(self):
        try:
            status = self.service.wizard.finish()
        except RuntimeError as exc:
            return QMessageBox.warning(self, "无法结束录制", str(exc))
        self._apply_wizard_status(status)
        self.wizard_pause.setEnabled(False)
        self.wizard_stop.setEnabled(False)
        self._job(self.service.wizard.finalize, self._wizard_complete)

    def _wizard_complete(self, recording_id):
        name = self.service.wizard.input.recording_name if self.service.wizard.input is not None else recording_id
        self.wizard_status.setText(f"当前阶段：完成 · {name}")
        self.wizard_start.setEnabled(True)
        self.wizard_pause.setEnabled(False)
        self.wizard_stop.setEnabled(False)
        self.refresh_recordings()

    def _run_qa(self):
        rid = self.corpus_table.selected_id()
        if not rid and self.annotation_fields[0].text().strip():
            rid = self.annotation_fields[0].text().strip()
        if not rid:
            return self._info("请先在“测试语料库”中选择一个样本，再进入本页检查。")
        self.current_sample_label.setText(f"当前样本：{rid}")
        self._job(
            lambda: self.service.run_qa(rid),
            self._show_qa_result,
        )

    def _show_qa_result(self, result: dict[str, Any]):
        passed = result.get("status") == "passed"
        failures = result.get("failures", [])
        reports = result.get("audio_reports", [])
        text = (
            "结论：质量检查通过，可以继续标注和数据集发布。" if passed else "结论：需要处理，样本已保留，不会自动删除。"
        )
        if failures:
            names = {
                "clipping_ratio": "存在削波",
                "dc_offset": "直流偏置过大",
                "low_rms": "通道电平过低",
            }
            text += "\n\n发现的问题：\n" + "\n".join(f"• {names.get(item, item)}" for item in failures)
        if reports:
            report = reports[0]
            text += f"\n\n音频长度：{report.get('sample_count', 0) / 48_000:.2f} 秒"
            text += f"\n通道数量：{report.get('channel_count', 0)}"
            duplicates = report.get("suspected_duplicate_channels", [])
            if duplicates:
                text += f"\n疑似重复通道：{duplicates}"
        text += "\n\n技术详情已保存在样本目录的 qa_report.json 中。"
        self.qa_output.setText(text)

    def _run_leakage(self):
        self._job(
            self.service.leakage_report,
            self._show_leakage_result,
        )

    def _show_leakage_result(self, result: dict[str, Any]):
        if result.get("passed"):
            text = "结论：未发现房间、采集会话或匿名说话人跨训练/验证/测试用途泄漏。"
        else:
            text = "结论：发现数据泄漏，暂时不能发布数据集版本。\n\n" + "\n".join(
                f"• {item['field']}={item['value']} 同时出现在 {', '.join(item['splits'])}"
                for item in result.get("leaks", [])
            )
        self.qa_output.setText(text)

    def _lock_dataset(self):
        dataset_id, ok = QInputDialog.getText(self, "预览数据用途分配", "输入数据集编号：")
        if not ok or not dataset_id.strip():
            return
        self._job(lambda: self.service.preview_dataset_split(dataset_id.strip()), self._show_split_preview)

    def _show_split_preview(self, preview: dict[str, Any]):
        counts = preview["counts"]
        durations = preview["duration_seconds"]
        leakage = preview["leakage_report"]
        text = (
            f"数据集：{preview['dataset_id']}\n样本总数：{preview['recording_count']}\n\n"
            f"建议分配：\n• 训练集：{counts['train']} 条，{durations['train']:.1f} 秒\n"
            f"• 验证集：{counts['validation']} 条，{durations['validation']:.1f} 秒\n"
            f"• 测试集：{counts['test']} 条，{durations['test']:.1f} 秒\n\n"
            f"泄漏检查：{'通过' if leakage['passed'] else '未通过'}"
        )
        self.qa_output.setText(text)
        if preview.get("already_locked"):
            return QMessageBox.information(self, "该版本已锁定", "这个数据集已经发布并锁定，不能再次原地修改。")
        if not leakage["passed"]:
            return QMessageBox.warning(self, "不能发布", "检测到数据泄漏，请先调整样本分组。")
        version, ok = QInputDialog.getText(self, "发布数据集版本", "输入新版本号（例如1.0.0）：")
        if not ok or not version.strip():
            return
        answer = QMessageBox.warning(
            self,
            "最后确认：发布并锁定",
            "发布后，该版本中的样本、标注和数据用途不能原地修改或删除。\n"
            "如需修改，必须创建新的数据集版本。\n\n确认现在发布吗？",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if answer == QMessageBox.Yes:
            self._job(
                lambda: self.service.assign_and_lock_dataset(preview["dataset_id"], version.strip()),
                lambda result: self.qa_output.setText(
                    f"数据集版本 {result['version']} 已发布并锁定。\n样本数量：{len(result['recordings'])}"
                ),
            )

    def _save_annotation(self):
        values = [field.text().strip() for field in self.annotation_fields]
        if not values[0] and self.corpus_table.selected_id():
            values[0] = self.corpus_table.selected_id() or ""
            self.annotation_fields[0].setText(values[0])
        try:
            start_seconds = float(values[1])
            end_seconds = float(values[2])
            if start_seconds < 0 or end_seconds <= start_seconds:
                raise ValueError("结束时间必须大于开始时间，且开始时间不能为负数")
            annotation = Annotation(
                str(uuid.uuid4()),
                values[0],
                round(start_seconds * 48_000),
                round(end_seconds * 48_000),
                self.annotation_type.currentData(),
                values[3],
                None if not values[4] else float(values[4]),
                float(values[5]),
                values[6],
                values[7],
            )
        except (ValueError, IndexError) as exc:
            return QMessageBox.warning(self, "标注字段错误", str(exc))
        self._job(lambda: self.service.add_annotation(annotation), lambda _: self._annotation_saved())

    def _annotation_saved(self):
        self.statusBar().showMessage("新标注版本已原子保存", 5000)
        self.refresh_recordings()

    def _mode_changed(self, mode: str):
        self.recording_command.emit(f"mode:{mode}")
        color = {"off": "#9ca3af", "manual": "#f59e0b", "continuous": "#ef4444", "event": "#22c55e"}[mode]
        label = {"off": "录音已关闭", "manual": "手动录音待命", "continuous": "连续录音", "event": "人声事件监听"}[mode]
        self.rec_badge.setText(f"● {label}")
        self.rec_badge.setStyleSheet(f"color:{color}")
        self._update_recording_buttons("idle" if mode in {"off", "manual"} else mode)

    def _recording_action(self, command: str):
        mode = self.mode_select.currentData()
        if command in {"record", "pause"} and mode != "manual":
            return self._info("“开始/暂停”只用于手动录音模式；连续录音和人声事件录音由模式规则自动控制。")
        self.recording_command.emit(command)
        if command == "record":
            self.rec_badge.setText("● 正在录音")
            self.rec_badge.setStyleSheet("color:#ef4444")
            self._update_recording_buttons("recording")
        elif command in {"pause", "stop"}:
            self.rec_badge.setText("● 已暂停" if command == "pause" else "● 当前片段已停止")
            self.rec_badge.setStyleSheet("color:#f59e0b" if command == "pause" else "color:#9ca3af")
            self._update_recording_buttons("paused" if command == "pause" else "idle")

    def update_recording_status(
        self, *, mode: str, session_id: str | None, duration_seconds: float, free_bytes: int
    ) -> None:
        self.session_label.setText(f"当前会话：{session_id or '—'}")
        seconds = max(0, int(duration_seconds))
        self.duration_label.setText(f"时长：{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}")
        self.disk_label.setText(f"磁盘余量：{free_bytes / 1024**3:.1f} GB")
        if self.mode_select.currentData() != mode:
            self.mode_select.blockSignals(True)
            index = self.mode_select.findData(mode)
            if index >= 0:
                self.mode_select.setCurrentIndex(index)
            self.mode_select.blockSignals(False)

    def _trash_session(self):
        sid = self.runtime_table.selected_id()
        if not sid:
            return self._info("请先在表格中选择一条运行录音。")
        if (
            QMessageBox.question(
                self,
                "确认移到回收站",
                f"运行录音 {sid} 将移到可恢复的回收站。\n原始文件不会立即永久删除。\n\n继续吗？",
            )
            == QMessageBox.Yes
        ):
            self._job(lambda: self.service.trash("session", sid), lambda _: self.refresh_all())

    def _trash_recording(self):
        recording_id = self.corpus_table.selected_id()
        if not recording_id:
            return self._info("请先在表格中选择一个测试样本。")
        row = next((item for item in self.service.recordings() if item["id"] == recording_id), None)
        if row is None:
            return
        answer = QMessageBox.warning(
            self,
            "确认移到回收站",
            f"样本编号：{recording_id}\n数据集编号：{row['dataset_id']}\n\n"
            "样本将进入可恢复的回收站。若它属于已锁定版本，系统会拒绝操作。",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return
        self._job(
            lambda: self.service.trash("recording", recording_id),
            lambda _: self._trash_complete("测试样本已移到可恢复的回收站。"),
        )

    def _rename_selected_recording(self):
        recording_id = self.corpus_table.selected_id()
        if not recording_id:
            return self._info("请先在表格中选择一个测试样本。")
        row = next(
            (item for item in self.service.recordings() if item["id"] == recording_id),
            None,
        )
        if row is None:
            return self._info("所选测试样本已不存在，请刷新列表。")
        current_name = str(row.get("display_name") or "")
        requested_name, ok = QInputDialog.getText(
            self,
            "修改音频名称",
            "新的标注名称：",
            QLineEdit.Normal,
            current_name,
        )
        if not ok:
            return None
        self._job(
            lambda: self.service.rename_recording(recording_id, requested_name),
            lambda name: self._recording_renamed(recording_id, name),
        )

    def _recording_renamed(self, recording_id: str, name: str) -> None:
        def loaded(rows: list[dict[str, Any]]) -> None:
            self.corpus_table.load(rows)
            self.corpus_table.select_id(recording_id)
            self.statusBar().showMessage(f"名称已修改：{name}", 5000)

        self._job(self.service.recordings, loaded)

    def show_default_window(self) -> None:
        """Open as a maximized normal window within the active screen."""
        self.showMaximized()

    def _trash_complete(self, message: str) -> None:
        self.refresh_all()
        self.statusBar().showMessage(message, 8000)

    def _restore_trash(self):
        operations = self.service.trash_operations()
        if not operations:
            return self._info("回收站中没有可恢复的操作。")
        type_names = {"session": "运行录音", "recording": "测试样本"}
        labels = [
            f"{type_names.get(item['entity_type'], item['entity_type'])} · {item['entity_id']}" for item in operations
        ]
        label, ok = QInputDialog.getItem(self, "从回收站恢复", "选择要恢复的内容：", labels, 0, False)
        if ok:
            operation_id = operations[labels.index(label)]["operation_id"]
            self._job(lambda: self.service.restore(operation_id), lambda _: self.refresh_all())

    def _export_path(self, source: str, suggested: str):
        target, _ = QFileDialog.getSaveFileName(self, "导出资产", suggested, "ZIP archive (*.zip)")
        if target:
            self._job(
                lambda: self.service.export([source], target),
                lambda path: self._info(f"已导出并写入校验清单：\n{path}"),
            )

    def _export_session(self):
        session_id = self.runtime_table.selected_id()
        row = next((item for item in self.service.runtime_sessions() if item["id"] == session_id), None)
        if row:
            self._export_path(row["path"], f"runtime-{session_id}.zip")
        else:
            self._info("请先选择一条运行录音。")

    def _export_recording(self):
        recording_id = self.corpus_table.selected_id()
        row = next((item for item in self.service.recordings() if item["id"] == recording_id), None)
        if row:
            self._export_path(row["path"], f"recording-{recording_id}.zip")
        else:
            self._info("请先选择一个测试样本。")

    def _simulate_selected_recording(self):
        recording_id = self.corpus_table.selected_id()
        row = next((item for item in self.service.recordings() if item["id"] == recording_id), None)
        if row is None:
            return self._info("请先在测试语料库中选择一个样本，再点击模拟测试。")
        root = Path(row["path"])
        try:
            manifest_path = (root / "recording_manifest.json").resolve(strict=True)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            kinds = {item.get("kind") for item in manifest.get("assets", [])}
            if "native_8ch" not in kinds:
                raise ValueError("该录音没有保存原始8通道音频")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return QMessageBox.warning(self, "无法启动模拟测试", f"所选录音不是完整麦克风输入：{exc}")
        project_root = Path(__file__).resolve().parents[2]
        self.channel_player.stop()
        try:
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "gui.dev_test_ui.app",
                    "--config",
                    str(project_root / "config" / "config.yaml"),
                    "--replay-recording",
                    str(manifest_path),
                    "--auto-start",
                ],
                cwd=project_root,
            )
        except OSError as exc:
            return QMessageBox.warning(self, "无法启动模拟测试", str(exc))
        self.statusBar().showMessage("Test UI已打开，正在实时模拟所选样本输入", 8000)
        self.showMinimized()

    def _listen_selected_channel(self):
        recording_id = self.corpus_table.selected_id()
        row = next((item for item in self.service.recordings() if item["id"] == recording_id), None)
        if row is None:
            return self._info("请先选择一条录音。")
        root = Path(row["path"])
        try:
            manifest = json.loads((root / "recording_manifest.json").read_text(encoding="utf-8"))
            asset = next(item for item in manifest.get("assets", ()) if item.get("kind") == "native_8ch")
            audio_path = (root / str(asset["path"])).resolve(strict=True)
            self.channel_player.play(audio_path, int(self.listen_channel.currentData()))
        except (OSError, ValueError, StopIteration, KeyError, json.JSONDecodeError) as exc:
            return QMessageBox.warning(self, "无法试听", str(exc))
        self.statusBar().showMessage(
            f"正在试听 {row.get('display_name', recording_id)} · 通道 {int(self.listen_channel.currentData()) + 1}",
            5000,
        )

    def _import_recording(self):
        source, _ = QFileDialog.getOpenFileName(self, "导入7/8通道48 kHz WAV", "", "WAV audio (*.wav)")
        if not source:
            return
        dialog = ImportMetadataDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            wizard = dialog.value()
            from data_management.contracts import RecordingMetadata
            from data_management.manifests import utc_now

            metadata = RecordingMetadata(
                wizard.dataset_id,
                "imported",
                utc_now(),
                wizard.environment_id,
                wizard.room_id,
                wizard.array_pose_id,
                wizard.source_count,
                wizard.source_categories,
                {
                    "consent_status": wizard.consent_status,
                    "license_id": wizard.license_id,
                    "allowed_uses": list(wizard.allowed_uses),
                    "expires_at_utc": wizard.expires_at_utc,
                },
                wizard.theta_degrees or None,
                wizard.distance_m or None,
                wizard.speaker_ids,
                wizard.language_tags,
            )
        except ValueError as exc:
            return QMessageBox.warning(self, "导入信息不完整", str(exc))
        self._job(
            lambda: self.service.import_recording(source, metadata),
            lambda recording_id: self._import_complete(recording_id),
        )

    def _import_complete(self, recording_id):
        self.statusBar().showMessage(f"导入和质量检查完成，样本编号：{recording_id}", 7000)
        self.refresh_recordings()

    def _create_experiment(self):
        values = {
            "name": self.experiment_name.text().strip(),
            "dataset_id": self.experiment_dataset.text().strip(),
            "dataset_version": self.experiment_dataset_version.text().strip(),
            "config_hash": self.experiment_config_hash.text().strip(),
            "model_version": self.experiment_model_version.text().strip(),
            "recording_ids": tuple(x.strip() for x in self.experiment_recordings.text().split(",") if x.strip()),
            "notes": self.experiment_notes.toPlainText().strip(),
        }
        self._job(
            lambda: self.service.create_experiment(**values),
            lambda experiment_id: self._experiment_created(experiment_id),
        )

    def _experiment_created(self, experiment_id):
        self.statusBar().showMessage(f"不可变实验快照已创建：{experiment_id}", 7000)
        self.refresh_all()

    def _info(self, text):
        QMessageBox.information(self, "Audio Data Manager", text)

    def closeEvent(self, event):
        self._closing = True
        if not self.capture_host.stop(timeout=1.0):
            self._closing = False
            event.ignore()
            QMessageBox.warning(
                self,
                "正在安全封存录音",
                "录音仍在安全封存中，窗口暂时不能关闭。请稍候几秒后再试。",
            )
            return
        self.channel_player.close()
        self.capture_host.close()
        self.pool.waitForDone(5000)
        self.service.close()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args(argv)
    app = QApplication(sys.argv)
    window = AudioDataManager(args.data_root)
    window.show_default_window()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
