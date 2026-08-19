from __future__ import annotations

import json
import os
import wave

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

from gui.production_ui.app import AudioDataManager, DataTable, ImportMetadataDialog


def app_instance() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_manager_uses_six_chinese_task_pages_and_clear_disconnected_state(tmp_path):
    app_instance()
    window = AudioDataManager(tmp_path)
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
        "操作首页",
        "运行录音",
        "测试语料库",
        "测试录制向导",
        "质量与标注",
        "系统维护",
    ]
    assert not window.mode_select.isEnabled()
    assert window.wizard_start.isEnabled()
    assert "未连接" in window.connection_badge.text()
    window.close()


def test_capture_connection_enables_only_valid_recording_actions(tmp_path):
    app_instance()
    window = AudioDataManager(tmp_path)
    window.set_capture_connected(True)
    manual = window.mode_select.findData("manual")
    window.mode_select.setCurrentIndex(manual)
    assert window.recording_buttons["record"].isEnabled()
    assert not window.recording_buttons["pause"].isEnabled()
    window._update_recording_buttons("recording")
    assert not window.recording_buttons["record"].isEnabled()
    assert window.recording_buttons["pause"].isEnabled()
    assert window.recording_buttons["stop"].isEnabled()
    assert window.mode_select.findData("event") == -1
    window.close()


def test_data_table_translates_internal_values_and_preserves_real_id():
    app_instance()
    table = DataTable([("id", "样本编号"), ("status", "状态"), ("split", "用途")])
    real_id = "12345678-1234-1234-1234-123456789abc"
    table.load([{"id": real_id, "status": "passed", "split": "train"}])
    table.selectRow(0)
    assert table.item(0, 1).text() == "检查通过"
    assert table.item(0, 2).text() == "训练集"
    assert table.selected_id() == real_id


def test_import_metadata_uses_chinese_allowed_use_choices():
    app_instance()
    dialog = ImportMetadataDialog()
    assert dialog.allowed_uses.currentText() == "仅用于内部研究"
    assert dialog.allowed_uses.currentData() == ("research",)
    dialog.close()


def test_annotation_form_uses_seconds_and_chinese_type_choices(tmp_path):
    app_instance()
    window = AudioDataManager(tmp_path)
    assert window.annotation_type.currentText() == "语音活动"
    assert window.annotation_type.currentData() == "voice_activity"
    assert window.annotation_fields[1].placeholderText() == "从0开始，例如：0.00"
    window.close()


def test_wizard_uses_chinese_allowed_use_choices(tmp_path):
    app_instance()
    window = AudioDataManager(tmp_path)
    assert set(window.wizard_fields) == {"recording_name", "notes"}
    assert window.wizard_start.text() == "开始录制"
    assert window.wizard_pause.text() == "暂停录制"
    assert window.wizard_stop.text() == "结束并保存"
    window.close()


def test_wizard_start_shows_validation_errors_instead_of_starting(tmp_path, monkeypatch):
    app_instance()
    window = AudioDataManager(tmp_path)
    messages: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, text: messages.append(text))
    window._start_wizard()
    assert messages
    assert window.service.wizard.phase.value == "idle"
    window.close()


def test_recording_page_has_only_requested_actions_and_can_listen_to_any_native_channel(tmp_path, monkeypatch):
    app_instance()
    window = AudioDataManager(tmp_path)
    root = tmp_path / "test_corpus" / "test-recordings" / "recordings" / "sample-listen"
    root.mkdir(parents=True)
    audio = root / "native_8ch.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(8)
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes(bytes(960 * 8 * 2))
    (root / "recording_manifest.json").write_text(
        json.dumps({"assets": [{"kind": "native_8ch", "path": audio.name}]}), encoding="utf-8"
    )
    row = {"id": "sample-listen", "display_name": "手工名称", "path": str(root)}
    monkeypatch.setattr(window.service, "recordings", lambda **_filters: [row])
    played = []
    monkeypatch.setattr(window.channel_player, "play", lambda path, channel: played.append((path, channel)))
    window.corpus_table.load([row])
    window.corpus_table.selectRow(0)
    window.listen_channel.setCurrentIndex(7)
    window._listen_selected_channel()
    assert played == [(audio.resolve(), 7)]
    button_texts = {button.text() for button in window.pages["corpus"].findChildren(QPushButton)}
    assert button_texts == {
        "试听所选通道", "停止试听", "用所选样本进行模拟测试", "移到回收站",
    }
    window.close()


def test_selected_database_sample_launches_complete_virtual_array_input(tmp_path, monkeypatch):
    app_instance()
    window = AudioDataManager(tmp_path)
    root = tmp_path / "test_corpus" / "test-recordings" / "recordings" / "sample-1"
    root.mkdir(parents=True)
    audio = root / "native_8ch.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(8)
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes(bytes(960 * 8 * 2))
    hotmaps = root / "hotmaps.jsonl"
    hotmaps.write_text("{}\n", encoding="utf-8")
    manifest_path = root / "recording_manifest.json"
    manifest_path.write_text(
        json.dumps({"assets": [
            {"kind": "native_8ch", "path": audio.name},
            {"kind": "cdc_hotmaps", "path": hotmaps.name},
        ]}), encoding="utf-8"
    )
    row = {"id": "sample-1", "path": str(root)}
    monkeypatch.setattr(window.service, "recordings", lambda **_filters: [row])
    launched = []
    monkeypatch.setattr("gui.production_ui.app.subprocess.Popen", lambda command, **options: launched.append((command, options)))
    window.corpus_table.load([row])
    window.corpus_table.selectRow(0)

    window._simulate_selected_recording()

    assert launched
    command = launched[0][0]
    assert command[1:3] == ["-m", "gui.dev_test_ui.app"]
    assert command[command.index("--replay-recording") + 1] == str(manifest_path.resolve())
    assert "--auto-start" in command
    window.close()


def test_close_defers_while_capture_is_still_stopping(tmp_path, monkeypatch):
    app_instance()
    window = AudioDataManager(tmp_path)
    warnings: list[str] = []
    real_stop = window.capture_host.stop
    monkeypatch.setattr(window.capture_host, "stop", lambda timeout=1.0: False)
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, text: warnings.append(text))
    window.close()
    assert window.isVisible() or warnings
    assert warnings and "安全封存" in warnings[0]
    monkeypatch.setattr(window.capture_host, "stop", real_stop)
    window.close()
