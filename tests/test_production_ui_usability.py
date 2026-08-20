from __future__ import annotations

import json
import os
import threading
import time
import wave

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton
from PySide6.QtCore import Qt

from gui.production_ui.app import AudioDataManager, DataTable, ImportMetadataDialog
from gui.production_ui.channel_player import NativeChannelPlayer


def app_instance() -> QApplication:
    return QApplication.instance() or QApplication([])


def process_events_until(predicate, timeout: float = 3.0) -> bool:
    app = app_instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(predicate())


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


def test_background_job_is_retained_until_its_ui_callback_finishes(tmp_path):
    app_instance()
    window = AudioDataManager(tmp_path)
    started = threading.Event()
    release = threading.Event()
    results = []

    def work():
        started.set()
        assert release.wait(3.0)
        return "loaded"

    window._job(work, results.append)
    assert started.wait(1.0)
    assert window._jobs
    release.set()
    assert process_events_until(lambda: results == ["loaded"])
    assert not window._jobs
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
    assert set(window.wizard_fields) == {"environment", "noise_source"}
    assert window.wizard_source_count.value() == 1
    assert len(window.wizard_source_rows) == 1
    assert window.wizard_start.text() == "开始录制"
    assert window.wizard_pause.text() == "暂停录制"
    assert window.wizard_stop.text() == "结束并保存"
    assert "无算法方向ID" in window.wizard_direction_id_status.text()
    window.close()


def test_runtime_detail_shows_public_tracks_and_plays_track_and_center(tmp_path, monkeypatch):
    app_instance()
    window = AudioDataManager(tmp_path)
    session_id = "session-v4"
    summary = {
        "session_id": session_id,
        "stream_epoch": 2,
        "track_id": 7,
        "first_sample": 960,
        "last_sample": 48_000,
        "duration_seconds": 1.0,
        "first_theta_deg": 359.0,
        "last_theta_deg": 1.0,
        "angle_change_deg": 2.0,
        "state": "confirmed",
        "latest_l4_probability": 0.91,
    }
    monkeypatch.setattr(window, "_job", lambda fn, done: done(fn()))
    monkeypatch.setattr(window.service, "runtime_session_tracks", lambda _sid: [summary])
    monkeypatch.setattr(
        window.service,
        "track_audio_assets",
        lambda _sid, _epoch, _track: [{"absolute_path": "track.wav", "decision_sample": 48_000}],
    )
    monkeypatch.setattr(
        window.service,
        "session_audio_assets",
        lambda _sid, _kind: [{"absolute_path": "logical.wav", "channel_count": 8}],
    )
    track_calls = []
    raw_calls = []
    monkeypatch.setattr(window.channel_player, "play_track_assets", lambda assets: track_calls.append(assets))
    monkeypatch.setattr(
        window.channel_player,
        "play_files",
        lambda paths, *, channel, channel_count: raw_calls.append((paths, channel, channel_count)),
    )

    window._load_sessions([{"id": session_id, "started_at": "", "ended_at": "", "mode": "continuous",
                            "status": "complete", "path": str(tmp_path)}])
    window.runtime_table.selectRow(0)
    assert window.runtime_track_table.rowCount() == 1
    assert "方向ID数量：1" in window.runtime_direction_status.text()
    window.runtime_track_table.selectRow(0)
    window._listen_runtime_track()
    window._listen_runtime_center()

    assert track_calls and track_calls[0][0]["absolute_path"] == "track.wav"
    assert raw_calls == [(["logical.wav"], 6, 8)]
    window.close()


def test_runtime_detail_explicitly_reports_no_algorithm_track_id(tmp_path, monkeypatch):
    app_instance()
    window = AudioDataManager(tmp_path)
    monkeypatch.setattr(window, "_job", lambda fn, done: done(fn()))
    monkeypatch.setattr(window.service, "runtime_session_tracks", lambda _sid: [])
    window._load_sessions([{"id": "l1-only", "started_at": "", "ended_at": "", "mode": "manual",
                            "status": "complete", "path": str(tmp_path)}])
    window.runtime_table.selectRow(0)
    assert "无算法方向ID" in window.runtime_direction_status.text()
    window.close()


def test_track_player_removes_320ms_overlap_between_decision_assets(tmp_path, monkeypatch):
    paths = []
    for index in range(2):
        path = tmp_path / f"track-{index}.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(48_000)
            output.writeframes(bytes(15_360 * 2))
        paths.append(path)
    player = NativeChannelPlayer()
    clips = []
    monkeypatch.setattr(player, "_start_clips", lambda values: clips.extend(values))

    player.play_track_assets([
        {"absolute_path": str(paths[0]), "decision_sample": 15_360},
        {"absolute_path": str(paths[1]), "decision_sample": 16_320},
    ])

    assert (clips[0].start_frame, clips[0].frame_count) == (0, 15_360)
    assert (clips[1].start_frame, clips[1].frame_count, clips[1].silence_before) == (14_400, 960, 0)


def test_wizard_builds_one_type_and_movement_row_per_source(tmp_path):
    app_instance()
    window = AudioDataManager(tmp_path)
    window.wizard_fields["environment"].setText("诊室")
    window.wizard_fields["noise_source"].setText("空调")
    window.wizard_source_count.setValue(2)
    assert len(window.wizard_source_rows) == 2
    window.wizard_source_rows[0][1].setText("医生人声")
    window.wizard_source_rows[0][2].setText("静止")
    window.wizard_source_rows[1][1].setText("患者人声")
    window.wizard_source_rows[1][2].setText("左右走动")

    data = window._wizard_input()

    assert data.environment_id == "诊室"
    assert data.source_count == 2
    assert data.source_categories == ("医生人声", "患者人声")
    assert data.source_movements == ("静止", "左右走动")
    assert data.noise_source == "空调"
    assert data.recording_name.startswith("诊室 · 2个声源 · ")
    window.close()


def test_wizard_start_shows_validation_errors_instead_of_starting(tmp_path, monkeypatch):
    app_instance()
    window = AudioDataManager(tmp_path)
    messages: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, text: messages.append(text))
    window._start_wizard()
    assert messages
    assert messages == ["请填写环境"]
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


def test_default_desktop_window_is_maximized_but_not_fullscreen(tmp_path):
    app = app_instance()
    window = AudioDataManager(tmp_path)
    assert window.size().width() <= 1200
    assert window.size().height() <= 760

    window.show_default_window()
    app.processEvents()

    assert window.windowState() & Qt.WindowState.WindowMaximized
    assert not window.isFullScreen()
    assert window.tabs.tabBar().expanding()
    window.close()


def test_selected_recording_moves_to_recoverable_trash_and_disappears(tmp_path, monkeypatch):
    app_instance()
    window = AudioDataManager(tmp_path)
    dataset = "test-recordings"
    recording_id = "sample-to-trash"
    root = tmp_path / "test_corpus" / dataset / "recordings" / recording_id
    root.mkdir(parents=True)
    manifest = {
        "schema_version": "test_recording_v1",
        "dataset_id": dataset,
        "recording_id": recording_id,
        "display_name": "待删除录音",
        "source_type": "dedicated",
        "quality_status": "pending",
        "split": "unset",
        "assets": [],
    }
    (root / "recording_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    window.service.catalog.upsert_dataset(dataset, root.parents[1])
    window.service.catalog.upsert_recording(manifest, root)
    window.corpus_table.load(window.service.recordings())
    window.corpus_table.selectRow(0)
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: QMessageBox.Yes)
    monkeypatch.setattr(window, "_job", lambda fn, done: done(fn()))

    window._trash_recording()

    assert not root.exists()
    assert window.corpus_table.rowCount() == 1
    assert window.corpus_table.selected_id() is None
    assert "还没有录音" in window.corpus_table.item(0, 0).text()
    operations = window.service.trash_operations()
    assert len(operations) == 1
    window.service.restore(operations[0]["operation_id"])
    assert root.exists()
    assert [row["id"] for row in window.service.recordings()] == [recording_id]
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
