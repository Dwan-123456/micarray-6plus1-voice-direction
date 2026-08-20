from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data_management.catalog import Catalog
from data_management.corpus_naming import (
    build_corpus_display_name,
    migrate_corpus_display_names,
    rename_corpus_recording,
    validate_manual_display_name,
)
from data_management.manifests import atomic_json, sha256_file, write_manifest


def test_display_name_contains_every_requested_component():
    name = build_corpus_display_name(
        "会议室",
        datetime(2026, 8, 20, 10, 29, tzinfo=timezone(timedelta(hours=8))),
        2,
        ("医生人声", "患者人声"),
        ("移动", "静止"),
        "风扇、空调",
    )

    assert name == (
        "会议室 · 0820-1029 · 2个声源 · "
        "声源1：医生人声（移动）；声源2：患者人声（静止） · 噪音：风扇、空调"
    )


def test_migration_updates_manifest_labels_hash_catalog_and_legacy_metadata(tmp_path: Path):
    recording_id = "legacy-recording"
    root = tmp_path / "test_corpus" / "test-recordings" / "recordings" / recording_id
    root.mkdir(parents=True)
    labels_path = atomic_json(
        root / "labels.json",
        {
            "schema_version": "test_recording_labels_v2",
            "recording_name": "会议室-单人声固定声源-风扇背景噪音",
            "duration_seconds": 1.0,
            "recorded_intervals": [],
        },
    )
    manifest = {
        "schema_version": "raw_microphone_recording_v1",
        "dataset_id": "test-recordings",
        "recording_id": recording_id,
        "display_name": "会议室-单人声固定声源-风扇背景噪音",
        "capture_time_utc": "2026-08-19T04:19:32Z",
        "source_type": "dedicated",
        "environment_id": "unspecified",
        "room_id": "unspecified",
        "source_count": 0,
        "source_categories": ["human_voice"],
        "source_movements": [],
        "noise_source": "",
        "quality_status": "passed",
        "split": "unset",
        "duration_samples": 48_000,
        "assets": [
            {"kind": "labels", "path": "labels.json", "sha256": sha256_file(labels_path)}
        ],
    }
    write_manifest(root / "recording_manifest.json", manifest)

    changes = migrate_corpus_display_names(tmp_path)

    assert len(changes) == 1
    updated = json.loads((root / "recording_manifest.json").read_text(encoding="utf-8"))
    expected = "会议室 · 0819-1219 · 1个声源 · 声源1：人声（固定） · 噪音：风扇"
    assert updated["display_name"] == expected
    assert updated["environment_id"] == "会议室"
    assert updated["source_count"] == 1
    assert updated["source_categories"] == ["人声"]
    assert updated["source_movements"] == ["固定"]
    assert updated["noise_source"] == "风扇"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert labels["schema_version"] == "test_recording_labels_v3"
    assert labels["recording_name"] == expected
    assert labels["sources"] == [{"index": 1, "movement": "固定", "type": "人声"}]
    assert updated["assets"][0]["sha256"] == sha256_file(labels_path)
    catalog = Catalog(tmp_path / "catalog.sqlite")
    try:
        row = next(item for item in catalog.list_recordings() if item["id"] == recording_id)
        assert json.loads(row["metadata_json"])["display_name"] == expected
    finally:
        catalog.close()


def test_manual_rename_updates_labels_hash_manifest_catalog_and_audit(tmp_path: Path):
    recording_id = "rename-recording"
    root = tmp_path / "test_corpus" / "test-recordings" / "recordings" / recording_id
    root.mkdir(parents=True)
    labels_path = atomic_json(
        root / "labels.json",
        {"schema_version": "test_recording_labels_v3", "recording_name": "原名称"},
    )
    manifest = {
        "schema_version": "raw_microphone_recording_v1",
        "dataset_id": "test-recordings",
        "recording_id": recording_id,
        "display_name": "原名称",
        "source_type": "dedicated",
        "quality_status": "passed",
        "split": "unset",
        "duration_samples": 48_000,
        "assets": [
            {"kind": "labels", "path": "labels.json", "sha256": sha256_file(labels_path)}
        ],
    }
    write_manifest(root / "recording_manifest.json", manifest)
    catalog = Catalog(tmp_path / "catalog.sqlite")
    try:
        catalog.upsert_dataset("test-recordings", root.parents[1])
        catalog.upsert_recording(manifest, root)
        result = rename_corpus_recording(
            tmp_path, recording_id, " 会议室双人对话（复核） ", catalog=catalog
        )

        assert result == "会议室双人对话（复核）"
        updated = json.loads((root / "recording_manifest.json").read_text(encoding="utf-8"))
        assert updated["display_name"] == result
        assert updated["display_name_source"] == "manual"
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        assert labels["recording_name"] == result
        assert updated["assets"][0]["sha256"] == sha256_file(labels_path)
        catalog_row = catalog.list_recordings()[0]
        assert json.loads(catalog_row["metadata_json"])["display_name"] == result
        actions = [
            row[0]
            for row in catalog.connection.execute(
                "SELECT action FROM audit_log WHERE entity_id=?", (recording_id,)
            )
        ]
        assert actions == ["recording_display_name_changed"]
        assert "recording_display_name_changed" in (root / "audit.jsonl").read_text(
            encoding="utf-8"
        )
    finally:
        catalog.close()


def test_manual_display_name_rejects_empty_control_characters_and_excess_length():
    for value in (None, "  ", "名称\n换行", "x" * 301):
        try:
            validate_manual_display_name(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid name accepted: {value!r}")
