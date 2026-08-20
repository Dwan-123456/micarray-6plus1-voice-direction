from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from .catalog import Catalog
from .manifests import append_audit, atomic_json, sha256_file, write_manifest


LOCAL_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")
_LEGACY_TIMESTAMP = re.compile(r"(?P<date>\d{8})[-_ ](?P<time>\d{4})(?:\d{2})?")
_CURRENT_TIMESTAMP = re.compile(r"(?:^|·)\s*(?P<date>\d{4})-(?P<time>\d{4})(?:\s*·|$)")
_LEGACY_SINGLE_SOURCE = re.compile(
    r"^(?P<environment>.+?)-单(?P<category>.+?)(?P<movement>固定|静止|移动)声源-"
    r"(?P<noise>.+?)背景噪音$"
)


def _clean(value: object, fallback: str) -> str:
    text = " ".join(str(value or "").split()).replace("·", "-").strip(" -")
    return text or fallback


def build_corpus_display_name(
    environment: str,
    recorded_at: datetime,
    source_count: int,
    source_categories: Sequence[str],
    source_movements: Sequence[str],
    noise_source: str,
) -> str:
    """Build the stable, operator-facing corpus name from structured labels."""

    if source_count < 0:
        raise ValueError("声源数量不能小于0")
    local_time = recorded_at
    if recorded_at.tzinfo is not None:
        local_time = recorded_at.astimezone(LOCAL_TIMEZONE)
    if source_count == 0:
        sources = "无声源"
    else:
        descriptions: list[str] = []
        for index in range(source_count):
            category = _clean(
                source_categories[index] if index < len(source_categories) else "",
                "未说明",
            )
            movement = _clean(
                source_movements[index] if index < len(source_movements) else "",
                "",
            )
            suffix = f"（{movement}）" if movement else ""
            descriptions.append(f"声源{index + 1}：{category}{suffix}")
        sources = "；".join(descriptions)
    return (
        f"{_clean(environment, '环境未说明')} · {local_time:%m%d-%H%M} · "
        f"{source_count}个声源 · {sources} · 噪音：{_clean(noise_source, '未说明')}"
    )


def _recorded_at(manifest: dict[str, Any], previous_name: str) -> datetime:
    match = _LEGACY_TIMESTAMP.search(previous_name)
    if match:
        return datetime.strptime(
            match.group("date") + match.group("time"), "%Y%m%d%H%M"
        ).replace(tzinfo=LOCAL_TIMEZONE)
    match = _CURRENT_TIMESTAMP.search(previous_name)
    if match:
        capture = _parse_capture_time(manifest.get("capture_time_utc"))
        return capture.replace(
            month=int(match.group("date")[:2]),
            day=int(match.group("date")[2:]),
            hour=int(match.group("time")[:2]),
            minute=int(match.group("time")[2:]),
        )
    return _parse_capture_time(manifest.get("capture_time_utc"))


def _parse_capture_time(value: object) -> datetime:
    if not value:
        raise ValueError("录音缺少capture_time_utc，无法生成月日时分")
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TIMEZONE)
    return parsed.astimezone(LOCAL_TIMEZONE)


def _metadata_for_name(manifest: dict[str, Any], previous_name: str) -> dict[str, Any]:
    metadata = {
        "environment_id": manifest.get("environment_id", ""),
        "source_count": int(manifest.get("source_count", 0)),
        "source_categories": list(manifest.get("source_categories") or ()),
        "source_movements": list(manifest.get("source_movements") or ()),
        "noise_source": manifest.get("noise_source", ""),
    }
    legacy = _LEGACY_SINGLE_SOURCE.fullmatch(previous_name.strip())
    if legacy and (
        _clean(metadata["environment_id"], "") in {"", "unspecified"}
        or metadata["source_count"] == 0
    ):
        metadata.update(
            environment_id=legacy.group("environment"),
            source_count=1,
            source_categories=[legacy.group("category")],
            source_movements=[legacy.group("movement")],
            noise_source=legacy.group("noise"),
        )
    return metadata


def migrate_corpus_display_names(
    data_root: str | Path, *, dry_run: bool = False
) -> list[dict[str, str]]:
    """Apply the current naming rule without renaming recording directories or assets."""

    root = Path(data_root)
    changes: list[dict[str, str]] = []
    catalog = None if dry_run else Catalog(root / "catalog.sqlite")
    try:
        for manifest_path in sorted(
            root.glob("test_corpus/*/recordings/*/recording_manifest.json")
        ):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous_name = str(manifest.get("display_name") or "").strip()
            if not previous_name:
                continue
            metadata = _metadata_for_name(manifest, previous_name)
            current_name = build_corpus_display_name(
                metadata["environment_id"],
                _recorded_at(manifest, previous_name),
                metadata["source_count"],
                metadata["source_categories"],
                metadata["source_movements"],
                metadata["noise_source"],
            )
            if current_name == previous_name:
                continue
            changes.append(
                {
                    "recording_id": str(manifest["recording_id"]),
                    "previous_name": previous_name,
                    "current_name": current_name,
                }
            )
            if dry_run:
                continue

            previous_metadata = {
                key: manifest.get(key)
                for key in (
                    "environment_id",
                    "source_count",
                    "source_categories",
                    "source_movements",
                    "noise_source",
                )
            }
            manifest.update(metadata)
            manifest["display_name"] = current_name
            labels_path = manifest_path.parent / "labels.json"
            if labels_path.is_file():
                labels = json.loads(labels_path.read_text(encoding="utf-8"))
                labels.update(
                    schema_version="test_recording_labels_v3",
                    recording_name=current_name,
                    environment=metadata["environment_id"],
                    source_count=metadata["source_count"],
                    sources=[
                        {
                            "index": index + 1,
                            "type": metadata["source_categories"][index]
                            if index < len(metadata["source_categories"])
                            else "未说明",
                            "movement": metadata["source_movements"][index]
                            if index < len(metadata["source_movements"])
                            else "",
                        }
                        for index in range(metadata["source_count"])
                    ],
                    noise_source=metadata["noise_source"],
                )
                atomic_json(labels_path, labels)
                for asset in manifest.get("assets", ()):
                    if asset.get("kind") == "labels":
                        asset["sha256"] = sha256_file(labels_path)

            append_audit(
                manifest_path.parent,
                "corpus_display_name_migrated",
                {
                    "previous_name": previous_name,
                    "current_name": current_name,
                    "previous_metadata": previous_metadata,
                },
            )
            write_manifest(manifest_path, manifest)
            assert catalog is not None
            catalog.upsert_recording(manifest, manifest_path.parent)
            catalog.audit(
                "recording",
                str(manifest["recording_id"]),
                "corpus_display_name_migrated",
                {"previous_name": previous_name, "current_name": current_name},
            )
    finally:
        if catalog is not None:
            catalog.close()
    return changes
