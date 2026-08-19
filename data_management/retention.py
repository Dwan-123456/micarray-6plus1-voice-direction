from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .catalog import Catalog
from .manifests import append_audit


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def move_to_trash(
    data_root: str | Path, asset_path: str | Path, *, entity_type: str, entity_id: str, catalog: Catalog
) -> Path:
    root = Path(data_root).resolve()
    source = Path(asset_path).resolve()
    if root not in source.parents or source in {root, root / "runtime_sessions", root / "test_corpus"}:
        raise ValueError("Trash目标越界")
    if not source.is_dir():
        raise FileNotFoundError(f"待移除的数据目录不存在：{source}")
    operation = str(uuid.uuid4())
    target = root / "trash" / operation / source.name
    target.parent.mkdir(parents=True, exist_ok=False)
    append_audit(source, "moved_to_trash", {"operation_id": operation})
    source.replace(target)
    catalog.audit(
        entity_type,
        entity_id,
        "trash",
        {"operation_id": operation, "original_path": str(source), "trash_path": str(target)},
    )
    (target.parent / "trash_metadata.json").write_text(
        json.dumps(
            {
                "operation_id": operation,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "original_path": str(source),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    catalog.mark_trashed(entity_type, entity_id)
    return target


def restore_from_trash(operation_root: str | Path, *, catalog: Catalog | None = None) -> Path:
    operation = Path(operation_root)
    meta = json.loads((operation / "trash_metadata.json").read_text(encoding="utf-8"))
    items = [x for x in operation.iterdir() if x.name != "trash_metadata.json"]
    if len(items) != 1:
        raise ValueError("Trash operation内容无效")
    required_manifest = {
        "recording": "recording_manifest.json",
        "session": "session_manifest.json",
    }.get(str(meta.get("entity_type")))
    if required_manifest is None or not (items[0] / required_manifest).is_file():
        raise ValueError("回收站内容不完整，不能恢复为有效录音")
    target = Path(meta["original_path"])
    if target.exists():
        raise FileExistsError("原路径已存在，不能覆盖恢复")
    target.parent.mkdir(parents=True, exist_ok=True)
    items[0].replace(target)
    if catalog is not None:
        catalog.mark_restored(str(meta["entity_type"]), str(meta["entity_id"]))
    shutil.rmtree(operation)
    return target


def retention_candidates(catalog: Catalog, *, retention_days: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    result = []
    for item in catalog.list_sessions(limit=100000):
        try:
            started = datetime.fromisoformat((item["started_at"] or "").replace("Z", "+00:00"))
        except ValueError:
            continue
        if started < cutoff and not item["pinned"] and not item["promoted"]:
            result.append(item)
    return result
