from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .manifests import sha256_file


SUPPORTED_DECISION_SCHEMAS = {
    "decision_record_v3", "decision_record_v4", "decision_record_v5",
}


def _session_asset_path(root: Path, asset: Mapping[str, Any]) -> Path:
    relative = asset.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("session资产缺少相对路径")
    path = (root / relative).resolve(strict=True)
    resolved_root = root.resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise ValueError("session资产路径越界")
    expected = asset.get("sha256")
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise ValueError(f"session资产hash校验失败：{relative}")
    return path


def load_session_manifest(root: str | Path) -> dict[str, Any]:
    session_root = Path(root).resolve()
    manifest_path = session_root / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "audio_session_v2":
        raise ValueError("不支持的运行录音manifest版本")
    return manifest


def iter_session_decisions(
    root: str | Path,
    *,
    include_v3: bool = True,
) -> Iterator[dict[str, Any]]:
    """Read current v5 decisions and legacy v3/v4 rows without rewriting them.

    Existing v3/v4 rows may call the current Layer 5 CNN stage ``l4``.  The
    read API deliberately preserves that raw field; presentation adapters are
    responsible for capability-based normalization.
    """

    session_root = Path(root).resolve()
    manifest = load_session_manifest(session_root)
    for chunk in manifest.get("chunks", ()):
        for asset in chunk.get("assets", ()):
            if asset.get("kind") != "results":
                continue
            path = _session_asset_path(session_root, asset)
            with path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("record_type") == "chunk_header":
                        continue
                    schema = row.get("schema_version")
                    if schema not in SUPPORTED_DECISION_SCHEMAS:
                        raise ValueError(f"不支持的DecisionRecord版本：{schema}（{path.name}:{line_number}）")
                    if schema == "decision_record_v3" and not include_v3:
                        continue
                    # A detached mapping prevents callers from mutating a
                    # cached manifest or performing an in-place v3 migration.
                    yield dict(row)


def enhanced_assets(root: str | Path) -> tuple[dict[str, Any], ...]:
    session_root = Path(root).resolve()
    manifest = load_session_manifest(session_root)
    rows: list[dict[str, Any]] = []
    for chunk in manifest.get("chunks", ()):
        for asset in chunk.get("assets", ()):
            if asset.get("kind") != "enhanced_audio":
                continue
            path = _session_asset_path(session_root, asset)
            rows.append({**dict(asset), "absolute_path": str(path)})
    rows.sort(
        key=lambda item: (
            int(item.get("stream_epoch", -1)),
            int(item.get("track_id", -1)),
            int(item.get("decision_sample", -1)),
            int(item.get("window_id", -1)),
        )
    )
    return tuple(rows)


__all__ = [
    "SUPPORTED_DECISION_SCHEMAS",
    "enhanced_assets",
    "iter_session_decisions",
    "load_session_manifest",
]
