from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as destination:
        json.dump(payload, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    partial.replace(target)
    return target


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    target = atomic_json(path, payload)
    sidecar = target.parent / "manifest.sha256"
    digest = sha256_file(target)
    partial = sidecar.with_name(sidecar.name + ".partial")
    with partial.open("w", encoding="ascii", newline="\n") as destination:
        destination.write(f"{digest}  {target.name}\n")
        destination.flush()
        os.fsync(destination.fileno())
    partial.replace(sidecar)
    return target


def append_audit(asset_root: Path, action: str, details: dict[str, Any] | None = None) -> None:
    asset_root.mkdir(parents=True, exist_ok=True)
    with (asset_root / "audit.jsonl").open("a", encoding="utf-8", newline="\n") as destination:
        destination.write(
            json.dumps({"at_utc": utc_now(), "action": action, "details": details or {}}, ensure_ascii=False) + "\n"
        )
        destination.flush()
        os.fsync(destination.fileno())
