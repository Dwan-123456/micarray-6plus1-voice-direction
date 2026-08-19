from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Iterable

from .manifests import sha256_file


def export_assets(asset_roots: Iterable[str | Path], destination: str | Path) -> Path:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        index = []
        for root_value in asset_roots:
            root = Path(root_value)
            for path in root.rglob("*"):
                if path.is_file():
                    arcname = f"{root.name}/{path.relative_to(root).as_posix()}"
                    archive.write(path, arcname)
                    index.append({"path": arcname, "sha256": sha256_file(path)})
        archive.writestr(
            "export_manifest.json",
            json.dumps({"schema_version": "asset_export_v1", "files": index}, ensure_ascii=False, indent=2),
        )
    return target


def verify_export(path: str | Path) -> bool:
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("export_manifest.json"))
        import hashlib

        return all(hashlib.sha256(archive.read(x["path"])).hexdigest() == x["sha256"] for x in manifest["files"])
