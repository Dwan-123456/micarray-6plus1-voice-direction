from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from .contracts import Annotation


def write_annotations(root: str | Path, annotations: Iterable[Annotation]) -> Path:
    items = tuple(annotations)
    if not items:
        raise ValueError("annotations不能为空")
    versions = {x.annotation_version for x in items}
    recordings = {x.recording_id for x in items}
    if len(versions) != 1 or len(recordings) != 1:
        raise ValueError("一次写入必须属于同一recording和version")
    folder = Path(root) / "annotations"
    folder.mkdir(parents=True, exist_ok=True)
    version = next(iter(versions))
    target = folder / f"{version}.jsonl"
    if target.exists():
        raise FileExistsError("标注版本不可覆盖")
    partial = Path(str(target) + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as out:
        for item in items:
            from dataclasses import asdict

            out.write(json.dumps({"schema_version": "annotation_v1", **asdict(item)}, ensure_ascii=False) + "\n")
        out.flush()
        os.fsync(out.fileno())
    partial.replace(target)
    return target
