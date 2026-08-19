from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .catalog import Catalog
from .contracts import Annotation
from .corpus_store import CorpusStore
from .dedicated_recording import DedicatedRecordingController
from .experiments import ExperimentStore
from .export import export_assets
from .manifests import sha256_file, utc_now, write_manifest
from .qa import leakage_check, qa_recording
from .retention import directory_size, move_to_trash, restore_from_trash
from .recording_store import RecordingStore
from .statistics import assign_grouped_splits, corpus_statistics
from .timeline import enhanced_assets, iter_session_decisions, load_session_manifest


class DataManagerService:
    """Transaction boundary used by the standalone Audio Data Manager UI."""

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)
        self.catalog = Catalog(self.data_root / "catalog.sqlite")
        self.wizard = DedicatedRecordingController(self.data_root, self.catalog)
        self.experiments_store = ExperimentStore(self.data_root, self.catalog)

    def dashboard(self) -> dict[str, Any]:
        usage = directory_size(self.data_root) if self.data_root.exists() else 0
        disk = shutil.disk_usage(self.data_root if self.data_root.exists() else self.data_root.parent)
        sessions = self.catalog.list_sessions()
        recordings = self.catalog.list_recordings()
        return {
            "sessions": len(sessions),
            "recordings": len(recordings),
            "storage_bytes": usage,
            "free_bytes": disk.free,
            "statistics": corpus_statistics(recordings),
        }

    def runtime_sessions(self) -> list[dict[str, Any]]:
        return self.catalog.list_sessions()

    def runtime_session_tracks(
        self, session_id: str, *, stream_epoch: int | None = None
    ) -> list[dict[str, Any]]:
        if self.catalog.get_session(session_id) is None:
            raise FileNotFoundError(session_id)
        return self.catalog.session_track_summaries(session_id, stream_epoch=stream_epoch)

    def track_timeline(self, session_id: str, stream_epoch: int, track_id: int) -> list[dict[str, Any]]:
        if self.catalog.get_session(session_id) is None:
            raise FileNotFoundError(session_id)
        return self.catalog.track_timeline(session_id, stream_epoch, track_id)

    def track_audio_assets(self, session_id: str, stream_epoch: int, track_id: int) -> list[dict[str, Any]]:
        row = self.catalog.get_session(session_id)
        if row is None:
            raise FileNotFoundError(session_id)
        return [
            item
            for item in enhanced_assets(row["path"])
            if int(item.get("stream_epoch", -1)) == int(stream_epoch)
            and int(item.get("track_id", -1)) == int(track_id)
        ]

    def session_audio_assets(self, session_id: str, kind: str) -> list[dict[str, Any]]:
        if kind not in {"native_8ch", "logical_8ch", "physical_7ch"}:
            raise ValueError("运行录音试听类型必须是native/logical/physical")
        row = self.catalog.get_session(session_id)
        if row is None:
            raise FileNotFoundError(session_id)
        root = Path(row["path"]).resolve()
        manifest = load_session_manifest(root)
        assets = []
        for chunk in manifest.get("chunks", ()):
            for asset in chunk.get("assets", ()):
                if asset.get("kind") != kind:
                    continue
                path = (root / str(asset["path"])).resolve(strict=True)
                if root != path and root not in path.parents:
                    raise ValueError("运行录音资产路径越界")
                if sha256_file(path) != asset.get("sha256"):
                    raise ValueError(f"运行录音资产hash校验失败：{asset['path']}")
                assets.append({**dict(asset), "absolute_path": str(path)})
        assets.sort(key=lambda item: (int(item["stream_epoch"]), int(item["start_sample"])))
        return assets

    def session_decisions(self, session_id: str, *, include_v3: bool = True) -> list[dict[str, Any]]:
        row = self.catalog.get_session(session_id)
        if row is None:
            raise FileNotFoundError(session_id)
        return list(iter_session_decisions(row["path"], include_v3=include_v3))

    def recordings(self, **filters: Any) -> list[dict[str, Any]]:
        rows = self.catalog.list_recordings(**filters)
        for row in rows:
            try:
                metadata = json.loads(row.get("metadata_json") or "{}")
            except json.JSONDecodeError:
                metadata = {}
            row["display_name"] = str(metadata.get("display_name") or row["id"])
        return rows

    def datasets(self) -> list[dict[str, Any]]:
        return self.catalog.list_datasets()

    def rebuild_catalog(self) -> dict[str, int]:
        return self.catalog.rebuild(self.data_root)

    def run_qa(self, recording_id: str) -> dict[str, Any]:
        matches = list(self.data_root.glob(f"test_corpus/*/recordings/{recording_id}/recording_manifest.json"))
        if len(matches) != 1:
            raise FileNotFoundError(recording_id)
        return qa_recording(matches[0].parent)

    def leakage_report(self) -> dict[str, Any]:
        items = []
        for row in self.catalog.list_recordings(limit=100000):
            items.append(json.loads(row["metadata_json"]))
        return leakage_check(items)

    def trash(self, entity_type: str, entity_id: str) -> Path:
        rows = (
            self.catalog.list_sessions(limit=100000)
            if entity_type == "session"
            else self.catalog.list_recordings(limit=100000)
        )
        row = next((x for x in rows if x["id"] == entity_id), None)
        if row is None:
            raise FileNotFoundError(entity_id)
        if entity_type == "recording":
            dataset = self.catalog.get_dataset(row["dataset_id"])
            if (dataset and dataset["locked"]) or self.catalog.recording_is_experiment_locked(entity_id):
                raise PermissionError("Recording属于锁定数据集或实验快照，必须先创建不含它的新数据集版本")
        return move_to_trash(
            self.data_root, row["path"], entity_type=entity_type, entity_id=entity_id, catalog=self.catalog
        )

    def restore(self, operation_id: str) -> Path:
        return restore_from_trash(self.data_root / "trash" / operation_id)

    def trash_operations(self) -> list[dict[str, Any]]:
        result = []
        for path in (self.data_root / "trash").glob("*/trash_metadata.json"):
            item = json.loads(path.read_text(encoding="utf-8"))
            item["operation_root"] = str(path.parent)
            result.append(item)
        return result

    def export(self, paths: list[str], destination: str) -> Path:
        return export_assets(paths, destination)

    def import_recording(self, source: str, metadata: Any) -> str:
        return CorpusStore(self.data_root, catalog=self.catalog).import_recording(source, metadata)

    def quarantine_partials(self) -> list[str]:
        store = RecordingStore(self.data_root, catalog=self.catalog)
        return [str(path) for path in store.recover_partials()]

    def add_annotation(self, annotation: Annotation) -> None:
        row = next((x for x in self.catalog.list_recordings(limit=100000) if x["id"] == annotation.recording_id), None)
        if row is None:
            raise FileNotFoundError(annotation.recording_id)
        dataset = self.catalog.get_dataset(row["dataset_id"])
        if dataset and dataset["locked"]:
            raise PermissionError("锁定数据集中的标注不可原地修改，请创建新数据集版本")
        CorpusStore(self.data_root, catalog=self.catalog).add_annotations(annotation.recording_id, (annotation,))

    def create_experiment(self, **values: Any) -> str:
        return self.experiments_store.create_snapshot(**values)

    def experiments(self) -> list[dict[str, Any]]:
        return self.experiments_store.list_snapshots()

    def recovery_status(self) -> dict[str, Any]:
        partials = [str(path) for path in self.data_root.rglob("*.partial")]
        corrupt = [
            row for row in self.catalog.list_sessions(limit=100000) if row["status"] in {"corrupt", "incomplete"}
        ]
        return {"partial_files": partials, "recoverable_sessions": corrupt, "trash_operations": self.trash_operations()}

    def assign_and_lock_dataset(self, dataset_id: str, version: str) -> dict[str, Any]:
        dataset = self.catalog.get_dataset(dataset_id)
        if dataset is None:
            raise FileNotFoundError(dataset_id)
        if dataset["locked"]:
            raise PermissionError("该数据集版本已经锁定")
        rows = self.catalog.list_recordings(dataset_id=dataset_id, limit=100000)
        if not rows:
            raise ValueError("数据集没有Recording")
        manifests = [json.loads(row["metadata_json"]) for row in rows]
        assignments = assign_grouped_splits([{"id": item["recording_id"], **item} for item in manifests])
        staged_manifests = [
            {
                **manifest,
                "split": assignments[row["id"]],
                "quality_status": "versioned",
            }
            for row, manifest in zip(rows, manifests, strict=True)
        ]
        report = leakage_check(staged_manifests)
        if not report["passed"]:
            raise ValueError(f"split泄漏检查失败: {report['leaks']}")

        recording_hashes = []
        for row, manifest in zip(rows, staged_manifests, strict=True):
            root = Path(row["path"])
            path = write_manifest(root / "recording_manifest.json", manifest)
            self.catalog.upsert_recording(manifest, root)
            recording_hashes.append({"recording_id": row["id"], "manifest_hash": sha256_file(path)})
        root = Path(dataset["path"])
        payload = {
            "schema_version": "dataset_manifest_v1",
            "dataset_id": dataset_id,
            "version": version,
            "created_at_utc": utc_now(),
            "locked": True,
            "recordings": recording_hashes,
            "assignments": assignments,
            "leakage_report": report,
        }
        path = write_manifest(root / "dataset_manifest.json", payload)
        with self.catalog._lock, self.catalog.connection:
            self.catalog.connection.execute(
                "UPDATE datasets SET version=?,locked=1,manifest_hash=?,metadata_json=?,updated_at=? WHERE id=?",
                (version, sha256_file(path), json.dumps(payload, ensure_ascii=False), utc_now(), dataset_id),
            )
        return payload

    def preview_dataset_split(self, dataset_id: str) -> dict[str, Any]:
        dataset = self.catalog.get_dataset(dataset_id)
        if dataset is None:
            raise FileNotFoundError(dataset_id)
        rows = self.catalog.list_recordings(dataset_id=dataset_id, limit=100000)
        if not rows:
            raise ValueError("数据集没有样本")
        manifests = [json.loads(row["metadata_json"]) for row in rows]
        assignments = assign_grouped_splits([{"id": item["recording_id"], **item} for item in manifests])
        counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
        duration: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
        leakage_items = []
        for manifest in manifests:
            split = assignments[manifest["recording_id"]]
            counts[split] += 1
            duration[split] += int(manifest.get("duration_samples", 0))
            leakage_items.append({**manifest, "split": split})
        return {
            "dataset_id": dataset_id,
            "already_locked": bool(dataset["locked"]),
            "recording_count": len(rows),
            "counts": counts,
            "duration_seconds": {key: value / 48_000 for key, value in duration.items()},
            "leakage_report": leakage_check(leakage_items),
            "assignments": assignments,
        }

    def close(self) -> None:
        self.catalog.close()
