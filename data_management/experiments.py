from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .catalog import Catalog
from .manifests import atomic_json, sha256_file, utc_now


class ExperimentStore:
    def __init__(self, data_root: str | Path, catalog: Catalog):
        self.data_root = Path(data_root)
        self.catalog = catalog

    def create_snapshot(
        self,
        *,
        name: str,
        dataset_id: str,
        dataset_version: str,
        config_hash: str,
        model_version: str,
        recording_ids: tuple[str, ...],
        notes: str = "",
    ) -> str:
        if not all((name, dataset_id, dataset_version, config_hash, model_version)) or not recording_ids:
            raise ValueError("实验名称、数据集版本、配置/模型版本和录音均不能为空")
        rows = self.catalog.list_recordings(dataset_id=dataset_id, limit=100000)
        available = {row["id"] for row in rows}
        if not set(recording_ids) <= available:
            raise ValueError("实验包含不属于所选数据集的Recording")
        experiment_id = str(uuid.uuid4())
        root = self.data_root / "experiments" / experiment_id
        root.mkdir(parents=True, exist_ok=False)
        payload = {
            "schema_version": "experiment_snapshot_v1",
            "experiment_id": experiment_id,
            "name": name,
            "created_at_utc": utc_now(),
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "config_hash": config_hash,
            "model_version": model_version,
            "recording_ids": list(recording_ids),
            "notes": notes,
        }
        path = atomic_json(root / "experiment_manifest.json", payload)
        now = utc_now()
        with self.catalog._lock, self.catalog.connection:
            self.catalog.connection.execute(
                "INSERT INTO experiments(id,created_at,updated_at,schema_version,metadata_json) VALUES(?,?,?,?,?)",
                (experiment_id, now, now, payload["schema_version"], json.dumps(payload, ensure_ascii=False)),
            )
            self.catalog.connection.executemany(
                "INSERT INTO experiment_items(experiment_id,recording_id) VALUES(?,?)",
                [(experiment_id, recording_id) for recording_id in recording_ids],
            )
            self.catalog.connection.execute(
                "UPDATE datasets SET locked=1,version=?,manifest_hash=?,updated_at=? WHERE id=?",
                (dataset_version, sha256_file(path), now, dataset_id),
            )
        return experiment_id

    def list_snapshots(self) -> list[dict[str, Any]]:
        with self.catalog._lock:
            rows = self.catalog.connection.execute("SELECT * FROM experiments ORDER BY created_at DESC")
            return [json.loads(row["metadata_json"]) for row in rows]
