from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .manifests import sha256_file, utc_now


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
 id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, schema_version TEXT NOT NULL,
 status TEXT NOT NULL, path TEXT NOT NULL UNIQUE, started_at TEXT, ended_at TEXT, mode TEXT, pinned INTEGER NOT NULL DEFAULT 0,
 promoted INTEGER NOT NULL DEFAULT 0, manifest_hash TEXT, metadata_json TEXT NOT NULL DEFAULT '{}');
CREATE INDEX IF NOT EXISTS ix_sessions_started ON sessions(started_at DESC);
CREATE TABLE IF NOT EXISTS datasets (
 id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, schema_version TEXT NOT NULL,
 name TEXT NOT NULL, version TEXT NOT NULL DEFAULT '0.1.0', locked INTEGER NOT NULL DEFAULT 0, path TEXT NOT NULL UNIQUE,
 manifest_hash TEXT, metadata_json TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS recordings (
 id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, schema_version TEXT NOT NULL,
 dataset_id TEXT NOT NULL, status TEXT NOT NULL, source_type TEXT NOT NULL, path TEXT NOT NULL UNIQUE,
 capture_session_id TEXT, room_id TEXT, environment_id TEXT, split TEXT NOT NULL DEFAULT 'unset', duration_samples INTEGER NOT NULL DEFAULT 0,
 manifest_hash TEXT, metadata_json TEXT NOT NULL DEFAULT '{}');
CREATE INDEX IF NOT EXISTS ix_recordings_filter ON recordings(dataset_id,status,split,room_id,environment_id);
CREATE TABLE IF NOT EXISTS annotations (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 schema_version TEXT NOT NULL, recording_id TEXT NOT NULL, version TEXT NOT NULL, path TEXT NOT NULL, metadata_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS quality_checks (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 schema_version TEXT NOT NULL, recording_id TEXT NOT NULL, status TEXT NOT NULL, metadata_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tags (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, schema_version TEXT NOT NULL, name TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS recording_tags (recording_id TEXT NOT NULL, tag_id TEXT NOT NULL, PRIMARY KEY(recording_id,tag_id));
CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, schema_version TEXT NOT NULL, metadata_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiment_items (experiment_id TEXT NOT NULL, recording_id TEXT NOT NULL, PRIMARY KEY(experiment_id,recording_id));
CREATE TABLE IF NOT EXISTS asset_lineage (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 schema_version TEXT NOT NULL, parent_type TEXT NOT NULL, parent_id TEXT NOT NULL, child_type TEXT NOT NULL, child_id TEXT NOT NULL, metadata_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, entity_type TEXT NOT NULL, entity_id TEXT NOT NULL, action TEXT NOT NULL, metadata_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(1,datetime('now'));
"""


class Catalog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def upsert_session(self, manifest: dict[str, Any], path: Path) -> None:
        now = utc_now()
        sid = manifest["session_id"]
        with self._lock, self.connection:
            self.connection.execute(
                """INSERT INTO sessions(id,created_at,updated_at,schema_version,status,path,started_at,ended_at,mode,manifest_hash,metadata_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at,status=excluded.status,
            path=excluded.path,ended_at=excluded.ended_at,mode=excluded.mode,manifest_hash=excluded.manifest_hash,metadata_json=excluded.metadata_json""",
                (
                    sid,
                    now,
                    now,
                    manifest["schema_version"],
                    manifest.get("status", "open"),
                    str(path),
                    manifest.get("started_at_utc"),
                    manifest.get("ended_at_utc"),
                    manifest.get("current_mode"),
                    sha256_file(path / "session_manifest.json"),
                    json.dumps(manifest, ensure_ascii=False),
                ),
            )

    def upsert_dataset(self, dataset_id: str, path: Path, name: str | None = None) -> None:
        now = utc_now()
        with self._lock, self.connection:
            self.connection.execute(
                """INSERT INTO datasets(id,created_at,updated_at,schema_version,name,path) VALUES(?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at,name=excluded.name,path=excluded.path""",
                (dataset_id, now, now, "dataset_manifest_v1", name or dataset_id, str(path)),
            )

    def upsert_recording(self, manifest: dict[str, Any], path: Path) -> None:
        now = utc_now()
        rid = manifest["recording_id"]
        with self._lock, self.connection:
            self.connection.execute(
                """INSERT INTO recordings(id,created_at,updated_at,schema_version,dataset_id,status,source_type,path,capture_session_id,room_id,environment_id,split,duration_samples,manifest_hash,metadata_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at,status=excluded.status,
            path=excluded.path,split=excluded.split,duration_samples=excluded.duration_samples,manifest_hash=excluded.manifest_hash,metadata_json=excluded.metadata_json""",
                (
                    rid,
                    now,
                    now,
                    manifest["schema_version"],
                    manifest["dataset_id"],
                    manifest.get("quality_status", "pending"),
                    manifest["source_type"],
                    str(path),
                    manifest.get("capture_session_id"),
                    manifest.get("room_id"),
                    manifest.get("environment_id"),
                    manifest.get("split", "unset"),
                    manifest.get("duration_samples", 0),
                    sha256_file(path / "recording_manifest.json"),
                    json.dumps(manifest, ensure_ascii=False),
                ),
            )

    def list_sessions(self, *, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(row)
                for row in self.connection.execute("SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,))
            ]

    def list_datasets(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.connection.execute("SELECT * FROM datasets ORDER BY updated_at DESC")]

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
            return None if row is None else dict(row)

    def recording_is_experiment_locked(self, recording_id: str) -> bool:
        with self._lock:
            row = self.connection.execute(
                "SELECT 1 FROM experiment_items WHERE recording_id=? LIMIT 1", (recording_id,)
            ).fetchone()
            return row is not None

    def list_recordings(
        self, *, dataset_id: str | None = None, status: str | None = None, split: str | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        clauses = []
        values: list[Any] = []
        for field, value in (("dataset_id", dataset_id), ("status", status), ("split", split)):
            if value:
                clauses.append(f"{field}=?")
                values.append(value)
        sql = (
            "SELECT * FROM recordings"
            + (" WHERE " + " AND ".join(clauses) if clauses else "")
            + " ORDER BY created_at DESC LIMIT ?"
        )
        values.append(limit)
        with self._lock:
            return [dict(row) for row in self.connection.execute(sql, values)]

    def rebuild(self, data_root: str | Path) -> dict[str, int]:
        root = Path(data_root)
        counts = {"sessions": 0, "recordings": 0}
        for path in root.glob("runtime_sessions/*/*/*/session_manifest.json"):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.upsert_session(manifest, path.parent)
            counts["sessions"] += 1
        for path in root.glob("test_corpus/*/recordings/*/recording_manifest.json"):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.upsert_dataset(manifest["dataset_id"], path.parents[2])
            self.upsert_recording(manifest, path.parent)
            counts["recordings"] += 1
        return counts

    def audit(self, entity_type: str, entity_id: str, action: str, metadata: dict[str, Any] | None = None) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT INTO audit_log(created_at,entity_type,entity_id,action,metadata_json) VALUES(?,?,?,?,?)",
                (utc_now(), entity_type, entity_id, action, json.dumps(metadata or {}, ensure_ascii=False)),
            )
