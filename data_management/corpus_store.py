from __future__ import annotations

import json
import uuid
import wave
from pathlib import Path
from typing import Iterable

import numpy as np

from layer1_input.pcm import pcm16_bytes

from .annotations import write_annotations
from .catalog import Catalog
from .contracts import Annotation, RecordingMetadata, public_mapping
from .manifests import append_audit, atomic_json, sha256_file, write_manifest
from .qa import qa_recording


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        rate = source.getframerate()
        width = source.getsampwidth()
        frames = source.readframes(source.getnframes())
    if width != 2:
        raise ValueError("仅支持PCM16 WAV")
    return np.frombuffer(frames, dtype="<i2").reshape(-1, channels).astype(np.float32) / 32768, rate


def _write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(samples.shape[1])
        out.setsampwidth(2)
        out.setframerate(48000)
        out.writeframes(pcm16_bytes(samples))


class CorpusStore:
    def __init__(self, data_root: str | Path, *, catalog: Catalog | None = None):
        self.data_root = Path(data_root)
        self.catalog = catalog or Catalog(self.data_root / "catalog.sqlite")
        self._owns_catalog = catalog is None

    def _root(self, dataset_id: str, recording_id: str) -> Path:
        return self.data_root / "test_corpus" / dataset_id / "recordings" / recording_id

    def _rights(self, meta: dict) -> None:
        rights = meta.get("rights", {})
        if rights.get("consent_status") not in {"granted", "not_applicable"} or not rights.get("allowed_uses"):
            raise PermissionError("缺少有效consent或allowed_uses，禁止正式录制/导入")

    def _save(self, root: Path, native: np.ndarray | None, physical: np.ndarray, meta: dict, lineage: dict) -> str:
        rid = root.name
        root.mkdir(parents=True, exist_ok=False)
        assets = []
        for kind, name, data in (
            ("native_8ch", "native_8ch.wav", native),
            ("physical_7ch", "physical_7ch.wav", physical),
        ):
            if data is not None:
                path = root / name
                _write_wav(path, data)
                assets.append(
                    {
                        "kind": kind,
                        "path": name,
                        "sha256": sha256_file(path),
                        "sample_count": len(data),
                        "channel_count": data.shape[1],
                        "sample_rate": 48000,
                        "dtype": "int16",
                    }
                )
        path = root / "physical_7ch_float.npy"
        np.save(path, np.asarray(physical, np.float32), allow_pickle=False)
        assets.append(
            {
                "kind": "physical_float",
                "path": path.name,
                "sha256": sha256_file(path),
                "sample_count": len(physical),
                "channel_count": 7,
                "sample_rate": 48000,
                "dtype": "float32",
            }
        )
        microphone_root = root / "microphones"
        for channel_index in range(physical.shape[1]):
            microphone_path = microphone_root / f"mic_{channel_index + 1:02d}.wav"
            microphone_path.parent.mkdir(parents=True, exist_ok=True)
            _write_wav(microphone_path, physical[:, channel_index : channel_index + 1])
            assets.append(
                {
                    "kind": f"microphone_{channel_index + 1:02d}",
                    "path": microphone_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(microphone_path),
                    "sample_count": len(physical),
                    "channel_count": 1,
                    "sample_rate": 48000,
                    "dtype": "int16",
                    "physical_channel_index": channel_index,
                }
            )
        labels_path = atomic_json(
            root / "labels.json",
            {
                "schema_version": "test_recording_labels_v1",
                "label_scope": "recording",
                "environment_description": meta["environment_id"],
                "source_count": meta["source_count"],
                "notes": meta.get("notes", ""),
                "duration_seconds": len(physical) / 48_000,
                "recorded_intervals": list(lineage.get("recorded_intervals", ())),
                "automatic_quality_report": "qa_report.json",
                "voice_activity_labels": [],
                "voice_activity_status": "not_annotated",
            },
        )
        assets.append(
            {
                "kind": "labels",
                "path": labels_path.relative_to(root).as_posix(),
                "sha256": sha256_file(labels_path),
            }
        )
        manifest = {
            "schema_version": "test_recording_v1",
            "dataset_id": meta["dataset_id"],
            "recording_id": rid,
            "capture_session_id": lineage.get("session_id"),
            "source_type": meta["source_type"],
            "lineage": lineage,
            "capture_time_utc": meta["capture_time_utc"],
            "environment_id": meta["environment_id"],
            "room_id": meta["room_id"],
            "array_pose_id": meta["array_pose_id"],
            "source_count": meta["source_count"],
            "source_categories": list(meta["source_categories"]),
            "known_theta_degrees": meta.get("known_theta_degrees"),
            "distance_m": meta.get("distance_m"),
            "speaker_ids_anonymous": list(meta.get("speaker_ids_anonymous", ())),
            "language_tags": list(meta.get("language_tags", ())),
            "rights": meta["rights"],
            "snr_db": meta.get("snr_db"),
            "notes": meta.get("notes", ""),
            "quality_status": "pending",
            "split": "unset",
            "assets": assets,
            "evaluation_references": [],
            "annotation_version": "",
            "duration_samples": len(physical),
        }
        append_audit(root, "recording_created", {"source_type": meta["source_type"]})
        write_manifest(root / "recording_manifest.json", manifest)
        report = qa_recording(root)
        manifest["quality_status"] = report["status"] if report["status"] == "passed" else "quarantine"
        write_manifest(root / "recording_manifest.json", manifest)
        dataset_root = root.parents[1]
        self.catalog.upsert_dataset(meta["dataset_id"], dataset_root)
        self.catalog.upsert_recording(manifest, root)
        return rid

    def register_raw_recording(
        self,
        root: Path,
        metadata: RecordingMetadata | dict,
        *,
        lineage: dict,
        assets: list[dict],
        duration_samples: int,
        run_quality_check: bool = True,
    ) -> str:
        """Register an already-streamed, lossless microphone-input recording."""

        meta = public_mapping(metadata)
        self._rights(meta)
        if duration_samples <= 0:
            raise ValueError("录音必须包含有效音频")
        native = next((item for item in assets if item.get("kind") == "native_8ch"), None)
        if native is None or int(native.get("channel_count", 0)) != 8:
            raise ValueError("完整测试录音必须包含原始8通道音频")
        for asset in assets:
            path = root / str(asset["path"])
            if not path.is_file() or sha256_file(path) != asset.get("sha256"):
                raise ValueError(f"录音资产不完整：{asset.get('path')}")

        labels_path = atomic_json(
            root / "labels.json",
            {
                "schema_version": "test_recording_labels_v3",
                "recording_name": meta.get("display_name", ""),
                "environment": meta["environment_id"],
                "source_count": int(meta["source_count"]),
                "sources": [
                    {
                        "index": index + 1,
                        "type": str(meta.get("source_categories", ())[index]),
                        "movement": (
                            str(meta.get("source_movements", ())[index])
                            if index < len(meta.get("source_movements", ()))
                            else ""
                        ),
                    }
                    for index in range(min(int(meta["source_count"]), len(meta.get("source_categories", ()))))
                ],
                "noise_source": meta.get("noise_source", ""),
                "duration_seconds": duration_samples / 48_000,
                "recorded_intervals": list(lineage.get("recorded_intervals", ())),
            },
        )
        assets.append(
            {
                "kind": "labels",
                "path": labels_path.name,
                "sha256": sha256_file(labels_path),
            }
        )
        manifest = {
            "schema_version": "raw_microphone_recording_v1",
            "dataset_id": meta["dataset_id"],
            "recording_id": root.name,
            "display_name": str(meta.get("display_name", "")).strip(),
            "capture_session_id": lineage.get("session_id"),
            "source_type": "dedicated",
            "lineage": lineage,
            "capture_time_utc": meta["capture_time_utc"],
            "environment_id": meta["environment_id"],
            "room_id": meta["room_id"],
            "array_pose_id": meta["array_pose_id"],
            "source_count": meta["source_count"],
            "source_categories": list(meta["source_categories"]),
            "source_movements": list(meta.get("source_movements", ())),
            "noise_source": meta.get("noise_source", ""),
            "rights": meta["rights"],
            "quality_status": "pending",
            "split": "unset",
            "assets": assets,
            "duration_samples": int(duration_samples),
            "input_contract": {
                "sample_rate": 48_000,
                "native_channel_count": 8,
                "audio_kind": "native_8ch",
                "hotmap_kind": "cdc_hotmaps",
            },
            "algorithm_direction_ids": {
                "status": "not_available",
                "reason": "l1_only_recording",
                "display_text": "无算法方向ID",
            },
        }
        append_audit(root, "raw_recording_created", {"display_name": manifest["display_name"]})
        write_manifest(root / "recording_manifest.json", manifest)
        if run_quality_check:
            report = qa_recording(root)
            manifest["quality_status"] = report["status"] if report["status"] == "passed" else "quarantine"
            write_manifest(root / "recording_manifest.json", manifest)
        dataset_root = root.parents[1]
        self.catalog.upsert_dataset(meta["dataset_id"], dataset_root)
        self.catalog.upsert_recording(manifest, root)
        return root.name

    def import_recording(self, source: str | Path, metadata: RecordingMetadata | dict) -> str:
        meta = public_mapping(metadata)
        self._rights(meta)
        src = Path(source)
        data, rate = _read_wav(src)
        if rate != 48000 or data.shape[1] not in {7, 8}:
            raise ValueError("非48kHz或非7/8通道文件必须先显式转换并进入quarantine")
        native = data if data.shape[1] == 8 else None
        physical = data[:, [0, 1, 2, 3, 4, 5, 7]] if data.shape[1] == 8 else data
        meta["source_type"] = "imported"
        rid = str(uuid.uuid4())
        return self._save(
            self._root(meta["dataset_id"], rid),
            native,
            physical,
            meta,
            {"source_uri": str(src.resolve()), "source_hash": sha256_file(src)},
        )

    def promote_runtime_segment(
        self, session_id: str, stream_epoch: int, start_sample: int, end_sample: int, metadata: RecordingMetadata | dict
    ) -> str:
        if start_sample < 0 or end_sample <= start_sample:
            raise ValueError("提升范围无效")
        meta = public_mapping(metadata)
        self._rights(meta)
        matches = list(self.data_root.glob(f"runtime_sessions/*/*/{session_id}/session_manifest.json"))
        if len(matches) != 1:
            raise FileNotFoundError("找不到唯一Runtime session")
        manifest = json.loads(matches[0].read_text(encoding="utf-8"))
        root = matches[0].parent
        chunks = [
            x
            for x in manifest["chunks"]
            if x["stream_epoch"] == stream_epoch and x["end_sample"] > start_sample and x["start_sample"] < end_sample
        ]
        if not chunks:
            raise ValueError("Runtime范围没有已完成音频")
        native_parts = []
        physical_parts = []
        for chunk in chunks:
            assets = {x["kind"]: x for x in chunk["assets"]}
            left = max(start_sample, chunk["start_sample"]) - chunk["start_sample"]
            right = min(end_sample, chunk["end_sample"]) - chunk["start_sample"]
            physical_parts.append(np.load(root / assets["physical_float"]["path"], allow_pickle=False)[left:right])
            if "native_8ch" in assets:
                native_parts.append(_read_wav(root / assets["native_8ch"]["path"])[0][left:right])
        physical = np.concatenate(physical_parts)
        native = np.concatenate(native_parts) if len(native_parts) == len(chunks) else None
        if len(physical) != end_sample - start_sample:
            raise ValueError("提升范围包含缺失sample，不能静默拼接")
        meta["source_type"] = "promoted_runtime"
        rid = str(uuid.uuid4())
        result = self._save(
            self._root(meta["dataset_id"], rid),
            native,
            physical,
            meta,
            {
                "session_id": session_id,
                "stream_epoch": stream_epoch,
                "start_sample": start_sample,
                "end_sample": end_sample,
            },
        )
        self.catalog.audit("recording", result, "promoted_runtime", {"session_id": session_id})
        return result

    def add_annotations(self, recording_id: str, annotations: Iterable[Annotation]) -> None:
        matches = list(self.data_root.glob(f"test_corpus/*/recordings/{recording_id}/recording_manifest.json"))
        if len(matches) != 1:
            raise FileNotFoundError(recording_id)
        items = tuple(annotations)
        write_annotations(matches[0].parent, items)
        manifest = json.loads(matches[0].read_text(encoding="utf-8"))
        manifest["annotation_version"] = items[0].annotation_version
        manifest["quality_status"] = "annotated"
        append_audit(matches[0].parent, "annotations_added", {"version": items[0].annotation_version})
        write_manifest(matches[0], manifest)
        self.catalog.upsert_recording(manifest, matches[0].parent)

    def close(self) -> None:
        if self._owns_catalog:
            self.catalog.close()
