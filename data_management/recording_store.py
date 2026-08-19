from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import wave
import zipfile
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from common.data_types import IngestedAudioBlock
from layer1_input.pcm import pcm16_bytes

from .catalog import Catalog
from .contracts import DecisionRecord, ResultWatermark, SessionMetadata, public_mapping
from .manifests import append_audit, atomic_json, sha256_file, utc_now, write_manifest


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _physical_samples(block: IngestedAudioBlock) -> np.ndarray:
    return np.asarray(block.samples[:, :7], dtype=np.float32)


def _logical_samples(block: IngestedAudioBlock) -> np.ndarray | None:
    samples = np.asarray(block.samples, dtype=np.float32)
    if samples.shape[1] == 8:
        return samples
    if block.native_samples is None:
        return None
    return np.column_stack((samples[:, :7], block.native_samples[:, 6])).astype(np.float32, copy=False)


def _slice_block(block: IngestedAudioBlock, start: int, end: int) -> IngestedAudioBlock:
    left, right = start - block.start_sample, end - block.start_sample
    return IngestedAudioBlock(
        block.session_id,
        block.stream_epoch,
        start,
        end,
        block.sample_rate,
        block.sequence_id,
        block.timestamp,
        block.samples[left:right],
        None if block.native_samples is None else block.native_samples[left:right],
        block.hotmap,
        block.noise_spectrum,
        block.imcra_hop if start == block.start_sample and end == block.end_sample else None,
    )


class _RawArraySpool:
    """Append-only ndarray storage with constant process-resident memory.

    Files deliberately end in ``.partial`` so an interrupted session is found
    by the existing recovery/quarantine pass.  The raw stream is converted to
    a standard NPY member without loading the chunk back into RAM.
    """

    _COPY_BYTES = 1024 * 1024

    def __init__(self, path: Path, *, dtype: np.dtype[Any], item_shape: tuple[int, ...]):
        self.path = path
        self.dtype = np.dtype(dtype)
        self.item_shape = tuple(int(value) for value in item_shape)
        if any(value <= 0 for value in self.item_shape):
            raise ValueError("streamed array item shape must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = path.open("wb", buffering=0)
        self.count = 0

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.count, *self.item_shape)

    def append(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=self.dtype)
        if array.shape == self.item_shape:
            array = array.reshape((1, *self.item_shape))
        if array.ndim != len(self.item_shape) + 1 or tuple(array.shape[1:]) != self.item_shape:
            raise ValueError(
                f"streamed array shape mismatch: expected [N,{self.item_shape}], got {array.shape}"
            )
        contiguous = np.ascontiguousarray(array, dtype=self.dtype)
        view = memoryview(contiguous).cast("B")
        while view:
            written = self._writer.write(view)
            if written is None or written <= 0:
                raise OSError("streamed array partial write failed")
            view = view[written:]
        self.count += int(contiguous.shape[0])

    def close_writer(self) -> None:
        if self._writer is None:
            return
        self._writer.flush()
        os.fsync(self._writer.fileno())
        self._writer.close()
        self._writer = None

    def write_npy(self, output: Any) -> None:
        self.close_writer()
        header = {
            "descr": np.lib.format.dtype_to_descr(self.dtype),
            "fortran_order": False,
            "shape": self.shape,
        }
        np.lib.format.write_array_header_1_0(output, header)
        expected = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize
        copied = 0
        with self.path.open("rb", buffering=0) as source:
            while True:
                block = source.read(self._COPY_BYTES)
                if not block:
                    break
                output.write(block)
                copied += len(block)
        if copied != expected:
            raise OSError(f"streamed array byte count mismatch: expected {expected}, got {copied}")

    def remove(self) -> None:
        self.close_writer()
        self.path.unlink(missing_ok=True)


def _write_streamed_npz(
    partial: Path,
    values: dict[str, np.ndarray | _RawArraySpool],
) -> None:
    """Write a conventional compressed NPZ using bounded streaming I/O."""

    with zipfile.ZipFile(
        partial,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
    ) as archive:
        for name, value in values.items():
            with archive.open(f"{name}.npy", mode="w", force_zip64=True) as member:
                if isinstance(value, _RawArraySpool):
                    value.write_npy(member)
                else:
                    np.lib.format.write_array(
                        member,
                        np.asarray(value),
                        allow_pickle=False,
                    )
    with partial.open("rb+") as output:
        os.fsync(output.fileno())


class _Chunk:
    _IMCRA_SPOOL_NAMES = {
        "noise_psd": "npsd",
        "smoothed_psd": "spsd",
        "conditional_smoothed_psd": "cspsd",
        "minimum_psd": "min",
        "conditional_minimum_psd": "cmin",
        "spp": "spp",
        "speech_absence_probability": "sap",
        "posterior_snr": "psnr",
        "prior_snr": "asnr",
        "noise_features": "nfeat",
        "noise_level_db": "nldb",
        "source_probability_per_mic": "spmic",
    }

    def __init__(
        self,
        root: Path,
        block: IngestedAudioBlock,
        end: int,
        settings: dict[str, Any],
        config_hash: str,
        calibration_hash: str,
    ):
        self.root = root
        self.epoch = block.stream_epoch
        self.start = block.start_sample
        self.target_end = end
        self.end = self.start
        self.settings = settings
        self.config_hash = config_hash
        self.calibration_hash = calibration_hash
        stem = f"epoch{self.epoch:03d}_start{self.start:012d}_end{self.target_end:012d}"
        self._stem = stem
        # Runtime session roots can already be deep on Windows.  Keep every
        # private staging component short so IMCRA tensor names cannot cross
        # the legacy MAX_PATH boundary; public final asset names are unchanged.
        self._staging_root = root / ".s"
        self.paths: dict[str, Path] = {}
        self._physical_float_spool: _RawArraySpool | None = None
        self.results: list[dict[str, Any]] = []
        self.spatial: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        self.noise_spectra: list[tuple[int, int, int, float, str]] = []
        self._noise_frequencies: np.ndarray | None = None
        self._noise_psd_spool: _RawArraySpool | None = None
        self.imcra_hops: list[dict[str, Any]] = []
        self._imcra_frequencies: np.ndarray | None = None
        self._imcra_spools: dict[str, _RawArraySpool] = {}
        self._commit_journal: Path | None = None
        self.clip_counts = np.zeros(7, dtype=np.int64)
        self.native = (
            self._wav("native_8ch", stem, 8)
            if settings["record_native_8ch"] and block.native_samples is not None
            else None
        )
        self.logical = (
            self._wav("logical_8ch", stem, 8)
            if settings["record_logical_8ch"] and _logical_samples(block) is not None
            else None
        )
        self.physical = self._wav("physical_7ch", stem, 7) if settings["record_physical_7ch"] else None
        if settings["record_physical_float32"]:
            self.paths["physical_float"] = root / "physical_7ch_float" / f"{stem}.npy"
            self._physical_float_spool = self._new_spool(
                "pf", dtype=np.dtype("<f4"), item_shape=(7,)
            )

    def _new_spool(
        self,
        name: str,
        *,
        dtype: np.dtype[Any],
        item_shape: tuple[int, ...],
    ) -> _RawArraySpool:
        return _RawArraySpool(
            self._staging_root / f"{self.epoch:03d}_{self.start:012d}_{name}.partial",
            dtype=dtype,
            item_shape=item_shape,
        )

    @property
    def resident_array_buffer_bytes(self) -> int:
        """Large per-hop/sample tensors are on disk; only frequency axes remain resident."""

        return sum(
            int(value.nbytes)
            for value in (self._noise_frequencies, self._imcra_frequencies)
            if value is not None
        )

    def _wav(self, key: str, stem: str, channels: int):
        path = self.root / key / f"{stem}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.paths[key] = path
        writer = wave.open(str(path) + ".partial", "wb")
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(48000)
        return writer

    def append(self, block: IngestedAudioBlock) -> None:
        if block.start_sample != self.end:
            raise ValueError("录音chunk sample不连续")
        if self.native is not None:
            if block.native_samples is None:
                raise ValueError("chunk内native可用性发生变化")
            self.native.writeframesraw(pcm16_bytes(block.native_samples))
        if self.logical is not None:
            logical = _logical_samples(block)
            if logical is None:
                raise ValueError("chunk内logical可用性发生变化")
            self.logical.writeframesraw(pcm16_bytes(logical))
        if self.physical is not None:
            self.physical.writeframesraw(pcm16_bytes(_physical_samples(block)))
        if self._physical_float_spool is not None:
            self._physical_float_spool.append(_physical_samples(block))
        physical = _physical_samples(block)
        clipped = np.logical_or(physical < -1, physical > 32767 / 32768)
        self.clip_counts += clipped.sum(axis=0)
        if self.settings["record_noise_spectrum"] and block.noise_spectrum is not None:
            noise = block.noise_spectrum
            frequencies = np.asarray(noise.frequencies_hz, np.float32)
            psd = np.asarray(noise.psd, np.float32)
            if self._noise_psd_spool is None:
                self._noise_frequencies = np.array(frequencies, dtype=np.float32, copy=True)
                self._noise_psd_spool = self._new_spool(
                    "np", dtype=np.dtype("<f4"), item_shape=tuple(psd.shape)
                )
            elif (
                self._noise_frequencies is None
                or not np.array_equal(frequencies, self._noise_frequencies)
                or tuple(psd.shape) != self._noise_psd_spool.item_shape
            ):
                raise ValueError("chunk内noise spectrum频率轴或shape发生变化")
            self._noise_psd_spool.append(psd)
            self.noise_spectra.append(
                (
                    block.start_sample,
                    block.end_sample,
                    int(noise.n_fft),
                    float(noise.timestamp),
                    str(noise.estimator),
                )
            )
        if self.settings["record_imcra"] and block.imcra_hop is not None:
            hop = block.imcra_hop
            frequencies = np.asarray(hop.frequencies_hz, np.float32)
            streamed = {
                "noise_psd": np.asarray(hop.noise_psd, np.float32),
                "smoothed_psd": np.asarray(hop.smoothed_psd, np.float32),
                "conditional_smoothed_psd": np.asarray(hop.conditional_smoothed_psd, np.float32),
                "minimum_psd": np.asarray(hop.minimum_psd, np.float32),
                "conditional_minimum_psd": np.asarray(hop.conditional_minimum_psd, np.float32),
                "spp": np.asarray(hop.spp, np.float32),
                "speech_absence_probability": np.asarray(hop.speech_absence_probability, np.float32),
                "posterior_snr": np.asarray(hop.posterior_snr, np.float32),
                "prior_snr": np.asarray(hop.prior_snr, np.float32),
                "noise_features": np.asarray(hop.noise_features, np.float32),
                "noise_level_db": np.asarray(hop.noise_level_db, np.float32),
                "source_probability_per_mic": np.asarray(
                    hop.source_probability_per_mic, np.float32
                ),
            }
            if not self._imcra_spools:
                self._imcra_frequencies = np.array(frequencies, dtype=np.float32, copy=True)
                self._imcra_spools = {
                    name: self._new_spool(
                        f"i{self._IMCRA_SPOOL_NAMES[name]}",
                        dtype=np.dtype("<f4"),
                        item_shape=tuple(value.shape),
                    )
                    for name, value in streamed.items()
                }
            elif self._imcra_frequencies is None or not np.array_equal(
                frequencies, self._imcra_frequencies
            ):
                raise ValueError("chunk内IMCRA频率轴发生变化")
            for name, value in streamed.items():
                self._imcra_spools[name].append(value)
            self.imcra_hops.append({
                "session_id": hop.session_id,
                "stream_epoch": hop.stream_epoch,
                "start_sample": hop.start_sample,
                "end_sample": hop.end_sample,
                "source_sequence_ids": hop.source_sequence_ids,
                "algorithm_version": hop.algorithm_version,
                "state": hop.state,
                "array_source_probability_20ms": hop.array_source_probability_20ms,
            })
        self.end = block.end_sample

    def add_result(self, item: dict[str, Any]) -> None:
        raw = item.pop("raw_scores", None)
        normalized = item.pop("normalized_scores", None)
        self.results.append(item)
        if raw is not None and normalized is not None:
            self.spatial.append(
                (
                    int(item["window_id"]),
                    int(item["decision_sample"]),
                    np.asarray(raw, np.float32),
                    np.asarray(normalized, np.float32),
                )
            )

    @property
    def commit_journal(self) -> Path | None:
        return self._commit_journal

    @staticmethod
    def _rename_prepared_asset(partial: Path, final: Path) -> None:
        """Single patch point used by crash-recovery fault-injection tests."""

        final.parent.mkdir(parents=True, exist_ok=True)
        partial.replace(final)

    def mark_manifest_committed(self) -> None:
        if self._commit_journal is not None:
            self._commit_journal.unlink(missing_ok=True)
            self._commit_journal = None

    def close(self, session_id: str) -> dict[str, Any]:
        frame_count = self.end - self.start
        actual_stem = f"epoch{self.epoch:03d}_start{self.start:012d}_end{self.end:012d}"
        assets: list[dict[str, Any]] = []
        prepared: list[tuple[Path, Path]] = []
        for writer, key in (
            (self.native, "native_8ch"),
            (self.logical, "logical_8ch"),
            (self.physical, "physical_7ch"),
        ):
            if writer is not None:
                writer.close()
                partial = Path(str(self.paths[key]) + ".partial")
                final = self.paths[key].with_name(actual_stem + self.paths[key].suffix)
                self.paths[key] = final
                prepared.append((partial, final))
        if "physical_float" in self.paths:
            initial = self.paths["physical_float"]
            path = initial.with_name(actual_stem + initial.suffix)
            self.paths["physical_float"] = path
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = Path(str(path) + ".partial")
            if self._physical_float_spool is None:
                raise RuntimeError("physical float stream is missing")
            with partial.open("wb") as out:
                self._physical_float_spool.write_npy(out)
                out.flush()
                os.fsync(out.fileno())
            prepared.append((partial, path))
        for key, path in self.paths.items():
            channels = 8 if key in {"native_8ch", "logical_8ch"} else 7
            partial = next(source for source, final in prepared if final == path)
            assets.append(
                {
                    "kind": key,
                    "path": str(path.relative_to(self.root)),
                    "sha256": sha256_file(partial),
                    "sample_count": frame_count,
                    "channel_count": channels,
                    "sample_rate": 48000,
                    "dtype": "float32" if key == "physical_float" else "int16",
                    "stream_epoch": self.epoch,
                    "start_sample": self.start,
                    "end_sample": self.end,
                    "calibration_hash": self.calibration_hash,
                }
            )
        if self.settings["record_noise_spectrum"] and self.noise_spectra:
            path = self.root / "noise_spectrum" / f"{actual_stem}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = Path(str(path) + ".partial")
            starts = np.asarray([x[0] for x in self.noise_spectra], np.int64)
            ends = np.asarray([x[1] for x in self.noise_spectra], np.int64)
            timestamps = np.asarray([x[3] for x in self.noise_spectra], np.float64)
            frequencies = self._noise_frequencies
            psd = self._noise_psd_spool
            if frequencies is None or psd is None:
                raise RuntimeError("noise spectrum stream is incomplete")
            _write_streamed_npz(partial, {
                "start_samples": starts,
                "end_samples": ends,
                "timestamps": timestamps,
                "frequencies_hz": frequencies,
                "psd": psd,
                "n_fft": np.asarray(self.noise_spectra[0][2], np.int64),
                "estimator": np.asarray(self.noise_spectra[0][4]),
            })
            prepared.append((partial, path))
            assets.append(
                {
                    "kind": "noise_spectrum",
                    "path": str(path.relative_to(self.root)),
                    "sha256": sha256_file(partial),
                    "record_count": len(self.noise_spectra),
                    "channel_count": 7,
                    "frequency_bin_count": int(frequencies.size),
                    "stream_epoch": self.epoch,
                    "start_sample": self.start,
                    "end_sample": self.end,
                }
            )
        if self.settings["record_imcra"] and self.imcra_hops:
            path = self.root / "imcra" / f"{actual_stem}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = Path(str(path) + ".partial")
            probabilities = np.asarray([
                np.nan if item["array_source_probability_20ms"] is None
                else item["array_source_probability_20ms"]
                for item in self.imcra_hops
            ], np.float32)
            if self._imcra_frequencies is None or not self._imcra_spools:
                raise RuntimeError("IMCRA streams are incomplete")
            _write_streamed_npz(partial, {
                "start_samples": np.asarray(
                    [item["start_sample"] for item in self.imcra_hops], np.int64
                ),
                "end_samples": np.asarray(
                    [item["end_sample"] for item in self.imcra_hops], np.int64
                ),
                "source_sequence_ids": np.asarray(
                    [json.dumps(item["source_sequence_ids"]) for item in self.imcra_hops]
                ),
                "algorithm_versions": np.asarray(
                    [item["algorithm_version"] for item in self.imcra_hops]
                ),
                "states": np.asarray([item["state"] for item in self.imcra_hops]),
                "frequencies_hz": self._imcra_frequencies,
                **self._imcra_spools,
                "array_source_probability_20ms": probabilities,
            })
            prepared.append((partial, path))
            assets.append({
                "kind": "imcra",
                "path": str(path.relative_to(self.root)),
                "sha256": sha256_file(partial),
                "record_count": len(self.imcra_hops),
                "channel_count": 7,
                "hop_samples": 960,
                "frequency_bin_count": int(self._imcra_frequencies.size),
                "stream_epoch": self.epoch,
                "start_sample": self.start,
                "end_sample": self.end,
            })

        # The journal is durable before the first final rename.  Therefore a
        # crash after any Nth rename is unambiguous: unless the complete chunk
        # is already indexed by the manifest, recovery quarantines every
        # prepared partial and every already-renamed final.
        journal = self.root / (
            f".cj_{self.epoch:03d}_{self.start:012d}_{self.end:012d}.json"
        )
        atomic_json(journal, {
            "schema_version": "chunk_asset_commit_v1",
            "session_id": session_id,
            "created_at_utc": utc_now(),
            "chunk": {
                "stream_epoch": self.epoch,
                "start_sample": self.start,
                "end_sample": self.end,
            },
            "entries": [
                {
                    "partial_path": str(partial.relative_to(self.root)),
                    "final_path": str(final.relative_to(self.root)),
                    "sha256": sha256_file(partial),
                }
                for partial, final in prepared
            ],
        })
        self._commit_journal = journal
        for partial, final in prepared:
            self._rename_prepared_asset(partial, final)

        # Raw spools are implementation staging, not public assets.  They are
        # removed only after every prepared public asset has reached its final
        # path.  On an interrupted commit they remain *.partial and are picked
        # up by recovery together with the journal transaction.
        if self._physical_float_spool is not None:
            self._physical_float_spool.remove()
        if self._noise_psd_spool is not None:
            self._noise_psd_spool.remove()
        for spool in self._imcra_spools.values():
            spool.remove()
        if self._staging_root.exists():
            try:
                self._staging_root.rmdir()
            except OSError:
                # Any leftover partial is intentionally retained for the
                # recovery/quarantine scan instead of being silently deleted.
                pass
        return {
            "stream_epoch": self.epoch,
            "start_sample": self.start,
            "end_sample": self.end,
            "frame_count": frame_count,
            "clip_count_by_channel": self.clip_counts.tolist(),
            "assets": assets,
            "result_count": len(self.results),
        }


class RecordingStore:
    MODES = {"off", "manual", "continuous", "event"}
    MAX_RESULT_QUEUE_ITEMS = 256

    def __init__(
        self,
        data_root: str | Path,
        *,
        catalog: Catalog | None = None,
        config: object | None = None,
        chunk_seconds: int = 60,
        audio_queue_seconds: int = 10,
        result_queue_capacity: int = 256,
        pre_roll_seconds: int = 2,
        post_roll_seconds: int = 3,
        min_free_storage_gb: float = 5,
        max_storage_gb: float = 200,
        settings: dict[str, bool] | None = None,
    ):
        self.data_root = Path(data_root)
        self.catalog = catalog or Catalog(self.data_root / "catalog.sqlite")
        self._owns_catalog = catalog is None
        self._mapping_contract = {
            "board_i2s": {"MIC_D0": ["MIC0", "MIC1"], "MIC_D1": ["MIC2", "MIC3"], "MIC_D2": ["MIC4", "MIC5"], "MIC_D3": ["Center"]},
            "native_host_order": ["CH0", "CH1", "CH2", "CH3", "CH4", "CH5", "HardwareMix", "Center"],
            "logical_from_native": [0, 1, 2, 3, 4, 5, 7, 6],
            "observation_face": "microphone_face_from_above",
            "theta_zero": "center_to_MIC0_positive_x",
            "theta_direction": "counterclockwise",
            "hardware_mix_has_physical_coordinate": False,
            "physical_angles_deg": [0, 60, 120, 180, 240, 300, None],
            "ring_radius_m": 0.04,
            "speed_of_sound_mps": 343.0,
        }
        if config is not None:
            runtime = config.recording.runtime
            event = config.recording.event
            chunk_seconds = runtime.chunk_seconds
            audio_queue_seconds = runtime.audio_queue_seconds
            result_queue_capacity = runtime.result_queue_capacity
            pre_roll_seconds = event.pre_roll_seconds
            post_roll_seconds = event.post_roll_seconds
            min_free_storage_gb = runtime.min_free_storage_gb
            max_storage_gb = runtime.max_storage_gb
            settings = {
                name: getattr(runtime, name)
                for name in (
                    "record_native_8ch",
                    "record_logical_8ch",
                    "record_physical_7ch",
                    "record_physical_float32",
                    "record_results_jsonl",
                    "record_spatial_response",
                    "record_hotmaps",
                    "record_imcra",
                    "record_noise_spectrum",
                )
            }
            self._mapping_contract["logical_from_native"] = list(config.device.logical_channel_map)
            self._mapping_contract["ring_radius_m"] = config.hardware.ring_radius_m
            self._mapping_contract["speed_of_sound_mps"] = config.hardware.speed_of_sound_mps
        self.settings = settings or {
            "record_native_8ch": True,
            "record_logical_8ch": True,
            "record_physical_7ch": True,
            "record_physical_float32": True,
            "record_results_jsonl": True,
            "record_spatial_response": True,
            "record_hotmaps": True,
            "record_imcra": True,
            "record_noise_spectrum": True,
        }
        self.chunk_samples = chunk_seconds * 48000
        self.pre_roll_samples = pre_roll_seconds * 48000
        self.post_roll_samples = post_roll_seconds * 48000
        self.min_free_bytes = int(min_free_storage_gb * 1024**3)
        self.max_storage_bytes = int(max_storage_gb * 1024**3)
        audio_capacity = max(1, int(audio_queue_seconds) * 50)
        result_capacity = min(
            self.MAX_RESULT_QUEUE_ITEMS,
            max(1, int(result_queue_capacity)),
        )
        self.audio_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=audio_capacity)
        self.result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=result_capacity)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._session_id = None
        self._root = None
        self._metadata = None
        self.mode = "off"
        self.manual_active = False
        self._ring: deque[IngestedAudioBlock] = deque()
        self._event_until: dict[int, int] = {}
        self._event_written_until: dict[int, int] = {}
        self._active_event_audit: dict[int, dict[str, Any]] = {}
        self._manifest: dict[str, Any] = {}
        self._chunk: _Chunk | None = None
        # Results are held only until their recorded chunk can be finalized.
        # The hard count limit is a last-resort guard for a stalled/missing
        # watermark; normal operation releases one chunk at a time.
        self._all_results: list[dict[str, Any]] = []
        self._max_pending_results = max(
            result_capacity,
            (self.chunk_samples + self.pre_roll_samples + self.post_roll_samples) // 960 + 16,
        )
        self._accepted_intervals: dict[int, list[tuple[int, int]]] = {}
        self._result_assets_finalized: set[tuple[int, int, int]] = set()
        self._last_watermark: dict[int, int] = {}
        self._writer_watermarks: dict[int, int] = {}
        self._hotmap_sequences: set[tuple[int, int]] = set()
        self._hotmap_file: Any | None = None
        self._hotmap_partial_path: Path | None = None
        self._hotmap_count = 0
        self._worker_error: Exception | None = None
        self._stop_requested = threading.Event()
        self._stopping = False

    @staticmethod
    def _discard_queue(target: queue.Queue[tuple[str, Any]]) -> None:
        while True:
            try:
                target.get_nowait()
            except queue.Empty:
                return

    def start_session(self, session_id: str, metadata: SessionMetadata | dict[str, Any]) -> None:
        with self._lock:
            if self._session_id:
                raise RuntimeError("已有录音session")
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("上一个录音writer仍在停止中")
            # A failed writer can leave commands from the previous session in
            # either bounded queue.  They must never be consumed under a new
            # session/root.
            self._discard_queue(self.audio_queue)
            self._discard_queue(self.result_queue)
            self._stop_requested.clear()
            self._stopping = False
            meta = public_mapping(metadata)
            self._session_id = session_id
            self._metadata = meta
            now = datetime.now(timezone.utc)
            self._root = self.data_root / "runtime_sessions" / f"{now:%Y}" / f"{now:%m}" / session_id
            if self._root.exists():
                self._session_id = None
                self._root = None
                raise FileExistsError(f"录音session目录已存在: {session_id}")
            self._manifest = {
                "schema_version": "audio_session_v2",
                "session_id": session_id,
                "status": "open",
                "started_at_utc": utc_now(),
                "ended_at_utc": None,
                "stop_reason": None,
                "device_format": {"sample_rate": 48000, "channels": 8, "pcm_format": "s16-le", "layout": "interleaved"},
                "physical_channel_map": [0, 1, 2, 3, 4, 5, 7],
                "hardware_mix_channel": 6,
                "logical_channel_map": [0, 1, 2, 3, 4, 5, 7, 6],
                "observation_face": "microphone_face_top_view_mic0_positive_x_ccw",
                "channel_layouts": {
                    "native_8ch": ["CH0", "CH1", "CH2", "CH3", "CH4", "CH5", "HardwareMix", "Center"],
                    "logical_8ch": ["MIC0", "MIC1", "MIC2", "MIC3", "MIC4", "MIC5", "Center", "HardwareMix"],
                    "logical_from_native": [0, 1, 2, 3, 4, 5, 7, 6],
                    "physical_7ch": ["MIC0", "MIC1", "MIC2", "MIC3", "MIC4", "MIC5", "Center"],
                },
                "mapping_contract": dict(self._mapping_contract),
                "calibration_hash": meta.get("calibration_hash", ""),
                "calibration_revision": int(meta.get("calibration_revision", 0)),
                "calibration_version": meta.get("calibration_version", "unversioned"),
                "geometry_version": meta.get("geometry_version", "r6plus1_mic_face_ccw_v1"),
                "config_hash": meta.get("config_hash", ""),
                "config_revision": int(meta.get("config_revision", 0)),
                "git_commit": meta.get("git_commit"),
                "runtime": meta.get("runtime", {}),
                "algorithm_versions": meta.get("algorithm_versions", {}),
                "current_mode": "off",
                "recording_mode_history": [],
                "recorded_intervals": [],
                "chunks": [],
                "missing_intervals": [],
                "result_gaps": [],
                "result_watermarks": {},
                "event_triggers": [],
            }
            self._all_results.clear()
            self._accepted_intervals.clear()
            self._result_assets_finalized.clear()
            self._hotmap_sequences.clear()
            self._hotmap_file = None
            self._hotmap_partial_path = None
            self._hotmap_count = 0
            self._last_watermark.clear()
            self._writer_watermarks.clear()
            self._ring.clear()
            self._event_until.clear()
            self._event_written_until.clear()
            self.mode = "off"
            self.manual_active = False
            self._worker_error = None
            self._chunk = None
            # Persist the open lifecycle state before a writer can produce any
            # asset.  Empty sessions and crashes before the first chunk are
            # therefore recoverable and visible in the Catalog as well.
            assert self._root is not None
            try:
                write_manifest(self._root / "session_manifest.json", self._manifest)
                self.catalog.upsert_session(self._manifest, self._root)
            except Exception:
                # This root was proven absent above and belongs exclusively to
                # this start attempt.  Roll back both indexes before exposing
                # an active session or starting a worker.
                try:
                    with self.catalog.connection:
                        self.catalog.connection.execute(
                            "DELETE FROM sessions WHERE id=?", (session_id,)
                        )
                finally:
                    if self._root.exists():
                        shutil.rmtree(self._root)
                    self._session_id = None
                    self._root = None
                    self._manifest = {}
                raise
            self._thread = threading.Thread(target=self._run, name="recording-writer", daemon=True)
            self._thread.start()

    def set_recording_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError("录音模式必须是off/manual/continuous/event")
        with self._lock:
            if mode == self.mode:
                return
        # Capacity scans can traverse a large database.  Never hold the ingest
        # hot lock while doing filesystem I/O.
        if mode == "continuous":
            self._ensure_storage()
        with self._lock:
            if mode == self.mode:
                return
            previous_mode = self.mode
            was_recording = (
                previous_mode == "continuous"
                or (previous_mode == "manual" and self.manual_active)
                or (previous_mode == "event" and bool(self._event_written_until))
            )
            if was_recording:
                self._enqueue_boundary("recording_mode_change")
            self.mode = mode
            self.manual_active = False
            if previous_mode == "event" or mode == "event":
                self._reset_event_state()
            self._manifest.get("recording_mode_history", []).append({"at_utc": utc_now(), "mode": mode})
            self._manifest["current_mode"] = mode

    def start_recording(self) -> None:
        with self._lock:
            if self.mode != "manual":
                raise RuntimeError("只有manual模式接受Record命令")
            if self.manual_active:
                return
        self._ensure_storage()
        with self._lock:
            if self.mode != "manual":
                raise RuntimeError("容量检查期间录音模式已改变")
            if self._stopping:
                raise RuntimeError("录音session正在停止")
            if self.manual_active:
                return
            self.manual_active = True

    def pause_recording(self) -> None:
        with self._lock:
            if self.mode != "manual":
                raise RuntimeError("只有manual模式接受Pause命令")
            if not self.manual_active:
                return
            self.manual_active = False
            self._enqueue_boundary("manual_pause")

    def _enqueue_boundary(self, reason: str) -> bool:
        """Request a chunk boundary without ever blocking an ingest caller."""

        try:
            self.audio_queue.put_nowait(("boundary", None))
        except queue.Full:
            self._record_missing_interval(
                None,
                None,
                None,
                "recording_boundary_overflow",
                boundary_reason=reason,
            )
            return False
        return True

    def _reset_event_state(self) -> None:
        self._ring.clear()
        self._event_until.clear()
        self._event_written_until.clear()
        self._active_event_audit.clear()

    def _ensure_storage(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        runtime_root = self.data_root / "runtime_sessions"
        used = (
            sum(path.stat().st_size for path in runtime_root.rglob("*") if path.is_file())
            if runtime_root.exists()
            else 0
        )
        if shutil.disk_usage(self.data_root).free < self.min_free_bytes or used >= self.max_storage_bytes:
            raise OSError("recording_storage_full")

    def _record_hotmap(self, block: IngestedAudioBlock) -> None:
        hotmap = block.hotmap
        sequence_id = int(getattr(hotmap, "sequence_id", block.sequence_id)) if hotmap is not None else -1
        sequence_key = (int(block.stream_epoch), sequence_id)
        if (
            not self.settings["record_hotmaps"]
            or hotmap is None
            or sequence_key in self._hotmap_sequences
        ):
            return
        value = np.asarray(getattr(hotmap, "matrix", hotmap))
        if value.shape == (16, 16):
            assert self._root is not None
            if self._hotmap_file is None:
                self._root.mkdir(parents=True, exist_ok=True)
                path = self._root / "hotmaps.jsonl"
                self._hotmap_partial_path = Path(str(path) + ".partial")
                self._hotmap_file = self._hotmap_partial_path.open(
                    "w", encoding="utf-8", newline="\n"
                )
            self._hotmap_sequences.add(sequence_key)
            item = {
                "schema_version": "hotmap_record_v1",
                "sequence_id": sequence_id,
                "timestamp": float(getattr(hotmap, "timestamp", block.timestamp)),
                "received_at_utc": str(getattr(hotmap, "received_at", utc_now())),
                "stream_epoch": block.stream_epoch,
                "start_sample": block.start_sample,
                "values": np.asarray(value, dtype=np.uint8).tolist(),
            }
            self._hotmap_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            self._hotmap_count += 1

    def _accept_interval(self, epoch: int, start: int, end: int) -> None:
        intervals = self._accepted_intervals.setdefault(int(epoch), [])
        if intervals and start <= intervals[-1][1]:
            previous_start, previous_end = intervals[-1]
            intervals[-1] = (previous_start, max(previous_end, end))
        else:
            intervals.append((int(start), int(end)))

    def _record_missing_interval(
        self,
        epoch: int | None,
        start: int | None,
        end: int | None,
        reason: str,
        **details: Any,
    ) -> None:
        """Coalesce adjacent overflow ranges so a stalled disk cannot grow the manifest unboundedly."""

        intervals = self._manifest.setdefault("missing_intervals", [])
        item = {
            "stream_epoch": epoch,
            "start_sample": start,
            "end_sample": end,
            "reason": reason,
            **details,
        }
        if intervals:
            previous = intervals[-1]
            same_metadata = all(
                previous.get(key) == value
                for key, value in item.items()
                if key not in {"start_sample", "end_sample"}
            ) and all(
                item.get(key) == value
                for key, value in previous.items()
                if key not in {"start_sample", "end_sample"}
            )
            if (
                same_metadata
                and start is not None
                and end is not None
                and previous.get("end_sample") is not None
                and start <= int(previous["end_sample"])
            ):
                previous["end_sample"] = max(int(previous["end_sample"]), int(end))
                return
        intervals.append(item)

    def _is_accepted_decision(self, epoch: int, decision: int) -> bool:
        return any(
            start < decision <= end
            for start, end in self._accepted_intervals.get(int(epoch), ())
        )

    def _enqueue_audio_slice(
        self,
        block: IngestedAudioBlock,
        *,
        start: int | None = None,
        end: int | None = None,
        deduplicate_event: bool = False,
    ) -> bool:
        start = block.start_sample if start is None else max(start, block.start_sample)
        end = block.end_sample if end is None else min(end, block.end_sample)
        if deduplicate_event:
            start = max(start, self._event_written_until.get(block.stream_epoch, start))
        if start >= end:
            return False
        queued = block if start == block.start_sample and end == block.end_sample else _slice_block(block, start, end)
        if queued.end_sample - queued.start_sample > 960:
            raise ValueError("recording queue accepts at most 960 samples per item")
        try:
            self.audio_queue.put_nowait(("audio", queued))
        except queue.Full:
            self._record_missing_interval(
                block.stream_epoch, start, end, "recording_overflow"
            )
            return False
        self._accept_interval(block.stream_epoch, start, end)
        if deduplicate_event:
            self._event_written_until[block.stream_epoch] = end
        return True

    def append_audio(self, block: IngestedAudioBlock) -> None:
        with self._lock:
            if block.session_id != self._session_id:
                raise ValueError("录音block不属于当前session")
            if self._stopping:
                raise RuntimeError("录音session正在停止")
            for start in range(block.start_sample, block.end_sample, 960):
                piece = _slice_block(block, start, min(start + 960, block.end_sample))
                if self.mode == "event":
                    self._ring.append(piece)
                    floor = piece.end_sample - self.pre_roll_samples
                    while self._ring and (
                        self._ring[0].stream_epoch != piece.stream_epoch
                        or self._ring[0].end_sample <= floor
                    ):
                        self._ring.popleft()
                    event_end = self._event_until.get(piece.stream_epoch, -1)
                    if piece.start_sample < event_end:
                        self._enqueue_audio_slice(
                            piece, end=event_end, deduplicate_event=True
                        )
                elif self.mode == "continuous" or (
                    self.mode == "manual" and self.manual_active
                ):
                    self._enqueue_audio_slice(piece)

    def _prepare_result(self, result: DecisionRecord | object) -> dict[str, Any]:
        item = public_mapping(result)
        if item.get("session_id") != self._session_id:
            raise ValueError("算法结果不属于当前录音session")
        schema = item.setdefault("schema_version", "decision_record_v4")
        if schema != "decision_record_v4":
            raise ValueError("旧DecisionRecord v3仅支持读取，不能写入新录音")
        for key in ("raw_scores", "normalized_scores"):
            if item.get(key) is not None:
                item[key] = np.asarray(item[key], np.float32).copy()
        item["enhanced_waveforms"] = tuple(
            np.asarray(value, np.float32).copy() for value in item.get("enhanced_waveforms", ())
        )
        metadata = tuple(item.get("enhanced_audio", ()))
        if len(metadata) != len(item["enhanced_waveforms"]):
            raise ValueError("增强音频元数据与波形数量不一致")
        track_ids = tuple(value.get("track_id") for value in metadata)
        if track_ids and any(type(value) is not int or value <= 0 for value in track_ids):
            raise ValueError("DecisionRecord v4增强音频必须包含正整数track_id")
        if len(set(track_ids)) != len(track_ids):
            raise ValueError("同一窗口的增强音频track_id不能重复")
        return item

    @staticmethod
    def _result_routing_fields(result: DecisionRecord | object) -> dict[str, Any]:
        if isinstance(result, dict):
            return result
        return {
            "session_id": getattr(result, "session_id", None),
            "stream_epoch": getattr(result, "stream_epoch", -1),
            "decision_sample": getattr(result, "decision_sample", getattr(result, "sample", None)),
            "window_id": getattr(result, "window_id", None),
        }

    def _should_retain_result(self, item: dict[str, Any]) -> bool:
        epoch = int(item.get("stream_epoch", -1))
        decision = item.get("decision_sample", item.get("sample"))
        if decision is None:
            return False
        decision = int(decision)
        if self._is_accepted_decision(epoch, decision):
            return not any(
                finalized_epoch == epoch and start < decision <= end
                for finalized_epoch, start, end in self._result_assets_finalized
            )
        # Event mode must retain a bounded result pre-roll before an event is
        # known.  All other non-recording modes discard immediately, before
        # copying large enhanced waveforms.
        return self.mode == "event"

    def _result_overflow_gap(
        self,
        *,
        item: dict[str, Any] | None = None,
        watermark: dict[str, Any] | None = None,
        reason: str = "result_overflow",
    ) -> dict[str, Any]:
        source = item or watermark or {}
        epoch = int(source.get("stream_epoch", -1))
        decision = None if item is None else item.get("decision_sample", item.get("sample"))
        sample = None if watermark is None else int(watermark["sample"])
        previous = self._last_watermark.get(epoch, -1)
        gap = {
            "reason": reason,
            "stream_epoch": epoch,
            "window_id": None if item is None else item.get("window_id"),
            "decision_sample": None if decision is None else int(decision),
            "gap_start_sample": previous,
            "gap_end_sample": sample if sample is not None else (
                None if decision is None else int(decision)
            ),
        }
        gaps = self._manifest["result_gaps"]
        if gaps:
            previous_gap = gaps[-1]
            if (
                previous_gap.get("reason") == reason
                and int(previous_gap.get("stream_epoch", -2)) == epoch
                and previous_gap.get("gap_end_sample") is not None
                and gap["gap_end_sample"] is not None
                and int(gap["gap_end_sample"]) >= int(previous_gap["gap_end_sample"])
            ):
                previous_gap["gap_end_sample"] = gap["gap_end_sample"]
                previous_gap["last_window_id"] = gap["window_id"]
                previous_gap["last_decision_sample"] = gap["decision_sample"]
                previous_gap["overflow_count"] = int(
                    previous_gap.get("overflow_count", 1)
                ) + 1
                return previous_gap
        gaps.append(gap)
        return gap

    def append_result(self, result: DecisionRecord | object) -> bool:
        raw = self._result_routing_fields(result)
        if raw.get("session_id") != self._session_id:
            raise ValueError("算法结果不属于当前录音session")
        with self._lock:
            if not self._should_retain_result(raw):
                return False
            if self.result_queue.full():
                self._result_overflow_gap(item=raw)
                return False
        item = self._prepare_result(result)
        try:
            self.result_queue.put_nowait(("result", item))
        except queue.Full:
            with self._lock:
                self._result_overflow_gap(item=item)
            return False
        return True

    def advance_result_watermark(self, watermark: ResultWatermark | object) -> bool:
        item = public_mapping(watermark)
        epoch = int(item["stream_epoch"])
        sample = int(item["sample"])
        with self._lock:
            if sample < self._last_watermark.get(epoch, -1):
                raise ValueError("result watermark不可倒退")
            if (
                not self._accepted_intervals.get(epoch)
                and self.mode not in {"continuous", "event"}
                and not (self.mode == "manual" and self.manual_active)
            ):
                return False
            try:
                self.result_queue.put_nowait(("watermark", item))
            except queue.Full:
                self._result_overflow_gap(watermark=item)
                return False
            # Producer-side monotonic state advances only after the command is
            # accepted.  A full queue can therefore never create a false
            # complete watermark.
            self._last_watermark[epoch] = sample
        return True

    def append_result_with_watermark(
        self,
        result: DecisionRecord | object,
        watermark: ResultWatermark | object,
    ) -> bool:
        """Atomically accept one decision and its progress watermark.

        The writer observes both or neither, eliminating the split-queue race
        where a result succeeds but its watermark fails (or vice versa).
        """

        raw_result = self._result_routing_fields(result)
        raw_watermark = public_mapping(watermark)
        if raw_result.get("session_id") != self._session_id:
            raise ValueError("算法结果不属于当前录音session")
        if raw_watermark.get("session_id") != self._session_id:
            raise ValueError("result watermark不属于当前录音session")
        result_epoch = int(raw_result["stream_epoch"])
        watermark_epoch = int(raw_watermark["stream_epoch"])
        if result_epoch != watermark_epoch:
            raise ValueError("result与watermark的stream_epoch不一致")
        sample = int(raw_watermark["sample"])
        decision = raw_result.get("decision_sample", raw_result.get("sample"))
        if decision is not None and int(decision) > sample:
            raise ValueError("result decision_sample不能晚于watermark")
        with self._lock:
            if sample < self._last_watermark.get(result_epoch, -1):
                raise ValueError("result watermark不可倒退")
            if not self._should_retain_result(raw_result):
                return False
            if self.result_queue.full():
                self._result_overflow_gap(item=raw_result, watermark=raw_watermark)
                return False
        item = self._prepare_result(result)
        with self._lock:
            try:
                self.result_queue.put_nowait(("result_with_watermark", (item, raw_watermark)))
            except queue.Full:
                self._result_overflow_gap(item=item, watermark=raw_watermark)
                return False
            self._last_watermark[result_epoch] = sample
        return True

    def trigger_event(self, result: DecisionRecord | object, reason: str = "voice_detection") -> None:
        item = public_mapping(result)
        if item.get("status") not in {"ok", "degraded"} or int(item.get("voice_direction_count", 0)) <= 0:
            return
        epoch = int(item["stream_epoch"])
        decision = int(item["decision_sample"])
        start = max(0, decision - self.pre_roll_samples)
        end = decision + self.post_roll_samples
        with self._lock:
            if self.mode != "event" or self._stopping:
                return
            # A trigger whose pre-roll overlaps the current post-roll belongs
            # to the same continuous event.  Extending an already-open event
            # must stay O(1): in particular it must not recursively rescan the
            # whole recording database at the 50 Hz decision rate.
            if start <= self._event_until.get(epoch, -1):
                self._extend_event_locked(item, reason, epoch, decision, end)
                return

        # Capacity scans may traverse a large database, so run one only when a
        # genuinely new event segment is about to start and never hold the
        # ingest hot lock while doing filesystem I/O.
        self._ensure_storage()
        with self._lock:
            if self.mode != "event" or self._stopping:
                return
            # Another trigger may have opened this event while the capacity
            # scan ran.  Recheck and merge instead of creating a duplicate.
            if start <= self._event_until.get(epoch, -1):
                self._extend_event_locked(item, reason, epoch, decision, end)
                return
            self._event_until[epoch] = end
            for block in tuple(self._ring):
                if block.stream_epoch == epoch:
                    self._enqueue_audio_slice(block, start=start, end=end, deduplicate_event=True)
            audit = {
                # Retain the original fields for manifest/API compatibility.
                "window_id": item.get("window_id"),
                "stream_epoch": epoch,
                "decision_sample": decision,
                "reason": reason,
                # One bounded record describes the whole merged event segment.
                "first_window_id": item.get("window_id"),
                "last_window_id": item.get("window_id"),
                "first_decision_sample": decision,
                "last_decision_sample": decision,
                "start_sample": start,
                "end_sample": end,
                "trigger_count": 1,
            }
            self._manifest["event_triggers"].append(audit)
            self._active_event_audit[epoch] = audit

    def _extend_event_locked(
        self,
        item: dict[str, Any],
        reason: str,
        epoch: int,
        decision: int,
        end: int,
    ) -> None:
        """Extend one active event while ``self._lock`` is held."""

        previous_end = self._event_until.get(epoch, -1)
        self._event_until[epoch] = max(previous_end, end)
        if decision > previous_end:
            # The new trigger may arrive after the previous post-roll ended
            # while its pre-roll still overlaps that event.  Replay only that
            # intervening ring-buffer interval; otherwise the audit would say
            # "merged" while the recorded audio contained a hole.
            for block in tuple(self._ring):
                if block.stream_epoch == epoch:
                    self._enqueue_audio_slice(
                        block,
                        start=max(0, previous_end),
                        end=end,
                        deduplicate_event=True,
                    )
        audit = self._active_event_audit.get(epoch)
        if audit is None:
            # Defensive compatibility for a session restored or constructed
            # before the merged-audit state was initialized.
            start = max(0, decision - self.pre_roll_samples)
            audit = {
                "window_id": item.get("window_id"),
                "stream_epoch": epoch,
                "decision_sample": decision,
                "reason": reason,
                "first_window_id": item.get("window_id"),
                "first_decision_sample": decision,
                "start_sample": start,
                "trigger_count": 0,
            }
            self._manifest["event_triggers"].append(audit)
            self._active_event_audit[epoch] = audit
        audit["last_window_id"] = item.get("window_id")
        audit["last_decision_sample"] = decision
        audit["end_sample"] = self._event_until[epoch]
        audit["trigger_count"] = int(audit.get("trigger_count", 0)) + 1
        if audit.get("reason") != reason:
            audit["reason"] = "multiple_reasons"

    def _new_chunk(self, block: IngestedAudioBlock) -> _Chunk:
        assert self._root is not None
        boundary = ((block.start_sample // self.chunk_samples) + 1) * self.chunk_samples
        return _Chunk(
            self._root,
            block,
            boundary,
            self.settings,
            self._metadata.get("config_hash", ""),
            self._metadata.get("calibration_hash", ""),
        )

    def _close_chunk(self) -> None:
        if self._chunk is None:
            return
        chunk = self._chunk
        info = chunk.close(self._session_id)
        self._manifest["chunks"].append(info)
        self._manifest["recorded_intervals"].append(
            {k: info[k] for k in ("stream_epoch", "start_sample", "end_sample")}
        )
        self._chunk = None
        self._hotmap_sequences.clear()
        self._finalize_ready_result_chunks()
        # Persist an open manifest at every sealed chunk.  If the process dies
        # before stop_session(), already-renamed assets are still indexed and
        # the next recovery/catalog pass can distinguish them from orphans.
        if self._root is not None:
            write_manifest(self._root / "session_manifest.json", self._manifest)
            chunk.mark_manifest_committed()

    def _write_audio(self, block: IngestedAudioBlock) -> None:
        self._record_hotmap(block)
        cursor = block.start_sample
        while cursor < block.end_sample:
            if self._chunk is None or self._chunk.epoch != block.stream_epoch or cursor != self._chunk.end:
                self._close_chunk()
                self._chunk = self._new_chunk(_slice_block(block, cursor, block.end_sample))
            end = min(block.end_sample, self._chunk.target_end)
            self._chunk.append(_slice_block(block, cursor, end))
            cursor = end
            if self._chunk.end >= self._chunk.target_end:
                self._close_chunk()

    def _run(self) -> None:
        try:
            while True:
                try:
                    kind, value = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    # Event mode can receive results for pre-roll before any
                    # audio is selected.  Drain them while idle so the bounded
                    # pre-roll ring never accumulates in the input queue.
                    self._drain_results()
                    if self._stop_requested.is_set():
                        break
                    continue
                if kind == "stop":
                    break
                if kind == "audio":
                    self._write_audio(value)
                elif kind == "boundary":
                    self._close_chunk()
                self._drain_results()
                if self._stop_requested.is_set() and self.audio_queue.empty():
                    break
            self._drain_results()
            self._close_chunk()
        except Exception as exc:
            self._worker_error = exc

    def _drain_results(self) -> None:
        while True:
            try:
                kind, value = self.result_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "result":
                self._consume_result(value)
            elif kind == "watermark":
                self._consume_watermark(value)
            elif kind == "result_with_watermark":
                result, watermark = value
                self._consume_result(result)
                self._consume_watermark(watermark)
        self._spill_ready_enhanced_audio()
        self._trim_pending_results()
        self._finalize_ready_result_chunks()

    def _consume_result(self, value: dict[str, Any]) -> None:
        self._all_results.append(value)

    def _consume_watermark(self, value: dict[str, Any]) -> None:
        epoch = int(value["stream_epoch"])
        sample = int(value["sample"])
        self._writer_watermarks[epoch] = max(sample, self._writer_watermarks.get(epoch, -1))
        self._manifest["result_watermarks"][str(epoch)] = self._writer_watermarks[epoch]
        for dropped in value.get("dropped_windows", ()):
            dropped_item = dict(dropped)
            # The staged runtime writes one terminal DecisionRecord for every
            # admitted window.  Legacy producers that emit only a watermark
            # marker still receive a standalone row.
            already_recorded = any(
                int(item.get("stream_epoch", -1)) == epoch
                and item.get("window_id") == dropped_item.get("window_id")
                for item in self._all_results
            )
            if not already_recorded:
                self._all_results.append(
                    {
                        "record_type": "dropped_window",
                        "schema_version": "decision_record_v4",
                        "session_id": value["session_id"],
                        "stream_epoch": value["stream_epoch"],
                        **dropped_item,
                    }
                )

    @staticmethod
    def _result_decision(item: dict[str, Any]) -> int | None:
        value = item.get("decision_sample", item.get("sample"))
        return None if value is None else int(value)

    def _decision_has_written_audio(self, epoch: int, decision: int) -> bool:
        if (
            self._chunk is not None
            and self._chunk.epoch == epoch
            and self._chunk.start < decision <= self._chunk.end
        ):
            return True
        return any(
            int(chunk["stream_epoch"]) == epoch
            and int(chunk["start_sample"]) < decision <= int(chunk["end_sample"])
            for chunk in self._manifest.get("chunks", ())
        )

    def _spill_ready_enhanced_audio(self) -> None:
        """Write large enhanced waveforms once their audio interval exists.

        Result metadata and small spatial arrays remain in memory until the
        chunk watermark completes, but 320 ms waveforms do not.  Event
        pre-roll results are intentionally kept in RAM only while provisional
        so an event that never triggers cannot create orphan assets.
        """

        if self._root is None:
            return
        for item in self._all_results:
            waveforms = item.get("enhanced_waveforms", ())
            if not waveforms:
                continue
            decision = self._result_decision(item)
            epoch = int(item.get("stream_epoch", -1))
            if decision is None or not self._decision_has_written_audio(epoch, decision):
                continue
            assets: list[dict[str, Any]] = []
            metadata = item.get("enhanced_audio", ())
            for index, (audio_meta, waveform) in enumerate(zip(metadata, waveforms, strict=True)):
                track_id = int(audio_meta["track_id"])
                theta_mdeg = int(round(float(audio_meta["theta_deg"]) * 1000))
                name = (
                    f"epoch{epoch:03d}_window{int(item['window_id']):012d}_"
                    f"decision{decision:012d}_track{track_id:06d}_theta{theta_mdeg:06d}_{index}.wav"
                )
                path = self._root / "enhanced_audio" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = Path(str(path) + ".partial")
                with wave.open(str(partial), "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(48_000)
                    writer.writeframes(pcm16_bytes(np.asarray(waveform, np.float32)[:, None]))
                assets.append({
                    "kind": "enhanced_audio",
                    "path": str(path.relative_to(self._root)),
                    "sha256": sha256_file(partial),
                    "sample_rate": 48_000,
                    "channel_count": 1,
                    "sample_count": int(len(waveform)),
                    "stream_epoch": epoch,
                    "window_id": int(item["window_id"]),
                    "decision_sample": decision,
                    "track_id": track_id,
                    # Internal transaction metadata.  It is removed before
                    # the public session manifest is committed.
                    "_partial_path": str(partial.relative_to(self._root)),
                    **_json_ready(dict(audio_meta)),
                })
            item["_enhanced_audio_assets"] = tuple(assets)
            item["enhanced_waveforms"] = ()

    def _trim_pending_results(self) -> None:
        if self.mode == "event" and self._all_results:
            latest_by_epoch: dict[int, int] = {}
            for item in self._all_results:
                decision = self._result_decision(item)
                if decision is not None:
                    epoch = int(item.get("stream_epoch", -1))
                    latest_by_epoch[epoch] = max(decision, latest_by_epoch.get(epoch, decision))
            kept: list[dict[str, Any]] = []
            for item in self._all_results:
                decision = self._result_decision(item)
                epoch = int(item.get("stream_epoch", -1))
                if decision is None:
                    continue
                floor = latest_by_epoch.get(epoch, decision) - self.pre_roll_samples
                if self._is_accepted_decision(epoch, decision) or decision > floor:
                    kept.append(item)
            self._all_results = kept
        if len(self._all_results) <= self._max_pending_results:
            return
        overflow = len(self._all_results) - self._max_pending_results
        removed = self._all_results[:overflow]
        del self._all_results[:overflow]
        for item in removed:
            self._result_overflow_gap(item=item, reason="result_retention_overflow")

    def _release_results_for_chunk(self, chunk: dict[str, Any]) -> None:
        epoch = int(chunk["stream_epoch"])
        start = int(chunk["start_sample"])
        end = int(chunk["end_sample"])
        self._all_results = [
            item
            for item in self._all_results
            if not (
                int(item.get("stream_epoch", -1)) == epoch
                and (decision := self._result_decision(item)) is not None
                and start < decision <= end
            )
        ]
        intervals = self._accepted_intervals.get(epoch, [])
        remaining: list[tuple[int, int]] = []
        for interval_start, interval_end in intervals:
            if interval_end <= end:
                continue
            if interval_start < end:
                remaining.append((end, interval_end))
            else:
                remaining.append((interval_start, interval_end))
        if remaining:
            self._accepted_intervals[epoch] = remaining
        else:
            self._accepted_intervals.pop(epoch, None)

    def _finalize_ready_result_chunks(self) -> None:
        ready = [
            chunk
            for chunk in self._manifest.get("chunks", ())
            if (
                (int(chunk["stream_epoch"]), int(chunk["start_sample"]), int(chunk["end_sample"]))
                not in self._result_assets_finalized
                and self._writer_watermarks.get(int(chunk["stream_epoch"]), -1)
                >= int(chunk["end_sample"])
            )
        ]
        if ready:
            self._rewrite_result_assets(ready)

    def _write_hotmaps(self) -> None:
        if not self._root or self._hotmap_count <= 0:
            return
        path = self._root / "hotmaps.jsonl"
        if self._hotmap_file is not None:
            self._hotmap_file.flush()
            os.fsync(self._hotmap_file.fileno())
            self._hotmap_file.close()
            self._hotmap_file = None
        partial = self._hotmap_partial_path
        if partial is not None and partial.exists():
            partial.replace(path)
        self._manifest["hotmaps"] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "count": self._hotmap_count,
        }

    def _commit_pending_enhanced_assets(self) -> Path | None:
        """Commit enhanced WAV partials under a crash-recovery journal."""

        if self._root is None:
            return None
        pending: list[tuple[dict[str, Any], Path, Path]] = []
        for chunk in self._manifest.get("chunks", ()):
            for asset in chunk.get("assets", ()):
                partial_name = asset.get("_partial_path")
                if asset.get("kind") != "enhanced_audio" or partial_name is None:
                    continue
                pending.append((asset, self._root / str(partial_name), self._root / str(asset["path"])))
        if not pending:
            return None
        journal = self._root / "enhanced_asset_commit.json"
        atomic_json(journal, {
            "schema_version": "enhanced_asset_commit_v2",
            "session_id": self._session_id,
            "created_at_utc": utc_now(),
            "entries": [
                {
                    "partial_path": str(partial.relative_to(self._root)),
                    "final_path": str(final.relative_to(self._root)),
                    "sha256": asset["sha256"],
                    "stream_epoch": asset["stream_epoch"],
                    "window_id": asset["window_id"],
                    "decision_sample": asset["decision_sample"],
                    "track_id": asset["track_id"],
                }
                for asset, partial, final in pending
            ],
        })
        for asset, partial, final in pending:
            final.parent.mkdir(parents=True, exist_ok=True)
            if partial.exists():
                partial.replace(final)
            elif not final.exists():
                raise FileNotFoundError(f"增强音频提交源丢失: {partial}")
            asset.pop("_partial_path", None)
        return journal

    def _rewrite_result_assets(self, chunks: list[dict[str, Any]] | None = None) -> None:
        if self._root is None:
            return
        targets = self._manifest["chunks"] if chunks is None else chunks
        for chunk in targets:
            epoch, start, end = chunk["stream_epoch"], chunk["start_sample"], chunk["end_sample"]
            chunk_key = (int(epoch), int(start), int(end))
            if chunk_key in self._result_assets_finalized:
                continue
            chosen = [
                item
                for item in self._all_results
                if int(item.get("stream_epoch", -1)) == epoch
                and (decision := item.get("decision_sample", item.get("sample"))) is not None
                and start < int(decision) <= end
            ]
            stem = f"epoch{epoch:03d}_start{start:012d}_end{end:012d}"
            assets = {x["kind"]: x for x in chunk["assets"]}
            if self.settings["record_results_jsonl"]:
                path = self._root / "results" / f"{stem}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                header = {
                    "record_type": "chunk_header",
                    "schema_version": "decision_chunk_v1",
                    "session_id": self._session_id,
                    "stream_epoch": epoch,
                    "start_sample": start,
                    "end_sample": end,
                    "config_hash": self._metadata.get("config_hash", ""),
                }
                partial = Path(str(path) + ".partial")
                with partial.open("w", encoding="utf-8", newline="\n") as out:
                    out.write(json.dumps(header, ensure_ascii=False) + "\n")
                    for original in chosen:
                        item = dict(original)
                        for key in (
                            "raw_scores",
                            "normalized_scores",
                            "enhanced_waveforms",
                            "_enhanced_audio_assets",
                        ):
                            item.pop(key, None)
                        item.setdefault("record_type", "decision")
                        item.setdefault("schema_version", "decision_record_v4")
                        out.write(json.dumps(_json_ready(item), ensure_ascii=False) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
                partial.replace(path)
                entry = {
                    "kind": "results",
                    "path": str(path.relative_to(self._root)),
                    "sha256": sha256_file(path),
                    "record_count": len(chosen),
                    "stream_epoch": epoch,
                    "start_sample": start,
                    "end_sample": end,
                }
                chunk["assets"].append(entry)
                assets["results"] = entry
            if self.settings["record_spatial_response"]:
                spatial = [
                    x for x in chosen if x.get("raw_scores") is not None and x.get("normalized_scores") is not None
                ]
                path = self._root / "spatial_response" / f"{stem}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = Path(str(path) + ".partial")
                ids = np.asarray([x["window_id"] for x in spatial], np.int64)
                samples = np.asarray([x["decision_sample"] for x in spatial], np.int64)
                doa_starts = np.asarray([x["doa_range"][0] for x in spatial], np.int64)
                doa_ends = np.asarray([x["doa_range"][1] for x in spatial], np.int64)
                raw = (
                    np.stack([np.asarray(x["raw_scores"], np.float32) for x in spatial])
                    if spatial
                    else np.empty((0, 360), np.float32)
                )
                normalized = (
                    np.stack([np.asarray(x["normalized_scores"], np.float32) for x in spatial])
                    if spatial
                    else np.empty((0, 360), np.float32)
                )
                with partial.open("wb") as out:
                    np.savez_compressed(
                        out,
                        window_ids=ids,
                        decision_samples=samples,
                        doa_start_samples=doa_starts,
                        doa_end_samples=doa_ends,
                        theta_degrees=np.arange(360, dtype=np.float32),
                        raw_scores=raw,
                        normalized_scores=normalized,
                    )
                partial.replace(path)
                entry = {
                    "kind": "spatial_response",
                    "path": str(path.relative_to(self._root)),
                    "sha256": sha256_file(path),
                    "record_count": len(spatial),
                    "angle_count": 360,
                    "stream_epoch": epoch,
                    "start_sample": start,
                    "end_sample": end,
                }
                chunk["assets"].append(entry)
                assets["spatial_response"] = entry
            def aligned(item: dict[str, Any]) -> dict[str, Any]:
                return {
                    "session_id": self._session_id,
                    "stream_epoch": epoch,
                    "window_id": item.get("window_id"),
                    "decision_sample": item.get("decision_sample"),
                    "doa_start_sample": item.get("doa_range", (None, None))[0],
                    "doa_end_sample": item.get("doa_range", (None, None))[1],
                    "context_start_sample": item.get("context_range", (None, None))[0],
                    "context_end_sample": item.get("context_range", (None, None))[1],
                }

            def write_jsonl(kind: str, records: list[dict[str, Any]], schema: str) -> None:
                if not records:
                    return
                path = self._root / kind / f"{stem}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = Path(str(path) + ".partial")
                with partial.open("w", encoding="utf-8", newline="\n") as out:
                    out.write(json.dumps({
                        "record_type": "chunk_header",
                        "schema_version": schema,
                        "session_id": self._session_id,
                        "stream_epoch": epoch,
                        "start_sample": start,
                        "end_sample": end,
                    }, ensure_ascii=False) + "\n")
                    for record in records:
                        out.write(json.dumps(_json_ready(record), ensure_ascii=False) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
                partial.replace(path)
                entry = {
                    "kind": kind,
                    "path": str(path.relative_to(self._root)),
                    "sha256": sha256_file(path),
                    "record_count": len(records),
                    "stream_epoch": epoch,
                    "start_sample": start,
                    "end_sample": end,
                }
                chunk["assets"].append(entry)

            gate_records = [
                {**aligned(item), "record_type": "gate", **dict(item["gate_decision"])}
                for item in chosen if item.get("gate_decision") is not None
            ]
            write_jsonl("gate", gate_records, "gate_chunk_v1")
            l4_records = [
                {**aligned(item), "record_type": "l4", "detections": item.get("detections", ()), **dict(item["l4_result"])}
                for item in chosen if item.get("l4_result") is not None
            ]
            write_jsonl("l4", l4_records, "l4_chunk_v1")

            for item in chosen:
                for asset in item.get("_enhanced_audio_assets", ()):
                    chunk["assets"].append(dict(asset))
                metadata = item.get("enhanced_audio", ())
                waveforms = item.get("enhanced_waveforms", ())
                pending_pairs = zip(metadata, waveforms, strict=True) if waveforms else ()
                for index, (audio_meta, waveform) in enumerate(pending_pairs):
                    track_id = int(audio_meta["track_id"])
                    theta_mdeg = int(round(float(audio_meta["theta_deg"]) * 1000))
                    name = (
                        f"epoch{epoch:03d}_window{int(item['window_id']):012d}_"
                        f"decision{int(item['decision_sample']):012d}_track{track_id:06d}_"
                        f"theta{theta_mdeg:06d}_{index}.wav"
                    )
                    path = self._root / "enhanced_audio" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    partial = Path(str(path) + ".partial")
                    with wave.open(str(partial), "wb") as writer:
                        writer.setnchannels(1)
                        writer.setsampwidth(2)
                        writer.setframerate(48_000)
                        writer.writeframes(pcm16_bytes(np.asarray(waveform, np.float32)[:, None]))
                    chunk["assets"].append({
                        "kind": "enhanced_audio",
                        "path": str(path.relative_to(self._root)),
                        "sha256": sha256_file(partial),
                        "sample_rate": 48_000,
                        "channel_count": 1,
                        "sample_count": int(len(waveform)),
                        "stream_epoch": epoch,
                        "window_id": int(item["window_id"]),
                        "decision_sample": int(item["decision_sample"]),
                        "track_id": track_id,
                        "_partial_path": str(partial.relative_to(self._root)),
                        **_json_ready(dict(audio_meta)),
                    })
            chunk["result_count"] = len(chosen)
            watermark = self._writer_watermarks.get(epoch)
            has_gap = self._chunk_has_result_gap(int(epoch), int(start), int(end))
            if watermark is not None and watermark >= end and not has_gap:
                chunk["result_status"] = "complete"
            elif chosen or watermark is not None or has_gap:
                chunk["result_status"] = "result_incomplete"
                self._manifest["result_gaps"].append({
                    "reason": "watermark_before_chunk_end",
                    "stream_epoch": epoch,
                    "chunk_end_sample": end,
                    "watermark_sample": watermark,
                    "covered_by_result_gap": has_gap,
                })
            else:
                chunk["result_status"] = "not_recorded"
            self._result_assets_finalized.add(chunk_key)
            self._release_results_for_chunk(chunk)

    def _chunk_has_result_gap(self, epoch: int, start: int, end: int) -> bool:
        for gap in self._manifest.get("result_gaps", ()):
            if gap.get("reason") not in {"result_overflow", "result_retention_overflow"}:
                continue
            gap_epoch = int(gap.get("stream_epoch", -1))
            if gap_epoch != epoch:
                continue
            decision = gap.get("decision_sample")
            if decision is not None and start < int(decision) <= end:
                return True
            gap_start = gap.get("gap_start_sample")
            gap_end = gap.get("gap_end_sample")
            if gap_start is not None and gap_end is not None:
                # Both intervals use the absolute timeline convention
                # (start, end] for completed decisions/watermarks.
                if int(gap_start) < end and int(gap_end) > start:
                    return True
        return False

    def stop_session(self, reason: str = "normal") -> dict[str, Any]:
        with self._lock:
            if not self._session_id:
                raise RuntimeError("没有活动session")
            self.manual_active = False
            self._stopping = True
            # Do not enqueue a sentinel here.  A failed writer can leave this
            # bounded queue full forever.  The event wakes an idle worker via
            # its bounded get timeout and lets a healthy worker drain all
            # commands already accepted before stopping.
            self._stop_requested.set()
            thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=10)
        if thread and thread.is_alive():
            self._manifest["status"] = "incomplete"
            self._manifest["writer_error"] = "recording writer did not stop within 10 seconds"
            self._manifest["ended_at_utc"] = utc_now()
            self._manifest["stop_reason"] = reason
            # The worker still owns open files and mutable manifest state.
            # Reporting a successful stop would let the runtime clear its
            # session flag and lose the only safe retry path.
            raise RuntimeError("recording writer did not stop within 10 seconds")
        elif self._worker_error:
            self._manifest["status"] = "corrupt"
            self._manifest["writer_error"] = repr(self._worker_error)
        else:
            self._manifest["status"] = "complete"
        # A worker that failed before its normal final drain can leave valid
        # results queued.  It is safe for the caller to drain them now because
        # the worker has exited.
        self._drain_results()
        self._rewrite_result_assets()
        self._write_hotmaps()
        enhanced_commit_journal = self._commit_pending_enhanced_assets()
        if self._manifest["status"] == "complete" and any(
            chunk.get("result_status") == "result_incomplete" for chunk in self._manifest["chunks"]
        ):
            self._manifest["status"] = "result_incomplete"
        self._manifest["ended_at_utc"] = utc_now()
        self._manifest["stop_reason"] = reason
        assert self._root is not None
        append_audit(self._root, "session_finalized", {"status": self._manifest["status"]})
        write_manifest(self._root / "session_manifest.json", self._manifest)
        if enhanced_commit_journal is not None:
            enhanced_commit_journal.unlink(missing_ok=True)
        self.catalog.upsert_session(self._manifest, self._root)
        result = dict(self._manifest)
        with self._lock:
            self._all_results.clear()
            self._accepted_intervals.clear()
            self._result_assets_finalized.clear()
            self._hotmap_sequences.clear()
            self._session_id = None
            self._thread = None
            self._root = None
            self._stopping = False
            self._discard_queue(self.audio_queue)
            self._discard_queue(self.result_queue)
        return result

    def recover_partials(self) -> list[Path]:
        quarantine = self.data_root / "quarantine" / "partial_recovery"
        recovered: list[Path] = []

        def quarantine_path(path: Path) -> None:
            if not path.exists() or quarantine in path.parents:
                return
            quarantine.mkdir(parents=True, exist_ok=True)
            target = quarantine / (path.name + "." + str(abs(hash(str(path.resolve())))))
            path.replace(target)
            recovered.append(target)

        def safe_path(root: Path, relative: object) -> Path | None:
            if not isinstance(relative, str) or not relative:
                return None
            candidate = (root / relative).resolve()
            return candidate if candidate.is_relative_to(root.resolve()) else None

        def indexed_assets(root: Path) -> dict[str, dict[str, Any]]:
            manifest_path = root / "session_manifest.json"
            if not manifest_path.exists():
                return {}
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                return {
                    str(asset["path"]): asset
                    for chunk in manifest.get("chunks", ())
                    for asset in chunk.get("assets", ())
                    if isinstance(asset, dict) and isinstance(asset.get("path"), str)
                }
            except (OSError, ValueError, TypeError, KeyError):
                return {}

        def valid_committed_entry(
            root: Path,
            entry: dict[str, Any],
            indexed: dict[str, dict[str, Any]],
            *,
            reject_internal_partial: bool = False,
        ) -> bool:
            relative = entry.get("final_path")
            asset = indexed.get(str(relative))
            final = safe_path(root, relative)
            expected = entry.get("sha256")
            if (
                asset is None
                or final is None
                or not final.is_file()
                or not isinstance(expected, str)
                or asset.get("sha256") != expected
                or (reject_internal_partial and asset.get("_partial_path") is not None)
            ):
                return False
            try:
                return sha256_file(final) == expected
            except OSError:
                return False

        # Ordinary chunk assets use one prepared transaction for WAV, NPY,
        # noise NPZ and IMCRA NPZ.  Atomic manifest replacement means the
        # chunk is either wholly indexed or wholly absent.  Anything left by
        # an interrupted Nth rename is quarantined as one transaction.
        for journal in self.data_root.rglob(".cj_*.json"):
            if quarantine in journal.parents:
                continue
            try:
                payload = json.loads(journal.read_text(encoding="utf-8"))
                entries = [dict(item) for item in payload.get("entries", ())]
            except (OSError, ValueError, TypeError):
                quarantine_path(journal)
                continue
            root = journal.parent
            indexed = indexed_assets(root)
            committed = bool(entries) and all(
                valid_committed_entry(root, entry, indexed) for entry in entries
            )
            if committed:
                journal.unlink(missing_ok=True)
                continue
            for entry in entries:
                partial = safe_path(root, entry.get("partial_path"))
                final = safe_path(root, entry.get("final_path"))
                if partial is not None:
                    quarantine_path(partial)
                if final is not None and not valid_committed_entry(root, entry, indexed):
                    quarantine_path(final)
            quarantine_path(journal)

        # An enhanced commit journal makes the rename-before-manifest crash
        # window recoverable.  A completed manifest wins; otherwise both the
        # prepared partial and any already-renamed final are quarantined.
        for journal in self.data_root.rglob("enhanced_asset_commit.json"):
            if quarantine in journal.parents:
                continue
            try:
                payload = json.loads(journal.read_text(encoding="utf-8"))
                entries = list(payload.get("entries", ()))
            except (OSError, ValueError, TypeError):
                quarantine_path(journal)
                continue
            root = journal.parent
            indexed = indexed_assets(root)
            committed = bool(entries) and all(
                valid_committed_entry(
                    root,
                    dict(entry),
                    indexed,
                    reject_internal_partial=True,
                )
                for entry in entries
            )
            if committed:
                journal.unlink(missing_ok=True)
                continue
            for entry in entries:
                for key in ("partial_path", "final_path"):
                    candidate = safe_path(root, entry.get(key))
                    if candidate is not None:
                        quarantine_path(candidate)
            quarantine_path(journal)

        for path in self.data_root.rglob("*.partial"):
            quarantine_path(path)

        # An open manifest is a checkpoint from a process that never reached
        # stop_session.  Validate every referenced chunk asset, remove any
        # provisional enhanced entries, and atomically convert the session to
        # an auditable incomplete state before rebuilding its Catalog row.
        for manifest_path in self.data_root.glob(
            "runtime_sessions/*/*/*/session_manifest.json"
        ):
            root = manifest_path.parent
            if self._root is not None and root.resolve() == self._root.resolve():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if manifest.get("status") != "open":
                continue
            recovery_errors: list[dict[str, Any]] = []
            for chunk in manifest.get("chunks", ()):
                valid_assets: list[dict[str, Any]] = []
                for original in chunk.get("assets", ()):
                    asset = dict(original)
                    kind = str(asset.get("kind", "unknown"))
                    relative = asset.get("path")
                    expected = asset.get("sha256")
                    candidate = safe_path(root, relative)
                    reason: str | None = None
                    if kind == "enhanced_audio" and asset.get("_partial_path") is not None:
                        reason = "uncommitted_enhanced_asset"
                    elif candidate is None or not candidate.is_file():
                        reason = "asset_missing"
                    elif not isinstance(expected, str):
                        reason = "asset_hash_missing"
                    else:
                        try:
                            if sha256_file(candidate) != expected:
                                reason = "asset_hash_mismatch"
                        except OSError:
                            reason = "asset_unreadable"
                    if reason is None:
                        valid_assets.append(asset)
                        continue
                    if candidate is not None and candidate.exists():
                        quarantine_path(candidate)
                    recovery_errors.append({
                        "reason": reason,
                        "kind": kind,
                        "path": relative,
                        "stream_epoch": chunk.get("stream_epoch"),
                        "start_sample": chunk.get("start_sample"),
                        "end_sample": chunk.get("end_sample"),
                        "window_id": asset.get("window_id"),
                        "decision_sample": asset.get("decision_sample"),
                    })
                    if kind == "enhanced_audio":
                        manifest.setdefault("result_gaps", []).append({
                            "reason": "crash_recovery_uncommitted_enhanced_asset",
                            "stream_epoch": asset.get(
                                "stream_epoch", chunk.get("stream_epoch")
                            ),
                            "window_id": asset.get("window_id"),
                            "decision_sample": asset.get("decision_sample"),
                        })
                    else:
                        manifest.setdefault("missing_intervals", []).append({
                            "reason": f"crash_recovery_{reason}",
                            "stream_epoch": chunk.get("stream_epoch"),
                            "start_sample": chunk.get("start_sample"),
                            "end_sample": chunk.get("end_sample"),
                            "asset_kind": kind,
                        })
                chunk["assets"] = valid_assets
                if recovery_errors:
                    chunk["recovery_status"] = "incomplete"
            manifest["status"] = "incomplete"
            manifest["ended_at_utc"] = utc_now()
            manifest["stop_reason"] = "crash_recovery"
            manifest["recovery_errors"] = recovery_errors
            write_manifest(manifest_path, manifest)
            self.catalog.upsert_session(manifest, root)
        return recovered

    def close(self) -> None:
        if self._session_id:
            self.stop_session("normal")
        writer_alive = self._thread is not None and self._thread.is_alive()
        if writer_alive or self._session_id is not None:
            # A timed-out writer still owns file handles and the active
            # manifest.  Reporting close success here would let callers tear
            # down the application while recording continues in the
            # background and would hide an unrecoverable lifecycle leak.
            raise RuntimeError(
                "RecordingStore cannot close while its writer/session is still active"
            )
        if self._owns_catalog:
            self.catalog.close()
