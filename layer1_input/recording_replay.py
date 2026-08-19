from __future__ import annotations

import hashlib
import json
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .interface import CdcHotmapFrame, DecodedAudio
from .sources import AudioSource, map_logical_channels, pcm16_to_float32


@dataclass(frozen=True, slots=True)
class ReplayStatus:
    state: str
    current_sample: int
    total_samples: int
    generation: int

    @property
    def current_seconds(self) -> float:
        return self.current_sample / 48_000

    @property
    def total_seconds(self) -> float:
        return self.total_samples / 48_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RecordingReplaySource(AudioSource):
    """Replay native USB audio and recorded CDC frames as one virtual array."""

    VALID_STATES = {"ready", "playing", "paused", "ended", "stopped", "error"}

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        logical_channel_map: tuple[int, ...],
        block_size: int = 960,
        autoplay: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.display_name = str(manifest.get("display_name") or self.root.name)
        assets = {str(item.get("kind")): item for item in manifest.get("assets", ())}
        if "native_8ch" not in assets or "cdc_hotmaps" not in assets:
            raise ValueError("完整模拟输入要求原始8通道音频和热力图")
        self.audio_path = self._validated_asset(assets["native_8ch"])
        self.hotmap_path = self._validated_asset(assets["cdc_hotmaps"])
        self.logical_channel_map = tuple(int(value) for value in logical_channel_map)
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("模拟输入块长度必须为正数")
        self._hotmap_rows = self._load_hotmaps()
        self._condition = threading.Condition(threading.RLock())
        self._wav: wave.Wave_read | None = None
        self._state = "ready"
        self._autoplay = bool(autoplay)
        self._sample_position = 0
        self._total_samples = 0
        self._sequence = 0
        self._generation = 0
        self._anchor = 0.0
        self._hotmap_index = 0
        self._current_hotmap: CdcHotmapFrame | None = None

    def _validated_asset(self, asset: dict[str, Any]) -> Path:
        path = (self.root / str(asset["path"])).resolve(strict=True)
        if self.root != path.parent and self.root not in path.parents:
            raise ValueError("录音资产路径越界")
        expected = str(asset.get("sha256", ""))
        if not expected or _sha256(path) != expected:
            raise ValueError(f"录音资产校验失败：{path.name}")
        return path

    def _load_hotmaps(self) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        with self.hotmap_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                matrix = np.asarray(item.get("matrix"), dtype=np.uint8)
                sample = int(item.get("playback_sample", -1))
                if matrix.shape != (16, 16) or sample < 0:
                    raise ValueError(f"热力图记录无效：第{line_number}行")
                rows.append({**item, "playback_sample": sample, "matrix": matrix})
        if not rows:
            raise ValueError("完整模拟输入要求至少一帧热力图")
        rows.sort(key=lambda item: int(item["playback_sample"]))
        return tuple(rows)

    @property
    def exhausted(self) -> bool:
        # EOF is an interactive replay state, not a request to tear down Runtime.
        return False

    @property
    def state(self) -> str:
        with self._condition:
            return self._state

    def status(self) -> ReplayStatus:
        with self._condition:
            return ReplayStatus(
                self._state,
                self._sample_position,
                self._total_samples,
                self._generation,
            )

    def start(self) -> None:
        with self._condition:
            if self._wav is not None:
                return
            source = wave.open(str(self.audio_path), "rb")
            if (
                source.getnchannels() != 8
                or source.getsampwidth() != 2
                or source.getframerate() != 48_000
            ):
                source.close()
                raise ValueError("模拟输入必须是48 kHz、原始8通道PCM16")
            self._wav = source
            self._total_samples = source.getnframes()
            self._state = "playing" if self._autoplay else "paused"
            self._anchor = time.monotonic()
            self._condition.notify_all()

    def _advance_hotmap(self, sample: int) -> None:
        while (
            self._hotmap_index < len(self._hotmap_rows)
            and int(self._hotmap_rows[self._hotmap_index]["playback_sample"]) <= sample
        ):
            item = self._hotmap_rows[self._hotmap_index]
            self._current_hotmap = CdcHotmapFrame(
                item["matrix"],
                int(item["sequence_id"]),
                float(item.get("timestamp", sample / 48_000)),
                None if item.get("received_at") is None else float(item["received_at"]),
            )
            self._hotmap_index += 1

    def read(self, timeout: float | None = None) -> DecodedAudio | None:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while self._state != "playing":
                if self._state in {"stopped", "error"}:
                    return None
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            wav = self._wav
            if wav is None:
                raise RuntimeError("模拟输入尚未启动")
            target = self._anchor + self._sample_position / 48_000
            while self._state == "playing" and time.monotonic() < target:
                remaining = target - time.monotonic()
                if deadline is not None:
                    remaining = min(remaining, max(0.0, deadline - time.monotonic()))
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if self._state != "playing":
                return None
            start_sample = self._sample_position
            payload = wav.readframes(self.block_size)
            if not payload:
                self._state = "ended"
                self._condition.notify_all()
                return None
            native = pcm16_to_float32(payload, 8)
            self._advance_hotmap(start_sample)
            frame = DecodedAudio(
                map_logical_channels(native, self.logical_channel_map),
                48_000,
                self._sequence,
                start_sample / 48_000,
                native_samples=native,
            )
            self._sample_position += len(native)
            self._sequence += 1
            return frame

    def latest_hotmap_frame(self) -> CdcHotmapFrame | None:
        with self._condition:
            return self._current_hotmap

    def pause(self) -> None:
        with self._condition:
            if self._state == "playing":
                self._state = "paused"
                self._condition.notify_all()

    def resume(self) -> None:
        with self._condition:
            if self._wav is None:
                self._autoplay = True
                return
            if self._state == "ended":
                return
            if self._state in {"ready", "paused"}:
                self._state = "playing"
                self._anchor = time.monotonic() - self._sample_position / 48_000
                self._condition.notify_all()

    def replay(self) -> None:
        with self._condition:
            if self._wav is None:
                self._autoplay = True
                self._state = "ready"
                return
            self._wav.rewind()
            self._sample_position = 0
            self._sequence = 0
            self._generation += 1
            self._hotmap_index = 0
            self._current_hotmap = None
            self._anchor = time.monotonic()
            self._state = "playing"
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._state = "stopped"
            wav, self._wav = self._wav, None
            self._condition.notify_all()
        if wav is not None:
            wav.close()


__all__ = ["RecordingReplaySource", "ReplayStatus"]
