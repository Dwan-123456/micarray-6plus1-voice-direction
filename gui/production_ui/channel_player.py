from __future__ import annotations

import threading
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd


@dataclass(frozen=True, slots=True)
class _Clip:
    path: Path
    channel: int
    channel_count: int
    start_frame: int
    frame_count: int
    silence_before: int = 0


class NativeChannelPlayer:
    """Stream raw chunk channels or overlap-free per-ID enhanced timelines."""

    def __init__(self, volume: float = 0.5):
        self.volume = float(volume)
        self._lock = threading.RLock()
        self._wave: wave.Wave_read | None = None
        self._stream: sd.OutputStream | None = None
        self._channel = 0
        self._clips: deque[_Clip] = deque()
        self._remaining_frames = 0
        self._silence_frames = 0

    def play(self, path: str | Path, channel: int) -> None:
        self.play_files((path,), channel=channel, channel_count=8)

    @staticmethod
    def _inspect(path: Path) -> tuple[int, int]:
        with wave.open(str(path), "rb") as source:
            if source.getsampwidth() != 2 or source.getframerate() != 48_000:
                raise ValueError("试听文件必须是48 kHz PCM16 WAV")
            return source.getnchannels(), source.getnframes()

    def play_files(
        self,
        paths: tuple[str | Path, ...] | list[str | Path],
        *,
        channel: int,
        channel_count: int,
    ) -> None:
        if not paths:
            raise ValueError("没有可试听的音频资产")
        if channel not in range(channel_count):
            raise ValueError(f"试听通道必须位于1到{channel_count}")
        clips: list[_Clip] = []
        for value in paths:
            path = Path(value).resolve(strict=True)
            actual_channels, frames = self._inspect(path)
            if actual_channels != channel_count:
                raise ValueError(f"试听文件应为{channel_count}通道")
            clips.append(_Clip(path, channel, channel_count, 0, frames))
        self._start_clips(clips)

    def play_track_assets(self, assets: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> None:
        if not assets:
            raise ValueError("该方向ID没有增强音频资产")
        ordered = sorted(assets, key=lambda item: int(item["decision_sample"]))
        clips: list[_Clip] = []
        previous_decision: int | None = None
        for item in ordered:
            path = Path(str(item["absolute_path"])).resolve(strict=True)
            channels, frames = self._inspect(path)
            if channels != 1:
                raise ValueError("方向ID增强资产必须是单通道")
            decision = int(item["decision_sample"])
            if previous_decision is None:
                start_frame, frame_count, silence = 0, frames, 0
            else:
                delta = max(0, decision - previous_decision)
                frame_count = min(frames, delta)
                start_frame = frames - frame_count
                silence = max(0, delta - frames)
            if frame_count > 0:
                clips.append(_Clip(path, 0, 1, start_frame, frame_count, silence))
            previous_decision = decision
        if not clips:
            raise ValueError("方向ID增强资产没有可播放的时间段")
        self._start_clips(clips)

    def _start_clips(self, clips: list[_Clip]) -> None:
        self.stop()
        with self._lock:
            self._clips = deque(clips)
            self._open_next_clip_locked()
        stream = sd.OutputStream(
            samplerate=48_000,
            channels=1,
            dtype="float32",
            callback=self._callback,
            finished_callback=self._finished,
        )
        with self._lock:
            self._stream = stream
        stream.start()

    def _open_next_clip_locked(self) -> bool:
        if self._wave is not None:
            self._wave.close()
            self._wave = None
        if not self._clips:
            return False
        clip = self._clips.popleft()
        source = wave.open(str(clip.path), "rb")
        source.setpos(clip.start_frame)
        self._wave = source
        self._channel = clip.channel
        self._remaining_frames = clip.frame_count
        self._silence_frames = clip.silence_before
        return True

    def _callback(self, outdata, frames, _time, status) -> None:
        outdata.fill(0)
        if status:
            raise sd.CallbackAbort
        with self._lock:
            output_offset = 0
            while output_offset < frames:
                if self._silence_frames > 0:
                    silent = min(frames - output_offset, self._silence_frames)
                    self._silence_frames -= silent
                    output_offset += silent
                    continue
                source = self._wave
                if source is None and not self._open_next_clip_locked():
                    raise sd.CallbackStop
                source = self._wave
                if source is None:
                    raise sd.CallbackStop
                requested = min(frames - output_offset, self._remaining_frames)
                payload = source.readframes(requested)
                values = np.frombuffer(payload, dtype="<i2")
                channel_count = source.getnchannels()
                if values.size % channel_count:
                    raise sd.CallbackAbort
                values = values.reshape(-1, channel_count)
                count = len(values)
                if count:
                    outdata[output_offset:output_offset + count, 0] = (
                        values[:, self._channel].astype(np.float32) * (self.volume / 32768.0)
                    )
                    output_offset += count
                    self._remaining_frames -= count
                if count < requested or self._remaining_frames <= 0:
                    if not self._open_next_clip_locked() and output_offset < frames:
                        raise sd.CallbackStop

    def _finished(self) -> None:
        with self._lock:
            if self._wave is not None:
                self._wave.close()
                self._wave = None
            self._clips.clear()
            self._remaining_frames = 0
            self._silence_frames = 0

    def stop(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
            source, self._wave = self._wave, None
            self._clips.clear()
            self._remaining_frames = 0
            self._silence_frames = 0
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        if source is not None:
            source.close()

    close = stop


__all__ = ["NativeChannelPlayer"]
