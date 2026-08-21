from __future__ import annotations

import threading
from pathlib import Path
import wave

import numpy as np
import sounddevice as sd


class PreviewPlayer:
    """Small non-blocking player for immutable configured engineering snapshots."""

    def __init__(
        self,
        *,
        sample_rate: int,
        volume: float,
        loop_gap_ms: int,
        autoplay: bool,
        peak_dbfs: float = -6.0,
        fade_ms: int = 5,
    ) -> None:
        self.sample_rate = sample_rate
        self.volume = float(volume)
        self.peak_amplitude = 10.0 ** (float(peak_dbfs) / 20.0)
        self.fade_samples = round(sample_rate * int(fade_ms) / 1000)
        self.loop_gap = round(sample_rate * loop_gap_ms / 1000)
        self.loop = bool(autoplay)
        self._lock = threading.Lock()
        self._audio = np.zeros(0, dtype=np.float32)
        self._position = 0
        self._playing = False
        self._gap_remaining = 0
        self._stream: sd.OutputStream | None = None
        self._stream_device: int | None = None
        self._stream_faulted = False
        self._last_error: str | None = None
        self._dc_offset = 0.0
        self._gain = 1.0
        self._callback_fade_samples = 0
        self._loaded_path: Path | None = None
        self._delete_on_release = False

    def set_volume(self, volume: float) -> None:
        value = float(volume)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("preview volume must be finite and non-negative")
        with self._lock:
            self.volume = value

    @property
    def playing(self) -> bool:
        with self._lock:
            return self._playing

    @property
    def playback_progress(self) -> float:
        """Return the loaded audio's current sample position as a 0..1 ratio."""
        with self._lock:
            sample_count = len(self._audio)
            if sample_count <= 0:
                return 0.0
            return float(np.clip(self._position / sample_count, 0.0, 1.0))

    @property
    def stream_active(self) -> bool:
        stream = self._stream
        if stream is None:
            return False
        try:
            return bool(stream.active)
        except Exception:
            return False

    def take_error(self) -> str | None:
        with self._lock:
            error = self._last_error
            self._last_error = None
            return error

    def load(self, waveform: np.ndarray) -> None:
        audio = np.asarray(waveform, dtype=np.float32).copy()
        audio -= np.mean(audio, dtype=np.float64)
        peak = float(np.max(np.abs(audio), initial=0.0))
        # Safety limiting may attenuate a hot preview, but listening must never
        # boost a quiet signal/noise floor to the configured peak target.
        if peak > self.peak_amplitude:
            audio *= self.peak_amplitude / peak
        fade = min(self.fade_samples, len(audio) // 2)
        if fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            audio[:fade] *= ramp
            audio[-fade:] *= ramp[::-1]
        with self._lock:
            self._release_audio_locked()
            self._audio, self._position, self._gap_remaining = audio, 0, 0
            self._last_error = None

    def load_file(
        self,
        path: str | Path,
        *,
        delete_on_release: bool = False,
        target_rms_dbfs: float | None = None,
        max_gain_db: float = 0.0,
    ) -> None:
        """Memory-map a Test UI raw-float32 cache instead of loading it into RAM."""
        path = Path(path)
        if (
            not path.is_file()
            or path.stat().st_size == 0
            or path.stat().st_size % np.dtype(np.float32).itemsize
        ):
            raise ValueError("invalid L3 audio cache file")
        if target_rms_dbfs is not None and (
            not np.isfinite(target_rms_dbfs) or target_rms_dbfs > 0.0
        ):
            raise ValueError("target RMS must be a finite non-positive dBFS value")
        if not np.isfinite(max_gain_db) or max_gain_db < 0.0:
            raise ValueError("maximum listening gain must be finite and non-negative")
        audio = np.memmap(path, mode="r", dtype=np.float32)
        # Two bounded-memory passes retain the same DC removal and attenuation-
        # only safety limiter as load(), without raising a quiet cache's noise.
        chunk_samples = 1 << 18
        total = 0.0
        for start in range(0, len(audio), chunk_samples):
            total += float(np.sum(audio[start:start + chunk_samples], dtype=np.float64))
        dc_offset = total / len(audio)
        peak = 0.0
        squared_total = 0.0
        for start in range(0, len(audio), chunk_samples):
            chunk = np.asarray(audio[start:start + chunk_samples], dtype=np.float32)
            centered = chunk.astype(np.float64) - dc_offset
            peak = max(peak, float(np.max(np.abs(centered), initial=0.0)))
            squared_total += float(np.sum(centered * centered, dtype=np.float64))
        rms = float(np.sqrt(squared_total / len(audio)))
        safety_gain = float("inf") if peak <= 0.0 else self.peak_amplitude / peak
        if target_rms_dbfs is None or rms <= 0.0:
            requested_gain = 1.0
        else:
            target_rms = 10.0 ** (float(target_rms_dbfs) / 20.0)
            requested_gain = max(1.0, target_rms / rms)
        maximum_gain = 10.0 ** (float(max_gain_db) / 20.0)
        playback_gain = min(requested_gain, maximum_gain, safety_gain)
        with self._lock:
            self._release_audio_locked()
            self._audio, self._position, self._gap_remaining = audio, 0, 0
            self._dc_offset = dc_offset
            self._gain = playback_gain
            self._callback_fade_samples = min(self.fade_samples, len(audio) // 2)
            self._loaded_path = path
            self._delete_on_release = bool(delete_on_release)
            self._last_error = None

    def load_wav_file(self, path: str | Path) -> None:
        """Decode a 48 kHz mono PCM16 WAV before sending float32 to PortAudio."""

        path = Path(path)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("invalid L4 WAV preview file")
        try:
            with wave.open(str(path), "rb") as source:
                channels = source.getnchannels()
                sample_width = source.getsampwidth()
                sample_rate = source.getframerate()
                frame_count = source.getnframes()
                payload = source.readframes(frame_count)
        except (EOFError, wave.Error) as exc:
            raise ValueError("invalid L4 WAV preview file") from exc
        if channels != 1 or sample_width != 2 or sample_rate != self.sample_rate:
            raise ValueError(
                f"L4 WAV preview must be {self.sample_rate} Hz mono PCM16"
            )
        if frame_count <= 0 or len(payload) != frame_count * sample_width:
            raise ValueError("invalid L4 WAV preview payload")
        audio = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        self.load(audio)

    def _release_audio_locked(self) -> None:
        if isinstance(self._audio, np.memmap):
            self._audio._mmap.close()
        path, delete = self._loaded_path, self._delete_on_release
        self._audio = np.zeros(0, dtype=np.float32)
        self._dc_offset = 0.0
        self._gain = 1.0
        self._callback_fade_samples = 0
        self._loaded_path = None
        self._delete_on_release = False
        if delete and path is not None:
            path.unlink(missing_ok=True)

    @staticmethod
    def _default_output_device() -> int:
        value = sd.default.device
        try:
            device = value[1]
        except (TypeError, IndexError):
            device = value
        if device is None or int(device) < 0:
            raise RuntimeError("Windows has no default audio output device")
        return int(device)

    @staticmethod
    def _dispose_stream(stream) -> None:
        if stream is None:
            return
        try:
            if stream.active:
                stream.stop()
        finally:
            stream.close()

    def _ensure_output_stream(self) -> None:
        desired_device = self._default_output_device()
        stream = self._stream
        stale = stream is None or self._stream_device != desired_device or self._stream_faulted
        if stream is not None and not stale:
            try:
                stale = not bool(stream.active)
            except Exception:
                stale = True
        if stale:
            self._stream = None
            self._stream_device = None
            self._stream_faulted = False
            try:
                self._dispose_stream(stream)
            except Exception:
                pass
            replacement = sd.OutputStream(
                device=desired_device,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            replacement.start()
            self._stream = replacement
            self._stream_device = desired_device

    def play(self) -> bool:
        with self._lock:
            if not len(self._audio):
                self._last_error = "No audio is loaded"
                return False
            if self._playing and self.stream_active:
                return True
        try:
            self._ensure_output_stream()
        except Exception as exc:
            with self._lock:
                self._playing = False
                self._last_error = f"Audio output failed: {exc}"
            return False
        with self._lock:
            self._playing = True
            self._last_error = None
        return True

    def pause(self) -> None:
        with self._lock:
            self._playing = False

    def toggle(self) -> bool:
        if self.playing:
            self.pause()
            return False
        return self.play()

    def validate_output(self) -> bool:
        """Synchronize a lost/stopped PortAudio stream back to the UI."""
        if not self.playing:
            return True
        if self.stream_active:
            return True
        with self._lock:
            self._playing = False
            self._last_error = "Audio output stream stopped; press play to reconnect"
        return False

    def stop(self) -> None:
        with self._lock:
            self._playing, self._position, self._gap_remaining = False, 0, 0

    def close(self) -> None:
        with self._lock:
            self._playing = False
        stream = self._stream
        self._stream = None
        self._stream_device = None
        self._stream_faulted = False
        try:
            self._dispose_stream(stream)
        except Exception:
            pass
        # Release the mapped L3 disk snapshot before its cache directory is
        # deleted by the Test UI shutdown path.
        with self._lock:
            self._release_audio_locked()
            self._position = 0
            self._gap_remaining = 0

    def _callback(self, outdata, frames, _time, _status) -> None:
        outdata.fill(0)
        with self._lock:
            if _status:
                self._last_error = f"Audio output status: {_status}"
                self._stream_faulted = True
                self._playing = False
                return
            if not self._playing or not len(self._audio):
                return
            if self._gap_remaining:
                skipped = min(frames, self._gap_remaining)
                self._gap_remaining -= skipped
                if skipped == frames:
                    return
                out_offset = skipped
            else:
                out_offset = 0
            remaining = len(self._audio) - self._position
            count = min(frames - out_offset, remaining)
            start = self._position
            samples = (
                np.asarray(self._audio[start:start + count], dtype=np.float32) - self._dc_offset
            ) * (self._gain * self.volume)
            fade = self._callback_fade_samples
            if fade:
                positions = np.arange(start, start + count)
                factors = np.minimum(1.0, positions / fade)
                factors = np.minimum(factors, (len(self._audio) - 1 - positions) / fade)
                samples = samples * np.clip(factors, 0.0, 1.0)
            outdata[out_offset:out_offset + count, 0] = samples
            self._position += count
            if self._position >= len(self._audio):
                if self.loop:
                    self._position = 0
                    self._gap_remaining = self.loop_gap
                else:
                    self._playing, self._position = False, 0
