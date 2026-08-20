from __future__ import annotations

import numpy as np

from common.data_types import DirectionalSignal, SpectrogramFeature

from .configuration import FeatureSettings
from .interface import Layer3Error


class FeatureExtractor:
    def __init__(
        self, settings: FeatureSettings, *, freq_mean: np.ndarray | None = None, freq_std: np.ndarray | None = None,
    ) -> None:
        self.settings = settings
        bins = settings.last_bin_inclusive - settings.first_bin + 1
        if (freq_mean is None) != (freq_std is None):
            raise ValueError("freq_mean与freq_std必须同时提供")
        self.freq_mean = None if freq_mean is None else np.asarray(freq_mean, dtype=np.float32)
        self.freq_std = None if freq_std is None else np.asarray(freq_std, dtype=np.float32)
        if self.freq_mean is not None:
            if self.freq_mean.shape != (bins,) or self.freq_std.shape != (bins,):
                raise ValueError("频率归一化统计shape无效")
            if not np.isfinite(self.freq_mean).all() or not np.isfinite(self.freq_std).all() or np.any(self.freq_std < 1e-6):
                raise ValueError("频率归一化统计无效")

    @property
    def standardized(self) -> bool:
        return self.freq_mean is not None

    def extract(self, signal: DirectionalSignal) -> SpectrogramFeature:
        selected = signal.stft_complex[self.settings.first_bin:self.settings.last_bin_inclusive + 1, :]
        feature = np.log(np.abs(selected) + self.settings.log_epsilon).T.astype(np.float32, copy=False)
        if self.freq_mean is not None:
            feature = (feature - self.freq_mean[None, :]) / self.freq_std[None, :]
        feature = np.ascontiguousarray(feature, dtype=np.float32)
        if feature.shape != (self.settings.frame_count, 169) or not np.isfinite(feature).all():
            raise Layer3Error(
                f"工程特征不是finite float32 [{self.settings.frame_count},169]"
            )
        return SpectrogramFeature(
            signal.session_id, signal.stream_epoch, signal.window_id, signal.decision_sample,
            signal.context_start_sample, signal.context_end_sample, signal.theta_deg,
            signal.beamformer_backend, self.settings.preprocessing_version, feature,
            signal.track_id,
        )
