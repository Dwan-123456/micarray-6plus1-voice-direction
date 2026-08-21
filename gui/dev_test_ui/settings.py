from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


class DevUiSettings:
    """Atomic persistent store for operator-tuned Development Test UI values."""

    SCHEMA_VERSION = "dev_test_ui_settings_v12"
    PREVIOUS_SCHEMA_VERSION = "dev_test_ui_settings_v11"
    OLDER_SCHEMA_VERSION = "dev_test_ui_settings_v2"
    LEGACY_SCHEMA_VERSION = "dev_test_ui_settings_v1"
    OBSOLETE_KEYS = {
        "layer2_iterative_peak_search_enabled",
    }

    def __init__(self, project_root: str | Path):
        self.path = Path(project_root).resolve() / "data" / "dev_test_ui" / "settings.json"

    def _load_payload(self) -> dict[str, object]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema_version") not in {
                self.SCHEMA_VERSION,
                self.PREVIOUS_SCHEMA_VERSION,
                "dev_test_ui_settings_v10",
                "dev_test_ui_settings_v9",
                "dev_test_ui_settings_v8",
                "dev_test_ui_settings_v7",
                "dev_test_ui_settings_v5",
                "dev_test_ui_settings_v4",
                "dev_test_ui_settings_v3",
                self.OLDER_SCHEMA_VERSION,
                self.LEGACY_SCHEMA_VERSION,
            }:
                return {}
            return dict(payload)
        except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError):
            return {}

    def _save_update(self, **updates: object) -> None:
        payload = self._load_payload()
        for key in self.OBSOLETE_KEYS:
            payload.pop(key, None)
        payload.update(updates)
        payload["schema_version"] = self.SCHEMA_VERSION
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def load_direction_threshold(self, default: float) -> float:
        fallback = self._validate_threshold(default)
        try:
            return self._validate_threshold(self._load_payload()["layer2_direction_threshold"])
        except (KeyError, TypeError, ValueError):
            return fallback

    def save_direction_threshold(self, value: float) -> float:
        threshold = self._validate_threshold(value)
        self._save_update(layer2_direction_threshold=threshold)
        return threshold

    def load_music_effective_order_limit(self, default: int = 3) -> int:
        fallback = self._validate_music_order_limit(default)
        try:
            return self._validate_music_order_limit(
                self._load_payload()["layer2_music_effective_order_limit"]
            )
        except (KeyError, TypeError, ValueError):
            return fallback

    def save_music_effective_order_limit(self, value: int) -> int:
        limit = self._validate_music_order_limit(value)
        self._save_update(layer2_music_effective_order_limit=limit)
        return limit

    def load_music_dpd_rank1_enabled(self, default: bool = False) -> bool:
        fallback = self._validate_bool(default)
        try:
            return self._validate_bool(self._load_payload()["layer2_music_dpd_rank1_enabled"])
        except (KeyError, TypeError, ValueError):
            return fallback

    def save_music_dpd_rank1_enabled(self, value: bool) -> bool:
        enabled = self._validate_bool(value)
        self._save_update(layer2_music_dpd_rank1_enabled=enabled)
        return enabled

    def load_music_noise_whitening_enabled(self, default: bool = False) -> bool:
        fallback = self._validate_bool(default)
        try:
            return self._validate_bool(
                self._load_payload()["layer2_music_noise_whitening_enabled"]
            )
        except (KeyError, TypeError, ValueError):
            return fallback

    def save_music_noise_whitening_enabled(self, value: bool) -> bool:
        enabled = self._validate_bool(value)
        self._save_update(layer2_music_noise_whitening_enabled=enabled)
        return enabled

    def load_direction_kalman_enabled(self, default: bool = False) -> bool:
        fallback = self._validate_bool(default)
        try:
            return self._validate_bool(self._load_payload()["layer2_direction_kalman_enabled"])
        except (KeyError, TypeError, ValueError):
            return fallback

    def save_direction_kalman_enabled(self, value: bool) -> bool:
        enabled = self._validate_bool(value)
        self._save_update(layer2_direction_kalman_enabled=enabled)
        return enabled

    def load_direction_id_tracking_enabled(self, default: bool = True) -> bool:
        fallback = self._validate_bool(default)
        try:
            return self._validate_bool(
                self._load_payload()["layer2_direction_id_tracking_enabled"]
            )
        except (KeyError, TypeError, ValueError):
            return fallback

    def save_direction_id_tracking_enabled(self, value: bool) -> bool:
        enabled = self._validate_bool(value)
        self._save_update(layer2_direction_id_tracking_enabled=enabled)
        return enabled

    def load_direction_kalman_q_scale(self, initial: float) -> float:
        fallback = self._validate_kalman_scale(initial)
        try:
            return self._validate_kalman_scale(
                self._load_payload()["layer2_direction_kalman_q_scale"]
            )
        except (KeyError, TypeError, ValueError):
            return fallback

    def save_direction_kalman_q_scale(self, value: float) -> float:
        scale = self._validate_kalman_scale(value)
        self._save_update(layer2_direction_kalman_q_scale=scale)
        return scale

    def load_direction_kalman_r_scale(self, initial: float) -> float:
        fallback = self._validate_kalman_scale(initial)
        try:
            return self._validate_kalman_scale(
                self._load_payload()["layer2_direction_kalman_r_scale"]
            )
        except (KeyError, TypeError, ValueError):
            return fallback

    def save_direction_kalman_r_scale(self, value: float) -> float:
        scale = self._validate_kalman_scale(value)
        self._save_update(layer2_direction_kalman_r_scale=scale)
        return scale

    def load_gate_probability_threshold(self, default: float = 0.60) -> float:
        fallback = self._validate_gate_threshold(default)
        try:
            return self._validate_gate_threshold(
                self._load_payload()["layer2_gate_probability_threshold"]
            )
        except (KeyError, TypeError, ValueError):
            return fallback

    def save_gate_probability_threshold(self, value: float) -> float:
        threshold = self._validate_gate_threshold(value)
        self._save_update(layer2_gate_probability_threshold=threshold)
        return threshold

    def load_l1_pre_denoise_enabled(self, default: bool = False) -> bool:
        fallback = self._validate_bool(default)
        try:
            return self._validate_bool(self._load_payload()["layer1_pre_denoise_enabled"])
        except (KeyError, TypeError, ValueError):
            return fallback

    def save_l1_pre_denoise_enabled(self, value: bool) -> bool:
        enabled = self._validate_bool(value)
        self._save_update(layer1_pre_denoise_enabled=enabled)
        return enabled

    def load_l4_input_gain_compensation_enabled(self, default: bool = True) -> bool:
        fallback = self._validate_bool(default)
        try:
            return self._validate_bool(
                self._load_payload()["layer4_input_gain_compensation_enabled"]
            )
        except (KeyError, TypeError, ValueError):
            return fallback

    def save_l4_input_gain_compensation_enabled(self, value: bool) -> bool:
        enabled = self._validate_bool(value)
        self._save_update(layer4_input_gain_compensation_enabled=enabled)
        return enabled

    @staticmethod
    def _validate_threshold(value: float) -> float:
        threshold = round(float(value), 2)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Layer 2 direction threshold must be in [0,1]")
        return threshold

    @staticmethod
    def _validate_bool(value: bool) -> bool:
        if type(value) is not bool:
            raise ValueError("switch setting must be bool")
        return value

    @staticmethod
    def _validate_music_order_limit(value: int) -> int:
        if type(value) is not int or value not in {1, 2, 3}:
            raise ValueError("MUSIC effective order limit must be 1, 2, or 3")
        return value

    @staticmethod
    def _validate_gate_threshold(value: float) -> float:
        threshold = round(float(value), 2)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Layer 2 Gate probability threshold must be in [0,1]")
        return threshold

    @staticmethod
    def _validate_kalman_scale(value: float) -> float:
        scale = round(float(value), 2)
        if not 0.02 <= scale <= 10.0 or (
            scale != 0.02 and abs(scale * 10.0 - round(scale * 10.0)) > 1.0e-9
        ):
            raise ValueError(
                "Layer 2 Kalman scale must be 0.02..10.00 in 0.1 steps "
                "(or the 0.02 minimum)"
            )
        return scale
