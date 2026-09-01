from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path


def main() -> int:
    required = ("numpy", "scipy", "yaml", "sounddevice", "serial", "pydantic", "PySide6")
    import_errors: dict[str, str] = {}
    for name in required:
        try:
            importlib.import_module(name)
        except Exception as exc:
            import_errors[name] = f"{type(exc).__name__}: {exc}"
    forbidden = [name for name in ("torch", "onnxruntime", "safetensors", "spectralcluster")
                 if importlib.util.find_spec(name) is not None]
    python_supported = sys.version_info[:2] == (3, 12)
    project_check_error: str | None = None
    try:
        from common.config import load_config

        importlib.import_module("gui.dev_test_ui.app")
        project_root = Path(__file__).resolve().parents[1]
        config = load_config(project_root / "config" / "config.yaml")
        if config.device.sample_rate != 48_000:
            raise ValueError("project config must use the 48 kHz runtime contract")
    except Exception as exc:
        project_check_error = f"{type(exc).__name__}: {exc}"
    report = {
        "python": sys.version.split()[0],
        "python_3_12": python_supported,
        "required_modules_importable": not import_errors,
        "import_errors": import_errors,
        "project_ui_and_config_loadable": project_check_error is None,
        "project_check_error": project_check_error,
        "neural_network_packages_present": forbidden,
        "profile": "v1.4-l1-l2-only",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if not python_supported or import_errors or project_check_error or forbidden else 0


if __name__ == "__main__":
    raise SystemExit(main())
