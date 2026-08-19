from __future__ import annotations

import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.production_ui.app import main  # noqa: E402 - desktop entry must add project root first

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        log_path = PROJECT_ROOT / "data" / "logs" / "audio_data_manager_startup.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise
