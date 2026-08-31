from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_app_import_enforces_single_thread_blas_before_runtime_import() -> None:
    environment = dict(os.environ)
    environment["OPENBLAS_NUM_THREADS"] = "8"
    environment["OMP_NUM_THREADS"] = "8"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import app, os; "
                "print(os.environ['OPENBLAS_NUM_THREADS'], "
                "os.environ['OMP_NUM_THREADS'])"
            ),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "1 1"
