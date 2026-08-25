"""Project-wide pytest safeguards."""

from __future__ import annotations

import os


# Automated Qt tests must never create native Windows surfaces.  An explicitly
# configured platform is preserved for developers who intentionally run a
# platform-specific or interactive test.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
