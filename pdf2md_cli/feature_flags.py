from __future__ import annotations

import os


def mock_backend_enabled() -> bool:
    v = os.getenv("PDF2MD_ENABLE_MOCK", "")
    return v.strip().lower() in {"1", "true", "yes", "on"}

