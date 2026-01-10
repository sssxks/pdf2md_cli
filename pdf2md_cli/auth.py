from __future__ import annotations

import os
import sys


def error(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)


def load_api_key(cli_key: str | None) -> str:
    key = (cli_key or "").strip() or os.environ.get("MISTRAL_API_KEY", "").strip()
    if not key:
        error("Mistral API key not provided. Use --api-key or set MISTRAL_API_KEY env var.")
        raise SystemExit(2)
    return key
