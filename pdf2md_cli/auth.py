from __future__ import annotations

import os
import sys

from pdf2md_cli.backends.registry import get_backend_spec, resolve_backend_api_key


def error(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)


def load_api_key(cli_key: str | None) -> str:
    key = (cli_key or "").strip() or os.environ.get("MISTRAL_API_KEY", "").strip()
    if not key:
        error("Mistral API key not provided. Use --api-key or set MISTRAL_API_KEY env var.")
        raise SystemExit(2)
    return key


def load_api_key_for_backend(backend: str, cli_key: str | None) -> str | None:
    spec = get_backend_spec(backend)
    if spec.api_key_env is None:
        return None

    key = resolve_backend_api_key(backend, cli_key)
    if key:
        return key

    aliases = ", ".join(spec.api_key_env_aliases)
    if aliases:
        env_hint = f"{spec.api_key_env} (or {aliases})"
    else:
        env_hint = spec.api_key_env

    error(f"{backend} API key not provided. Use --api-key or set {env_hint} env var.")
    raise SystemExit(2)
