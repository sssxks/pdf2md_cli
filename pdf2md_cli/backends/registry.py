from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from pdf2md_cli.backends.glm import make_glm_runner
from pdf2md_cli.backends.mistral import make_mistral_runner
from pdf2md_cli.feature_flags import mock_backend_enabled
from pdf2md_cli.pipeline import OcrRunner
from pdf2md_cli.retry import BackoffConfig

BackendName = Literal["mistral", "glm", "mock"]


@dataclass(frozen=True, slots=True)
class BackendSpec:
    name: BackendName
    default_model: str
    api_key_env: str | None = None
    api_key_env_aliases: tuple[str, ...] = ()
    available: bool = True


def _all_specs() -> tuple[BackendSpec, ...]:
    return (
        BackendSpec(
            name="mistral",
            default_model="mistral-ocr-latest",
            api_key_env="MISTRAL_API_KEY",
        ),
        BackendSpec(
            name="glm",
            default_model="glm-ocr",
            api_key_env="BIGMODEL_API_KEY",
            api_key_env_aliases=("ZHIPUAI_API_KEY",),
        ),
        BackendSpec(
            name="mock",
            default_model="mock-ocr-latest",
            available=mock_backend_enabled(),
        ),
    )


def list_backend_specs(*, include_unavailable: bool = False) -> list[BackendSpec]:
    specs = list(_all_specs())
    if include_unavailable:
        return specs
    return [s for s in specs if s.available]


def backend_choices() -> list[str]:
    return [s.name for s in list_backend_specs()]


def backend_default_model(name: str) -> str:
    spec = get_backend_spec(name)
    return spec.default_model


def get_backend_spec(name: str) -> BackendSpec:
    for spec in _all_specs():
        if spec.name == name:
            return spec
    raise ValueError(f"Unsupported backend: {name!r}")


def _resolve_api_key_from_spec(spec: BackendSpec, cli_key: str | None) -> str | None:
    if spec.api_key_env is None:
        return None
    if cli_key is not None and cli_key.strip():
        return cli_key.strip()
    env_names = (spec.api_key_env, *spec.api_key_env_aliases)
    for env_name in env_names:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return None


def resolve_backend_api_key(backend: str, cli_key: str | None) -> str | None:
    spec = get_backend_spec(backend)
    return _resolve_api_key_from_spec(spec, cli_key)


def make_backend_runner(
    *,
    backend: str = "mistral",
    api_key: str | None = None,
    backoff: BackoffConfig = BackoffConfig(),
    mock: object | None = None,
) -> OcrRunner:
    if backend == "mistral":
        key = resolve_backend_api_key("mistral", api_key)
        if not key:
            raise ValueError("Mistral API key not provided (pass api_key=... or set env var MISTRAL_API_KEY).")
        return make_mistral_runner(api_key=key, backoff=backoff)

    if backend == "glm":
        key = resolve_backend_api_key("glm", api_key)
        if not key:
            raise ValueError(
                "GLM API key not provided (pass api_key=... or set BIGMODEL_API_KEY / ZHIPUAI_API_KEY env var)."
            )
        return make_glm_runner(api_key=key, backoff=backoff)

    if backend == "mock":
        if not mock_backend_enabled():
            raise ValueError('mock backend is disabled (set env var PDF2MD_ENABLE_MOCK=1 to enable).')
        from pdf2md_cli.backends.mock import MockConfig, make_mock_runner

        mock_cfg = mock if isinstance(mock, MockConfig) else MockConfig()
        return make_mock_runner(mock=mock_cfg, backoff=backoff)

    raise ValueError(f"Unsupported backend: {backend!r}")
