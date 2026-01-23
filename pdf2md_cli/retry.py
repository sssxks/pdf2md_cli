from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Protocol, TypeVar

from pdf2md_cli.types import NO_PROGRESS, Progress

T = TypeVar("T")


class _Rng(Protocol):
    def uniform(self, a: float, b: float) -> float: ...


@dataclass(frozen=True, slots=True)
class BackoffConfig:
    """Configuration for retry attempts and exponential backoff."""

    max_retries: int = 5
    initial_delay_s: float = 0.5
    max_delay_s: float = 20.0
    multiplier: float = 2.0
    jitter: float = 0.2


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _parse_retry_after_seconds(exc: BaseException) -> float | None:
    """
    Best-effort parse of Retry-After from exceptions coming from httpx / SDK wrappers.

    - If we can see a response.headers["Retry-After"], honor it.
    - Supports both seconds and HTTP-date formats per RFC 9110.
    """

    response = getattr(exc, "response", None)
    if response is None:
        return None

    headers = getattr(response, "headers", None)
    if not headers:
        return None

    raw = headers.get("Retry-After")
    if not raw:
        return None

    raw = str(raw).strip()
    if not raw:
        return None

    try:
        return float(raw)
    except Exception:
        pass

    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        return max(0.0, (dt - now).total_seconds())
    except Exception:
        return None


def _get_status_code_from_exc(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            try:
                return int(status_code)
            except Exception:
                return None

    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        try:
            return int(status_code)
        except Exception:
            return None

    return None


def is_retryable_exception(exc: BaseException) -> bool:
    """
    Decide whether an exception is likely transient and worth retrying.

    Uses httpx exception types when present, otherwise falls back to status code
    heuristics and message matching.
    """

    try:
        import httpx  # type: ignore

        if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
            return True

        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return status in {408, 409, 425, 429} or 500 <= status < 600
    except Exception:
        pass

    status_code = _get_status_code_from_exc(exc)
    if status_code is not None:
        return status_code in {408, 409, 425, 429} or 500 <= status_code < 600

    msg = str(exc).lower()
    return any(
        needle in msg
        for needle in (
            "rate limit",
            "too many requests",
            "timeout",
            "timed out",
            "temporarily unavailable",
            "service unavailable",
            "connection reset",
            "connection aborted",
            "remote protocol error",
        )
    )


def with_backoff(
    fn: Callable[[], T],
    *,
    what: str,
    cfg: BackoffConfig,
    progress: Progress = NO_PROGRESS,
    sleep_fn: Callable[[float], None] = time.sleep,
    rng: _Rng = random,
) -> T:
    """
    Retry wrapper with exponential backoff + jitter.

    cfg.max_retries is the number of retries after the initial attempt.
    """

    if cfg.max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if cfg.initial_delay_s < 0:
        raise ValueError("initial_delay_s must be >= 0")
    if cfg.max_delay_s <= 0:
        raise ValueError("max_delay_s must be > 0")
    if cfg.multiplier <= 0:
        raise ValueError("multiplier must be > 0")

    jitter = _clamp(cfg.jitter, 0.0, 1.0)
    delay_s = cfg.initial_delay_s
    attempt = 0

    while True:
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if attempt >= cfg.max_retries or not is_retryable_exception(e):
                raise

            attempt += 1

            retry_after_s = _parse_retry_after_seconds(e)
            sleep_s = retry_after_s if retry_after_s is not None else delay_s
            sleep_s = _clamp(float(sleep_s), 0.0, cfg.max_delay_s)

            if jitter > 0:
                sleep_s *= rng.uniform(1.0 - jitter, 1.0 + jitter)
                sleep_s = _clamp(sleep_s, 0.0, cfg.max_delay_s)

            def _msg() -> str:
                status = _get_status_code_from_exc(e)
                status_txt = f" status={status}" if status is not None else ""
                return (
                    f"{what} failed{status_txt} ({type(e).__name__}: {e}); "
                    f"retrying in {sleep_s:.1f}s (attempt {attempt}/{cfg.max_retries})..."
                )

            progress.emit_lazy(_msg)

            sleep_fn(sleep_s)
            delay_s = _clamp(delay_s * cfg.multiplier, 0.0, cfg.max_delay_s)
