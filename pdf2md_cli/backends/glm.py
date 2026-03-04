from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import cast

from pdf2md_cli.pipeline import OcrRunner
from pdf2md_cli.retry import BackoffConfig, with_backoff
from pdf2md_cli.types import InputKind, OcrPage, OcrResult, Progress, TableFormat

_GLM_LAYOUT_PARSING_URL = "https://open.bigmodel.cn/api/paas/v4/layout_parsing"
_GLM_SUPPORTED_IMAGE_MIMES = {"image/jpeg", "image/png"}
_GLM_SUPPORTED_DOC_SUFFIXES = {".pdf"}


class _HttpResponseLike:
    __slots__ = ("status_code", "headers")

    def __init__(self, *, status_code: int, headers: Mapping[str, str]) -> None:
        self.status_code = status_code
        self.headers = headers


class _HttpStatusError(RuntimeError):
    __slots__ = ("status_code", "response")

    def __init__(self, *, status_code: int, body: str, headers: Mapping[str, str]) -> None:
        msg = body.strip().replace("\n", " ")
        if len(msg) > 400:
            msg = msg[:400] + "…"
        super().__init__(f"HTTP {status_code}: {msg}")
        self.status_code = status_code
        self.response = _HttpResponseLike(status_code=status_code, headers=headers)


def _post_json(
    *,
    url: str,
    api_key: str,
    payload: Mapping[str, object],
    timeout_s: float = 180.0,
) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            raw = resp.read()
            return cast(dict[str, object], json.loads(raw.decode("utf-8")))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = str(e)
        headers = {k: v for k, v in (e.headers.items() if e.headers else [])}
        raise _HttpStatusError(status_code=int(getattr(e, "code", 0) or 0), body=body, headers=headers) from e


def _validate_glm_input(*, file_name: str, input_kind: InputKind, mime_type: str | None) -> None:
    lower_name = file_name.lower()
    if input_kind == InputKind.DOCUMENT:
        if not any(lower_name.endswith(suffix) for suffix in _GLM_SUPPORTED_DOC_SUFFIXES):
            supported = ", ".join(sorted(_GLM_SUPPORTED_DOC_SUFFIXES))
            raise ValueError(f"GLM backend currently supports document inputs: {supported}")
        return

    if input_kind == InputKind.IMAGE:
        if mime_type not in _GLM_SUPPORTED_IMAGE_MIMES:
            supported = ", ".join(sorted(_GLM_SUPPORTED_IMAGE_MIMES))
            raise ValueError(f"GLM backend currently supports image MIME types: {supported}")
        return

    raise ValueError(f"Unsupported input_kind: {input_kind!r}")


def _extract_fallback_markdown(layout_details_obj: object) -> str:
    if not isinstance(layout_details_obj, list):
        return ""
    layout_details = cast(list[object], layout_details_obj)
    lines: list[str] = []
    for page_obj in layout_details:
        if not isinstance(page_obj, list):
            continue
        page_items = cast(list[object], page_obj)
        for item in page_items:
            if not isinstance(item, Mapping):
                continue
            item_map = cast(Mapping[str, object], item)
            label = str(item_map.get("label", "") or "").strip().lower()
            content = str(item_map.get("content", "") or "").strip()
            if label in {"text", "formula"} and content:
                lines.append(content)
    return "\n\n".join(lines).strip()


def make_glm_runner(*, api_key: str, backoff: BackoffConfig) -> OcrRunner:
    def run(
        *,
        file_name: str,
        content: bytes,
        model: str,
        delete_remote_file: bool,
        input_kind: InputKind,
        mime_type: str | None,
        table_format: TableFormat | None,
        extract_header: bool,
        extract_footer: bool,
        progress: Progress,
    ) -> OcrResult:
        _ = (delete_remote_file, table_format, extract_header, extract_footer)
        _validate_glm_input(file_name=file_name, input_kind=input_kind, mime_type=mime_type)

        progress.emit("Encoding file...")
        payload = {
            "model": model,
            "file": base64.b64encode(content).decode("utf-8"),
        }

        progress.emit("Running OCR (this can take a while)...")
        response = with_backoff(
            lambda: _post_json(url=_GLM_LAYOUT_PARSING_URL, api_key=api_key, payload=payload),
            what="GLM layout_parsing request",
            cfg=backoff,
            progress=progress,
        )

        markdown = str(response.get("md_results", "") or "").strip()
        if not markdown:
            markdown = _extract_fallback_markdown(response.get("layout_details"))

        return OcrResult(pages=[OcrPage(markdown=markdown, images=[])])

    return run
