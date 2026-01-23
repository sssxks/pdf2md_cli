from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import cast

from mistralai import Mistral

from pdf2md_cli.pipeline import OcrRunner
from pdf2md_cli.retry import BackoffConfig, with_backoff
from pdf2md_cli.types import InputKind, OcrImage, OcrPage, OcrResult, OcrTable, Progress, TableFormat


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
    *, url: str, api_key: str, payload: Mapping[str, object], timeout_s: float = 120.0
) -> dict[str, object]:
    """Send a JSON POST request and decode the response as a JSON object."""
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


def make_mistral_runner(*, api_key: str, backoff: BackoffConfig) -> OcrRunner:
    """
    Build an OCR runner backed by Mistral OCR + the Files API (for documents).

    Behavior:
        - Images are sent directly to the OCR API as a `data:` URL.
        - Documents are uploaded to the Files API, a signed URL is obtained, then OCR is called.
        - When `delete_remote_file=True`, uploaded documents are deleted best-effort after OCR.
        - Transient failures are retried according to `backoff`.
    """
    def _get(obj: object, key: str, default: object | None = None) -> object | None:
        if isinstance(obj, Mapping):
            obj_map = cast(Mapping[str, object], obj)
            return obj_map.get(key, default)
        return cast(object | None, getattr(obj, key, default))

    def _to_opt_str(v: object | None) -> str | None:
        return None if v is None else str(v)

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
        with Mistral(api_key=api_key) as client:
            uploaded_file_id: str | None = None

            if input_kind == InputKind.IMAGE:
                if not mime_type:
                    raise ValueError("mime_type is required for image inputs")
                progress.emit("Encoding image...")
                b64 = base64.b64encode(content).decode("utf-8")
                progress.emit("Running OCR (this can take a while)...")

                ocr_response: dict[str, object] = with_backoff(
                    lambda: _post_json(
                        url="https://api.mistral.ai/v1/ocr",
                        api_key=api_key,
                        payload={
                            "model": model,
                            "document": {"type": "image_url", "image_url": f"data:{mime_type};base64,{b64}"},
                            "include_image_base64": True,
                            **({} if table_format is None else {"table_format": table_format.value}),
                            **({} if not extract_header else {"extract_header": True}),
                            **({} if not extract_footer else {"extract_footer": True}),
                        },
                    ),
                    what="OCR request",
                    cfg=backoff,
                    progress=progress,
                )
            elif input_kind == InputKind.DOCUMENT:
                try:
                    progress.emit("Uploading document...")

                    def _upload() -> object:
                        return client.files.upload(
                            file={"file_name": file_name, "content": content},
                            purpose="ocr",
                        )

                    uploaded: object = with_backoff(
                        _upload,
                        what="Upload document",
                        cfg=backoff,
                        progress=progress,
                    )

                    uploaded_file_id_obj = _get(uploaded, "id")
                    uploaded_file_id = str(uploaded_file_id_obj or "").strip()
                    if not uploaded_file_id:
                        raise RuntimeError(f"Upload succeeded but returned no file id: {uploaded!r}")

                    def _get_signed_url() -> object:
                        return client.files.get_signed_url(file_id=uploaded_file_id)

                    signed: object = with_backoff(
                        _get_signed_url,
                        what="Get signed URL",
                        cfg=backoff,
                        progress=progress,
                    )
                    doc_url_obj = _get(signed, "url")
                    doc_url = str(doc_url_obj or "").strip()
                    if not doc_url:
                        raise RuntimeError(f"Signed URL request returned no url: {signed!r}")

                    progress.emit("Running OCR (this can take a while)...")

                    ocr_response = with_backoff(
                        lambda: _post_json(
                            url="https://api.mistral.ai/v1/ocr",
                            api_key=api_key,
                            payload={
                                "model": model,
                                "document": {"type": "document_url", "document_url": doc_url},
                                "include_image_base64": True,
                                **({} if table_format is None else {"table_format": table_format.value}),
                                **({} if not extract_header else {"extract_header": True}),
                                **({} if not extract_footer else {"extract_footer": True}),
                            },
                        ),
                        what="OCR request",
                        cfg=backoff,
                        progress=progress,
                    )
                finally:
                    if delete_remote_file and uploaded_file_id:
                        try:
                            # Keep cleanup retries bounded; failures should not fail the whole run.
                            cleanup_cfg = BackoffConfig(
                                max_retries=min(3, backoff.max_retries),
                                initial_delay_s=min(1.0, backoff.initial_delay_s),
                                max_delay_s=min(10.0, backoff.max_delay_s),
                                multiplier=backoff.multiplier,
                                jitter=backoff.jitter,
                            )
                            with_backoff(
                                lambda: client.files.delete(file_id=uploaded_file_id),
                                what="Delete remote file",
                                cfg=cleanup_cfg,
                                progress=progress,
                            )
                        except Exception:
                            # Best-effort cleanup.
                            pass
            else:
                raise ValueError(f"Unsupported input_kind: {input_kind!r}")

        pages_src_obj = ocr_response.get("pages", [])
        pages_src: list[object] = cast(list[object], pages_src_obj) if isinstance(pages_src_obj, list) else []

        pages: list[OcrPage] = []
        for page in pages_src:
            markdown = str(_get(page, "markdown", "") or "")
            images_src_obj = _get(page, "images")
            tables_src_obj = _get(page, "tables")
            header = _to_opt_str(_get(page, "header"))
            footer = _to_opt_str(_get(page, "footer"))

            images_src: list[object] = cast(list[object], images_src_obj) if isinstance(images_src_obj, list) else []
            tables_src: list[object] = cast(list[object], tables_src_obj) if isinstance(tables_src_obj, list) else []

            images: list[OcrImage] = []
            for img in images_src:
                image_id = str(_get(img, "id", "") or "")
                image_base64 = _to_opt_str(_get(img, "image_base64"))
                images.append(OcrImage(id=image_id, image_base64=image_base64))

            tables: list[OcrTable] = []
            for tbl in tables_src:
                table_id = str(_get(tbl, "id", "") or "")
                table_content = str(_get(tbl, "content", "") or "")
                fmt = _to_opt_str(_get(tbl, "format"))
                tables.append(OcrTable(id=table_id, content=table_content, format=fmt))

            pages.append(OcrPage(markdown=markdown, images=images, tables=tables, header=header, footer=footer))

        return OcrResult(pages=pages)

    return run
