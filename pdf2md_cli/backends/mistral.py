from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Mapping

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


def _post_json(*, url: str, api_key: str, payload: dict[str, object], timeout_s: float = 120.0) -> dict:
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
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = str(e)
        headers = {k: v for k, v in (e.headers.items() if e.headers else [])}
        raise _HttpStatusError(status_code=int(getattr(e, "code", 0) or 0), body=body, headers=headers) from e


def make_mistral_runner(*, api_key: str, backoff: BackoffConfig) -> OcrRunner:
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
        include_image_base64: bool,
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

                ocr_response = with_backoff(
                    lambda: _post_json(
                        url="https://api.mistral.ai/v1/ocr",
                        api_key=api_key,
                        payload={
                            "model": model,
                            "document": {"type": "image_url", "image_url": f"data:{mime_type};base64,{b64}"},
                            "include_image_base64": include_image_base64,
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

                    uploaded = with_backoff(
                        lambda: client.files.upload(
                            file={"file_name": file_name, "content": content},
                            purpose="ocr",
                        ),
                        what="Upload document",
                        cfg=backoff,
                        progress=progress,
                    )

                    uploaded_file_id_obj = getattr(uploaded, "id", None)
                    if uploaded_file_id_obj is None and isinstance(uploaded, dict):
                        uploaded_file_id_obj = uploaded.get("id")
                    uploaded_file_id = str(uploaded_file_id_obj or "").strip()
                    if not uploaded_file_id:
                        raise RuntimeError(f"Upload succeeded but returned no file id: {uploaded!r}")

                    signed = with_backoff(
                        lambda: client.files.get_signed_url(file_id=uploaded_file_id),
                        what="Get signed URL",
                        cfg=backoff,
                        progress=progress,
                    )
                    doc_url_obj = getattr(signed, "url", None)
                    if doc_url_obj is None and isinstance(signed, dict):
                        doc_url_obj = signed.get("url")
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
                                "include_image_base64": include_image_base64,
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

        pages_src = ocr_response.get("pages", [])

        pages: list[OcrPage] = []
        for page in pages_src:
            if isinstance(page, dict):
                markdown = str(page.get("markdown", ""))
                images_src = page.get("images", []) or []
                tables_src = page.get("tables", []) or []
                header = page.get("header")
                footer = page.get("footer")
            else:
                markdown = str(getattr(page, "markdown", ""))
                images_src = getattr(page, "images", []) or []
                tables_src = getattr(page, "tables", []) or []
                header = getattr(page, "header", None)
                footer = getattr(page, "footer", None)

            images: list[OcrImage] = []
            for img in images_src:
                if isinstance(img, dict):
                    images.append(OcrImage(id=str(img.get("id")), image_base64=img.get("image_base64")))
                else:
                    images.append(OcrImage(id=str(getattr(img, "id", "")), image_base64=getattr(img, "image_base64", None)))

            tables: list[OcrTable] = []
            for tbl in tables_src:
                if isinstance(tbl, dict):
                    tables.append(
                        OcrTable(
                            id=str(tbl.get("id", "")),
                            content=str(tbl.get("content", "")),
                            format=tbl.get("format"),
                        )
                    )
                else:
                    tables.append(
                        OcrTable(
                            id=str(getattr(tbl, "id", "")),
                            content=str(getattr(tbl, "content", "")),
                            format=getattr(tbl, "format", None),
                        )
                    )

            pages.append(OcrPage(markdown=markdown, images=images, tables=tables, header=header, footer=footer))

        return OcrResult(pages=pages)

    return run
