from __future__ import annotations

from typing import Optional

from mistralai import Mistral

from pdf2md_cli.pipeline import OcrRunner
from pdf2md_cli.retry import BackoffConfig, with_backoff
from pdf2md_cli.types import OcrImage, OcrPage, OcrResult, ProgressFn


def make_mistral_runner(*, api_key: str, backoff: BackoffConfig) -> OcrRunner:
    def run(
        *,
        file_name: str,
        content: bytes,
        model: str,
        delete_remote_file: bool,
        progress: Optional[ProgressFn],
    ) -> OcrResult:
        with Mistral(api_key=api_key) as client:
            uploaded_file_id: Optional[str] = None

            try:
                if progress:
                    progress("Uploading PDF...")

                uploaded = with_backoff(
                    lambda: client.files.upload(
                        file={"file_name": file_name, "content": content},
                        purpose="ocr",
                    ),
                    what="Upload PDF",
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

                if progress:
                    progress("Running OCR (this can take a while)...")

                ocr_response = with_backoff(
                    lambda: client.ocr.process(
                        model=model,
                        document={"type": "document_url", "document_url": doc_url},
                        include_image_base64=True,
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

        pages: list[OcrPage] = []
        for page in ocr_response.pages:
            images: list[OcrImage] = []
            for img in page.images:
                images.append(OcrImage(id=str(img.id), image_base64=getattr(img, "image_base64", None)))
            pages.append(OcrPage(markdown=str(page.markdown), images=images))

        return OcrResult(pages=pages)

    return run

