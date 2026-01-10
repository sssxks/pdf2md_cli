from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from pdf2md_cli.pipeline import OcrRunner
from pdf2md_cli.retry import BackoffConfig, with_backoff
from pdf2md_cli.types import OcrImage, OcrPage, OcrResult, ProgressFn


_TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "pfZFQAAAAABJRU5ErkJggg=="
)


@dataclass(frozen=True, slots=True)
class MockConfig:
    pages: int = 1
    images_per_page: int = 1
    delay_ms: int = 0
    fail_first: int = 0


def make_mock_runner(*, mock: MockConfig, backoff: BackoffConfig) -> OcrRunner:
    def run(
        *,
        file_name: str,
        content: bytes,
        model: str,
        delete_remote_file: bool,
        progress: Optional[ProgressFn],
    ) -> OcrResult:
        # Parameters exist to match the real runner; most are unused but useful for UX parity.
        _ = (file_name, content, model, delete_remote_file)

        remaining_failures = mock.fail_first

        def _attempt() -> OcrResult:
            nonlocal remaining_failures
            if remaining_failures > 0:
                remaining_failures -= 1
                # Message includes "timeout" so retry heuristics treat it as transient.
                raise RuntimeError("mock timeout")

            if mock.delay_ms > 0:
                if progress:
                    progress(f"Mock backend sleeping {mock.delay_ms}ms...")
                time.sleep(mock.delay_ms / 1000.0)

            pages: list[OcrPage] = []
            img_global = 1
            for p in range(max(1, mock.pages)):
                images: list[OcrImage] = []
                md_lines = [f"# Mock OCR", f"", f"- page: {p + 1}", f"- backend: mock", ""]

                for _i in range(max(0, mock.images_per_page)):
                    img_id = f"mock_image_{img_global:03d}"
                    img_global += 1
                    images.append(OcrImage(id=img_id, image_base64=_TINY_PNG_BASE64))
                    md_lines.append(f"![{img_id}]({img_id})")

                pages.append(OcrPage(markdown="\n".join(md_lines).strip(), images=images))

            return OcrResult(pages=pages)

        return with_backoff(_attempt, what="Mock OCR", cfg=backoff, progress=progress)

    return run
