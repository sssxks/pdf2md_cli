from __future__ import annotations

import json
import urllib.request
import unittest
from typing import cast
from unittest.mock import patch

from pdf2md_cli.api import make_runner
from pdf2md_cli.retry import BackoffConfig
from pdf2md_cli.types import InputKind, NO_PROGRESS


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        _ = (exc_type, exc, tb)
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class TestGlmBackend(unittest.TestCase):
    def test_make_runner_glm_requires_api_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "GLM API key not provided"):
            make_runner(
                backend="glm",
                api_key=None,
                backoff=BackoffConfig(max_retries=0, initial_delay_s=0.0, max_delay_s=0.01, jitter=0.0),
            )

    def test_glm_runner_posts_layout_parsing_request_and_maps_markdown(self) -> None:
        runner = make_runner(
            backend="glm",
            api_key="test-glm-key",
            backoff=BackoffConfig(max_retries=0, initial_delay_s=0.0, max_delay_s=0.01, jitter=0.0),
        )

        captured: dict[str, object] = {}

        def _fake_urlopen(req: urllib.request.Request, timeout: float = 180.0) -> _FakeResponse:
            captured["url"] = req.full_url
            captured["auth"] = req.headers.get("Authorization")
            captured["timeout"] = timeout
            raw_data = req.data
            if isinstance(raw_data, bytes):
                body_bytes = raw_data
            elif raw_data is None:
                body_bytes = b"{}"
            elif isinstance(raw_data, bytearray):
                body_bytes = bytes(raw_data)
            elif isinstance(raw_data, memoryview):
                body_bytes = raw_data.tobytes()
            else:
                raise AssertionError(f"Unexpected request body type: {type(raw_data)!r}")
            captured["body"] = json.loads(body_bytes.decode("utf-8"))
            return _FakeResponse({"md_results": "# GLM OCR\n\nok"})

        with patch("pdf2md_cli.backends.glm.urllib.request.urlopen", _fake_urlopen):
            result = runner(
                file_name="sample.pdf",
                content=b"%PDF-1.4\n%glm\n",
                model="glm-ocr",
                delete_remote_file=True,
                input_kind=InputKind.DOCUMENT,
                mime_type=None,
                table_format=None,
                extract_header=False,
                extract_footer=False,
                progress=NO_PROGRESS,
            )

        self.assertEqual(captured["url"], "https://open.bigmodel.cn/api/paas/v4/layout_parsing")
        self.assertEqual(captured["auth"], "Bearer test-glm-key")
        self.assertEqual(cast(dict[str, object], captured["body"])["model"], "glm-ocr")
        self.assertIn("file", cast(dict[str, object], captured["body"]))
        self.assertEqual(result.pages[0].markdown, "# GLM OCR\n\nok")

    def test_glm_runner_rejects_unsupported_document_extensions(self) -> None:
        runner = make_runner(
            backend="glm",
            api_key="test-glm-key",
            backoff=BackoffConfig(max_retries=0, initial_delay_s=0.0, max_delay_s=0.01, jitter=0.0),
        )
        with self.assertRaisesRegex(ValueError, "supports document inputs"):
            runner(
                file_name="sample.docx",
                content=b"docx",
                model="glm-ocr",
                delete_remote_file=True,
                input_kind=InputKind.DOCUMENT,
                mime_type=None,
                table_format=None,
                extract_header=False,
                extract_footer=False,
                progress=NO_PROGRESS,
            )


if __name__ == "__main__":
    unittest.main()
