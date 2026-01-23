from __future__ import annotations

import tempfile
import unittest
import base64
from pathlib import Path

from pdf2md_cli.backends.mock import MockConfig, make_mock_runner
from pdf2md_cli.pipeline import convert_file_to_markdown, convert_pdf_to_markdown
from pdf2md_cli.retry import BackoffConfig


class TestMockBackend(unittest.TestCase):
    def test_mock_backend_writes_markdown_and_images(self) -> None:
        runner = make_mock_runner(
            mock=MockConfig(pages=2, images_per_page=2, delay_ms=0, fail_first=0),
            backoff=BackoffConfig(max_retries=0, initial_delay_s=0.0, max_delay_s=0.01, jitter=0.0),
        )

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pdf_path = tmp / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%mock\n")

            outdir = tmp / "out"
            res = convert_file_to_markdown(
                input_file=pdf_path,
                outdir=outdir,
                runner=runner,
                model="mock-model",
                delete_remote_file=True,
                progress=None,
            )

            self.assertTrue(res.markdown_path.exists())
            md = res.markdown_path.read_text(encoding="utf-8")

            # Should have generated 4 images total (2 pages x 2 images/page).
            self.assertEqual(len(res.image_id_to_filename), 4)
            for _img_id, fname in res.image_id_to_filename.items():
                self.assertIn(fname, md)
                self.assertTrue((outdir / fname).exists())

    def test_mock_backend_can_fail_then_succeed_with_retries(self) -> None:
        runner = make_mock_runner(
            mock=MockConfig(pages=1, images_per_page=0, delay_ms=0, fail_first=2),
            backoff=BackoffConfig(max_retries=2, initial_delay_s=0.0, max_delay_s=0.01, jitter=0.0),
        )

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pdf_path = tmp / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%mock\n")

            outdir = tmp / "out"
            res = convert_file_to_markdown(
                input_file=pdf_path,
                outdir=outdir,
                runner=runner,
                model="mock-model",
                delete_remote_file=True,
                progress=None,
            )
            self.assertTrue(res.markdown_path.exists())

    def test_mock_backend_accepts_image_inputs(self) -> None:
        runner = make_mock_runner(
            mock=MockConfig(pages=1, images_per_page=1, delay_ms=0, fail_first=0),
            backoff=BackoffConfig(max_retries=0, initial_delay_s=0.0, max_delay_s=0.01, jitter=0.0),
        )

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            img_path = tmp / "sample.png"
            img_path.write_bytes(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
                    "pfZFQAAAAABJRU5ErkJggg=="
                )
            )

            outdir = tmp / "out"
            res = convert_file_to_markdown(
                input_file=img_path,
                outdir=outdir,
                runner=runner,
                model="mock-model",
                delete_remote_file=True,
                progress=None,
            )

            self.assertTrue(res.markdown_path.exists())

    def test_convert_pdf_to_markdown_rejects_non_pdf(self) -> None:
        runner = make_mock_runner(
            mock=MockConfig(pages=1, images_per_page=0, delay_ms=0, fail_first=0),
            backoff=BackoffConfig(max_retries=0, initial_delay_s=0.0, max_delay_s=0.01, jitter=0.0),
        )

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            img_path = tmp / "not_a_pdf.png"
            img_path.write_bytes(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
                    "pfZFQAAAAABJRU5ErkJggg=="
                )
            )

            with self.assertRaises(ValueError):
                convert_pdf_to_markdown(
                    pdf_file=img_path,
                    outdir=tmp / "out",
                    runner=runner,
                    model="mock-model",
                    delete_remote_file=True,
                    progress=None,
                )


if __name__ == "__main__":
    unittest.main()
