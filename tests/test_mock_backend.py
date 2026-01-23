from __future__ import annotations

import io
import os
import tempfile
import unittest
import base64
from contextlib import redirect_stdout
from pathlib import Path

from pdf2md_cli.cli import main as cli_main
from pdf2md_cli.backends.mock import MockConfig, make_mock_runner
from pdf2md_cli.pipeline import convert_file_to_markdown, convert_pdf_to_markdown
from pdf2md_cli.retry import BackoffConfig
from pdf2md_cli.types import NO_PROGRESS, InputKind, OcrPage, OcrResult, OcrTable, TableFormat


class TestMockBackend(unittest.TestCase):
    def test_cli_default_inlines_html_tables(self) -> None:
        os.environ["PDF2MD_ENABLE_MOCK"] = "1"

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pdf_path = tmp / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%mock\n")

            outdir = tmp / "out"
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli_main(
                    [
                        str(pdf_path),
                        "-o",
                        str(outdir),
                        "--backend",
                        "mock",
                        "--mock-pages",
                        "1",
                        "--mock-images",
                        "0",
                    ]
                )

            md_path = outdir / "sample.md"
            self.assertTrue(md_path.exists())
            md = md_path.read_text(encoding="utf-8")
            self.assertIn("<table>", md)
            self.assertIn("<!-- table: tbl-1.html -->", md)
            self.assertNotIn("[tbl-1.html](tbl-1.html)", md)

            # When inlining, do not keep table sidecar files.
            self.assertFalse((outdir / "tbl-1.html").exists())

    def test_inline_tables_splits_paragraph_when_link_has_surrounding_text(self) -> None:
        def runner(
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
            progress: object,
        ) -> OcrResult:
            _ = (
                file_name,
                content,
                model,
                delete_remote_file,
                input_kind,
                mime_type,
                table_format,
                extract_header,
                extract_footer,
                include_image_base64,
                progress,
            )
            return OcrResult(
                pages=[
                    OcrPage(
                        markdown="Before [tbl-1.html](tbl-1.html) after",
                        images=[],
                        tables=[
                            OcrTable(
                                id="tbl-1.html",
                                content="<table><tr><td>1</td></tr></table>",
                                format=TableFormat.HTML.value,
                            )
                        ],
                    )
                ]
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
                table_format=TableFormat.HTML,
                inline_tables=True,
                progress=NO_PROGRESS,
            )

            md = res.markdown_path.read_text(encoding="utf-8")
            self.assertIn("Before", md)
            self.assertIn("after", md)
            self.assertIn("<table>", md)
            self.assertIn("<!-- table: tbl-1.html -->", md)
            self.assertNotIn("[tbl-1.html](tbl-1.html)", md)
            self.assertFalse((outdir / "tbl-1.html").exists())

    def test_cli_table_format_markdown_keeps_inline_markdown_table(self) -> None:
        os.environ["PDF2MD_ENABLE_MOCK"] = "1"

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pdf_path = tmp / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%mock\n")

            outdir = tmp / "out"
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli_main(
                    [
                        str(pdf_path),
                        "-o",
                        str(outdir),
                        "--backend",
                        "mock",
                        "--mock-pages",
                        "1",
                        "--mock-images",
                        "0",
                        "--table-format",
                        "markdown",
                    ]
                )

            md_path = outdir / "sample.md"
            self.assertTrue(md_path.exists())
            md = md_path.read_text(encoding="utf-8")
            self.assertIn("| A | B |", md)
            self.assertFalse((outdir / "tbl-1.html").exists())
            self.assertFalse((outdir / "tbl-1.md").exists())

    def test_cli_extract_table_writes_sidecar_and_keeps_link(self) -> None:
        os.environ["PDF2MD_ENABLE_MOCK"] = "1"

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pdf_path = tmp / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%mock\n")

            outdir = tmp / "out"
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli_main(
                    [
                        str(pdf_path),
                        "-o",
                        str(outdir),
                        "--backend",
                        "mock",
                        "--mock-pages",
                        "1",
                        "--mock-images",
                        "0",
                        "--extract-table",
                        "--table-format",
                        "html",
                    ]
                )

            md_path = outdir / "sample.md"
            self.assertTrue(md_path.exists())
            md = md_path.read_text(encoding="utf-8")
            self.assertIn("[tbl-1.html](tbl-1.html)", md)
            self.assertNotIn("<table>", md)

            tbl_path = outdir / "tbl-1.html"
            self.assertTrue(tbl_path.exists())
            self.assertIn("<table>", tbl_path.read_text(encoding="utf-8"))

    def test_cli_extract_table_markdown_writes_sidecar_and_keeps_link(self) -> None:
        os.environ["PDF2MD_ENABLE_MOCK"] = "1"

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pdf_path = tmp / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%mock\n")

            outdir = tmp / "out"
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli_main(
                    [
                        str(pdf_path),
                        "-o",
                        str(outdir),
                        "--backend",
                        "mock",
                        "--mock-pages",
                        "1",
                        "--mock-images",
                        "0",
                        "--extract-table",
                        "--table-format",
                        "markdown",
                    ]
                )

            md_path = outdir / "sample.md"
            self.assertTrue(md_path.exists())
            md = md_path.read_text(encoding="utf-8")
            self.assertIn("[tbl-1.md](tbl-1.md)", md)
            self.assertNotIn("<table>", md)

            tbl_path = outdir / "tbl-1.md"
            self.assertTrue(tbl_path.exists())
            self.assertIn("| A | B |", tbl_path.read_text(encoding="utf-8"))

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
                progress=NO_PROGRESS,
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
                progress=NO_PROGRESS,
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
                progress=NO_PROGRESS,
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
                    progress=NO_PROGRESS,
                )


if __name__ == "__main__":
    unittest.main()
