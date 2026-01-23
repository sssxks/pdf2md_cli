from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from pdf2md_cli.api import convert_file, convert_files


class TestApi(unittest.TestCase):
    def test_convert_file_with_mock_backend(self) -> None:
        os.environ["PDF2MD_ENABLE_MOCK"] = "1"

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pdf_path = tmp / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%mock\n")

            outdir = tmp / "out"
            res = convert_file(
                pdf_path,
                outdir=outdir,
                backend="mock",
            )
            self.assertTrue(res.markdown_path.exists())

    def test_convert_files_outdir_matches_cli_semantics(self) -> None:
        os.environ["PDF2MD_ENABLE_MOCK"] = "1"

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "a.pdf").write_bytes(b"%PDF-1.4\n%mock\n")
            (tmp / "b.pdf").write_bytes(b"%PDF-1.4\n%mock\n")

            base_out = tmp / "out"
            batch = convert_files(
                [tmp / "*.pdf"],
                outdir=base_out,
                backend="mock",
                workers=2,
            )
            self.assertTrue(batch.ok)
            self.assertEqual(len(batch.succeeded), 2)
            self.assertTrue((base_out / "a_ocr" / "a.md").exists())
            self.assertTrue((base_out / "b_ocr" / "b.md").exists())
