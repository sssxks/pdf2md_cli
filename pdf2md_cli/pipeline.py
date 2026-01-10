from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, Optional, Protocol, Tuple

from PIL import Image

from pdf2md_cli.types import OcrResult, ProgressFn


class OcrRunner(Protocol):
    def __call__(
        self,
        *,
        file_name: str,
        content: bytes,
        model: str,
        delete_remote_file: bool,
        progress: Optional[ProgressFn],
    ) -> OcrResult: ...


@dataclass(frozen=True, slots=True)
class ConvertResult:
    markdown_path: Path
    image_id_to_filename: Dict[str, str]


VALID_DOCUMENT_EXTENSIONS = {".pdf"}


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def clean_base64_payload(base64_str: str) -> str:
    payload = base64_str.strip()
    if "," in payload:
        payload = payload.split(",", 1)[1]
    return "".join(payload.split())


def write_placeholder_png(path: Path) -> None:
    Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(path, format="PNG")


def decode_and_save_images(ocr_result: OcrResult, outdir: Path, stem: str) -> Dict[str, str]:
    """
    Save images to disk and return a map from OCR image id -> saved filename.

    Images are saved next to the markdown file as: {stem}_image_XXX.png
    """

    id_to_filename: Dict[str, str] = {}
    img_counter = 1

    for page in ocr_result.pages:
        for img in page.images:
            filename = f"{stem}_image_{img_counter:03d}.png"
            img_counter += 1
            out_path = outdir / filename

            if not img.image_base64:
                id_to_filename[img.id] = filename
                if not out_path.exists():
                    write_placeholder_png(out_path)
                continue

            payload = clean_base64_payload(img.image_base64)
            img_bytes = base64.b64decode(payload)
            pil = Image.open(BytesIO(img_bytes)).convert("RGBA")
            pil.save(out_path, format="PNG")
            id_to_filename[img.id] = filename

    return id_to_filename


def rewrite_markdown(markdown_text: str, id_to_filename: Dict[str, str]) -> str:
    pattern = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<target>[^)]+)\)")

    def repl(match: re.Match[str]) -> str:
        target = match.group("target")
        fname = id_to_filename.get(target)
        if not fname:
            return match.group(0)
        alt = match.group("alt")
        if not alt or alt == target:
            alt = fname
        return f"![{alt}]({fname})"

    return pattern.sub(repl, markdown_text)


def convert_pdf_to_markdown(
    *,
    pdf_file: Path,
    outdir: Path,
    runner: OcrRunner,
    model: str,
    delete_remote_file: bool,
    progress: Optional[ProgressFn] = None,
) -> ConvertResult:
    if pdf_file.suffix.lower() not in VALID_DOCUMENT_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {pdf_file.suffix}. Only PDFs are supported.")

    ensure_outdir(outdir)

    if progress:
        progress("Reading PDF...")
    pdf_bytes = pdf_file.read_bytes()

    if progress:
        progress("Running OCR...")
    ocr_result = runner(
        file_name=pdf_file.name,
        content=pdf_bytes,
        model=model,
        delete_remote_file=delete_remote_file,
        progress=progress,
    )

    markdown_text = "\n\n".join(page.markdown for page in ocr_result.pages).strip()

    if progress:
        progress("Saving images and markdown...")
    stem = pdf_file.stem
    id_to_filename = decode_and_save_images(ocr_result, outdir, stem)
    rewritten_markdown = rewrite_markdown(markdown_text, id_to_filename)

    md_path = outdir / f"{stem}.md"
    md_path.write_text(rewritten_markdown, encoding="utf-8")

    return ConvertResult(markdown_path=md_path, image_id_to_filename=id_to_filename)

