from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pdf2md_cli.types import OcrResult, ProgressFn


class OcrRunner(Protocol):
    def __call__(
        self,
        *,
        file_name: str,
        content: bytes,
        model: str,
        delete_remote_file: bool,
        input_kind: str,
        mime_type: str | None,
        progress: ProgressFn | None,
    ) -> OcrResult: ...


@dataclass(frozen=True, slots=True)
class ConvertResult:
    markdown_path: Path
    image_id_to_filename: dict[str, str]


VALID_DOCUMENT_EXTENSIONS = {".pdf"}
VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".avif",
    ".tiff",
    ".tif",
    ".gif",
    ".heic",
    ".heif",
    ".bmp",
    ".webp",
}

_EXT_TO_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".avif": "image/avif",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".gif": "image/gif",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def clean_base64_payload(base64_str: str) -> str:
    payload = base64_str.strip()
    if "," in payload:
        payload = payload.split(",", 1)[1]
    return "".join(payload.split())


def write_placeholder_png(path: Path) -> None:
    path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg=="
        )
    )


def decode_and_save_images(ocr_result: OcrResult, outdir: Path, stem: str) -> dict[str, str]:
    """
    Save images to disk and return a map from OCR image id -> saved filename.

    Images are saved next to the markdown file as: {stem}_image_XXX.png
    """

    id_to_filename: dict[str, str] = {}
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
            if img_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                out_path.write_bytes(img_bytes)
            else:
                try:
                    from io import BytesIO

                    from PIL import Image  # type: ignore[import-not-found]
                except Exception as e:  # noqa: BLE001
                    raise RuntimeError(
                        "Pillow is required to convert extracted images to PNG. Install with: pip install pillow"
                    ) from e

                pil = Image.open(BytesIO(img_bytes)).convert("RGBA")
                pil.save(out_path, format="PNG")
            id_to_filename[img.id] = filename

    return id_to_filename


def rewrite_markdown(markdown_text: str, id_to_filename: dict[str, str]) -> str:
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
    progress: ProgressFn | None = None,
) -> ConvertResult:
    input_kind, _mime_type = _classify_input(pdf_file)
    if input_kind != "pdf":
        raise ValueError(f"pdf_file must be a PDF (.pdf). Got: {pdf_file.name}")
    return convert_file_to_markdown(
        input_file=pdf_file,
        outdir=outdir,
        runner=runner,
        model=model,
        delete_remote_file=delete_remote_file,
        progress=progress,
    )


def _classify_input(input_file: Path) -> tuple[str, str | None]:
    ext = input_file.suffix.lower()
    if ext in VALID_DOCUMENT_EXTENSIONS:
        return "pdf", None
    if ext in VALID_IMAGE_EXTENSIONS:
        mime = _EXT_TO_MIME.get(ext)
        if not mime:
            raise ValueError(f"Unsupported image type: {ext}")
        return "image", mime
    raise ValueError(
        f"Unsupported file type: {ext}. Supported: PDFs ({', '.join(sorted(VALID_DOCUMENT_EXTENSIONS))}) "
        f"and images ({', '.join(sorted(VALID_IMAGE_EXTENSIONS))})."
    )


def convert_file_to_markdown(
    *,
    input_file: Path,
    outdir: Path,
    runner: OcrRunner,
    model: str,
    delete_remote_file: bool,
    progress: ProgressFn | None = None,
) -> ConvertResult:
    input_kind, mime_type = _classify_input(input_file)

    ensure_outdir(outdir)

    if progress:
        progress("Reading input...")
    content = input_file.read_bytes()

    if progress:
        progress("Running OCR...")
    ocr_result = runner(
        file_name=input_file.name,
        content=content,
        model=model,
        delete_remote_file=delete_remote_file,
        input_kind=input_kind,
        mime_type=mime_type,
        progress=progress,
    )

    markdown_text = "\n\n".join(page.markdown for page in ocr_result.pages).strip()

    if progress:
        progress("Saving images and markdown...")
    stem = input_file.stem
    id_to_filename = decode_and_save_images(ocr_result, outdir, stem)
    rewritten_markdown = rewrite_markdown(markdown_text, id_to_filename)

    md_path = outdir / f"{stem}.md"
    md_path.write_text(rewritten_markdown, encoding="utf-8")

    return ConvertResult(markdown_path=md_path, image_id_to_filename=id_to_filename)
