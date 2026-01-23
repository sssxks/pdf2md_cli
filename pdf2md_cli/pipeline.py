from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from pathlib import PurePath
from typing import Protocol

import marko
from marko.block import HTMLBlock, Paragraph
from marko.inline import Image, Link, RawText
from marko.md_renderer import MarkdownRenderer

from pdf2md_cli.types import NO_PROGRESS, InputKind, OcrResult, Progress, TableFormat


class OcrRunner(Protocol):
    def __call__(
        self,
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


def _sanitize_filename(name: str) -> str:
    safe = PurePath(str(name)).name
    safe = safe.strip()
    if not safe or safe in {".", ".."}:
        raise ValueError(f"Invalid filename: {name!r}")
    return safe


def write_tables(ocr_result: OcrResult, outdir: Path) -> None:
    """
    Save extracted tables to disk (when present).

    The Markdown from Mistral OCR typically links to tables as: [tbl-4.html](tbl-4.html).
    To keep these links working, this function writes each table using its OCR-provided id
    as the filename (sanitized).
    """

    for page in ocr_result.pages:
        for tbl in getattr(page, "tables", []):
            if not getattr(tbl, "id", None):
                continue
            filename = _sanitize_filename(tbl.id)
            (outdir / filename).write_text(tbl.content or "", encoding="utf-8")


def write_extracted_headers_and_footers(ocr_result: OcrResult, outdir: Path, stem: str) -> None:
    """
    If header/footer extraction is enabled, write a sidecar markdown file with the extracted content.
    """

    chunks: list[str] = []
    for idx, page in enumerate(ocr_result.pages, start=1):
        header = getattr(page, "header", None)
        footer = getattr(page, "footer", None)
        if header:
            chunks.append(f"## Page {idx} header\n\n{header}".strip())
        if footer:
            chunks.append(f"## Page {idx} footer\n\n{footer}".strip())

    if not chunks:
        return

    extras_path = outdir / f"{stem}_headers_footers.md"
    extras_path.write_text("\n\n".join(chunks).strip() + "\n", encoding="utf-8")


def _yaml_quote(value: str) -> str:
    needs_quotes = (
        value == ""
        or value.strip() != value
        or any(ch in value for ch in [":", "#", "\n", "\r", "\t", '"', "'"])
    )
    if not needs_quotes:
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f"\"{escaped}\""


def build_front_matter(
    *,
    input_file: Path,
    model: str,
    ocr_result: OcrResult,
) -> str:
    try:
        tool_version = metadata.version("pdf2md-cli")
    except Exception:
        tool_version = "unknown"

    pages = len(ocr_result.pages)
    images = sum(len(p.images) for p in ocr_result.pages)

    generated_at_utc = datetime.now(UTC).isoformat(timespec="seconds")

    lines = [
        "---",
        "tool: 2md",
        f"tool_version: {_yaml_quote(tool_version)}",
        f"ocr_model: {_yaml_quote(model)}",
        f"source_file: {_yaml_quote(input_file.name)}",
        f"generated_at_utc: {_yaml_quote(generated_at_utc)}",
        f"pages: {pages}",
        f"images: {images}",
        "---",
        "",
    ]
    return "\n".join(lines)


def _alt_text(img: Image) -> str | None:
    if not hasattr(img, "children"):
        return None
    if not isinstance(img.children, list):
        return None
    if len(img.children) != 1:
        return None
    child = img.children[0]
    if not isinstance(child, RawText):
        return None
    if not isinstance(child.children, str):
        return None
    return child.children


def _set_alt_text(img: Image, alt: str) -> None:
    img.children = [RawText(alt)]


def _rewrite_images_in_place(node: object, id_to_filename: dict[str, str]) -> None:
    if isinstance(node, Image):
        old_dest = str(getattr(node, "dest", "") or "")
        new_dest = id_to_filename.get(old_dest)
        if new_dest:
            node.dest = new_dest
            alt = _alt_text(node)
            if alt is not None and (alt == "" or alt == old_dest):
                _set_alt_text(node, new_dest)

    children = getattr(node, "children", None)
    if isinstance(children, list):
        for child in children:
            _rewrite_images_in_place(child, id_to_filename)


def _inline_tables_in_blocks(node: object, id_to_table_html: dict[str, str]) -> None:
    children = getattr(node, "children", None)
    if not isinstance(children, list):
        return

    def _split_has_significant_inline(inlines: list[object]) -> bool:
        for inline in inlines:
            if isinstance(inline, RawText) and isinstance(inline.children, str):
                if inline.children.strip() != "":
                    return True
                continue
            return True
        return False

    def _table_html_for_inline(inline: object) -> tuple[str, str] | None:
        if not isinstance(inline, Link) or isinstance(inline, Image):
            return None
        dest = str(getattr(inline, "dest", "") or "")
        filename = PurePath(dest.split("#", 1)[0].split("?", 1)[0]).name
        html = id_to_table_html.get(filename)
        if not html:
            return None
        return filename, html

    for child in children:
        _inline_tables_in_blocks(child, id_to_table_html)

    new_children: list[object] = []
    for child in children:
        if not isinstance(child, Paragraph):
            new_children.append(child)
            continue

        inlines = getattr(child, "children", None)
        if not isinstance(inlines, list) or not inlines:
            new_children.append(child)
            continue

        segment: list[object] = []
        replaced_any = False
        for inline in inlines:
            tbl = _table_html_for_inline(inline)
            if tbl is None:
                segment.append(inline)
                continue

            filename, html = tbl
            if _split_has_significant_inline(segment):
                p = Paragraph([])
                p.children = segment
                new_children.append(p)
            segment = []
            new_children.append(HTMLBlock(f"<!-- table: {filename} -->\n\n{html}"))
            replaced_any = True

        if replaced_any:
            if _split_has_significant_inline(segment):
                p = Paragraph([])
                p.children = segment
                new_children.append(p)
        else:
            new_children.append(child)

    node.children = new_children


def rewrite_markdown(
    markdown_text: str,
    id_to_filename: dict[str, str],
    *,
    ocr_result: OcrResult | None = None,
    inline_tables: bool = False,
    table_format: TableFormat | None = None,
) -> str:
    md = marko.Markdown(renderer=MarkdownRenderer)
    doc = md.parse(markdown_text)

    _rewrite_images_in_place(doc, id_to_filename)

    if inline_tables and table_format == TableFormat.HTML and ocr_result is not None:
        id_to_table_html: dict[str, str] = {}
        for page in ocr_result.pages:
            for tbl in getattr(page, "tables", []) or []:
                if getattr(tbl, "id", None) and getattr(tbl, "content", None):
                    id_to_table_html[str(tbl.id)] = str(tbl.content)
        if id_to_table_html:
            _inline_tables_in_blocks(doc, id_to_table_html)

    return md.render(doc).strip()


def convert_pdf_to_markdown(
    *,
    pdf_file: Path,
    outdir: Path,
    runner: OcrRunner,
    model: str,
    delete_remote_file: bool,
    table_format: TableFormat | None = None,
    extract_header: bool = False,
    extract_footer: bool = False,
    include_image_base64: bool = True,
    progress: Progress = NO_PROGRESS,
) -> ConvertResult:
    input_kind, _mime_type = _classify_input(pdf_file)
    if input_kind != InputKind.PDF:
        raise ValueError(f"pdf_file must be a PDF (.pdf). Got: {pdf_file.name}")
    return convert_file_to_markdown(
        input_file=pdf_file,
        outdir=outdir,
        runner=runner,
        model=model,
        delete_remote_file=delete_remote_file,
        table_format=table_format,
        extract_header=extract_header,
        extract_footer=extract_footer,
        include_image_base64=include_image_base64,
        progress=progress,
    )


def _classify_input(input_file: Path) -> tuple[InputKind, str | None]:
    ext = input_file.suffix.lower()
    if ext in VALID_DOCUMENT_EXTENSIONS:
        return InputKind.PDF, None
    if ext in VALID_IMAGE_EXTENSIONS:
        mime = _EXT_TO_MIME.get(ext)
        if not mime:
            raise ValueError(f"Unsupported image type: {ext}")
        return InputKind.IMAGE, mime
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
    table_format: TableFormat | None = None,
    extract_header: bool = False,
    extract_footer: bool = False,
    include_image_base64: bool = True,
    add_front_matter: bool = True,
    add_page_markers: bool = True,
    inline_tables: bool = True,
    progress: Progress = NO_PROGRESS,
) -> ConvertResult:
    input_kind, mime_type = _classify_input(input_file)

    ensure_outdir(outdir)

    progress.emit("Reading input...")
    content = input_file.read_bytes()

    progress.emit("Running OCR...")
    ocr_result = runner(
        file_name=input_file.name,
        content=content,
        model=model,
        delete_remote_file=delete_remote_file,
        input_kind=input_kind,
        mime_type=mime_type,
        table_format=table_format,
        extract_header=extract_header,
        extract_footer=extract_footer,
        include_image_base64=include_image_base64,
        progress=progress,
    )

    page_chunks: list[str] = []
    for idx, page in enumerate(ocr_result.pages, start=1):
        page_md = (page.markdown or "").strip()
        if add_page_markers:
            page_chunks.append(f"<!-- page: {idx} -->\n\n{page_md}".strip())
        else:
            page_chunks.append(page_md)

    markdown_text = "\n\n".join([c for c in page_chunks if c]).strip()

    progress.emit("Saving images and markdown...")
    stem = input_file.stem
    id_to_filename = decode_and_save_images(ocr_result, outdir, stem)
    if table_format is not None and not inline_tables:
        write_tables(ocr_result, outdir)
    write_extracted_headers_and_footers(ocr_result, outdir, stem)
    rewritten_markdown = rewrite_markdown(
        markdown_text,
        id_to_filename,
        ocr_result=ocr_result,
        inline_tables=inline_tables,
        table_format=table_format,
    )

    if add_front_matter:
        rewritten_markdown = (
            build_front_matter(
                input_file=input_file,
                model=model,
                ocr_result=ocr_result,
            )
            + rewritten_markdown.lstrip()
        )

    md_path = outdir / f"{stem}.md"
    md_path.write_text(rewritten_markdown, encoding="utf-8")

    return ConvertResult(markdown_path=md_path, image_id_to_filename=id_to_filename)
