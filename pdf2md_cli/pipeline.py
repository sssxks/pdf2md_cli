from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path, PurePath
from typing import Protocol, cast

import marko
from marko.block import HTMLBlock, Paragraph
from marko.element import Element
from marko.inline import Image, Link, RawText
from marko.md_renderer import MarkdownRenderer

from pdf2md_cli.types import NO_PROGRESS, HeaderFooterMode, InputKind, OcrResult, Progress, TableFormat


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
        progress: Progress,
    ) -> OcrResult: ...


@dataclass(frozen=True, slots=True)
class ConvertResult:
    """Output of a single-file conversion."""

    markdown_path: Path
    image_id_to_filename: dict[str, str]


VALID_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".txt",
    ".epub",
    ".xml",
    ".rtf",
    ".odt",
    ".bib",
    ".fb2",
    ".ipynb",
    ".tex",
    ".opml",
    ".1",
    ".man",
}
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

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def ensure_outdir(outdir: Path) -> None:
    """Create the output directory if it doesn't exist."""
    outdir.mkdir(parents=True, exist_ok=True)


def clean_base64_payload(base64_str: str) -> str:
    """Normalize a base64 payload that may include a data-URL prefix and whitespace."""
    payload = base64_str.strip()
    if "," in payload:
        payload = payload.split(",", 1)[1]
    return "".join(payload.split())


def write_placeholder_png(path: Path) -> None:
    """Write a 1×1 PNG placeholder to `path`."""
    path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg=="
        )
    )


def _to_png_bytes(image_bytes: bytes) -> bytes:
    """Convert arbitrary image bytes to PNG bytes (no-op if already a PNG)."""
    if image_bytes.startswith(_PNG_SIGNATURE):
        return image_bytes

    try:
        from io import BytesIO

        from PIL import Image  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Pillow is required to convert extracted images to PNG. Install with: pip install pillow"
        ) from e

    pil_image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    out = BytesIO()
    pil_image.save(out, format="PNG")
    return out.getvalue()


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
            out_path.write_bytes(_to_png_bytes(img_bytes))
            id_to_filename[img.id] = filename

    return id_to_filename


def _sanitize_filename(name: str) -> str:
    """Turn an arbitrary table id / path-like string into a safe filename."""
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


def write_extracted_headers_and_footers(
    ocr_result: OcrResult,
    outdir: Path,
    stem: str,
    *,
    include_headers: bool,
    include_footers: bool,
) -> None:
    """
    If header/footer extraction is enabled, write a sidecar markdown file with the extracted content.
    """

    chunks: list[str] = []
    for idx, page in enumerate(ocr_result.pages, start=1):
        header = getattr(page, "header", None) if include_headers else None
        footer = getattr(page, "footer", None) if include_footers else None
        if header:
            chunks.append(f"## Page {idx} header\n\n{header}".strip())
        if footer:
            chunks.append(f"## Page {idx} footer\n\n{footer}".strip())

    if not chunks:
        return

    extras_path = outdir / f"{stem}_headers_footers.md"
    extras_path.write_text("\n\n".join(chunks).strip() + "\n", encoding="utf-8")


def _yaml_quote(value: str) -> str:
    """Quote a string for YAML front-matter when needed (minimal escaping)."""
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
    """Build a YAML front-matter header describing the conversion output."""
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
    """Return the alt text for an Image element, when it is a single RawText child."""
    if not hasattr(img, "children"):
        return None
    if not isinstance(img.children, list):
        return None
    if len(img.children) != 1:
        return None
    child = img.children[0]
    if not isinstance(child, RawText):
        return None
    return child.children


def _set_alt_text(img: Image, alt: str) -> None:
    """Set the alt text for an Image element as a single RawText child."""
    img.children = [RawText(alt)]


def _rewrite_images_in_place(node: object, id_to_filename: dict[str, str]) -> None:
    """Recursively rewrite image destinations and alt text to local filenames."""
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
        for child in cast(list[object], children):
            _rewrite_images_in_place(child, id_to_filename)


def _inline_tables_in_blocks(node: object, id_to_table_html: dict[str, str]) -> None:
    """
    Inline extracted HTML tables into the parsed Markdown AST.

    Mistral OCR can emit links like `[tbl-4.html](tbl-4.html)` while also returning the
    table HTML in the API response. When `--table-format html` (default) and not extracting
    tables to sidecar files, we replace those link-only paragraphs with HTML blocks so the
    rendered Markdown contains the tables inline.
    """
    children_any = getattr(node, "children", None)
    if not isinstance(children_any, list):
        return
    children = cast(list[Element], children_any)

    def _has_significant_inline(inlines: list[Element]) -> bool:
        for inline in inlines:
            if isinstance(inline, RawText):
                if inline.children.strip() != "":
                    return True
                continue
            return True
        return False

    def _linked_table_html(inline: Element) -> tuple[str, str] | None:
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

    new_children: list[Element] = []
    for child in children:
        if not isinstance(child, Paragraph):
            new_children.append(child)
            continue

        inlines = getattr(child, "children", None)
        if not isinstance(inlines, list) or not inlines:
            new_children.append(child)
            continue
        inline_children = cast(list[Element], inlines)

        segment: list[Element] = []
        replaced_any = False
        for inline in inline_children:
            tbl = _linked_table_html(inline)
            if tbl is None:
                segment.append(inline)
                continue

            filename, html = tbl
            if _has_significant_inline(segment):
                p = Paragraph([])
                p.children = segment
                new_children.append(p)
            segment = []
            new_children.append(HTMLBlock(f"<!-- table: {filename} -->\n\n{html}"))
            replaced_any = True

        if replaced_any:
            if _has_significant_inline(segment):
                p = Paragraph([])
                p.children = segment
                new_children.append(p)
        else:
            new_children.append(child)

    setattr(node, "children", new_children)


def _collect_table_html(ocr_result: OcrResult) -> dict[str, str]:
    """Collect `{table_filename -> html}` mappings from an OCR result."""
    id_to_table_html: dict[str, str] = {}
    for page in ocr_result.pages:
        for tbl in getattr(page, "tables", []) or []:
            if getattr(tbl, "id", None) and getattr(tbl, "content", None):
                id_to_table_html[str(tbl.id)] = str(tbl.content)
    return id_to_table_html


def _classify_input(input_file: Path) -> tuple[InputKind, str | None]:
    """Classify an input file as document vs image and return its MIME type (images only)."""
    ext = input_file.suffix.lower()
    if ext in VALID_DOCUMENT_EXTENSIONS:
        return InputKind.DOCUMENT, None
    if ext in VALID_IMAGE_EXTENSIONS:
        mime = _EXT_TO_MIME.get(ext)
        if not mime:
            raise ValueError(f"Unsupported image type: {ext}")
        return InputKind.IMAGE, mime
    raise ValueError(
        f"Unsupported file type: {ext}. Supported: documents ({', '.join(sorted(VALID_DOCUMENT_EXTENSIONS))}) "
        f"and images ({', '.join(sorted(VALID_IMAGE_EXTENSIONS))})."
    )


def _escape_html_comment_payload(text: str) -> str:
    # Keep comments reasonably safe to embed as HTML comments.
    # Avoid closing the comment and the troublesome `--` sequence.
    out = text.replace("-->", "-- >").replace("--", "- -")
    return out.replace("\r\n", "\n").replace("\r", "\n")


def _comment_block(*, label: str, text: str) -> str:
    payload = _escape_html_comment_payload(text).strip()
    if not payload:
        return f"<!-- {label} (empty) -->"
    return f"<!-- {label}\n{payload}\n-->"


def _build_markdown_from_pages(
    ocr_result: OcrResult,
    *,
    add_page_markers: bool,
    header_mode: HeaderFooterMode,
    footer_mode: HeaderFooterMode,
) -> str:
    """Concatenate page markdown, optionally adding HTML page markers and header/footer comments."""
    chunks: list[str] = []
    for page_index, page in enumerate(ocr_result.pages, start=1):
        page_md = (page.markdown or "").strip()
        header = getattr(page, "header", None)
        footer = getattr(page, "footer", None)

        parts: list[str] = []
        if add_page_markers:
            parts.append(f"<!-- page: {page_index} -->")

        if header_mode == HeaderFooterMode.COMMENT and header:
            parts.append(_comment_block(label=f"header: page {page_index}", text=header))

        if page_md:
            parts.append(page_md)

        if footer_mode == HeaderFooterMode.COMMENT and footer:
            parts.append(_comment_block(label=f"footer: page {page_index}", text=footer))

        joined = "\n\n".join(p for p in parts if p.strip()).strip()
        if joined:
            chunks.append(joined)
    return "\n\n".join(chunks).strip()


def _resolve_table_handling(
    *, table_format: TableFormat, extract_table: bool
) -> tuple[TableFormat | None, bool, bool]:
    """
    Map user-intent flags to runner inputs + markdown post-processing behavior.

    Returns:
        (runner_table_format, should_write_tables, should_inline_html_tables)

    Semantics:
        - `table_format=html`:
            - default: request HTML table extraction and inline tables into Markdown output
            - with `extract_table=True`: write `tbl-*.html` sidecars and keep links in Markdown
        - `table_format=markdown`:
            - default: keep tables inline as markdown (API default; do not send table_format)
            - with `extract_table=True`: request extracted markdown tables and write `tbl-*.md` sidecars
    """

    if table_format == TableFormat.HTML:
        runner_table_format = TableFormat.HTML
        should_inline_html_tables = not extract_table
        should_write_tables = extract_table
        return runner_table_format, should_write_tables, should_inline_html_tables

    # markdown mode:
    # - default (no extract): keep tables inline as markdown (API default => do not send table_format)
    # - with extract: request extracted markdown tables separately
    runner_table_format = TableFormat.MARKDOWN if extract_table else None
    should_write_tables = runner_table_format is not None
    should_inline_html_tables = False
    return runner_table_format, should_write_tables, should_inline_html_tables


def convert_file_to_markdown(
    *,
    input_file: Path,
    outdir: Path,
    runner: OcrRunner,
    model: str,
    delete_remote_file: bool,
    table_format: TableFormat = TableFormat.HTML,
    extract_table: bool = False,
    header_mode: HeaderFooterMode = HeaderFooterMode.COMMENT,
    footer_mode: HeaderFooterMode = HeaderFooterMode.COMMENT,
    add_front_matter: bool = True,
    add_page_markers: bool = True,
    progress: Progress = NO_PROGRESS,
) -> ConvertResult:
    """
    Convert a single input file to Markdown, writing outputs under `outdir`.

    Side effects:
        - writes `<stem>.md`
        - writes extracted images as PNG files
        - optionally writes extracted tables / headers+footers as sidecar files
    """
    input_stem = input_file.stem
    input_kind, mime_type = _classify_input(input_file)
    runner_table_format, should_write_tables, should_inline_html_tables = _resolve_table_handling(
        table_format=table_format,
        extract_table=extract_table,
    )

    effective_header_mode = header_mode
    effective_footer_mode = footer_mode
    runner_extract_header = effective_header_mode != HeaderFooterMode.INLINE
    runner_extract_footer = effective_footer_mode != HeaderFooterMode.INLINE
    should_write_headers = effective_header_mode == HeaderFooterMode.EXTRACT
    should_write_footers = effective_footer_mode == HeaderFooterMode.EXTRACT

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
        table_format=runner_table_format,
        extract_header=runner_extract_header,
        extract_footer=runner_extract_footer,
        progress=progress,
    )


    progress.emit("Saving sidecars and markdown...")

    ensure_outdir(outdir)

    # save sidecars
    id_to_filename = decode_and_save_images(ocr_result, outdir, input_stem)
    write_extracted_headers_and_footers(
        ocr_result,
        outdir,
        input_stem,
        include_headers=should_write_headers,
        include_footers=should_write_footers,
    )
    if should_write_tables:
        write_tables(ocr_result, outdir)

    # rewrite markdown
    markdown_text = _build_markdown_from_pages(
        ocr_result,
        add_page_markers=add_page_markers,
        header_mode=effective_header_mode,
        footer_mode=effective_footer_mode,
    )
    md = marko.Markdown(renderer=MarkdownRenderer)
    doc: object = md.parse(markdown_text)
    _rewrite_images_in_place(doc, id_to_filename)
    if should_inline_html_tables:
        id_to_table_html = _collect_table_html(ocr_result)
        if id_to_table_html:
            _inline_tables_in_blocks(doc, id_to_table_html)
    rewritten_markdown = md.render(doc).strip()

    if add_front_matter:
        rewritten_markdown = (
            build_front_matter(
                input_file=input_file,
                model=model,
                ocr_result=ocr_result,
            )
            + rewritten_markdown.lstrip()
        )

    # save markdown
    markdown_path = outdir / f"{input_stem}.md"
    markdown_path.write_text(rewritten_markdown, encoding="utf-8")

    return ConvertResult(markdown_path=markdown_path, image_id_to_filename=id_to_filename)
