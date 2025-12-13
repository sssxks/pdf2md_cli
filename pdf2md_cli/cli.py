import argparse
import base64
import concurrent.futures
import glob
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
import threading
import time

from mistralai import Mistral
from PIL import Image

VALID_DOCUMENT_EXTENSIONS = {".pdf"}


def _error(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def _load_api_key(cli_key: Optional[str]) -> str:
    key = (cli_key or "").strip() or os.environ.get("MISTRAL_API_KEY", "").strip()
    if not key:
        _error("Mistral API key not provided. Use --api-key or set MISTRAL_API_KEY env var.")
        sys.exit(2)
    return key


def _upload_pdf(pdf_path: Path, api_key: str) -> str:
    client = Mistral(api_key=api_key)
    content = pdf_path.read_bytes()
    uploaded_file = client.files.upload(
        file={"file_name": pdf_path.name, "content": content},
        purpose="ocr",
    )
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id)
    return signed_url.url


def _process_ocr(document_url: str, api_key: str):
    client = Mistral(api_key=api_key)
    return client.ocr.process(
        model="mistral-ocr-2505",
        document={"type": "document_url", "document_url": document_url},
        include_image_base64=True,
    )


def _decode_and_save_images(ocr_response, outdir: Path, stem: str) -> Dict[str, str]:
    """Save images to disk and return a map from image id -> saved filename.

    Images are saved next to the markdown file as: {stem}_image_XXX.png
    """
    id_to_filename: Dict[str, str] = {}
    img_counter = 1

    for page in ocr_response.pages:
        for img in page.images:
            filename = f"image_{img_counter:03d}.png"
            img_counter += 1
            out_path = outdir / filename

            if not getattr(img, "image_base64", None):
                # No base64 content available; skip but keep a placeholder mapping
                id_to_filename[img.id] = filename
                continue

            base64_str = img.image_base64
            if "," in base64_str:
                # Strip data URL prefix if any
                base64_str = base64_str.split(",", 1)[1]

            try:
                img_bytes = base64.b64decode(base64_str)
                pil = Image.open(BytesIO(img_bytes)).convert("RGBA")
                pil.save(out_path, format="PNG")
                id_to_filename[img.id] = filename
            except Exception as e:
                _error(f"Failed to decode/save image {img.id}: {e}")
                id_to_filename[img.id] = filename  # still map so markdown link resolves to a path

    return id_to_filename


def _rewrite_markdown(markdown_text: str, id_to_filename: Dict[str, str]) -> str:
    # Replace occurrences of ![<id>](<id>) with ![<id>](<filename>)
    rewritten = markdown_text
    for img_id, fname in id_to_filename.items():
        rewritten = rewritten.replace(f"![{img_id}]({img_id})", f"![{fname}]({fname})")
    return rewritten


class _Spinner:
    """A simple spinner that writes transient progress to stderr.

    - Uses a background thread to animate while long operations run.
    - update(msg) changes the message shown next to the spinner.
    - stop(clear=True) stops and clears the line to avoid clutter.
    """

    _glyphs = "|/-\\"

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and sys.stderr.isatty()
        self._msg = ""
        self._running = False
        self._t: Optional[threading.Thread] = None

    def start(self, msg: str) -> None:
        if not self.enabled:
            return
        self._msg = msg
        self._running = True
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def update(self, msg: str) -> None:
        if not self.enabled:
            return
        self._msg = msg

    def stop(self, clear: bool = True) -> None:
        if not self.enabled:
            return
        self._running = False
        if self._t:
            self._t.join(timeout=1.0)
        if clear:
            self._clear_line()

    def _run(self) -> None:
        i = 0
        while self._running:
            ch = self._glyphs[i % len(self._glyphs)]
            line = f"\r{self._msg} {ch}"
            try:
                sys.stderr.write(line)
                sys.stderr.flush()
            except Exception:
                # If writing fails, disable spinner to avoid crash
                self.enabled = False
                return
            time.sleep(0.1)
            i += 1

    def _clear_line(self) -> None:
        try:
            # Overwrite the current line with spaces and return carriage
            cols = 120
            sys.stderr.write("\r" + (" " * cols) + "\r")
            sys.stderr.flush()
        except Exception:
            pass


def convert_pdf_to_markdown(
    pdf_file: Path,
    outdir: Path,
    api_key: str,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[Path, Dict[str, str]]:
    if pdf_file.suffix.lower() not in VALID_DOCUMENT_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {pdf_file.suffix}. Only PDFs are supported.")

    _ensure_outdir(outdir)

    # 1) Upload and get signed URL
    if progress:
        progress("Uploading PDF...")
    doc_url = _upload_pdf(pdf_file, api_key)

    # 2) OCR
    try:
        if progress:
            progress("Running OCR (this can take a while)...")
        ocr_response = _process_ocr(doc_url, api_key)
    except Exception as e:
        raise RuntimeError(f"OCR processing failed: {e}")

    # 3) Join page markdown
    markdown_text = "\n\n".join(page.markdown for page in ocr_response.pages).strip()

    # 4) Save images and rewrite links
    if progress:
        progress("Saving images and markdown...")
    stem = pdf_file.stem
    id_to_filename = _decode_and_save_images(ocr_response, outdir, stem)
    rewritten_markdown = _rewrite_markdown(markdown_text, id_to_filename)

    # 5) Write markdown file
    md_path = outdir / f"{stem}.md"
    md_path.write_text(rewritten_markdown, encoding="utf-8")

    return md_path, id_to_filename


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one or more PDFs to Markdown using Mistral OCR. Images are saved alongside "
            "the produced markdown files. Batch runs are processed concurrently."
        )
    )
    parser.add_argument(
        "pdf_files",
        nargs="+",
        help="One or more PDF files to process (supports glob patterns, e.g. docs/*.pdf)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=None,
        help=(
            "Output directory. For a single file, this is the exact output folder. "
            "For multiple files, this folder is used as a base and each file writes to "
            "<outdir>/<pdf_stem>_ocr. Default: <PDF_DIR>/<PDF_STEM>_ocr"
        ),
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="Mistral API key (or set MISTRAL_API_KEY env var)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of concurrent workers (default: min(4, number of files))",
    )

    args = parser.parse_args()

    def _expand_inputs(inputs: list[str]) -> list[Path]:
        files: list[Path] = []
        errors: list[str] = []

        for raw in inputs:
            pattern = os.path.expanduser(raw)
            if any(ch in pattern for ch in "*?["):
                matches = [Path(m) for m in glob.glob(pattern, recursive=True)]
                if not matches:
                    errors.append(f"No files match pattern: {raw}")
                else:
                    files.extend(matches)
            else:
                files.append(Path(pattern))

        if errors:
            for msg in errors:
                _error(msg)
            sys.exit(2)

        return files

    pdf_files = _expand_inputs(args.pdf_files)
    missing = [p for p in pdf_files if not p.exists()]
    if missing:
        for p in missing:
            _error(f"Input file not found: {p}")
        sys.exit(2)

    api_key = _load_api_key(args.api_key)

    # Single-file path: keep spinner UX
    if len(pdf_files) == 1:
        pdf_path = pdf_files[0]
        outdir: Path = args.outdir or (pdf_path.parent / f"{pdf_path.stem}_ocr")

        spinner = _Spinner(enabled=True)
        try:
            spinner.start("Starting...")
            md_path, _ = convert_pdf_to_markdown(pdf_path, outdir, api_key, progress=spinner.update)
            spinner.stop(clear=True)
        except Exception as e:
            spinner.stop(clear=True)
            _error(str(e))
            sys.exit(1)

        print(str(md_path))
        return

    # Batch path: concurrent processing
    base_outdir = args.outdir
    max_workers = args.workers or min(4, len(pdf_files))

    def _outdir_for(pdf: Path) -> Path:
        if base_outdir:
            return base_outdir / f"{pdf.stem}_ocr"
        return pdf.parent / f"{pdf.stem}_ocr"

    def _task(pdf: Path) -> Tuple[Path, Optional[str]]:
        try:
            outdir = _outdir_for(pdf)
            md_path, _ = convert_pdf_to_markdown(pdf, outdir, api_key)
            return md_path, None
        except Exception as e:  # noqa: BLE001
            return pdf, str(e)

    results: list[Tuple[Path, Optional[str]]] = [None] * len(pdf_files)  # type: ignore[assignment]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_task, pdf): idx for idx, pdf in enumerate(pdf_files)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:  # pragma: no cover - defensive
                results[idx] = (pdf_files[idx], str(e))

    failed = False
    for (pdf, maybe_err), original_pdf in zip(results, pdf_files):
        if maybe_err is None:
            print(str(pdf))
        else:
            failed = True
            _error(f"Failed to process {original_pdf}: {maybe_err}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
