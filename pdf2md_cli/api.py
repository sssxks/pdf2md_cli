from __future__ import annotations

from pathlib import Path

from pdf2md_cli.backends.mistral import make_mistral_runner
from pdf2md_cli.pipeline import convert_pdf_to_markdown
from pdf2md_cli.retry import BackoffConfig
from pdf2md_cli.types import NO_PROGRESS, Progress

DEFAULT_OCR_MODEL = "mistral-ocr-2505"


def convert_pdf_to_markdown_mistral(
    pdf_file: Path,
    outdir: Path,
    api_key: str,
    *,
    model: str = DEFAULT_OCR_MODEL,
    delete_remote_file: bool = True,
    max_retries: int = 5,
    backoff_initial_s: float = 0.5,
    backoff_max_s: float = 20.0,
    backoff_multiplier: float = 2.0,
    backoff_jitter: float = 0.2,
    progress: Progress = NO_PROGRESS,
) -> tuple[Path, dict[str, str]]:
    backoff = BackoffConfig(
        max_retries=max_retries,
        initial_delay_s=backoff_initial_s,
        max_delay_s=backoff_max_s,
        multiplier=backoff_multiplier,
        jitter=backoff_jitter,
    )
    runner = make_mistral_runner(api_key=api_key, backoff=backoff)
    res = convert_pdf_to_markdown(
        pdf_file=pdf_file,
        outdir=outdir,
        runner=runner,
        model=model,
        delete_remote_file=delete_remote_file,
        progress=progress,
    )
    return res.markdown_path, res.image_id_to_filename


# Backwards-compatible alias for older code importing pdf2md_cli.cli.convert_pdf_to_markdown.
convert_pdf_to_markdown = convert_pdf_to_markdown_mistral
