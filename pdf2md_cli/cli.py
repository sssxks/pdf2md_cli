from __future__ import annotations

import argparse
import concurrent.futures
import sys
from pathlib import Path

from pdf2md_cli.auth import error, load_api_key
from pdf2md_cli.backends.mistral import make_mistral_runner
from pdf2md_cli.backends.mock import MockConfig, make_mock_runner
from pdf2md_cli.inputs import expand_inputs, validate_input_paths
from pdf2md_cli.pipeline import convert_file_to_markdown
from pdf2md_cli.retry import BackoffConfig
from pdf2md_cli.types import Progress
from pdf2md_cli.ui import Spinner

DEFAULT_OCR_MODEL = "mistral-ocr-2505"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one or more PDFs/images to Markdown using Mistral OCR. Images are saved alongside "
            "the produced markdown files. Batch runs are processed concurrently."
        )
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more files to process (PDF or supported image; supports glob patterns, e.g. docs/*.*)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=None,
        help=(
            "Output directory. For a single file, this is the exact output folder. "
            "For multiple files, this folder is used as a base and each file writes to "
            "<outdir>/<input_stem>_ocr. Default: <INPUT_DIR>/<INPUT_STEM>_ocr"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["mistral", "mock"],
        default="mistral",
        help="OCR backend to use (default: mistral)",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="Mistral API key (or set MISTRAL_API_KEY env var); required for --backend mistral",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of concurrent workers (default: min(16, number of files))",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OCR_MODEL,
        help=f"OCR model to use (default: {DEFAULT_OCR_MODEL})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Number of retries for transient failures (default: 5; 0 disables retries)",
    )
    parser.add_argument(
        "--backoff-initial-ms",
        type=int,
        default=500,
        help="Initial backoff delay in milliseconds (default: 500)",
    )
    parser.add_argument(
        "--backoff-max-ms",
        type=int,
        default=20000,
        help="Maximum backoff delay in milliseconds (default: 20000)",
    )
    parser.add_argument(
        "--backoff-multiplier",
        type=float,
        default=2.0,
        help="Backoff multiplier per retry (default: 2.0)",
    )
    parser.add_argument(
        "--backoff-jitter",
        type=float,
        default=0.2,
        help="Jitter fraction added to delays (0..1, default: 0.2)",
    )
    parser.add_argument(
        "--keep-remote-file",
        action="store_true",
        help="(mistral backend) Do not delete the uploaded file from Mistral after OCR completes",
    )

    # Mock-only knobs for UX testing.
    parser.add_argument(
        "--mock-pages",
        type=int,
        default=1,
        help="(mock backend) Number of pages to generate (default: 1)",
    )
    parser.add_argument(
        "--mock-images",
        type=int,
        default=1,
        help="(mock backend) Images per page to generate (default: 1)",
    )
    parser.add_argument(
        "--mock-delay-ms",
        type=int,
        default=0,
        help="(mock backend) Artificial delay per file in milliseconds (default: 0)",
    )
    parser.add_argument(
        "--mock-fail-first",
        type=int,
        default=0,
        help="(mock backend) Fail N times before succeeding to exercise retries (default: 0)",
    )

    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be a positive integer")
    if args.retries < 0:
        raise ValueError("--retries must be >= 0")
    if args.backoff_initial_ms < 0:
        raise ValueError("--backoff-initial-ms must be >= 0")
    if args.backoff_max_ms <= 0:
        raise ValueError("--backoff-max-ms must be > 0")
    if args.backoff_multiplier <= 0:
        raise ValueError("--backoff-multiplier must be > 0")
    if not (0.0 <= args.backoff_jitter <= 1.0):
        raise ValueError("--backoff-jitter must be between 0 and 1")
    if args.backend == "mock":
        if args.mock_pages <= 0:
            raise ValueError("--mock-pages must be > 0")
        if args.mock_images < 0:
            raise ValueError("--mock-images must be >= 0")
        if args.mock_delay_ms < 0:
            raise ValueError("--mock-delay-ms must be >= 0")
        if args.mock_fail_first < 0:
            raise ValueError("--mock-fail-first must be >= 0")


def main(argv: list[str] | None = None) -> None:
    try:
        args = _parse_args(argv)
        _validate_args(args)
    except ValueError as e:
        error(str(e))
        raise SystemExit(2) from e

    try:
        input_files = expand_inputs(args.files)
        validate_input_paths(input_files)
    except ValueError as e:
        error(str(e))
        raise SystemExit(2) from e

    backoff = BackoffConfig(
        max_retries=args.retries,
        initial_delay_s=args.backoff_initial_ms / 1000.0,
        max_delay_s=args.backoff_max_ms / 1000.0,
        multiplier=args.backoff_multiplier,
        jitter=args.backoff_jitter,
    )

    runner = None
    if args.backend == "mistral":
        api_key = load_api_key(args.api_key)
        runner = make_mistral_runner(api_key=api_key, backoff=backoff)
    else:
        mock_cfg = MockConfig(
            pages=args.mock_pages,
            images_per_page=args.mock_images,
            delay_ms=args.mock_delay_ms,
            fail_first=args.mock_fail_first,
        )
        runner = make_mock_runner(mock=mock_cfg, backoff=backoff)

    # Single-file path: keep spinner UX.
    if len(input_files) == 1:
        input_path = input_files[0]
        outdir: Path = args.outdir or (input_path.parent / f"{input_path.stem}_ocr")

        spinner = Spinner(enabled=True)
        try:
            spinner.start("Starting...")
            result = convert_file_to_markdown(
                input_file=input_path,
                outdir=outdir,
                runner=runner,
                model=args.model,
                delete_remote_file=not args.keep_remote_file,
                progress=Progress(spinner.update),
            )
            spinner.stop(clear=True)
        except Exception as e:  # noqa: BLE001
            spinner.stop(clear=True)
            error(str(e))
            raise SystemExit(1) from e

        print(str(result.markdown_path))
        return

    # Batch path: concurrent processing.
    base_outdir = args.outdir
    max_workers = args.workers or min(16, len(input_files))
    total_files = len(input_files)

    def _outdir_for(p: Path) -> Path:
        if base_outdir:
            return base_outdir / f"{p.stem}_ocr"
        return p.parent / f"{p.stem}_ocr"

    spinner = Spinner(enabled=True)

    def _task(p: Path, idx_1based: int) -> tuple[Path, str | None]:
        try:
            outdir = _outdir_for(p)

            def _progress(msg: str) -> None:
                spinner.update(f"[{idx_1based}/{total_files}] {p.name}: {msg}")

            res = convert_file_to_markdown(
                input_file=p,
                outdir=outdir,
                runner=runner,
                model=args.model,
                delete_remote_file=not args.keep_remote_file,
                progress=Progress(_progress),
            )
            return res.markdown_path, None
        except Exception as e:  # noqa: BLE001
            return p, str(e)

    results: list[tuple[Path, str | None]] = [(p, "not started") for p in input_files]

    spinner.start(f"Starting batch ({total_files} files, {max_workers} workers)...")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {pool.submit(_task, p, idx + 1): idx for idx, p in enumerate(input_files)}
            completed = 0
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:  # pragma: no cover
                    results[idx] = (input_files[idx], str(e))
                completed += 1
                spinner.update(f"Completed {completed}/{total_files}: {input_files[idx].name}")
    finally:
        spinner.stop(clear=True)

    failed = False
    for (md_path, maybe_err), original_input in zip(results, input_files):
        if maybe_err is None:
            print(str(md_path))
        else:
            failed = True
            error(f"Failed to process {original_input}: {maybe_err}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
