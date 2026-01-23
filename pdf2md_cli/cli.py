from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import sys
from difflib import get_close_matches
from pathlib import Path

from pdf2md_cli.auth import error, load_api_key
from pdf2md_cli.backends.mistral import make_mistral_runner
from pdf2md_cli.feature_flags import mock_backend_enabled
from pdf2md_cli.inputs import expand_inputs, validate_input_paths
from pdf2md_cli.pipeline import convert_file_to_markdown
from pdf2md_cli.retry import BackoffConfig
from pdf2md_cli.types import Progress, TableFormat
from pdf2md_cli.ui import Spinner

DEFAULT_OCR_MODEL = "mistral-ocr-latest"

_TABLE_HELP = """\
Table behavior (Mistral OCR):

| Flags | Mistral `table_format` | Markdown output | Sidecar files |
| --- | --- | --- | --- |
| (default) `--table-format html` | `html` | HTML tables are inlined | none |
| `--table-format markdown` | (not sent) | Tables stay inline as markdown | none |
| `--extract-table --table-format html` | `html` | Links to `tbl-*.html` | writes `tbl-*.html` |
| `--extract-table --table-format markdown` | `markdown` | Links to `tbl-*.md` | writes `tbl-*.md` |
"""

_ANSI_RESET = "\x1b[0m"


def _ansi(text: str, code: str) -> str:
    return f"\x1b[{code}m{text}{_ANSI_RESET}"


def _color_enabled(file: object) -> bool:
    # Respect https://no-color.org/ and common CLICOLOR conventions.
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("CLICOLOR") == "0":
        return False
    if os.environ.get("CLICOLOR_FORCE") not in (None, "", "0"):
        return True
    term = os.environ.get("TERM", "")
    if term.lower() == "dumb":
        return False
    isatty = getattr(file, "isatty", None)
    return bool(isatty and isatty())


_RE_FLAG = re.compile(r"(^|[\s\[,(/])(--?[A-Za-z0-9][A-Za-z0-9-]*)(?![\w-])")
_RE_METAVAR = re.compile(r"\b([A-Z][A-Z0-9_-]{1,})\b")


def _reorder_help_sections(text: str) -> str:
    """Move 'Examples:' above 'options:' for easier scanning."""
    lines = text.splitlines(keepends=True)

    def _is_heading(line: str) -> bool:
        stripped = line.strip()
        return bool(stripped) and (not line.startswith((" ", "\t"))) and stripped.endswith(":")

    def _find_line(pred) -> int | None:
        for i, ln in enumerate(lines):
            if pred(ln):
                return i
        return None

    examples_i = _find_line(lambda ln: ln.strip() == "Examples:")
    options_i = _find_line(lambda ln: ln.strip() == "options:")
    if examples_i is None or options_i is None or examples_i < options_i:
        return text

    # Slice the Examples block until the next heading (e.g. "Help topics:") or EOF.
    end = examples_i + 1
    while end < len(lines) and not _is_heading(lines[end]):
        end += 1
    examples_block = lines[examples_i:end]
    del lines[examples_i:end]

    # Re-find options after deletion and insert Examples before it.
    options_i = _find_line(lambda ln: ln.strip() == "options:")
    if options_i is None:
        return "".join(lines) + "".join(examples_block)
    lines[options_i:options_i] = examples_block + (["\n"] if examples_block and not examples_block[-1].endswith("\n") else [])
    return "".join(lines)


def _colorize_help(text: str) -> str:
    in_examples = False

    def _color_line(line: str) -> str:
        nonlocal in_examples
        stripped = line.lstrip()

        # Headings like "usage:", "options:", "Input/Output:".
        if line and not line.startswith((" ", "\t")) and stripped.rstrip().endswith(":"):
            label = stripped.rstrip()
            if label == "Examples:":
                in_examples = True
                return line[: len(line) - len(stripped)] + _ansi(label, "1;35") + line[len(line.rstrip()) :]
            in_examples = False
            return line[: len(line) - len(stripped)] + _ansi(label, "1;36") + line[len(line.rstrip()) :]

        # "usage: ..." line
        if stripped.startswith("usage:"):
            prefix, rest = stripped.split(":", 1)
            colored = _ansi(f"{prefix}:", "1;36") + rest
            return line[: len(line) - len(stripped)] + colored

        # Option lines usually start with indentation then "-" or "--".
        if stripped.startswith("-"):
            line = _RE_FLAG.sub(lambda m: f"{m.group(1)}{_ansi(m.group(2), '32')}", line)  # green
            line = _RE_METAVAR.sub(lambda m: _ansi(m.group(1), "33"), line)  # yellow
            return line

        # Example command lines.
        if in_examples and (stripped.startswith("2md ") or stripped.startswith("$ 2md ")):
            return _ansi(line.rstrip("\n"), "35") + ("\n" if line.endswith("\n") else "")

        return line

    return "".join(_color_line(ln) for ln in text.splitlines(keepends=True))


class _FriendlyArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # noqa: A003
        error(message)

        unknown: list[str] = []
        prefix = "unrecognized arguments:"
        if prefix in message:
            tail = message.split(prefix, 1)[1]
            unknown = [tok.strip() for tok in tail.split() if tok.strip().startswith("-")]

        if unknown:
            known = sorted(self._option_string_actions.keys())
            for tok in unknown:
                suggestions = get_close_matches(tok, known, n=3, cutoff=0.6)
                if suggestions:
                    error(f"Did you mean: {', '.join(suggestions)} ?")

        error("Run `2md -h` for help, or `2md help` for topics.")
        raise SystemExit(2)

    def print_help(self, file=None) -> None:  # type: ignore[override]
        if file is None:
            file = sys.stdout
        msg = _reorder_help_sections(self.format_help())
        if _color_enabled(file):
            msg = _colorize_help(msg)
        self._print_message(msg, file)

    def print_usage(self, file=None) -> None:  # type: ignore[override]
        if file is None:
            file = sys.stdout
        msg = self.format_usage()
        if _color_enabled(file):
            msg = _colorize_help(msg)
        self._print_message(msg, file)


def _build_parser(*, advanced: bool) -> argparse.ArgumentParser:
    enable_mock = mock_backend_enabled()
    parser = _FriendlyArgumentParser(
        prog="2md",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Convert documents/images to Markdown using Mistral OCR.",
        epilog=(
            "Examples:\n"
            "  2md file.pdf\n"
            "  2md docs/*.pdf --workers 4\n"
            "  2md file.docx\n"
            "  2md slides.pptx\n"
            "  2md file.pdf -o out\n"
            "  2md file.pdf --table-format markdown\n"
            "  2md file.pdf --extract-table --table-format html\n"
            "  2md file.pdf --extract-table --table-format markdown\n"
            "\n"
            "Help topics:\n"
            "  2md help tables\n"
            "  2md help advanced\n"
        ),
    )

    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "files",
        nargs="+",
        help=(
            "One or more files (supported document or image); supports glob patterns (e.g. docs/*.*).\n"
            "Documents: .pdf .docx .pptx .txt .epub .xml .rtf .odt .bib .fb2 .ipynb .tex .opml .1 .man"
        ),
    )
    io_group.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=None,
        help=(
            "Output directory.\n"
            "  - single file: exact output folder\n"
            "  - multiple files: base folder; writes <outdir>/<input_stem>_ocr\n"
            "Default: <INPUT_DIR>/<INPUT_STEM>_ocr"
        ),
    )

    backend_group = parser.add_argument_group("Backend")
    backend_group.add_argument(
        "--backend",
        choices=(["mistral", "mock"] if enable_mock else ["mistral"]),
        default="mistral",
        help="OCR backend to use (default: mistral)",
    )
    backend_group.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="Mistral API key (or set MISTRAL_API_KEY env var); required for --backend mistral",
    )
    backend_group.add_argument(
        "--model",
        default=DEFAULT_OCR_MODEL,
        help=f"OCR model to use (default: {DEFAULT_OCR_MODEL})",
    )
    backend_group.add_argument(
        "--keep-remote-file",
        action="store_true",
        help="(mistral) Do not delete uploaded documents from Mistral after OCR completes",
    )

    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of concurrent workers (default: min(16, number of files))",
    )

    tables_group = parser.add_argument_group("Tables")
    tables_group.add_argument(
        "--extract-table",
        action="store_true",
        help="Extract tables into separate files and keep links like [tbl-3.html](tbl-3.html)",
    )
    tables_group.add_argument(
        "--table-format",
        choices=["html", "markdown"],
        default="html",
        help=(
            "(mistral) Table formatting mode (default: html).  \n"
            "See: `2md help tables`"
        ),
    )

    reliability_group = parser.add_argument_group("Reliability")
    reliability_group.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Number of retries for transient failures (default: 5; 0 disables retries)",
    )

    if advanced:
        advanced_group = parser.add_argument_group("Advanced")
        advanced_group.add_argument(
            "--backoff-initial-ms",
            type=int,
            default=500,
            help="Initial backoff delay in milliseconds (default: 500)",
        )
        advanced_group.add_argument(
            "--backoff-max-ms",
            type=int,
            default=20000,
            help="Maximum backoff delay in milliseconds (default: 20000)",
        )
        advanced_group.add_argument(
            "--backoff-multiplier",
            type=float,
            default=2.0,
            help="Backoff multiplier per retry (default: 2.0)",
        )
        advanced_group.add_argument(
            "--backoff-jitter",
            type=float,
            default=0.2,
            help="Jitter fraction added to delays (0..1, default: 0.2)",
        )
        advanced_group.add_argument(
            "--no-front-matter",
            action="store_true",
            help="Do not add YAML front matter metadata to the top of the markdown output",
        )
        advanced_group.add_argument(
            "--no-page-markers",
            action="store_true",
            help="Do not insert per-page markers like <!-- page: N --> when concatenating OCR pages",
        )
        advanced_group.add_argument(
            "--extract-header",
            action="store_true",
            help="(mistral) Extract headers separately (writes <stem>_headers_footers.md when present)",
        )
        advanced_group.add_argument(
            "--extract-footer",
            action="store_true",
            help="(mistral) Extract footers separately (writes <stem>_headers_footers.md when present)",
        )
        advanced_group.add_argument(
            "--no-image-base64",
            action="store_true",
            help="(mistral) Do not request image base64 payloads from OCR (image placeholders may be blank)",
        )

        if enable_mock:
            mock_group = parser.add_argument_group("Mock backend (advanced)")
            mock_group.add_argument(
                "--mock-pages",
                type=int,
                default=1,
                help="(mock) Number of pages to generate (default: 1)",
            )
            mock_group.add_argument(
                "--mock-images",
                type=int,
                default=1,
                help="(mock) Images per page to generate (default: 1)",
            )
            mock_group.add_argument(
                "--mock-delay-ms",
                type=int,
                default=0,
                help="(mock) Artificial delay per file in milliseconds (default: 0)",
            )
            mock_group.add_argument(
                "--mock-fail-first",
                type=int,
                default=0,
                help="(mock) Fail N times before succeeding to exercise retries (default: 0)",
            )
    else:
        parser.add_argument("--backoff-initial-ms", type=int, default=500, help=argparse.SUPPRESS)
        parser.add_argument("--backoff-max-ms", type=int, default=20000, help=argparse.SUPPRESS)
        parser.add_argument("--backoff-multiplier", type=float, default=2.0, help=argparse.SUPPRESS)
        parser.add_argument("--backoff-jitter", type=float, default=0.2, help=argparse.SUPPRESS)
        parser.add_argument("--no-front-matter", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument("--no-page-markers", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument("--extract-header", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument("--extract-footer", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument("--no-image-base64", action="store_true", help=argparse.SUPPRESS)

        if enable_mock:
            parser.add_argument("--mock-pages", type=int, default=1, help=argparse.SUPPRESS)
            parser.add_argument("--mock-images", type=int, default=1, help=argparse.SUPPRESS)
            parser.add_argument("--mock-delay-ms", type=int, default=0, help=argparse.SUPPRESS)
            parser.add_argument("--mock-fail-first", type=int, default=0, help=argparse.SUPPRESS)

    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser(advanced=False).parse_args(argv)


def _parse_args_advanced(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser(advanced=True).parse_args(argv)


def _print_tables_help() -> None:
    print(_TABLE_HELP)


def _print_advanced_help() -> None:
    _build_parser(advanced=True).print_help()


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
        if not mock_backend_enabled():
            raise ValueError('mock backend is disabled (set env var PDF2MD_ENABLE_MOCK=1 to enable)')
        if getattr(args, "mock_pages", 1) <= 0:
            raise ValueError("--mock-pages must be > 0")
        if getattr(args, "mock_images", 1) < 0:
            raise ValueError("--mock-images must be >= 0")
        if getattr(args, "mock_delay_ms", 0) < 0:
            raise ValueError("--mock-delay-ms must be >= 0")
        if getattr(args, "mock_fail_first", 0) < 0:
            raise ValueError("--mock-fail-first must be >= 0")


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]

    if raw_argv and raw_argv[0] == "help":
        topic = raw_argv[1] if len(raw_argv) > 1 else ""
        if topic in {"tables", "table"}:
            _print_tables_help()
            return
        if topic in {"advanced", "adv"}:
            _print_advanced_help()
            return
        if topic == "":
            _build_parser(advanced=False).print_help()
            return
        error(f"Unknown help topic: {topic!r} (try: tables, advanced)")
        raise SystemExit(2)

    if "--help-tables" in raw_argv:
        _print_tables_help()
        return
    if "--help-advanced" in raw_argv:
        _print_advanced_help()
        return

    try:
        args = _parse_args(raw_argv)
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
        from pdf2md_cli.backends.mock import MockConfig, make_mock_runner

        mock_cfg = MockConfig(
            pages=getattr(args, "mock_pages", 1),
            images_per_page=getattr(args, "mock_images", 1),
            delay_ms=getattr(args, "mock_delay_ms", 0),
            fail_first=getattr(args, "mock_fail_first", 0),
        )
        runner = make_mock_runner(mock=mock_cfg, backoff=backoff)

    parsed_table_format = TableFormat(args.table_format)

    table_format: TableFormat | None
    inline_tables: bool
    if parsed_table_format == TableFormat.HTML:
        table_format = TableFormat.HTML
        inline_tables = not bool(args.extract_table)
    else:
        # markdown mode:
        # - default (no --extract-table): keep tables inline (API default => do not send table_format)
        # - with --extract-table: request extracted markdown tables separately
        table_format = TableFormat.MARKDOWN if bool(args.extract_table) else None
        inline_tables = False

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
                table_format=table_format,
                extract_header=bool(args.extract_header),
                extract_footer=bool(args.extract_footer),
                include_image_base64=not bool(args.no_image_base64),
                add_front_matter=not bool(args.no_front_matter),
                add_page_markers=not bool(args.no_page_markers),
                inline_tables=inline_tables,
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
                table_format=table_format,
                extract_header=bool(args.extract_header),
                extract_footer=bool(args.extract_footer),
                include_image_base64=not bool(args.no_image_base64),
                add_front_matter=not bool(args.no_front_matter),
                add_page_markers=not bool(args.no_page_markers),
                inline_tables=inline_tables,
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
    ok = 0
    for (md_path, maybe_err), original_input in zip(results, input_files):
        if maybe_err is None:
            print(str(md_path))
            ok += 1
        else:
            failed = True
            error(f"Failed to process {original_input}: {maybe_err}")

    fail = total_files - ok
    print(f"Done: {ok}/{total_files} succeeded, {fail} failed.", file=sys.stderr)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
