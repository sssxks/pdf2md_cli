from __future__ import annotations

import concurrent.futures
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from pdf2md_cli.backends.mistral import make_mistral_runner
from pdf2md_cli.feature_flags import mock_backend_enabled
from pdf2md_cli.inputs import expand_inputs, validate_input_paths
from pdf2md_cli.pipeline import ConvertResult, OcrRunner, convert_file_to_markdown
from pdf2md_cli.retry import BackoffConfig
from pdf2md_cli.types import HeaderFooterMode, NO_PROGRESS, Progress, ProgressFn, TableFormat

DEFAULT_OCR_MODEL = "mistral-ocr-latest"


@dataclass(frozen=True, slots=True)
class ConvertOptions:
    model: str = DEFAULT_OCR_MODEL
    table_format: TableFormat = TableFormat.HTML
    extract_table: bool = False
    header: HeaderFooterMode = HeaderFooterMode.COMMENT
    footer: HeaderFooterMode = HeaderFooterMode.COMMENT
    add_front_matter: bool = True
    add_page_markers: bool = True


@dataclass(frozen=True, slots=True)
class FailedConversion:
    input_file: Path
    error: str


@dataclass(frozen=True, slots=True)
class BatchResult:
    succeeded: list[ConvertResult]
    failed: list[FailedConversion]

    @property
    def ok(self) -> bool:
        return not self.failed


def resolve_api_key(api_key: str | None) -> str:
    key = (api_key or "").strip() or os.environ.get("MISTRAL_API_KEY", "").strip()
    if not key:
        raise ValueError("Mistral API key not provided (pass api_key=... or set env var MISTRAL_API_KEY).")
    return key


def make_runner(
    *,
    backend: Literal["mistral", "mock"] = "mistral",
    api_key: str | None = None,
    backoff: BackoffConfig = BackoffConfig(),
    mock: object | None = None,
) -> OcrRunner:
    """
    Build an OCR runner compatible with `convert_file_to_markdown`.

    Notes:
      - For `backend="mistral"`, `api_key` is required (or MISTRAL_API_KEY env var).
      - For `backend="mock"`, the feature flag must be enabled with PDF2MD_ENABLE_MOCK=1.
    """

    if backend == "mistral":
        return make_mistral_runner(api_key=resolve_api_key(api_key), backoff=backoff)

    if backend == "mock":
        if not mock_backend_enabled():
            raise ValueError('mock backend is disabled (set env var PDF2MD_ENABLE_MOCK=1 to enable).')
        from pdf2md_cli.backends.mock import MockConfig, make_mock_runner

        mock_cfg = mock if isinstance(mock, MockConfig) else MockConfig()
        return make_mock_runner(mock=mock_cfg, backoff=backoff)

    raise ValueError(f"Unsupported backend: {backend!r}")


def _progress_sink(progress: Progress | ProgressFn | None) -> Progress:
    if progress is None:
        return NO_PROGRESS
    if isinstance(progress, Progress):
        return progress
    return Progress(progress)


def _outdir_for_single(input_file: Path, outdir: Path | None) -> Path:
    return outdir if outdir is not None else (input_file.parent / f"{input_file.stem}_ocr")


def _outdir_for_batch(input_file: Path, base_outdir: Path | None) -> Path:
    if base_outdir is not None:
        return base_outdir / f"{input_file.stem}_ocr"
    return input_file.parent / f"{input_file.stem}_ocr"


def convert_file(
    input_file: str | Path,
    *,
    outdir: str | Path | None = None,
    backend: Literal["mistral", "mock"] = "mistral",
    api_key: str | None = None,
    keep_remote_file: bool = False,
    backoff: BackoffConfig = BackoffConfig(),
    options: ConvertOptions = ConvertOptions(),
    progress: Progress | ProgressFn | None = None,
    runner: OcrRunner | None = None,
) -> ConvertResult:
    """
    Convert a single input file to Markdown and write outputs to disk.

    If `runner` is provided, `backend/api_key/backoff` are ignored.
    """

    p = Path(input_file).expanduser().resolve(strict=False)
    validate_input_paths([p])

    resolved_outdir = _outdir_for_single(p, Path(outdir) if outdir is not None else None)
    resolved_runner = runner or make_runner(backend=backend, api_key=api_key, backoff=backoff)
    return convert_file_to_markdown(
        input_file=p,
        outdir=resolved_outdir,
        runner=resolved_runner,
        model=options.model,
        delete_remote_file=not keep_remote_file,
        table_format=options.table_format,
        extract_table=options.extract_table,
        header_mode=options.header,
        footer_mode=options.footer,
        add_front_matter=options.add_front_matter,
        add_page_markers=options.add_page_markers,
        progress=_progress_sink(progress),
    )


def convert_files(
    inputs: Sequence[str | Path],
    *,
    outdir: str | Path | None = None,
    backend: Literal["mistral", "mock"] = "mistral",
    api_key: str | None = None,
    keep_remote_file: bool = False,
    backoff: BackoffConfig = BackoffConfig(),
    workers: int | None = None,
    options: ConvertOptions = ConvertOptions(),
    progress: Progress | ProgressFn | None = None,
    runner: OcrRunner | None = None,
) -> BatchResult:
    """
    Convert multiple inputs (supports glob patterns) concurrently.

    Output directory behavior matches the CLI:
      - if `outdir` is set: writes to `<outdir>/<input_stem>_ocr/`
      - otherwise: writes to `<input_dir>/<input_stem>_ocr/`

    If `runner` is provided, `backend/api_key/backoff` are ignored.
    """

    expanded = expand_inputs([str(x) for x in inputs])
    validate_input_paths(expanded)
    if not expanded:
        return BatchResult(succeeded=[], failed=[])

    base_outdir = Path(outdir) if outdir is not None else None
    max_workers = workers or min(16, len(expanded))
    progress_sink = _progress_sink(progress)
    resolved_runner = runner or make_runner(backend=backend, api_key=api_key, backoff=backoff)
    def _task(p: Path, idx_1based: int) -> tuple[int, ConvertResult | None, str | None]:
        try:
            outdir_for_p = _outdir_for_batch(p, base_outdir)

            def _emit(msg: str) -> None:
                progress_sink.emit(f"[{idx_1based}/{len(expanded)}] {p.name}: {msg}")

            res = convert_file_to_markdown(
                input_file=p,
                outdir=outdir_for_p,
                runner=resolved_runner,
                model=options.model,
                delete_remote_file=not keep_remote_file,
                table_format=options.table_format,
                extract_table=options.extract_table,
                header_mode=options.header,
                footer_mode=options.footer,
                add_front_matter=options.add_front_matter,
                add_page_markers=options.add_page_markers,
                progress=Progress(_emit) if progress_sink.enabled else NO_PROGRESS,
            )
            return idx_1based - 1, res, None
        except Exception as e:  # noqa: BLE001
            return idx_1based - 1, None, str(e)

    succeeded: list[ConvertResult] = []
    failed: list[FailedConversion] = []

    if max_workers <= 1:
        for idx, p in enumerate(expanded, start=1):
            _, res, err = _task(p, idx)
            if err is None and res is not None:
                succeeded.append(res)
            else:
                failed.append(FailedConversion(input_file=p, error=err or "unknown error"))
        return BatchResult(succeeded=succeeded, failed=failed)

    results: list[tuple[ConvertResult | None, str | None]] = [(None, "not started") for _ in expanded]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {pool.submit(_task, p, i + 1): i for i, p in enumerate(expanded)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                _idx0, res, err = future.result()
                results[idx] = (res, err)
            except Exception as e:  # pragma: no cover
                results[idx] = (None, str(e))

    for p, (res, err) in zip(expanded, results):
        if err is None and res is not None:
            succeeded.append(res)
        else:
            failed.append(FailedConversion(input_file=p, error=err or "unknown error"))

    return BatchResult(succeeded=succeeded, failed=failed)
