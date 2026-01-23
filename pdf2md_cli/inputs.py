from __future__ import annotations

import glob
import os
from pathlib import Path

from pdf2md_cli.pipeline import VALID_DOCUMENT_EXTENSIONS, VALID_IMAGE_EXTENSIONS


def expand_inputs(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    errors: list[str] = []

    for raw in inputs:
        pattern = os.path.expanduser(raw)
        if any(ch in pattern for ch in "*?["):
            matches = sorted(Path(m) for m in glob.glob(pattern, recursive=True))
            if not matches:
                errors.append(f"No files match pattern: {raw}")
            else:
                files.extend(matches)
        else:
            files.append(Path(pattern))

    if errors:
        raise ValueError("\n".join(errors))

    normalized: list[Path] = []
    seen: set[Path] = set()
    for p in files:
        rp = p.expanduser().resolve(strict=False)
        if rp in seen:
            continue
        seen.add(rp)
        normalized.append(rp)

    return normalized


def validate_input_paths(input_files: list[Path]) -> None:
    missing = [p for p in input_files if not p.exists()]
    if missing:
        raise ValueError("\n".join(f"Input file not found: {p}" for p in missing))

    not_files = [p for p in input_files if p.exists() and not p.is_file()]
    if not_files:
        raise ValueError("\n".join(f"Input path is not a file: {p}" for p in not_files))

    supported = VALID_DOCUMENT_EXTENSIONS | VALID_IMAGE_EXTENSIONS
    unsupported = [p for p in input_files if p.suffix.lower() not in supported]
    if unsupported:
        raise ValueError(
            "\n".join(
                f"Unsupported file type (supported: {', '.join(sorted(supported))}): {p}" for p in unsupported
            )
        )


def validate_pdf_paths(pdf_files: list[Path]) -> None:
    supported = VALID_DOCUMENT_EXTENSIONS
    unsupported = [p for p in pdf_files if p.suffix.lower() not in supported]
    if unsupported:
        raise ValueError("\n".join(f"Unsupported file type (expected .pdf): {p}" for p in unsupported))
    validate_input_paths(pdf_files)
