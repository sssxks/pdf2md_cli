"""
Purge (delete) files from your Mistral account using the Files API.

This is intentionally a standalone utility script (not part of the pdf2md_cli module).

Usage:
  # Dry-run (lists what would be deleted)
  python scripts/purge_mistral_files.py --dry-run

  # Actually delete (prompts for confirmation)
  python scripts/purge_mistral_files.py

  # Non-interactive delete (use with care)
  python scripts/purge_mistral_files.py --yes

Env:
  - MISTRAL_API_KEY (required unless --api-key is provided)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from mistralai import Mistral


def _error(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)


def _load_api_key(cli_key: str | None) -> str:
    key = (cli_key or "").strip() or os.environ.get("MISTRAL_API_KEY", "").strip()
    if not key:
        _error("Mistral API key not provided. Use --api-key or set MISTRAL_API_KEY env var.")
        sys.exit(2)
    return key


def _to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        # pydantic v2
        return obj.model_dump()
    if hasattr(obj, "dict"):
        # pydantic v1
        return obj.dict()
    return dict(obj)  # last resort; may raise


def _get_items(list_response: Any) -> list[dict[str, Any]]:
    data = getattr(list_response, "data", None)
    if data is None:
        data = _to_dict(list_response).get("data", [])
    items: list[dict[str, Any]] = []
    for item in data or []:
        items.append(_to_dict(item))
    return items


def _fmt_ts(ts: Any) -> str:
    try:
        ts_int = int(ts)
    except Exception:
        return "unknown"
    dt = datetime.fromtimestamp(ts_int, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _iter_all_files(
    client: Mistral, page_size: int, max_pages: int | None
) -> Iterable[dict[str, Any]]:
    page = 0
    seen_ids: set[str] = set()

    while True:
        if max_pages is not None and page >= max_pages:
            return

        res = client.files.list(page=page, page_size=page_size)
        items = _get_items(res)
        if not items:
            return

        for item in items:
            file_id = str(item.get("id", "")).strip()
            if not file_id or file_id in seen_ids:
                continue
            seen_ids.add(file_id)
            yield item

        page += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete files from Mistral Files API (paginated).")
    parser.add_argument("--api-key", default=None, help="Mistral API key (or set MISTRAL_API_KEY env var)")
    parser.add_argument("--page-size", type=int, default=100, help="Page size for listing (default: 100)")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Safety cap on number of pages to scan (default: no limit)",
    )
    parser.add_argument(
        "--purpose",
        default=None,
        help="Only delete files matching this purpose (e.g. 'ocr', 'batch')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not delete; only list what would be deleted",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not prompt for confirmation before deleting",
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=0,
        help="Optional delay between deletes (helps avoid rate limits). Default: 0",
    )

    args = parser.parse_args()

    if args.page_size <= 0 or args.page_size > 1000:
        _error("--page-size must be between 1 and 1000")
        sys.exit(2)

    api_key = _load_api_key(args.api_key)
    client = Mistral(api_key=api_key)

    candidates: list[dict[str, Any]] = []
    for item in _iter_all_files(client, page_size=args.page_size, max_pages=args.max_pages):
        if args.purpose and str(item.get("purpose", "")).strip() != args.purpose:
            continue
        candidates.append(item)

    if not candidates:
        print("No files found (or none match the provided filters).")
        return

    print(f"Found {len(candidates)} file(s) to delete:")
    for f in candidates:
        file_id = f.get("id", "unknown")
        filename = f.get("filename", "unknown")
        purpose = f.get("purpose", "unknown")
        created_at = _fmt_ts(f.get("created_at"))
        print(f"- {file_id} | {filename} | purpose={purpose} | created_at={created_at}")

    if args.dry_run:
        print("\nDry-run enabled; no deletions performed.")
        return

    if not args.yes:
        answer = input("\nType 'DELETE' to confirm deletion of ALL listed files: ").strip()
        if answer != "DELETE":
            print("Aborted.")
            sys.exit(1)

    deleted = 0
    failed = 0
    for f in candidates:
        file_id = str(f.get("id", "")).strip()
        if not file_id:
            failed += 1
            continue
        try:
            res = client.files.delete(file_id=file_id)
            deleted_flag = getattr(res, "deleted", None)
            if deleted_flag is None:
                deleted_flag = _to_dict(res).get("deleted")
            if deleted_flag is True:
                deleted += 1
            else:
                failed += 1
                _error(f"Delete returned unexpected response for {file_id}: {res!r}")
        except Exception as e:
            failed += 1
            _error(f"Failed to delete {file_id}: {e}")

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    print(f"\nDone. Deleted={deleted}, Failed={failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
