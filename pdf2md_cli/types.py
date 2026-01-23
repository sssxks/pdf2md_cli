from __future__ import annotations

from enum import StrEnum
from dataclasses import dataclass
from dataclasses import field
from typing import Callable


ProgressFn = Callable[[str], None]


class Progress:
    """Progress sink used to stream human-readable status messages."""

    __slots__ = ("_fn",)

    def __init__(self, fn: ProgressFn | None = None) -> None:
        self._fn = fn

    @property
    def enabled(self) -> bool:
        return self._fn is not None

    def emit(self, msg: str) -> "Progress":
        if self._fn is not None:
            self._fn(msg)
        return self

    def emit_lazy(self, supplier: Callable[[], str]) -> "Progress":
        if self._fn is not None:
            self._fn(supplier())
        return self

    def map(self, f: Callable[[ProgressFn], None]) -> "Progress":
        if self._fn is not None:
            f(self._fn)
        return self

    def __call__(self, msg: str) -> "Progress":
        return self.emit(msg)

    def __bool__(self) -> bool:
        return self._fn is not None


NO_PROGRESS = Progress(None)


class TableFormat(StrEnum):
    """Table formatting mode exposed by the CLI."""

    HTML = "html"
    MARKDOWN = "markdown"


class InputKind(StrEnum):
    """Classifies an input as a document upload vs an image payload."""

    DOCUMENT = "document"
    IMAGE = "image"


class HeaderFooterMode(StrEnum):
    """
    Header/footer handling mode.

    - inline: keep headers/footers in the main markdown (do not request extraction)
    - discard: request extraction but drop them
    - extract: request extraction and write a sidecar file
    - comment: request extraction and add them back as HTML comments in the markdown
    """

    DISCARD = "discard"
    EXTRACT = "extract"
    COMMENT = "comment"
    INLINE = "inline"


@dataclass(frozen=True, slots=True)
class OcrImage:
    """An extracted image returned by OCR."""

    id: str
    image_base64: str | None = None


@dataclass(frozen=True, slots=True)
class OcrTable:
    """An extracted table returned by OCR."""

    id: str
    content: str
    format: str | None = None


def _default_tables() -> list[OcrTable]:
    return []


@dataclass(frozen=True, slots=True)
class OcrPage:
    """A single OCR page with markdown and extracted assets."""

    markdown: str
    images: list[OcrImage]
    tables: list[OcrTable] = field(default_factory=_default_tables)
    header: str | None = None
    footer: str | None = None


@dataclass(frozen=True, slots=True)
class OcrResult:
    """OCR result containing all pages."""

    pages: list[OcrPage]
