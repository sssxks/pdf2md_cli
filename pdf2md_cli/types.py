from __future__ import annotations

from enum import StrEnum
from dataclasses import dataclass
from dataclasses import field
from typing import Callable


ProgressFn = Callable[[str], None]


class Progress:
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
    HTML = "html"
    MARKDOWN = "markdown"


class InputKind(StrEnum):
    PDF = "pdf"
    IMAGE = "image"


@dataclass(frozen=True, slots=True)
class OcrImage:
    id: str
    image_base64: str | None = None


@dataclass(frozen=True, slots=True)
class OcrTable:
    id: str
    content: str
    format: str | None = None


@dataclass(frozen=True, slots=True)
class OcrPage:
    markdown: str
    images: list[OcrImage]
    tables: list[OcrTable] = field(default_factory=list)
    header: str | None = None
    footer: str | None = None


@dataclass(frozen=True, slots=True)
class OcrResult:
    pages: list[OcrPage]
