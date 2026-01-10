from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


ProgressFn = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class OcrImage:
    id: str
    image_base64: str | None = None


@dataclass(frozen=True, slots=True)
class OcrPage:
    markdown: str
    images: list[OcrImage]


@dataclass(frozen=True, slots=True)
class OcrResult:
    pages: list[OcrPage]
