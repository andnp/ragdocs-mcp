"""Bounded MIME-aware extraction for Google Drive content."""

from __future__ import annotations

import importlib
import io
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ExtractionStatus(StrEnum):
    INDEXED = "indexed"
    UNSUPPORTED = "unsupported"
    TOO_LARGE = "too-large"
    TRUNCATED = "truncated"
    RETRYABLE_ERROR = "retryable-error"


@dataclass(frozen=True, slots=True)
class ExtractionLimits:
    max_download_bytes: int = 25 * 1024 * 1024
    max_text_bytes: int = 4 * 1024 * 1024
    max_items: int = 100_000
    max_pages: int = 500
    max_seconds: float = 10.0


@dataclass(frozen=True, slots=True)
class ExtractionProfile:
    name: str
    version: str
    export_mime_type: str | None
    extractor: str


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    status: ExtractionStatus
    text: str | None
    profile: str
    profile_version: str
    reason: str | None = None
    item_count: int = 0
    page_count: int = 0
    bytes_read: int = 0


EXTRACTION_PROFILES: dict[str, ExtractionProfile] = {
    "google-docs": ExtractionProfile("google-docs", "v1", "text/plain", "text"),
    "google-sheets": ExtractionProfile(
        "google-sheets",
        "v1",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xlsx",
    ),
    "google-slides": ExtractionProfile("google-slides", "v1", "application/pdf", "pdf"),
    "markdown": ExtractionProfile("markdown", "v1", None, "text"),
    "plain-text": ExtractionProfile("plain-text", "v1", None, "text"),
    "pdf": ExtractionProfile("pdf", "v1", None, "pdf"),
    "docx": ExtractionProfile("docx", "v1", None, "docx"),
    "xlsx": ExtractionProfile("xlsx", "v1", None, "xlsx"),
    "pptx": ExtractionProfile("pptx", "v1", None, "pptx"),
}
DEFAULT_EXTRACTION_LIMITS = ExtractionLimits()

_MIME_PROFILES = {
    "application/vnd.google-apps.document": "google-docs",
    "application/vnd.google-apps.spreadsheet": "google-sheets",
    "application/vnd.google-apps.presentation": "google-slides",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "text/plain": "plain-text",
    "text/markdown": "markdown",
    "text/x-markdown": "markdown",
}


class _BoundExceeded(Exception):
    def __init__(self, reason: str, *, page_count: int = 0, item_count: int = 0) -> None:
        super().__init__(reason)
        self.reason = reason
        self.page_count = page_count
        self.item_count = item_count


class _Budget:
    def __init__(self, limits: ExtractionLimits, byte_count: int, clock: Callable[[], float]) -> None:
        self.limits = limits
        self.byte_count = byte_count
        self.clock = clock
        self.started = clock()
        if byte_count > limits.max_download_bytes:
            raise _BoundExceeded("download exceeds max_download_bytes")

    def check(self) -> None:
        if self.clock() - self.started > self.limits.max_seconds:
            raise _BoundExceeded("extraction exceeds max_seconds")

    def text(self, value: str) -> str:
        self.check()
        if len(value.encode("utf-8")) > self.limits.max_text_bytes:
            raise _BoundExceeded("text exceeds max_text_bytes")
        return value

    def items(self, count: int) -> None:
        self.check()
        if count > self.limits.max_items:
            raise _BoundExceeded("items exceed max_items", item_count=count)

    def pages(self, count: int) -> None:
        self.check()
        if count > self.limits.max_pages:
            raise _BoundExceeded("pages exceed max_pages", page_count=count)


def _status_result(profile: ExtractionProfile, status: ExtractionStatus, budget: _Budget, **kwargs: Any) -> ExtractionResult:
    return ExtractionResult(status, None, profile.name, profile.version, bytes_read=budget.byte_count, **kwargs)


def _extract_pdf(payload: bytes, budget: _Budget) -> tuple[str, int, int]:
    reader = importlib.import_module("pypdf").PdfReader(io.BytesIO(payload))
    parts: list[str] = []
    for number, page in enumerate(reader.pages, start=1):
        budget.pages(number)
        parts.append(page.extract_text() or "")
        budget.items(sum(len(part.splitlines()) for part in parts))
        budget.text("\n".join(parts))
    return budget.text("\n".join(parts)), sum(len(part.splitlines()) for part in parts), len(reader.pages)


def _extract_docx(payload: bytes, budget: _Budget) -> tuple[str, int, int]:
    document = importlib.import_module("docx").Document(io.BytesIO(payload))
    parts: list[str] = []
    for number, paragraph in enumerate(document.paragraphs, start=1):
        budget.items(number)
        parts.append(paragraph.text)
        budget.text("\n".join(parts))
    return budget.text("\n".join(parts)), len(parts), 0


def _extract_xlsx(payload: bytes, budget: _Budget) -> tuple[str, int, int]:
    workbook = importlib.import_module("openpyxl").load_workbook(
        io.BytesIO(payload), read_only=True, data_only=True
    )
    parts: list[str] = []
    item_count = 0
    for page, sheet in enumerate(workbook.worksheets, start=1):
        budget.pages(page)
        parts.append(f"# Sheet: {sheet.title}")
        for row in sheet.iter_rows(values_only=True):
            item_count += 1
            budget.items(item_count)
            parts.append("\t".join("" if value is None else str(value) for value in row))
            budget.text("\n".join(parts))
    return budget.text("\n".join(parts)), item_count, len(workbook.worksheets)


def extract_content(
    payload: bytes,
    mime_type: str,
    *,
    profile: ExtractionProfile | None = None,
    limits: ExtractionLimits = DEFAULT_EXTRACTION_LIMITS,
    monotonic: Callable[[], float] = time.monotonic,
) -> ExtractionResult:
    """Return complete extracted text, or a status without partial content."""
    selected = profile or EXTRACTION_PROFILES.get(_MIME_PROFILES.get(mime_type, ""))
    selected = selected or ExtractionProfile("unknown", "v1", None, "unknown")
    try:
        budget = _Budget(limits, len(payload), monotonic)
        if selected.extractor == "unknown":
            return _status_result(selected, ExtractionStatus.UNSUPPORTED, budget, reason=mime_type)
        try:
            if selected.extractor == "text":
                text, item_count, page_count = budget.text(payload.decode("utf-8-sig")), 0, 0
            elif selected.extractor == "pdf":
                text, item_count, page_count = _extract_pdf(payload, budget)
            elif selected.extractor == "docx":
                text, item_count, page_count = _extract_docx(payload, budget)
            elif selected.extractor == "xlsx":
                text, item_count, page_count = _extract_xlsx(payload, budget)
            else:
                return _status_result(selected, ExtractionStatus.UNSUPPORTED, budget, reason="no extractor")
        except ImportError as error:
            return _status_result(selected, ExtractionStatus.UNSUPPORTED, budget, reason=f"optional dependency unavailable: {error.name}")
        except (OSError, ValueError, TypeError, KeyError) as error:
            return _status_result(selected, ExtractionStatus.RETRYABLE_ERROR, budget, reason=str(error) or type(error).__name__)
        return ExtractionResult(
            ExtractionStatus.INDEXED,
            text,
            selected.name,
            selected.version,
            item_count=item_count,
            page_count=page_count,
            bytes_read=budget.byte_count,
        )
    except _BoundExceeded as error:
        status = ExtractionStatus.TOO_LARGE if "max_download_bytes" in error.reason or "max_text_bytes" in error.reason else ExtractionStatus.TRUNCATED
        return ExtractionResult(status, None, selected.name, selected.version, reason=error.reason, item_count=error.item_count, page_count=error.page_count, bytes_read=len(payload))


__all__ = [
    "DEFAULT_EXTRACTION_LIMITS",
    "EXTRACTION_PROFILES",
    "ExtractionLimits",
    "ExtractionProfile",
    "ExtractionResult",
    "ExtractionStatus",
    "extract_content",
]
