"""Tests for bounded Google Drive MIME extraction."""

from mcp_markdown_ragdocs.gdrive.extraction import (
    ExtractionLimits,
    ExtractionStatus,
    extract_content,
)


def test_text_extraction_rejects_payload_over_download_bound() -> None:
    """
    Reject oversized provider payloads before any content is published.
    """
    result = extract_content(b"12345", "text/plain", limits=ExtractionLimits(max_download_bytes=4))

    assert result.status is ExtractionStatus.TOO_LARGE
    assert result.text is None
    assert result.bytes_read == 5


def test_text_extraction_rejects_decoded_text_over_text_bound() -> None:
    """
    Return no partial text when decoded content exceeds its text budget.
    """
    result = extract_content(b"hello", "text/plain", limits=ExtractionLimits(max_text_bytes=4))

    assert result.status is ExtractionStatus.TOO_LARGE
    assert result.text is None
    assert result.reason == "text exceeds max_text_bytes"


def test_text_extraction_reports_elapsed_time_bound() -> None:
    """
    Stop extraction when the injected monotonic clock exceeds its budget.
    """
    clock_values = iter((10.0, 10.1))
    result = extract_content(
        b"hello",
        "text/plain",
        limits=ExtractionLimits(max_seconds=0.05),
        monotonic=lambda: next(clock_values),
    )

    assert result.status is ExtractionStatus.TRUNCATED
    assert result.text is None
    assert result.reason == "extraction exceeds max_seconds"


def test_unknown_mime_is_status_only() -> None:
    """
    Preserve unsupported MIME metadata without fabricating extracted content.
    """
    result = extract_content(b"binary", "image/png")

    assert result.status is ExtractionStatus.UNSUPPORTED
    assert result.text is None
    assert result.reason == "image/png"
