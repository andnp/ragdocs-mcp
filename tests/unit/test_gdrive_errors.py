"""Tests for Google Drive provider-error classification."""

import pytest

from mcp_markdown_ragdocs.gdrive.errors import (
    ProviderErrorClassification,
    classify_provider_error,
)


class _ApiError(RuntimeError):
    def __init__(self, status: int, reason: str | None = None) -> None:
        super().__init__(reason or "provider failure")
        self.resp = type("Response", (), {"status": status})()
        self.reason = reason


@pytest.mark.parametrize("status", [401, 429, 500, 502, 503, 504])
def test_transient_provider_statuses_are_retryable(status: int) -> None:
    """
    Keep transient authentication, quota, and server failures recoverable.
    """
    result = classify_provider_error(_ApiError(status))

    assert result.classification is ProviderErrorClassification.RETRYABLE
    assert result.retryable is True


@pytest.mark.parametrize(
    ("status", "reason"),
    [(403, "rateLimitExceeded"), (403, "backendError"), (403, "authError")],
)
def test_recoverable_provider_reasons_are_retryable(status: int, reason: str) -> None:
    """
    Classify Google reasons that can accompany an otherwise ambiguous 403.
    """
    result = classify_provider_error(_ApiError(status, reason))

    assert result.retryable is True
    assert result.reason == reason


@pytest.mark.parametrize(
    ("status", "reason"),
    [(400, "badRequest"), (403, "insufficientFilePermissions"), (404, "notFound"), (410, None)],
)
def test_definitive_provider_failures_are_not_retryable(
    status: int,
    reason: str | None,
) -> None:
    """
    Keep malformed, inaccessible, and removed resources out of retry loops.
    """
    result = classify_provider_error(_ApiError(status, reason))

    assert result.classification is ProviderErrorClassification.DEFINITIVE
    assert result.retryable is False
