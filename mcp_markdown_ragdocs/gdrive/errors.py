"""Classification for failures crossing the Google Drive provider boundary."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ProviderErrorClassification(StrEnum):
    RETRYABLE = "retryable"
    DEFINITIVE = "definitive"


@dataclass(frozen=True, slots=True)
class ProviderErrorInfo:
    classification: ProviderErrorClassification
    status_code: int | None
    reason: str | None
    message: str

    @property
    def retryable(self) -> bool:
        return self.classification is ProviderErrorClassification.RETRYABLE


def _provider_value(error: BaseException, name: str) -> Any:
    value = getattr(error, name, None)
    if value is not None:
        return value
    response = getattr(error, "resp", None)
    return getattr(response, name, None)


def _status_code(error: BaseException) -> int | None:
    value = _provider_value(error, "status")
    if value is None:
        value = getattr(error, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def classify_provider_error(error: BaseException) -> ProviderErrorInfo:
    """Return a stable retry decision from Google-style error attributes."""
    status_code = _status_code(error)
    reason = getattr(error, "reason", None)
    retryable_reasons = {
        "backendError",
        "internalError",
        "rateLimitExceeded",
        "userRateLimitExceeded",
        "authError",
        "invalidCredentials",
    }
    retryable_statuses = {401, 429, 500, 502, 503, 504}
    retryable = status_code in retryable_statuses or (
        status_code == 403 and reason in retryable_reasons
    )
    message = " ".join(str(error).split())
    return ProviderErrorInfo(
        classification=(
            ProviderErrorClassification.RETRYABLE
            if retryable
            else ProviderErrorClassification.DEFINITIVE
        ),
        status_code=status_code,
        reason=str(reason) if reason else None,
        message=message,
    )


__all__ = [
    "ProviderErrorClassification",
    "ProviderErrorInfo",
    "classify_provider_error",
]
