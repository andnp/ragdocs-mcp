"""Ragdocs-owned Google Drive authorized-user credential session."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from mcp_markdown_ragdocs.config import DEFAULT_GDRIVE_SCOPE
from mcp_markdown_ragdocs.gdrive.credentials import (
    validate_gdrive_credentials_path,
)

CredentialFactory = Callable[[str, Sequence[str]], Credentials]
RequestFactory = Callable[[], Request]

# Mirrors gdrive.client.DRIVE_REQUEST_TIMEOUT_SECONDS. Importing it from
# client.py would create a circular import: client.py imports this module
# for DriveCredentialSession.
DRIVE_REQUEST_TIMEOUT_SECONDS = 30


class _BoundedRequest(Request):
    """Request transport that bounds an unbounded OAuth token refresh.

    ``Request.__call__`` defaults ``timeout`` to google-auth's internal
    120s constant when no caller supplies one, and credential refresh
    call sites never pass timeout explicitly. Default it here so a
    stalled refresh cannot park a huey worker thread indefinitely.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", DRIVE_REQUEST_TIMEOUT_SECONDS)
        return super().__call__(*args, **kwargs)


class DriveCredentialSession(Protocol):
    """Narrow credential boundary required by a future Drive client."""

    def get_credentials(self) -> Credentials:
        """Return usable authorized-user credentials for a Drive request."""
        ...


class AuthorizedUserSession:
    """Load and refresh one ragdocs-owned authorized-user credential file."""

    def __init__(
        self,
        credentials_path: str | Path,
        source_root: str | Path,
        *,
        scopes: Sequence[str] = (DEFAULT_GDRIVE_SCOPE,),
        credential_factory: CredentialFactory = Credentials.from_authorized_user_file,
        request_factory: RequestFactory = _BoundedRequest,
    ) -> None:
        self._credentials_path = validate_gdrive_credentials_path(
            credentials_path,
            source_root,
        )
        self._scopes = tuple(scopes)
        self._credential_factory = credential_factory
        self._request_factory = request_factory
        self._credentials: Credentials | None = None

    def get_credentials(self) -> Credentials:
        """Load credentials once and refresh them when they are expired."""
        if self._credentials is None:
            self._credentials = self._credential_factory(
                str(self._credentials_path),
                list(self._scopes),
            )

        if self._credentials.valid:
            return self._credentials
        if not self._credentials.expired or not self._credentials.refresh_token:
            raise ValueError(
                "Google Drive authorized-user credentials cannot be refreshed"
            )

        self._credentials.refresh(self._request_factory())
        return self._credentials
