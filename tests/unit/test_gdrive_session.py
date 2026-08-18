"""Tests for the ragdocs-owned Google Drive credential session."""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
import requests
from google.auth import exceptions as google_auth_exceptions
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from mcp_markdown_ragdocs.gdrive.session import (
    DRIVE_REQUEST_TIMEOUT_SECONDS,
    AuthorizedUserSession,
    _BoundedRequest,
)


class _FakeCredentials:
    def __init__(
        self,
        *,
        valid: bool,
        expired: bool,
        refresh_token: str | None,
    ) -> None:
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self.refresh_requests: list[object] = []

    def refresh(self, request: object) -> None:
        self.refresh_requests.append(request)
        self.valid = True


def _credential_path(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    credential_path = tmp_path / "state" / "authorized-user.json"
    credential_path.parent.mkdir()
    credential_path.write_text('{"type":"authorized_user"}', encoding="utf-8")
    credential_path.chmod(0o600)
    return source_root, credential_path


def test_loads_valid_authorized_user_credentials_once(tmp_path: Path):
    """
    Load a valid credential file through the narrow session boundary.
    Cache the result so repeated Drive requests reuse one session credential.
    """
    source_root, credential_path = _credential_path(tmp_path)
    credentials = _FakeCredentials(valid=True, expired=False, refresh_token=None)
    calls: list[tuple[str, Sequence[str]]] = []

    def load(path: str, scopes: Sequence[str]) -> Credentials:
        calls.append((path, scopes))
        return cast(Credentials, credentials)

    session = AuthorizedUserSession(
        credential_path,
        source_root,
        scopes=("scope-a",),
        credential_factory=load,
    )

    assert session.get_credentials() is credentials
    assert session.get_credentials() is credentials
    assert calls == [(str(credential_path.resolve()), ["scope-a"])]
    assert credentials.refresh_requests == []


def test_refreshes_expired_credentials_with_request_factory(tmp_path: Path):
    """
    Refresh an expired authorized-user credential with its refresh token.
    Keep request construction injectable so tests never contact Google.
    """
    source_root, credential_path = _credential_path(tmp_path)
    credentials = _FakeCredentials(valid=False, expired=True, refresh_token="refresh")
    request = object()

    session = AuthorizedUserSession(
        credential_path,
        source_root,
        credential_factory=lambda _path, _scopes: cast(Credentials, credentials),
        request_factory=lambda: cast(Request, request),
    )

    assert session.get_credentials() is credentials
    assert credentials.refresh_requests == [request]


def test_rejects_invalid_credentials_without_refresh_token(tmp_path: Path):
    """
    Fail clearly when an invalid authorized-user file cannot be refreshed.
    Do not silently fall back to another application's credential path.
    """
    source_root, credential_path = _credential_path(tmp_path)
    credentials = _FakeCredentials(valid=False, expired=True, refresh_token=None)

    session = AuthorizedUserSession(
        credential_path,
        source_root,
        credential_factory=lambda _path, _scopes: cast(Credentials, credentials),
    )

    with pytest.raises(ValueError, match="cannot be refreshed"):
        session.get_credentials()


def test_default_request_factory_bounds_refresh_timeout() -> None:
    """
    Bound the transport timeout that an OAuth token refresh actually uses.

    A hung refresh can park a huey worker thread indefinitely, so assert
    on the timeout reaching the underlying requests.Session call, not
    merely that some request object was constructed.
    """
    captured: dict[str, object] = {}

    class _RecordingSession:
        def request(self, method: str, url: str, **kwargs: object) -> object:
            captured.update(kwargs)
            raise requests.exceptions.ConnectionError("no network in tests")

        def close(self) -> None:
            pass

    request = _BoundedRequest(session=cast(requests.Session, _RecordingSession()))

    with pytest.raises(google_auth_exceptions.TransportError):
        request("https://oauth2.googleapis.com/token")

    assert captured["timeout"] == DRIVE_REQUEST_TIMEOUT_SECONDS
