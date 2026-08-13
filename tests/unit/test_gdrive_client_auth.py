"""Integration tests for Drive client authentication and failure bounds."""

from pathlib import Path
from typing import cast

import pytest
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from mcp_markdown_ragdocs.gdrive.client import GoogleDriveClient
from mcp_markdown_ragdocs.gdrive.client import DriveService
from mcp_markdown_ragdocs.gdrive.errors import (
    ProviderErrorClassification,
    classify_provider_error,
)
from mcp_markdown_ragdocs.gdrive.models import DriveScope
from mcp_markdown_ragdocs.gdrive.session import AuthorizedUserSession


class _Credentials:
    def __init__(self, *, valid: bool, expired: bool, refresh_token: str | None) -> None:
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self.refresh_requests: list[object] = []

    def refresh(self, request: object) -> None:
        self.refresh_requests.append(request)
        self.valid = True


class _Request:
    def __init__(self, result: object) -> None:
        self.result = result
        self.executions = 0

    def execute(self) -> object:
        self.executions += 1
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _Files:
    def __init__(self, request: _Request) -> None:
        self.request = request

    def list(self, **kwargs: object) -> _Request:
        del kwargs
        return self.request


class _Service:
    def __init__(self, request: _Request) -> None:
        self._files = _Files(request)

    def files(self) -> _Files:
        return self._files


def _credential_path(tmp_path: Path) -> tuple[Path, Path]:
    """Create a valid isolated source root and owner-only credential file."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    credentials = tmp_path / "state" / "authorized-user.json"
    credentials.parent.mkdir()
    credentials.write_text("{}", encoding="utf-8")
    credentials.chmod(0o600)
    return source_root, credentials


@pytest.mark.asyncio
async def test_client_passes_session_credentials_to_service_factory(tmp_path: Path) -> None:
    """
    Use the ragdocs-owned session as the only credential source for the client.
    """
    source_root, credentials_path = _credential_path(tmp_path)
    credentials = _Credentials(valid=True, expired=False, refresh_token=None)
    session = AuthorizedUserSession(
        credentials_path,
        source_root,
        scopes=("scope-a",),
        credential_factory=lambda _path, _scopes: cast(Credentials, credentials),
    )
    seen: list[object] = []

    def build_service(received: Credentials) -> DriveService:
        seen.append(received)
        return cast(DriveService, _Service(_Request({"files": []})))

    client = GoogleDriveClient(
        session,
        service_factory=build_service,
    )

    await client.list_files_page(DriveScope("workspace"))

    assert seen == [credentials]


@pytest.mark.asyncio
async def test_client_refreshes_expired_session_before_service_creation(tmp_path: Path) -> None:
    """
    Refresh expired credentials through the session before making Drive calls.
    """
    source_root, credentials_path = _credential_path(tmp_path)
    credentials = _Credentials(valid=False, expired=True, refresh_token="refresh")
    refresh_request = object()
    session = AuthorizedUserSession(
        credentials_path,
        source_root,
        credential_factory=lambda _path, _scopes: cast(Credentials, credentials),
        request_factory=lambda: cast(Request, refresh_request),
    )
    seen: list[object] = []

    def build_service(received: Credentials) -> DriveService:
        seen.append(received)
        return cast(DriveService, _Service(_Request({"files": []})))

    client = GoogleDriveClient(
        session,
        service_factory=build_service,
    )

    await client.list_files_page(DriveScope("workspace"))

    assert credentials.refresh_requests == [refresh_request]
    assert seen == [credentials]


@pytest.mark.asyncio
async def test_client_does_not_retry_a_provider_failure_unboundedly(tmp_path: Path) -> None:
    """
    Execute one failed provider request and expose its classification to policy.
    """
    source_root, credentials_path = _credential_path(tmp_path)
    credentials = _Credentials(valid=True, expired=False, refresh_token=None)
    session = AuthorizedUserSession(
        credentials_path,
        source_root,
        credential_factory=lambda _path, _scopes: cast(Credentials, credentials),
    )
    class _ProviderError(RuntimeError):
        resp = type("Response", (), {"status": 503})()

    error = _ProviderError("temporary provider outage")
    request = _Request(error)
    client = GoogleDriveClient(session, service=cast(DriveService, _Service(request)))

    with pytest.raises(RuntimeError, match="temporary provider outage"):
        await client.list_files_page(DriveScope("workspace"))

    assert request.executions == 1
    assert classify_provider_error(error).classification is ProviderErrorClassification.RETRYABLE
