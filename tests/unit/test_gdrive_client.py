"""Tests for the bounded Google Drive transport client."""

import asyncio
import io
import time
from pathlib import Path
from typing import Any, cast

import pytest
from google.oauth2.credentials import Credentials

from mcp_markdown_ragdocs.gdrive.client import DRIVE_REQUEST_TIMEOUT_SECONDS, GoogleDriveClient
from mcp_markdown_ragdocs.gdrive.gate import DriveRequestGate
from mcp_markdown_ragdocs.gdrive.models import DriveScope


class _Session:
    def __init__(self) -> None:
        self.credentials = object()
        self.calls = 0

    def get_credentials(self) -> Any:
        self.calls += 1
        return self.credentials


class _Request:
    def __init__(self, result: object) -> None:
        self.result = result

    def execute(self) -> object:
        return self.result


class _MediaHttp:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def request(
        self,
        _uri: str,
        _method: str,
        headers: dict[str, str],
    ) -> tuple["_MediaResponse", bytes]:
        start, end = (int(value) for value in headers["range"][6:].split("-"))
        payload = self.content[start : end + 1]
        response = _MediaResponse(
            {
                "content-range": f"bytes {start}-{start + len(payload) - 1}/{len(self.content)}",
                "content-length": str(len(payload)),
            }
        )
        return response, payload


class _MediaResponse(dict[str, str]):
    @property
    def status(self) -> int:
        return 206


class _MediaRequest(_Request):
    def __init__(self, content: bytes) -> None:
        super().__init__(b"")
        self.uri = "https://drive.example/media"
        self.headers: dict[str, str] = {}
        self.http = _MediaHttp(content)

    def execute(self) -> bytes:
        raise AssertionError("bounded media requests must use the streaming transport")


class _MediaResult:
    content = b"body"


class _Files:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def list(self, **kwargs: object) -> _Request:
        self.calls.append(kwargs)
        return _Request({"files": [], "nextPageToken": "next"})

    def get(self, **kwargs: object) -> _Request:
        return _Request({"id": kwargs["fileId"], "name": "Notes", "mimeType": "text/plain"})

    def get_media(self, **kwargs: object) -> _Request:
        del kwargs
        return _Request(b"body")

    def export(self, **kwargs: object) -> _Request:
        del kwargs
        return _Request(b"exported")


class _Changes:
    def getStartPageToken(self, **kwargs: object) -> _Request:
        del kwargs
        return _Request({"startPageToken": "start"})

    def list(self, **kwargs: object) -> _Request:
        del kwargs
        return _Request({"changes": []})

    def watch(self, **kwargs: object) -> _Request:
        del kwargs
        return _Request({"id": "channel", "resourceId": "resource", "expiration": "10"})


class _Channels:
    def stop(self, **kwargs: object) -> _Request:
        del kwargs
        return _Request({})


class _Service:
    def __init__(self) -> None:
        self.file_resource = _Files()

    def files(self) -> _Files:
        return self.file_resource

    def changes(self) -> _Changes:
        return _Changes()

    def channels(self) -> _Channels:
        return _Channels()


@pytest.mark.asyncio
async def test_client_uses_session_once_and_caps_provider_page_size() -> None:
    """
    Build the service from session credentials once and cap list requests.
    """
    session = _Session()
    service = _Service()
    seen_credentials: list[object] = []

    def factory(credentials: object) -> _Service:
        seen_credentials.append(credentials)
        return service

    client = GoogleDriveClient(
        cast(Any, session),
        service_factory=factory,
        max_page_size=25,
    )

    await client.list_files_page(DriveScope("workspace"), page_size=1000)
    await client.get_file_metadata("file-1")

    assert service.file_resource.calls[0]["pageSize"] == 25
    assert session.calls == 1
    assert seen_credentials == [session.credentials]


@pytest.mark.asyncio
async def test_client_rejects_media_over_configured_bound() -> None:
    """
    Refuse oversized provider media before returning it to extraction code.
    """
    service = _Service()
    service.file_resource.get_media = lambda **_kwargs: _MediaRequest(b"12345")
    client = GoogleDriveClient(cast(Any, _Session()), service=service, max_download_bytes=4)

    with pytest.raises(ValueError, match="byte limit"):
        await client.download_file("file-1")


@pytest.mark.asyncio
async def test_client_accepts_media_at_configured_bound() -> None:
    """
    Return media whose size exactly matches the configured limit.
    """
    service = _Service()
    service.file_resource.get_media = lambda **_kwargs: _MediaRequest(b"1234")
    client = GoogleDriveClient(cast(Any, _Session()), service=service, max_download_bytes=4)

    assert await client.download_file("file-1") == b"1234"


@pytest.mark.asyncio
async def test_client_supports_legacy_media_response_shapes() -> None:
    """
    Preserve fake-service compatibility for supported media result shapes.
    """
    results: list[object] = [
        b"body",
        bytearray(b"body"),
        memoryview(b"body"),
        io.BytesIO(b"body"),
        {"body": b"body"},
        {"content": b"body"},
        {"data": b"body"},
        _MediaResult(),
    ]

    for result in results:
        service = _Service()
        service.file_resource.get_media = lambda **_kwargs: _Request(result)
        client = GoogleDriveClient(cast(Any, _Session()), service=service)

        assert await client.download_file("file-1") == b"body"


@pytest.mark.asyncio
async def test_client_serializes_provider_requests_through_private_gate(
    tmp_path: Path,
) -> None:
    """
    Keep concurrent Drive requests out of the shared index transaction.
    """
    gate = DriveRequestGate(tmp_path / "drive-gate.db", min_interval_seconds=0)
    active = 0
    maximum = 0

    class _ConcurrentRequest:
        def execute(self) -> dict[str, object]:
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            time.sleep(0.02)
            active -= 1
            return {"files": []}

    class _ConcurrentFiles:
        def list(self, **kwargs: object) -> _ConcurrentRequest:
            del kwargs
            return _ConcurrentRequest()

    class _ConcurrentService:
        def files(self) -> _ConcurrentFiles:
            return _ConcurrentFiles()

    client = GoogleDriveClient(
        cast(Any, _Session()),
        service=cast(Any, _ConcurrentService()),
        request_gate=gate,
    )

    await asyncio.gather(
        client.list_files_page(DriveScope("workspace")),
        client.list_files_page(DriveScope("workspace")),
    )

    assert maximum == 1


def test_build_service_applies_request_timeout_to_transport() -> None:
    credentials = Credentials(token="tok")

    service = GoogleDriveClient._build_service(credentials)

    transport = cast(Any, service)._http
    assert transport.http.timeout == DRIVE_REQUEST_TIMEOUT_SECONDS
