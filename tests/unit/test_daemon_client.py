from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.daemon import DaemonMetadata
from src.daemon.client import (
    raise_daemon_request_error,
    request_daemon_json,
    should_retry_daemon_request,
)


def _metadata(
    *,
    status: str = "ready",
    socket_path: str = "/tmp/ragdocs.sock",
) -> DaemonMetadata:
    return DaemonMetadata(
        pid=123,
        status=status,
        started_at=1.0,
        socket_path=socket_path,
    )


def test_should_retry_daemon_request_only_for_transport_retryable_errors() -> None:
    assert (
        should_retry_daemon_request(
            {"status": "error", "error": "daemon_request_timed_out"}
        )
        is True
    )
    assert (
        should_retry_daemon_request(
            {"status": "error", "error": "invalid_response"}
        )
        is True
    )
    assert (
        should_retry_daemon_request(
            {"status": "error", "error": "git_indexing_unavailable"}
        )
        is False
    )
    assert should_retry_daemon_request({"status": "ok"}) is False


def test_raise_daemon_request_error_uses_specific_messages() -> None:
    with pytest.raises(RuntimeError, match="Start it with 'ragdocs daemon start'"):
        raise_daemon_request_error(None)

    with pytest.raises(RuntimeError, match="Git history search is not available"):
        raise_daemon_request_error(
            {"status": "error", "error": "git_indexing_unavailable"}
        )

    with pytest.raises(RuntimeError, match="timed out"):
        raise_daemon_request_error(
            {"status": "error", "error": "daemon_request_timed_out"}
        )

    with pytest.raises(RuntimeError, match="custom details"):
        raise_daemon_request_error(
            {
                "status": "error",
                "error": "unknown_error",
                "details": "custom details",
            }
        )


def test_request_daemon_json_auto_starts_and_waits_for_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr("src.daemon.client.RuntimePaths.resolve", lambda: SimpleNamespace())
    monkeypatch.setattr(
        "src.daemon.client.start_daemon",
        lambda cwd, project_override, paths: _metadata(status="starting"),
    )
    monkeypatch.setattr(
        "src.daemon.client.wait_for_daemon_ready",
        lambda paths: _metadata(status="ready", socket_path="/tmp/ready.sock"),
    )
    monkeypatch.setattr(
        "src.daemon.client.request_daemon_socket",
        lambda socket_path, path, payload, timeout_seconds: captured.update(
            {
                "socket_path": socket_path,
                "path": path,
                "payload": payload,
            }
        )
        or {"status": "ok", "results": []},
    )

    response = request_daemon_json(
        "/api/search/query",
        {"query": "router"},
        project_override="proj-a",
        auto_start=True,
    )

    assert response == {"status": "ok", "results": []}
    assert captured == {
        "socket_path": Path("/tmp/ready.sock"),
        "path": "/api/search/query",
        "payload": {"query": "router"},
    }


def test_request_daemon_json_returns_none_for_non_retryable_error_when_not_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.daemon.client.RuntimePaths.resolve", lambda: SimpleNamespace())
    monkeypatch.setattr(
        "src.daemon.client.inspect_daemon",
        lambda paths: SimpleNamespace(running=True, metadata=_metadata()),
    )
    monkeypatch.setattr(
        "src.daemon.client.request_daemon_socket",
        lambda socket_path, path, payload, timeout_seconds: {
            "status": "error",
            "error": "git_indexing_unavailable",
        },
    )

    response = request_daemon_json(
        "/api/search/git-history",
        {"query": "history"},
        project_override=None,
        auto_start=False,
    )

    assert response is None


def test_request_daemon_json_returns_retryable_error_payload_when_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.daemon.client.RuntimePaths.resolve", lambda: SimpleNamespace())
    monkeypatch.setattr(
        "src.daemon.client.start_daemon",
        lambda cwd, project_override, paths: _metadata(),
    )
    monkeypatch.setattr(
        "src.daemon.client.request_daemon_socket",
        lambda socket_path, path, payload, timeout_seconds: {
            "status": "error",
            "error": "daemon_request_timed_out",
        },
    )

    response = request_daemon_json(
        "/api/search/query",
        {"query": "slow"},
        project_override=None,
        auto_start=True,
        allow_error=True,
    )

    assert response == {"status": "error", "error": "daemon_request_timed_out"}
