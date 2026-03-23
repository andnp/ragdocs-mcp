from __future__ import annotations

import inspect
from pathlib import Path

from src.daemon import RuntimePaths
from src.daemon.health import (
    DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS,
    request_daemon_socket,
)
from src.daemon.management import inspect_daemon, start_daemon, wait_for_daemon_ready


DAEMON_PENDING_READY_STATUSES = {"starting", "initializing"}


def call_with_supported_kwargs(func, /, **kwargs):
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return func(**kwargs)

    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return func(**filtered_kwargs)


def request_daemon_json_with_dependencies(
    path: str,
    payload: dict[str, object],
    *,
    project_override: str | None,
    auto_start: bool,
    allow_error: bool = False,
    runtime_paths_resolver=RuntimePaths.resolve,
    start_daemon_fn=start_daemon,
    wait_for_daemon_ready_fn=wait_for_daemon_ready,
    inspect_daemon_fn=inspect_daemon,
    request_daemon_socket_fn=request_daemon_socket,
    cwd_provider=Path.cwd,
) -> dict[str, object] | None:
    runtime_paths = runtime_paths_resolver()
    response: dict[str, object] | None = None

    if auto_start:
        metadata = call_with_supported_kwargs(
            start_daemon_fn,
            cwd=cwd_provider(),
            project_override=project_override,
            paths=runtime_paths,
        )
        if metadata.status in DAEMON_PENDING_READY_STATUSES:
            metadata = call_with_supported_kwargs(
                wait_for_daemon_ready_fn,
                paths=runtime_paths,
            )
    else:
        inspection = call_with_supported_kwargs(inspect_daemon_fn, paths=runtime_paths)
        metadata = inspection.metadata if inspection.running else None

    if metadata is not None and metadata.socket_path:
        response = request_daemon_socket_fn(
            Path(metadata.socket_path),
            path,
            payload,
            timeout_seconds=DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS,
        )
        if not should_retry_daemon_request(response) or not auto_start:
            if response.get("status") == "error" and not allow_error:
                return None
            return response

    if response is None:
        return None

    if response.get("status") == "error" and not allow_error:
        return None
    return response


def request_daemon_json(
    path: str,
    payload: dict[str, object],
    *,
    project_override: str | None,
    auto_start: bool,
    allow_error: bool = False,
) -> dict[str, object] | None:
    return request_daemon_json_with_dependencies(
        path,
        payload,
        project_override=project_override,
        auto_start=auto_start,
        allow_error=allow_error,
    )


def should_retry_daemon_request(response: dict[str, object]) -> bool:
    return response.get("status") == "error" and response.get("error") in {
        "daemon_request_timed_out",
        "daemon_socket_unavailable",
        "empty_response",
        "invalid_response",
    }


def raise_daemon_request_error(response: dict[str, object] | None) -> None:
    if response is None:
        raise RuntimeError(
            "Daemon unavailable. Start it with 'ragdocs daemon start' and retry."
        )

    error = str(response.get("error", "unknown_error"))
    details = response.get("details")

    if error == "git_indexing_unavailable":
        raise RuntimeError(
            "Git history search is not available (git binary not found or disabled in config)"
        )

    if error == "daemon_request_timed_out":
        raise RuntimeError(
            "Daemon request timed out while waiting for a response. The daemon may still be initializing or performing a long-running operation."
        )

    if isinstance(details, str) and details:
        raise RuntimeError(details)

    raise RuntimeError(f"Daemon request failed: {error}")
