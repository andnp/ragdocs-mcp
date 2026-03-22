from __future__ import annotations

import time
from collections.abc import Callable

import click

from src.daemon.client import raise_daemon_request_error, request_daemon_json
from src.indexing.rebuild_service import (
    REBUILD_ACTIVE_STATUSES,
    REBUILD_TERMINAL_STATUSES,
)

type EnsureRuntimeAutoRegistration = Callable[[str | None], None]
type EmitMessage = Callable[[str], None]
type Sleep = Callable[[float], None]

DEFAULT_REBUILD_POLL_INTERVAL_SECONDS = 0.2


def resolve_rebuild_project_scope(
    *,
    project: str | None,
    all_projects: bool,
) -> str | None:
    if all_projects and project is not None:
        raise click.UsageError("--all-projects cannot be used with --project")
    if project is not None:
        return project
    return None


def request_rebuild_submit_payload(
    *,
    project_override: str | None,
) -> dict[str, object]:
    payload = request_daemon_json(
        "/api/admin/rebuild/submit",
        {"project": project_override},
        project_override=project_override,
        auto_start=True,
        allow_error=True,
    )
    if payload is None or payload.get("status") == "error":
        raise_daemon_request_error(payload)
    return payload


def request_rebuild_status_payload(
    *,
    project_override: str | None,
) -> dict[str, object]:
    payload = request_daemon_json(
        "/api/admin/rebuild/status",
        {},
        project_override=project_override,
        auto_start=False,
        allow_error=True,
    )
    if payload is None or payload.get("status") == "error":
        raise_daemon_request_error(payload)
    return payload


def render_rebuild_messages(
    payload: dict[str, object],
    *,
    printed_count: int,
    emit: EmitMessage,
) -> int:
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        return printed_count

    normalized_messages = [item for item in messages if isinstance(item, str)]
    for message in normalized_messages[printed_count:]:
        emit(message)
    return len(normalized_messages)


def run_rebuild_command(
    *,
    project: str | None,
    all_projects: bool,
    ensure_runtime_auto_registration: EnsureRuntimeAutoRegistration,
    emit: EmitMessage,
    sleep: Sleep = time.sleep,
    poll_interval_seconds: float = DEFAULT_REBUILD_POLL_INTERVAL_SECONDS,
) -> None:
    ensure_runtime_auto_registration(project)

    effective_project = resolve_rebuild_project_scope(
        project=project,
        all_projects=all_projects,
    )
    submit_payload = request_rebuild_submit_payload(
        project_override=effective_project,
    )
    if bool(submit_payload.get("already_running")):
        emit("ℹ️  Rebuild already in progress; attaching to daemon-owned status")

    printed_messages = 0
    while True:
        status_payload = request_rebuild_status_payload(
            project_override=effective_project,
        )
        printed_messages = render_rebuild_messages(
            status_payload,
            printed_count=printed_messages,
            emit=emit,
        )

        rebuild_status = str(status_payload.get("status", "idle"))
        if rebuild_status in REBUILD_TERMINAL_STATUSES:
            if rebuild_status != "succeeded":
                raise RuntimeError(
                    str(status_payload.get("error", "Daemon rebuild failed"))
                )
            return

        if rebuild_status not in REBUILD_ACTIVE_STATUSES:
            raise RuntimeError(
                f"Unexpected daemon rebuild status: {rebuild_status}"
            )

        sleep(poll_interval_seconds)