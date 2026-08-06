from __future__ import annotations

import time
from collections.abc import Callable

import click

from mcp_markdown_ragdocs.daemon.client import raise_daemon_request_error, request_daemon_json
from mcp_markdown_ragdocs.indexing.rebuild_service import (
    REBUILD_ACTIVE_STATUSES,
    REBUILD_RECOVERABLE_STATUSES,
    REBUILD_TERMINAL_STATUSES,
)

type EmitMessage = Callable[[str], None]
type Sleep = Callable[[float], None]

DEFAULT_REBUILD_POLL_INTERVAL_SECONDS = 0.2


def _require_rebuild_payload(payload: dict[str, object] | None) -> dict[str, object]:
    if payload is None or payload.get("status") == "error":
        raise_daemon_request_error(payload)
    assert payload is not None
    return payload


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
    return _require_rebuild_payload(payload)


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
    return _require_rebuild_payload(payload)


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


def render_rebuild_progress(
    payload: dict[str, object],
    *,
    emit: EmitMessage,
) -> None:
    telemetry_fields = (
        "phase",
        "current_document_path",
        "current_git_repository",
        "documents_total",
        "documents_completed",
        "git_batches_completed",
        "processing_rate",
        "eta_seconds",
        "queue_wait_seconds",
    )
    if not any(field in payload for field in telemetry_fields):
        return

    phase = str(payload.get("phase", "unknown"))
    details = [f"phase={phase}"]
    document_total = payload.get("documents_total")
    document_completed = payload.get("documents_completed")
    if isinstance(document_total, int) and isinstance(document_completed, int):
        details.append(f"documents={document_completed}/{document_total}")
    git_total = payload.get("git_records_total")
    git_completed = payload.get("git_records_completed")
    if isinstance(git_completed, int):
        git_progress = f"git={git_completed}"
        if isinstance(git_total, int):
            git_progress += f"/{git_total}"
        details.append(git_progress)

    current_item = payload.get("current_item") or payload.get(
        "current_document_path"
    ) or payload.get("current_git_repository")
    if isinstance(current_item, str) and current_item:
        details.append(f"current={current_item}")
    rate = payload.get("processing_rate")
    if isinstance(rate, (int, float)):
        details.append(f"rate={rate:.2f} records/s")
    eta = payload.get("eta_seconds")
    if isinstance(eta, (int, float)):
        details.append(f"eta={eta:.1f}s")
    wait = payload.get("writer_wait_seconds")
    if isinstance(wait, (int, float)):
        details.append(f"writer_wait={wait:.2f}s")
    emit("Rebuild progress: " + " ".join(details))


def run_rebuild_command(
    *,
    project: str | None,
    all_projects: bool,
    emit: EmitMessage,
    sleep: Sleep = time.sleep,
    poll_interval_seconds: float = DEFAULT_REBUILD_POLL_INTERVAL_SECONDS,
) -> None:
    effective_project = resolve_rebuild_project_scope(
        project=project,
        all_projects=all_projects,
    )
    while True:
        submit_payload = request_rebuild_submit_payload(
            project_override=effective_project,
        )
        if not bool(submit_payload.get("retry_later")):
            break
        sleep(poll_interval_seconds)

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
        render_rebuild_progress(status_payload, emit=emit)

        rebuild_status = str(status_payload.get("status", "idle"))
        if rebuild_status in REBUILD_TERMINAL_STATUSES:
            if rebuild_status != "succeeded":
                raise RuntimeError(
                    str(status_payload.get("error", "Daemon rebuild failed"))
                )
            return

        if rebuild_status not in REBUILD_ACTIVE_STATUSES:
            if rebuild_status in REBUILD_RECOVERABLE_STATUSES:
                raise RuntimeError(
                    str(
                        status_payload.get(
                            "error",
                            "Rebuild status is corrupt and must be recovered",
                        )
                    )
                )
            raise RuntimeError(
                f"Unexpected daemon rebuild status: {rebuild_status}"
            )

        sleep(poll_interval_seconds)
