"""Formatting and filtering helpers for the `ragdocs queue` CLI commands."""

import click

_QUEUE_DETAIL_STATES = ("all", "pending", "scheduled", "failed")


def _coerce_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    return {}


def _coerce_list_of_dicts(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _coerce_int(value: object, default: int = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _apply_queue_detail_limit(
    items: list[dict[str, object]],
    *,
    limit: int | None,
) -> list[dict[str, object]]:
    if limit is None:
        return items
    return items[:limit]


def _filter_queue_status_payload(
    payload: dict[str, object],
    *,
    state: str,
    limit: int | None,
) -> dict[str, object]:
    filtered_payload = dict(payload)
    pending_tasks = _coerce_list_of_dicts(payload.get("pending_tasks"))
    scheduled_tasks = _coerce_list_of_dicts(payload.get("scheduled_tasks"))
    recent_failures = _coerce_list_of_dicts(payload.get("recent_failures"))

    filtered_payload["pending_tasks"] = _apply_queue_detail_limit(
        pending_tasks if state in ("all", "pending") else [],
        limit=limit,
    )
    filtered_payload["scheduled_tasks"] = _apply_queue_detail_limit(
        scheduled_tasks if state in ("all", "scheduled") else [],
        limit=limit,
    )
    filtered_payload["recent_failures"] = _apply_queue_detail_limit(
        recent_failures if state in ("all", "failed") else [],
        limit=limit,
    )
    filtered_payload["detail_state_filter"] = state
    filtered_payload["detail_limit"] = limit
    return filtered_payload


def _format_queue_task_summary(
    task: dict[str, object],
    *,
    details: bool,
) -> list[str]:
    task_name = str(task.get("task_name") or "unknown")
    task_id = str(task.get("task_id") or "unknown")

    if not details:
        parts = [f"id={task_id}"]
        if task.get("priority") is not None:
            parts.append(f"priority={task['priority']}")
        if task.get("eta"):
            parts.append(f"eta={task['eta']}")
        return [f"  {task_name} ({', '.join(parts)})"]

    lines = [f"  {task_name}", f"    id: {task_id}"]
    state = task.get("state")
    if state:
        lines.append(f"    state: {state}")
    source_queue = task.get("source_queue")
    if source_queue:
        lines.append(f"    queue: {source_queue}")
    priority = task.get("priority")
    if priority is not None:
        lines.append(f"    priority: {priority}")
    eta = task.get("eta")
    if eta:
        lines.append(f"    eta: {eta}")
    lines.append(f"    retries: {_coerce_int(task.get('retries'))}")
    return lines


def _emit_queue_task_section(
    title: str,
    tasks: list[dict[str, object]],
    *,
    details: bool,
) -> None:
    if not tasks:
        return

    click.echo(title)
    for task in tasks:
        for line in _format_queue_task_summary(task, details=details):
            click.echo(line)


def _emit_git_refresh_progress(payload: object) -> None:
    if not isinstance(payload, list):
        return
    rows = [row for row in payload if isinstance(row, dict)]
    if not rows:
        return

    click.echo("Git refresh:")
    for row in rows:
        repository = str(row.get("repository_path") or "unknown")
        state = str(row.get("state") or "unknown")
        processed = row.get("processed_count")
        discovered = row.get("discovered_count")
        if isinstance(processed, int) and isinstance(discovered, int):
            click.echo(f"  {repository}: {state} ({processed}/{discovered})")
        else:
            click.echo(f"  {repository}: {state}")
