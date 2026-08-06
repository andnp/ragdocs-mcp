from __future__ import annotations

import click
import pytest

from mcp_markdown_ragdocs.daemon import rebuild_commands as rebuild_module


def test_resolve_rebuild_project_scope_rejects_conflicting_flags() -> None:
    with pytest.raises(click.UsageError, match="--all-projects cannot be used with --project"):
        rebuild_module.resolve_rebuild_project_scope(
            project="project-a",
            all_projects=True,
        )


def test_render_rebuild_messages_emits_only_unprinted_strings() -> None:
    emitted: list[str] = []

    printed_count = rebuild_module.render_rebuild_messages(
        {
            "messages": ["first", "second", 3, "third"],
        },
        printed_count=1,
        emit=emitted.append,
    )

    assert printed_count == 3
    assert emitted == ["second", "third"]


def test_render_rebuild_progress_exposes_current_item_and_timing() -> None:
    emitted: list[str] = []

    rebuild_module.render_rebuild_progress(
        {
            "phase": "indexing_documents",
            "documents_completed": 2,
            "documents_total": 4,
            "current_document_path": "/docs/two.md",
            "processing_rate": 1.5,
            "eta_seconds": 1.333,
            "writer_wait_seconds": 0.25,
        },
        emit=emitted.append,
    )

    assert emitted == [
        "Rebuild progress: phase=indexing_documents documents=2/4 "
        "current=/docs/two.md rate=1.50 records/s eta=1.3s writer_wait=0.25s"
    ]


def test_run_rebuild_command_polls_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    emitted: list[str] = []
    sleeps: list[float] = []
    submit_projects: list[str | None] = []
    status_projects: list[str | None] = []
    statuses = iter(
        [
            {"status": "running", "messages": ["queued", "working"]},
            {"status": "succeeded", "messages": ["queued", "working", "done"]},
        ]
    )

    monkeypatch.setattr(
        rebuild_module,
        "request_rebuild_submit_payload",
        lambda *, project_override: submit_projects.append(project_override)
        or {"status": "ok", "already_running": True},
    )
    monkeypatch.setattr(
        rebuild_module,
        "request_rebuild_status_payload",
        lambda *, project_override: status_projects.append(project_override)
        or next(statuses),
    )

    rebuild_module.run_rebuild_command(
        project="project-a",
        all_projects=False,
        emit=emitted.append,
        sleep=sleeps.append,
        poll_interval_seconds=0.25,
    )

    assert submit_projects == ["project-a"]
    assert status_projects == ["project-a", "project-a"]
    assert sleeps == [0.25]
    assert emitted == [
        "ℹ️  Rebuild already in progress; attaching to daemon-owned status",
        "queued",
        "working",
        "done",
    ]


def test_run_rebuild_command_retries_transient_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[str] = []
    sleeps: list[float] = []
    submissions = iter(
        [
            {"status": "ok", "retry_later": True},
            {"status": "ok", "already_running": False},
        ]
    )
    statuses = iter(
        [
            {"status": "running", "messages": []},
            {"status": "succeeded", "messages": []},
        ]
    )

    monkeypatch.setattr(
        rebuild_module,
        "request_rebuild_submit_payload",
        lambda *, project_override: next(submissions),
    )
    monkeypatch.setattr(
        rebuild_module,
        "request_rebuild_status_payload",
        lambda *, project_override: next(statuses),
    )

    rebuild_module.run_rebuild_command(
        project=None,
        all_projects=False,
        emit=emitted.append,
        sleep=sleeps.append,
        poll_interval_seconds=0.25,
    )

    assert sleeps == [0.25, 0.25]
    assert emitted == []


def test_run_rebuild_command_raises_failed_terminal_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rebuild_module,
        "request_rebuild_submit_payload",
        lambda *, project_override: {"status": "ok", "already_running": False},
    )
    monkeypatch.setattr(
        rebuild_module,
        "request_rebuild_status_payload",
        lambda *, project_override: {"status": "failed", "error": "boom", "messages": []},
    )

    with pytest.raises(RuntimeError, match="boom"):
        rebuild_module.run_rebuild_command(
            project=None,
            all_projects=False,
            emit=lambda message: None,
        )
