"""Instance-owned task runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from mcp_markdown_ragdocs.coordination.queue import QueueRuntime
from mcp_markdown_ragdocs.coordination.task_submission import TaskSubmissionPort


@dataclass(frozen=True)
class TaskRuntime:
    """Own one queue, its submission port, and registered task handles."""

    queue_runtime: QueueRuntime
    submission: TaskSubmissionPort
    task_handles: Mapping[str, object] = field(default_factory=dict)
    gdrive_task_handles: Mapping[str, object] = field(default_factory=dict)

