"""Huey registration helpers for indexing command handlers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def register_huey_tasks(
    huey: Any,
    handlers: dict[str, Callable[..., Any]],
) -> dict[str, Any]:
    """Register command handlers with Huey without changing their behavior."""

    return {
        name: huey.task()(handler)
        for name, handler in handlers.items()
    }
