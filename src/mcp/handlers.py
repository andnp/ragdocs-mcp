"""MCP handler infrastructure - context and registration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Awaitable

from mcp.types import TextContent

if TYPE_CHECKING:
    from src.context import ApplicationContext
    from src.lifecycle import LifecycleCoordinator
    from src.reader.context import ReadOnlyContext

logger = logging.getLogger(__name__)

MIN_TOP_N = 1
MAX_TOP_N = 100

ToolHandler = Callable[["HandlerContext", dict], Awaitable[list[TextContent]]]

_TOOL_HANDLERS: dict[str, ToolHandler] = {}


def tool_handler(name: str):
    """Decorator to register a tool handler function."""
    def decorator(func: ToolHandler) -> ToolHandler:
        _TOOL_HANDLERS[name] = func
        return func

    return decorator


def get_handler(name: str) -> ToolHandler | None:
    """Get a registered tool handler by name."""
    return _TOOL_HANDLERS.get(name)


class HandlerContext:
    """Context passed to tool handlers containing server state."""

    def __init__(
        self,
        ctx: ApplicationContext | ReadOnlyContext | None,
        coordinator: LifecycleCoordinator,
    ):
        self.ctx = ctx
        self.coordinator = coordinator

    def require_ctx(self) -> ApplicationContext | ReadOnlyContext:
        """Get the application context, raising an error if not initialized."""
        if self.ctx is None:
            raise RuntimeError("Server not initialized")
        return self.ctx

    async def wait_for_ready(self, timeout: float = 60.0) -> None:
        """Wait for indices to be ready if still initializing."""
        if self.ctx and not self.ctx.is_ready():
            logger.info("Query received while initializing, waiting for indices...")
            await self.coordinator.wait_ready(timeout=timeout)
