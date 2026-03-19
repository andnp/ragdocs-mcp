"""MCP handler infrastructure - context and registration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Awaitable

from mcp.types import TextContent

if TYPE_CHECKING:
    from src.context import ApplicationContext
    from src.lifecycle import LifecycleCoordinator

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
        ctx_getter: Callable[[], ApplicationContext | None],
        coordinator: LifecycleCoordinator,
    ):
        self._ctx_getter = ctx_getter
        self.coordinator = coordinator

    @property
    def ctx(self) -> ApplicationContext | None:
        return self._ctx_getter()

    def require_ctx(self) -> ApplicationContext:
        """Get the application context, raising an error if not initialized."""
        ctx = self._ctx_getter()
        if ctx is None:
            raise RuntimeError("Server not initialized")
        return ctx

    async def wait_for_ready(self, timeout: float = 60.0) -> None:
        """Wait for indices to be ready if still initializing."""
        if self.ctx is None or not self.ctx.is_ready():
            logger.info("Query received while initializing, waiting for indices...")
            await self.coordinator.wait_ready(timeout=timeout)

    def get_nonblocking_search_payload(
        self,
        *,
        query: str,
        include_git_metadata: bool = False,
    ) -> dict[str, object] | None:
        """Return a structured cold-start payload when search is not yet queryable."""
        ctx = self.ctx
        if ctx is None:
            logger.info("Search requested before application context was available")
            return {
                "status": "initializing",
                "message": "Search indices are still initializing. Retry shortly.",
                "query": query,
                "results": [],
                "lifecycle": self.coordinator.state.value,
                "daemon_scope": "global",
                "project_context_mode": "request_only",
                "configured_root_count": 0,
                "index_state": {
                    "status": "uninitialized",
                    "indexed_count": 0,
                    "total_count": 0,
                    "last_error": None,
                },
                **(
                    {"total_commits_indexed": 0}
                    if include_git_metadata
                    else {
                        "compression_stats": {},
                        "strategy_stats": {},
                    }
                ),
            }

        if ctx.is_ready():
            return None

        index_state = ctx.get_index_state()
        if index_state.status in {"failed", "partial"}:
            logger.info("Search requested while indices were unavailable: %s", index_state.status)
            return {
                "status": "error",
                "error": "index_initialization_failed",
                "details": index_state.last_error or "Search indices are not queryable.",
                "query": query,
                "results": [],
                "lifecycle": self.coordinator.state.value,
                "daemon_scope": "global",
                "project_context_mode": "request_only",
                "configured_root_count": len(ctx.documents_roots),
                "index_state": index_state.to_dict(),
                **(
                    {
                        "total_commits_indexed": (
                            ctx.commit_indexer.get_total_commits()
                            if ctx.commit_indexer is not None
                            else 0
                        )
                    }
                    if include_git_metadata
                    else {
                        "compression_stats": {},
                        "strategy_stats": {},
                    }
                ),
            }

        logger.info("Search requested during cold start; returning initializing payload")
        return {
            "status": "initializing",
            "message": "Search indices are still initializing. Retry shortly.",
            "query": query,
            "results": [],
            "lifecycle": self.coordinator.state.value,
            "daemon_scope": "global",
            "project_context_mode": "request_only",
            "configured_root_count": len(ctx.documents_roots),
            "index_state": index_state.to_dict(),
            **(
                {
                    "total_commits_indexed": (
                        ctx.commit_indexer.get_total_commits()
                        if ctx.commit_indexer is not None
                        else 0
                    )
                }
                if include_git_metadata
                else {
                    "compression_stats": {},
                    "strategy_stats": {},
                }
            ),
        }


def format_search_status_text(
    result_header: str,
    payload: dict[str, object],
    *,
    include_git_metadata: bool = False,
) -> str:
    """Render a structured text response for non-queryable MCP search states."""
    index_state = payload.get("index_state", {})
    index_status = "unknown"
    indexed_count = 0
    total_count = 0
    if isinstance(index_state, dict):
        index_status = str(index_state.get("status", "unknown"))
        indexed_count = int(index_state.get("indexed_count", 0) or 0)
        total_count = int(index_state.get("total_count", 0) or 0)

    results = payload.get("results", [])
    results_returned = len(results) if isinstance(results, list) else 0
    message = payload.get("message") or payload.get("details") or ""
    status = str(payload.get("status", "unknown"))

    output_lines = [
        f"# {result_header}",
        "",
        f"**Status:** {status}",
    ]

    if message:
        output_lines.append(f"**Message:** {message}")

    query = payload.get("query")
    if isinstance(query, str) and query:
        output_lines.append(f"**Query:** {query}")

    output_lines.extend(
        [
            f"**Lifecycle:** {payload.get('lifecycle', 'unknown')}",
            f"**Configured Roots:** {int(payload.get('configured_root_count', 0) or 0)}",
            f"**Index State:** {index_status} ({indexed_count}/{total_count})",
        ]
    )

    if include_git_metadata:
        output_lines.append(
            f"**Total Commits Indexed:** {int(payload.get('total_commits_indexed', 0) or 0)}"
        )

    output_lines.extend(
        [
            "",
            "## Results",
            "",
            f"**Results Returned:** {results_returned}",
        ]
    )

    if status == "initializing":
        output_lines.append(
            "_No results yet — background initialization is still running._"
        )
    else:
        output_lines.append("_No results available._")

    return "\n".join(output_lines)
