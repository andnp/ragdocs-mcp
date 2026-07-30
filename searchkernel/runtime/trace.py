"""Lightweight query tracing with named span timing.

A QueryTrace records end-to-end query latency and per-stage timing using
context managers. No external dependencies.
"""

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    """A named timing span within a query."""

    name: str
    """Name of the stage or operation."""

    start_time: float
    """Wall-clock time when the span started (perf_counter)."""

    end_time: float | None = None
    """Wall-clock time when the span ended; None if not yet closed."""

    duration_ms: float | None = None
    """Duration in milliseconds; computed on close."""

    def close(self) -> None:
        """Mark the span as closed and compute duration."""
        if self.end_time is None:
            self.end_time = time.perf_counter()
            self.duration_ms = (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
        }


@dataclass
class QueryTrace:
    """End-to-end trace of a single query execution.

    Records latency of the overall query and per-stage timing for different
    search phases (vector, keyword, reranking, etc.).
    """

    query_text: str
    """The query that was executed."""

    start_time: float = field(default_factory=time.perf_counter)
    """Wall-clock time when the query started."""

    end_time: float | None = None
    """Wall-clock time when the query ended."""

    spans: dict[str, Span] = field(default_factory=dict)
    """Named timing spans for each stage."""

    provenance: dict[str, Any] | None = None
    """Optional SearchResultProvenance or similar metadata."""

    def close(self) -> None:
        """Mark the query as completed and compute overall duration."""
        if self.end_time is None:
            self.end_time = time.perf_counter()
        for span in self.spans.values():
            if span.end_time is None:
                span.close()

    @property
    def total_duration_ms(self) -> float | None:
        """Total query duration in milliseconds; None if not closed."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def add_span(self, name: str, span: Span) -> None:
        """Register a span by name."""
        self.spans[name] = span

    @contextmanager
    def span(self, name: str) -> Generator[Span]:
        """Context manager for timing a named span.

        Usage:
            trace = QueryTrace("my query")
            with trace.span("vector_search"):
                # ... do vector search ...
            trace.close()

        Args:
            name: Name of the stage.

        Yields:
            The Span object (for introspection if needed).
        """
        start = time.perf_counter()
        span_obj = Span(name=name, start_time=start)
        try:
            yield span_obj
        finally:
            span_obj.close()
            self.add_span(name, span_obj)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        result = {
            "query": self.query_text,
            "total_duration_ms": self.total_duration_ms,
            "spans": [span.to_dict() for span in self.spans.values()],
        }
        if self.provenance is not None:
            result["provenance"] = self.provenance
        return result
