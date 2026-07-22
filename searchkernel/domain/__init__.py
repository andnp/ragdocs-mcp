"""Core domain types for the search kernel.

This module contains pure data types that form the contract between the kernel
and the outside world. Types here are source-agnostic and I/O-free.
"""

from searchkernel.domain.models import (
    Record,
    RecordStatus,
    Chunk,
    SearchResult,
    ScoredRef,
    Cursor,
    ChangeSignal,
    Vector,
    Tier,
    Filters,
)

__all__ = [
    "Record",
    "RecordStatus",
    "Chunk",
    "SearchResult",
    "ScoredRef",
    "Cursor",
    "ChangeSignal",
    "Vector",
    "Tier",
    "Filters",
]
