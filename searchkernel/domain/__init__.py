"""Core domain types for the search kernel.

This module contains pure data types that form the contract between the kernel
and the outside world. Types here are source-agnostic and I/O-free.
"""

from searchkernel.domain.models import (
    ChangeSignal,
    Chunk,
    Cursor,
    Filters,
    Record,
    RecordStatus,
    ScoredRef,
    SearchResult,
    Tier,
    Vector,
)

__all__ = [
    "ChangeSignal",
    "Chunk",
    "Cursor",
    "Filters",
    "Record",
    "RecordStatus",
    "ScoredRef",
    "SearchResult",
    "Tier",
    "Vector",
]
