"""ChunkTuningConfig port: the config shape ChunkingStrategy implementations need.

Decouples chunking strategies from any concrete, app-specific config module.
Implementations only ever read the size/overlap knobs captured here; a
composition root can pass its full app config's chunking section directly
(it structurally satisfies this Protocol) or a purpose-built value object.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ChunkTuningConfig(Protocol):
    """Size and overlap knobs used by chunking strategies."""

    min_chunk_chars: int
    max_chunk_chars: int
    overlap_chars: int
    parent_chunk_min_chars: int
    parent_chunk_max_chars: int
