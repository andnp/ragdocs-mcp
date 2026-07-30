"""OrchestratorConfig port: the config shape SearchOrchestrator needs.

Decouples the search orchestrator from any concrete, app-specific config
module. The orchestrator only ever reads the handful of tuning knobs and
paths captured here; a composition root can pass its full app config object
directly (it structurally satisfies this Protocol) or a purpose-built value
object.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class OrchestratorSearchTuning(Protocol):
    """Search-strategy weights and pipeline tuning knobs."""

    semantic_weight: float
    keyword_weight: float
    min_confidence: float
    max_chunks_per_doc: int
    dedup_threshold: float
    reranking_enabled: bool
    rerank_top_n: int
    project_uplift_multiplier: float


@runtime_checkable
class OrchestratorIndexingPaths(Protocol):
    """Indexing paths the orchestrator falls back to when none are given explicitly."""

    documents_path: str


@runtime_checkable
class OrchestratorConfig(Protocol):
    """Minimal config surface required by BaseSearchOrchestrator/SearchOrchestrator."""

    search: OrchestratorSearchTuning
    indexing: OrchestratorIndexingPaths
    detected_project: str | None
