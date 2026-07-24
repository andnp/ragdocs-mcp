"""RAGFusionStage: multi-query RAG-Fusion retrieval, OFF by default (flagged).

RAG-Fusion rewrites the user's query into several variants, retrieves
candidates for each variant independently, then reciprocal-rank-fuses
the per-variant result lists into one combined candidate list -- meant
to broaden recall for queries a single retrieval pass under-serves.

Unlike the rest of the query toolkit, this is new capability rather than
a lift of existing orchestrator behavior, so it ships disabled by
default (`RAGFusionConfig.enabled=False`): with the flag off, `run` is a
no-op passthrough, so registering or spec-listing this stage changes
nothing for existing callers unless they explicitly opt in.

Reuses `ScorePipeline.fuse` for the cross-variant RRF combine -- each
variant's result list is treated as its own equally-weighted "strategy"
(`ScorePipeline`'s semantic/keyword dynamic-weight adjustment only
triggers for those two specific keys, so arbitrary variant keys just
fall through to a plain weighted RRF sum).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path

from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.score_pipeline import ScorePipeline
from searchkernel.search.types import SearchResultDict

GenerateQueryVariants = Callable[[str, int], Awaitable[list[str]]]
Retriever = Callable[
    [str, int, "set[str] | None", Path], Awaitable[list[SearchResultDict]]
]

_TOP_K_KEY = "top_k"
_EXCLUDED_FILES_KEY = "excluded_files"
_DOCS_ROOT_KEY = "docs_root"


@dataclass(frozen=True)
class RAGFusionConfig:
    """RAG-Fusion is off by default; set `enabled=True` to opt in."""

    enabled: bool = False
    num_variants: int = 3


class RAGFusionStage:
    """Retrieve over LLM-generated query variants, then RRF-fuse them.

    Expects `context.metadata` to carry `top_k` (int), `excluded_files`
    (set[str] | None) and `docs_root` (Path) -- the same shape
    `RetrieveStage` expects. When `config.enabled` is `False` (the
    default), `run` returns `context` unchanged. When enabled, writes
    the fused `(chunk_id, score)` list to `context.candidates`.
    """

    name = "rag_fusion"

    def __init__(
        self,
        generate_query_variants: GenerateQueryVariants,
        retrieve: Retriever,
        config: RAGFusionConfig | None = None,
    ):
        self._generate_query_variants = generate_query_variants
        self._retrieve = retrieve
        self._config = config or RAGFusionConfig()

    async def run(self, context: SearchContext) -> SearchContext:
        if not self._config.enabled:
            return context

        top_k = context.metadata[_TOP_K_KEY]
        excluded_files = context.metadata.get(_EXCLUDED_FILES_KEY)
        docs_root = context.metadata[_DOCS_ROOT_KEY]

        variants = await self._generate_query_variants(
            context.query, self._config.num_variants
        )
        queries = [context.query, *variants]

        results_per_query = await asyncio.gather(
            *(
                self._retrieve(query, top_k, excluded_files, docs_root)
                for query in queries
            )
        )

        strategy_results = {
            f"rag_fusion_variant_{index}": [
                (result["chunk_id"], result.get("score", 0.0)) for result in results
            ]
            for index, results in enumerate(results_per_query)
        }
        fused = ScorePipeline().fuse(strategy_results)

        return replace(context, candidates=fused)
