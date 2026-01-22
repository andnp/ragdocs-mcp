import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar

from src.config import Config
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.search.score_pipeline import ScorePipeline, ScorePipelineConfig
from src.search.tag_expansion import expand_query_with_tags

logger = logging.getLogger(__name__)

ResultT = TypeVar("ResultT")


@dataclass
class HybridSearchContext:
    vector_results: list[dict] = field(default_factory=list)
    keyword_results: list[dict] = field(default_factory=list)
    chunk_id_to_doc_id: dict[str, str] = field(default_factory=dict)
    all_doc_ids: set[str] = field(default_factory=set)

    def merge_results(self, results: list[dict]) -> None:
        for result in results:
            chunk_id = result["chunk_id"]
            doc_id = result["doc_id"]
            self.all_doc_ids.add(doc_id)
            self.chunk_id_to_doc_id[chunk_id] = doc_id


class BaseSearchOrchestrator(ABC, Generic[ResultT]):
    def __init__(
        self,
        vector: VectorIndex,
        keyword: KeywordIndex,
        graph: GraphStore,
        config: Config,
        documents_path: Path | None = None,
    ):
        self._vector = vector
        self._keyword = keyword
        self._graph = graph
        self._config = config
        self._documents_path = documents_path

    async def _execute_parallel_search(
        self,
        query: str,
        top_k: int,
    ) -> HybridSearchContext:
        search_tasks = [
            self._search_vector_base(query, top_k),
            self._search_keyword_base(query, top_k),
        ]

        results = await asyncio.gather(*search_tasks)

        ctx = HybridSearchContext(
            vector_results=results[0],
            keyword_results=results[1],
        )

        ctx.merge_results(ctx.vector_results)
        ctx.merge_results(ctx.keyword_results)

        return ctx

    def _apply_tag_expansion(
        self,
        ctx: HybridSearchContext,
        top_k: int,
    ) -> None:
        if not self._config.search.tag_expansion_enabled:
            return

        combined_initial_results = ctx.vector_results + ctx.keyword_results
        tag_expanded_results = expand_query_with_tags(
            initial_results=combined_initial_results,
            graph=self._graph,
            vector=self._vector,
            top_k=top_k,
            max_related_tags=self._config.search.tag_expansion_max_tags,
            max_depth=self._config.search.tag_expansion_depth,
        )

        existing_chunk_ids = {r["chunk_id"] for r in ctx.vector_results}
        for result in tag_expanded_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in existing_chunk_ids:
                ctx.merge_results([result])
                ctx.vector_results.append(result)

    def _build_strategy_results(
        self,
        ctx: HybridSearchContext,
    ) -> dict[str, list[tuple[str, float]]]:
        return {
            "semantic": [(r["chunk_id"], r.get("score", 0.0)) for r in ctx.vector_results],
            "keyword": [(r["chunk_id"], r.get("score", 0.0)) for r in ctx.keyword_results],
        }

    def _get_base_weights(self) -> dict[str, float]:
        return {
            "semantic": self._config.search.semantic_weight,
            "keyword": self._config.search.keyword_weight,
        }

    def _apply_score_pipeline(
        self,
        strategy_results: dict[str, list[tuple[str, float]]],
        weights: dict[str, float],
    ) -> list[tuple[str, float]]:
        config = self._build_score_pipeline_config(weights)
        pipeline = ScorePipeline(config)
        return pipeline.run(strategy_results)

    @abstractmethod
    def _build_score_pipeline_config(
        self, weights: dict[str, float]
    ) -> ScorePipelineConfig:
        ...

    async def _search_vector_base(self, query: str, top_k: int) -> list[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._vector.search, query, top_k, None, self._documents_path
        )

    async def _search_keyword_base(self, query: str, top_k: int) -> list[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._keyword.search, query, top_k, None, self._documents_path
        )
