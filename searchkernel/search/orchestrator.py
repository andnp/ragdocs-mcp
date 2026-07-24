import asyncio
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from searchkernel.config import Config
from searchkernel.indices.graph import GraphStore
from searchkernel.indices.keyword import KeywordIndex
from searchkernel.indices.vector import VectorIndex
from searchkernel.indexing.manager import IndexManager
from searchkernel.models import (
    ChunkResult,
    CompressionStats,
    SearchResultProvenance,
    SearchStrategyStats,
)
from searchkernel.search.base_orchestrator import BaseSearchOrchestrator
from searchkernel.search.chunk_hydrator import ChunkHydrator
from searchkernel.search.classifier import QueryType
from searchkernel.search.filters import normalize_project_filter
from searchkernel.search.graph_expansion import build_graph_chunk_candidates
from searchkernel.search.path_utils import extract_doc_id_from_chunk_id
from searchkernel.pipeline.executor import PipelineExecutor
from searchkernel.pipeline.registry import DEFAULT_QUERY_STAGE_REGISTRY, StageDeps
from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.dedup_rerank import DedupRerankStage
from searchkernel.search.pipeline import SearchPipelineConfig
from searchkernel.search.query_execution import QueryExecutionContext
from searchkernel.search.result_cache import QueryResultCache, QueryResultCacheKey
from searchkernel.search.score_pipeline import ScorePipelineConfig
from searchkernel.search.tag_expansion import expand_query_with_tags

logger = logging.getLogger(__name__)

_ACTIVE_PROJECT_UPLIFT = 1.2
_FACTUAL_QUERY_CLEAR_CANDIDATE_LIMIT = 6
_FACTUAL_QUERY_CONSENSUS_DEPTH = 2
_FACTUAL_QUERY_CONTRACTED_TOP_K_FLOOR = 8
_FACTUAL_QUERY_CONTRACTED_TOP_K_MULTIPLIER = 2
_FACTUAL_QUERY_TOP_N_CONTRACTION_LIMIT = 5
_RESULT_CACHE_MAX_ENTRIES = 64


@dataclass
class CachedQueryResult:
    chunk_results: list[ChunkResult]
    compression_stats: CompressionStats
    strategy_stats: SearchStrategyStats
    query_execution_stats: dict[str, int] | None


class SearchOrchestrator(BaseSearchOrchestrator[ChunkResult]):
    def __init__(
        self,
        vector_index: VectorIndex,
        keyword_index: KeywordIndex,
        graph_store: GraphStore,
        config: Config,
        index_manager: IndexManager | None = None,
        documents_path: Path | None = None,
    ):
        super().__init__(
            vector_index, keyword_index, graph_store, config, documents_path
        )
        self._documents_path: Path = (
            documents_path
            if documents_path is not None
            else Path(config.indexing.documents_path)
        )
        self._index_manager = index_manager
        self._pipeline: DedupRerankStage | None = None
        self._pending_reindex: set[str] = set()
        self._reindex_tasks: set[asyncio.Task] = set()
        self._chunk_hydrator = ChunkHydrator(
            vector_index,
            keyword_index,
            self._documents_path,
            self._queue_reindex_for_chunks,
        )
        self._result_cache: QueryResultCache[CachedQueryResult] = QueryResultCache(
            max_entries=_RESULT_CACHE_MAX_ENTRIES
        )
        self._last_query_execution_stats: dict[str, int] | None = None
        self._executor = PipelineExecutor(DEFAULT_QUERY_STAGE_REGISTRY)

    @property
    def documents_path(self) -> Path:
        return self._documents_path

    def _build_dedup_rerank_stage(
        self, config: SearchPipelineConfig
    ) -> DedupRerankStage:
        return DEFAULT_QUERY_STAGE_REGISTRY["dedup_rerank"](asdict(config), StageDeps())

    def _get_pipeline(self) -> DedupRerankStage:
        if self._pipeline is None:
            self._pipeline = self._build_dedup_rerank_stage(
                self._build_pipeline_config()
            )
        return self._pipeline

    def _build_pipeline_config(self) -> SearchPipelineConfig:
        return SearchPipelineConfig(
            min_confidence=self._config.search.min_confidence,
            max_chunks_per_doc=self._config.search.max_chunks_per_doc,
            dedup_threshold=self._config.search.dedup_threshold,
            reranking_enabled=self._config.search.reranking_enabled,
            rerank_top_n=self._config.search.rerank_top_n,
        )

    def _resolve_pipeline(
        self,
        pipeline_config: SearchPipelineConfig | None,
        *,
        disable_reranking: bool = False,
    ) -> DedupRerankStage:
        if pipeline_config is None and not disable_reranking:
            return self._get_pipeline()

        effective_config = pipeline_config or self._build_pipeline_config()
        if disable_reranking and effective_config.reranking_enabled:
            effective_config = replace(effective_config, reranking_enabled=False)
        return self._build_dedup_rerank_stage(effective_config)

    def _get_result_cache_key(
        self,
        *,
        query_text: str,
        top_k: int,
        top_n: int,
        pipeline_config: SearchPipelineConfig | None,
        excluded_files: set[str] | None,
        project_filter: list[str] | None,
        project_context: str | None,
        source_filter: list[str] | None,
    ) -> QueryResultCacheKey | None:
        if self._index_manager is None:
            return None

        effective_pipeline_config = pipeline_config or self._build_pipeline_config()
        normalized_project_filter = normalize_project_filter(project_filter)
        effective_project_context = project_context or self._config.detected_project

        return QueryResultCacheKey(
            query_text=query_text,
            top_k=top_k,
            top_n=top_n,
            min_confidence=effective_pipeline_config.min_confidence,
            max_chunks_per_doc=effective_pipeline_config.max_chunks_per_doc,
            dedup_threshold=effective_pipeline_config.dedup_threshold,
            reranking_enabled=effective_pipeline_config.reranking_enabled,
            rerank_top_n=effective_pipeline_config.rerank_top_n,
            excluded_files=tuple(sorted(excluded_files or set())),
            project_filter=tuple(sorted(normalized_project_filter or set())),
            project_context=effective_project_context,
            index_state_version=self._index_manager.get_state_version(),
            source_filter=tuple(sorted(source_filter or ())),
        )

    def _should_skip_expensive_factual_enrichments(
        self,
        query_type: QueryType,
        vector_results: list[dict[str, object]],
        keyword_results: list[dict[str, object]],
    ) -> bool:
        if query_type is not QueryType.FACTUAL:
            return False

        unique_chunk_ids = {
            str(result["chunk_id"])
            for result in vector_results + keyword_results
            if isinstance(result.get("chunk_id"), str) and result.get("chunk_id")
        }
        if len(unique_chunk_ids) <= 1:
            return True
        if len(unique_chunk_ids) > _FACTUAL_QUERY_CLEAR_CANDIDATE_LIMIT:
            return False

        vector_top = [
            str(result["chunk_id"])
            for result in vector_results[:_FACTUAL_QUERY_CONSENSUS_DEPTH]
            if isinstance(result.get("chunk_id"), str) and result.get("chunk_id")
        ]
        keyword_top = [
            str(result["chunk_id"])
            for result in keyword_results[:_FACTUAL_QUERY_CONSENSUS_DEPTH]
            if isinstance(result.get("chunk_id"), str) and result.get("chunk_id")
        ]

        if not vector_top or not keyword_top:
            return False

        if vector_top[0] == keyword_top[0]:
            return True

        return bool(set(vector_top) & set(keyword_top))

    def _resolve_effective_stage_top_k(
        self,
        *,
        requested_top_k: int,
        top_n: int,
        query_type: QueryType,
        project_filter: list[str] | None,
    ) -> int:
        if requested_top_k <= 0:
            return requested_top_k

        if project_filter:
            return requested_top_k

        if query_type is not QueryType.FACTUAL:
            return requested_top_k

        if top_n > _FACTUAL_QUERY_TOP_N_CONTRACTION_LIMIT:
            return requested_top_k

        contracted_top_k = max(
            _FACTUAL_QUERY_CONTRACTED_TOP_K_FLOOR,
            top_n * _FACTUAL_QUERY_CONTRACTED_TOP_K_MULTIPLIER,
        )
        return min(requested_top_k, contracted_top_k)

    async def query(
        self,
        query_text: str,
        top_k: int = 10,
        top_n: int = 5,
        pipeline_config: SearchPipelineConfig | None = None,
        excluded_files: set[str] | None = None,
        project_filter: list[str] | None = None,
        project_context: str | None = None,
        source_filter: list[str] | None = None,
    ) -> tuple[list[ChunkResult], CompressionStats, SearchStrategyStats]:
        if not query_text or not query_text.strip():
            return (
                [],
                CompressionStats(
                    original_count=0,
                    after_threshold=0,
                    after_content_dedup=0,
                    after_ngram_dedup=0,
                    after_dedup=0,
                    after_doc_limit=0,
                    clusters_merged=0,
                ),
                SearchStrategyStats(),
            )

        cache_key = self._get_result_cache_key(
            query_text=query_text,
            top_k=top_k,
            top_n=top_n,
            pipeline_config=pipeline_config,
            excluded_files=excluded_files,
            project_filter=project_filter,
            project_context=project_context,
            source_filter=source_filter,
        )
        if cache_key is not None:
            cached_result = self._result_cache.get(cache_key)
            if cached_result is not None:
                self._last_query_execution_stats = cached_result.query_execution_stats
                return (
                    cached_result.chunk_results,
                    cached_result.compression_stats,
                    cached_result.strategy_stats,
                )

        docs_root = self._documents_path
        query_context = self._create_query_execution_context()
        base_semantic = self._config.search.semantic_weight
        base_keyword = self._config.search.keyword_weight
        base_graph = 1.0
        routing_context = await self._route(
            query_text, base_semantic, base_keyword, base_graph
        )
        query_type = routing_context.metadata["query_type"]
        weights: dict[str, float] = routing_context.metadata["strategy_weights"]
        effective_stage_top_k = self._resolve_effective_stage_top_k(
            requested_top_k=top_k,
            top_n=top_n,
            query_type=query_type,
            project_filter=project_filter,
        )

        retrieve_context = await self._retrieve(
            query_text, effective_stage_top_k, excluded_files, docs_root
        )
        vector_results = retrieve_context.metadata["vector_results"]
        keyword_results = retrieve_context.metadata["keyword_results"]
        skip_expensive_factual_enrichments = (
            self._should_skip_expensive_factual_enrichments(
                query_type,
                vector_results,
                keyword_results,
            )
        )

        all_doc_ids = set()
        chunk_id_to_doc_id = {}

        for result in vector_results:
            chunk_id = result["chunk_id"]
            doc_id = result["doc_id"]
            all_doc_ids.add(doc_id)
            chunk_id_to_doc_id[chunk_id] = doc_id

        for result in keyword_results:
            chunk_id = result["chunk_id"]
            doc_id = result["doc_id"]
            all_doc_ids.add(doc_id)
            chunk_id_to_doc_id[chunk_id] = doc_id

        graph_seed_scores = self._build_graph_seed_scores(
            vector_results, keyword_results
        )

        tag_expansion_context = self._apply_tag_expansion(
            vector_results,
            keyword_results,
            chunk_id_to_doc_id,
            all_doc_ids,
            effective_stage_top_k,
            skip=skip_expensive_factual_enrichments,
        )
        vector_results = tag_expansion_context.metadata["vector_results"]
        chunk_id_to_doc_id = tag_expansion_context.metadata["chunk_id_to_doc_id"]
        all_doc_ids = tag_expansion_context.metadata["all_doc_ids"]
        tag_expansion_count = tag_expansion_context.metadata["tag_expansion_count"]
        applied_tag_expansion_results = tag_expansion_context.metadata[
            "applied_tag_expansion_results"
        ]

        graph_context = await self._graph_expand(
            graph_seed_scores,
            effective_stage_top_k,
            excluded_chunk_ids=set(chunk_id_to_doc_id),
        )
        graph_chunk_ids = graph_context.metadata["graph_chunk_ids"]
        graph_doc_scores = graph_context.metadata["graph_doc_scores"]

        # Build strategy stats
        strategy_stats = SearchStrategyStats(
            vector_count=len(vector_results),
            keyword_count=len(keyword_results),
            graph_count=len(graph_chunk_ids),
            tag_expansion_count=tag_expansion_count,
        )

        # Build strategy results with scores for ScorePipeline
        strategy_results: dict[str, list[tuple[str, float]]] = {
            "semantic": [(r["chunk_id"], r.get("score", 0.0)) for r in vector_results],
            "keyword": [(r["chunk_id"], r.get("score", 0.0)) for r in keyword_results],
            "graph": [
                (
                    cid,
                    graph_doc_scores.get(extract_doc_id_from_chunk_id(cid), 0.0),
                )
                for cid in graph_chunk_ids
            ],
        }
        provenance_results = dict(strategy_results)
        if applied_tag_expansion_results:
            provenance_results["tag_expansion"] = [
                (r["chunk_id"], r.get("score", 0.0))
                for r in applied_tag_expansion_results
            ]
        result_provenance = self._build_result_provenance(provenance_results)

        fused = await self._apply_score_pipeline(strategy_results, weights)

        fused = self._apply_community_boost(
            fused,
            all_doc_ids,
            chunk_id_to_doc_id,
            result_provenance=result_provenance,
        )
        fused = self._apply_project_uplift(
            fused,
            query_context=query_context,
            project_context=project_context,
            result_provenance=result_provenance,
        )
        fused = self._apply_project_filter(
            fused,
            query_context=query_context,
            project_filter=project_filter,
        )
        fused = self._apply_source_filter(
            fused,
            query_context=query_context,
            source_filter=source_filter,
        )

        pipeline = self._resolve_pipeline(
            pipeline_config,
            disable_reranking=skip_expensive_factual_enrichments,
        )

        dedup_context = pipeline.run(
            SearchContext(
                query=query_text,
                candidates=fused,
                metadata={
                    "get_embedding": query_context.get_chunk_embedding,
                    "get_content": query_context.get_chunk_content,
                    "top_n": top_n,
                },
            )
        )
        final = dedup_context.candidates
        compression_stats = dedup_context.metadata["compression_stats"]

        # Parent expansion: always expand child chunks to parent chunks
        final = self._expand_to_parents(
            final,
            query_context=query_context,
            result_provenance=result_provenance,
        )

        chunk_results = self._materialize_chunk_results(
            final,
            query_context=query_context,
            result_provenance=result_provenance,
        )

        self._last_query_execution_stats = query_context.stats.to_dict()

        if cache_key is not None:
            self._result_cache.set(
                cache_key,
                CachedQueryResult(
                    chunk_results=chunk_results,
                    compression_stats=compression_stats,
                    strategy_stats=strategy_stats,
                    query_execution_stats=self._last_query_execution_stats,
                ),
            )

        return chunk_results, compression_stats, strategy_stats

    def _create_query_execution_context(self) -> QueryExecutionContext:
        return QueryExecutionContext(self._vector, self._keyword, self._chunk_hydrator)

    def _expand_to_parents(
        self,
        results: list[tuple[str, float]],
        query_context: QueryExecutionContext | None = None,
        result_provenance: dict[str, SearchResultProvenance] | None = None,
    ) -> list[tuple[str, float]]:
        get_chunk = (
            query_context.get_vector_chunk
            if query_context is not None
            else self._vector.get_chunk_by_id
        )
        get_parent_chunk = (
            query_context.get_parent_chunk
            if query_context is not None
            else self._vector.get_chunk_by_id
        )
        stage = DEFAULT_QUERY_STAGE_REGISTRY["parent_expansion"](
            {},
            StageDeps(get_chunk=get_chunk, get_parent_chunk=get_parent_chunk),
        )
        metadata: dict[str, object] = {}
        if result_provenance is not None:
            metadata["result_provenance"] = result_provenance
        context = stage.run(
            SearchContext(query="", candidates=results, metadata=metadata)
        )

        for chunk_id in context.metadata["missing_chunk_ids"]:
            self._queue_reindex_for_chunks(
                [chunk_id], "docstore lookup failed during parent expansion"
            )
        for parent_chunk_id in context.metadata["missing_parent_chunk_ids"]:
            self._queue_reindex_for_chunks(
                [parent_chunk_id],
                "parent chunk lookup failed during parent expansion",
            )

        return context.candidates

    def _build_result_provenance(
        self,
        strategy_results: dict[str, list[tuple[str, float]]],
    ) -> dict[str, SearchResultProvenance]:
        stage = DEFAULT_QUERY_STAGE_REGISTRY["provenance"]({}, StageDeps())
        context = stage.run(
            SearchContext(query="", strategy_results=strategy_results)
        )
        return context.metadata["result_provenance"]

    def _build_score_pipeline_config(
        self, weights: dict[str, float]
    ) -> ScorePipelineConfig:
        return ScorePipelineConfig(
            strategy_weights=weights,
        )

    async def _apply_score_pipeline(
        self,
        strategy_results: dict[str, list[tuple[str, float]]],
        weights: dict[str, float],
    ) -> list[tuple[str, float]]:
        config = self._build_score_pipeline_config(weights)
        context = SearchContext(query="", strategy_results=strategy_results)
        result = await self._executor.run_stage(
            "fusion", {"strategy_weights": config.strategy_weights}, context
        )
        return result.candidates

    def _get_chunk_embedding(self, chunk_id: str) -> list[float] | None:
        return self._vector.get_embedding_for_chunk(chunk_id)

    def _get_chunk_content(self, chunk_id: str) -> str | None:
        return self._chunk_hydrator.get_content(chunk_id)

    async def _route(
        self,
        query_text: str,
        base_semantic: float,
        base_keyword: float,
        base_graph: float,
    ) -> SearchContext:
        return await self._executor.run_stage(
            "routing",
            {},
            SearchContext(
                query=query_text,
                metadata={
                    "base_semantic_weight": base_semantic,
                    "base_keyword_weight": base_keyword,
                    "base_graph_weight": base_graph,
                },
            ),
        )

    async def _graph_expand(
        self,
        seed_scores: dict[str, float],
        top_k: int,
        *,
        excluded_chunk_ids: set[str] | None,
    ) -> SearchContext:
        deps = StageDeps(
            rank_neighbors=self._get_ranked_graph_neighbors,
            build_chunk_candidates=self._build_graph_chunk_candidates,
        )
        return await self._executor.run_stage(
            "graph_expand",
            {},
            SearchContext(
                query="",
                metadata={
                    "seed_scores": seed_scores,
                    "top_k": top_k,
                    "excluded_chunk_ids": excluded_chunk_ids,
                },
            ),
            deps,
        )

    def _build_graph_chunk_candidates(
        self,
        neighbor_doc_ids: list[str],
        top_k: int,
        excluded_chunk_ids: set[str] | None,
    ) -> list[str]:
        return build_graph_chunk_candidates(
            neighbor_doc_ids,
            self._vector,
            top_k,
            excluded_chunk_ids=excluded_chunk_ids,
        )

    async def _retrieve(
        self,
        query_text: str,
        top_k: int,
        excluded_files: set[str] | None,
        docs_root: Path,
    ) -> SearchContext:
        deps = StageDeps(
            search_vector=self._search_vector, search_keyword=self._search_keyword
        )
        return await self._executor.run_stage(
            "retrieve",
            {},
            SearchContext(
                query=query_text,
                metadata={
                    "top_k": top_k,
                    "excluded_files": excluded_files,
                    "docs_root": docs_root,
                },
            ),
            deps,
        )

    async def _search_vector(
        self,
        query_text: str,
        top_k: int,
        excluded_files: set[str] | None,
        docs_root: Path,
    ):
        expanded_query = self._vector.expand_query(query_text)

        results = await asyncio.to_thread(
            self._vector.search, expanded_query, top_k, excluded_files, docs_root
        )
        logger.info(
            f"Vector search returned {len(results)} results with chunk_ids: {[r['chunk_id'] for r in results[:3]]}"
        )
        return results

    async def _search_keyword(
        self,
        query_text: str,
        top_k: int,
        excluded_files: set[str] | None,
        docs_root: Path,
    ):
        results = await asyncio.to_thread(
            self._keyword.search, query_text, top_k, excluded_files, docs_root
        )
        logger.info(
            f"Keyword search returned {len(results)} results with chunk_ids: {[r['chunk_id'] for r in results[:3]]}"
        )
        return results

    def _build_graph_seed_scores(
        self,
        vector_results: list[dict[str, object]],
        keyword_results: list[dict[str, object]],
    ):
        seed_scores: dict[str, float] = {}

        for result in vector_results + keyword_results:
            doc_id_obj = result.get("doc_id")
            if not isinstance(doc_id_obj, str) or not doc_id_obj:
                continue

            raw_score = result.get("score", 0.0)
            score = float(raw_score) if isinstance(raw_score, int | float) else 0.0
            current_score = seed_scores.get(doc_id_obj, 0.0)
            if score > current_score:
                seed_scores[doc_id_obj] = score

        return seed_scores

    def _run_tag_expansion(
        self,
        combined_initial_results: list[dict[str, object]],
        top_k: int,
    ) -> list[dict[str, object]]:
        return expand_query_with_tags(
            initial_results=combined_initial_results,
            graph=self._graph,
            vector=self._vector,
            top_k=top_k,
            max_related_tags=5,
            max_depth=2,
        )

    def _apply_tag_expansion(
        self,
        vector_results: list[dict[str, object]],
        keyword_results: list[dict[str, object]],
        chunk_id_to_doc_id: dict[str, str],
        all_doc_ids: set[str],
        top_k: int,
        *,
        skip: bool,
    ) -> SearchContext:
        stage = DEFAULT_QUERY_STAGE_REGISTRY["tag_expansion"](
            {}, StageDeps(expand_query_with_tags=self._run_tag_expansion)
        )
        return stage.run(
            SearchContext(
                query="",
                metadata={
                    "vector_results": vector_results,
                    "keyword_results": keyword_results,
                    "chunk_id_to_doc_id": chunk_id_to_doc_id,
                    "all_doc_ids": all_doc_ids,
                    "top_k": top_k,
                    "skip_tag_expansion": skip,
                },
            )
        )

    def _get_ranked_graph_neighbors(self, seed_scores: dict[str, float]):
        neighbors = self._graph.rank_neighbors(seed_scores)
        logger.info(
            "Graph traversal for %s returned %d ranked neighbors: %s",
            list(seed_scores)[:3],
            len(neighbors),
            neighbors[:5],
        )
        return neighbors

    def _apply_community_boost(
        self,
        fused: list[tuple[str, float]],
        seed_doc_ids: set[str],
        chunk_id_to_doc_id: dict[str, str],
        result_provenance: dict[str, SearchResultProvenance] | None = None,
    ) -> list[tuple[str, float]]:
        stage = DEFAULT_QUERY_STAGE_REGISTRY["community_boost"](
            {}, StageDeps(boost_by_community=self._graph.boost_by_community)
        )
        metadata: dict[str, object] = {
            "seed_doc_ids": seed_doc_ids,
            "chunk_id_to_doc_id": chunk_id_to_doc_id,
        }
        if result_provenance is not None:
            metadata["result_provenance"] = result_provenance
        context = stage.run(
            SearchContext(query="", candidates=fused, metadata=metadata)
        )
        return context.candidates

    def _apply_project_uplift(
        self,
        fused: list[tuple[str, float]],
        *,
        query_context: QueryExecutionContext | None = None,
        project_context: str | None = None,
        result_provenance: dict[str, SearchResultProvenance] | None = None,
    ) -> list[tuple[str, float]]:
        active_project = project_context or self._config.detected_project
        get_chunk = (
            query_context.get_vector_chunk
            if query_context is not None
            else self._vector.get_chunk_by_id
        )
        stage = DEFAULT_QUERY_STAGE_REGISTRY["project_uplift"](
            {"uplift": _ACTIVE_PROJECT_UPLIFT}, StageDeps(get_chunk=get_chunk)
        )
        metadata: dict[str, object] = {"active_project": active_project}
        if result_provenance is not None:
            metadata["result_provenance"] = result_provenance
        context = stage.run(
            SearchContext(query="", candidates=fused, metadata=metadata)
        )
        return context.candidates

    def _apply_project_filter(
        self,
        fused: list[tuple[str, float]],
        *,
        query_context: QueryExecutionContext | None = None,
        project_filter: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        get_chunk = (
            query_context.get_vector_chunk
            if query_context is not None
            else self._vector.get_chunk_by_id
        )
        stage = DEFAULT_QUERY_STAGE_REGISTRY["project_filter"](
            {}, StageDeps(get_chunk=get_chunk)
        )
        context = stage.run(
            SearchContext(
                query="",
                candidates=fused,
                metadata={"project_filter": project_filter},
            )
        )
        return context.candidates

    def _apply_source_filter(
        self,
        fused: list[tuple[str, float]],
        *,
        query_context: QueryExecutionContext | None = None,
        source_filter: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        get_chunk = (
            query_context.get_vector_chunk
            if query_context is not None
            else self._vector.get_chunk_by_id
        )
        stage = DEFAULT_QUERY_STAGE_REGISTRY["source_filter"](
            {}, StageDeps(get_chunk=get_chunk)
        )
        context = stage.run(
            SearchContext(
                query="",
                candidates=fused,
                metadata={"source_filter": source_filter},
            )
        )
        return context.candidates

    def _queue_reindex_for_chunks(self, chunk_ids: list[str], reason: str):
        doc_ids = {
            extract_doc_id_from_chunk_id(chunk_id) for chunk_id in chunk_ids if chunk_id
        }

        if not doc_ids:
            return

        pending: list[str] = []
        for doc_id in doc_ids:
            if doc_id and doc_id not in self._pending_reindex:
                self._pending_reindex.add(doc_id)
                pending.append(doc_id)

        if not pending:
            return

        logger.warning(
            "Detected %d missing chunks; scheduling reindex for %d documents (%s)",
            len(chunk_ids),
            len(pending),
            reason,
        )
        try:
            task = asyncio.create_task(self._run_reindex(pending, reason))
        except RuntimeError:
            self._reindex_documents_sync(pending, reason)
            return

        self._reindex_tasks.add(task)
        task.add_done_callback(lambda finished: self._reindex_tasks.discard(finished))

    async def drain_reindex(self, timeout: float | None = None):
        tasks = [task for task in self._reindex_tasks if not task.done()]
        if not tasks:
            return 0

        if timeout is None:
            await asyncio.gather(*tasks, return_exceptions=True)
            return len(tasks)

        done, _pending = await asyncio.wait(tasks, timeout=timeout)
        return len(done)

    async def _run_reindex(self, doc_ids: list[str], reason: str):
        try:
            await asyncio.to_thread(self._reindex_documents_sync, doc_ids, reason)
        finally:
            for doc_id in doc_ids:
                self._pending_reindex.discard(doc_id)

    def _reindex_documents_sync(self, doc_ids: list[str], reason: str):
        if self._index_manager is None:
            return

        reindexed = 0
        for doc_id in doc_ids:
            if self._index_manager.reindex_document(doc_id, reason=reason):
                reindexed += 1

        if reindexed > 0:
            try:
                self._index_manager.persist()
                logger.info(
                    "Reindexed %d documents after missing chunk recovery", reindexed
                )
            except TimeoutError as e:
                logger.warning("Reindex persist skipped (lock busy): %s", e)

    async def query_with_hypothesis(
        self,
        hypothesis: str,
        top_k: int = 10,
        top_n: int = 5,
        excluded_files: set[str] | None = None,
        project_filter: list[str] | None = None,
        project_context: str | None = None,
    ) -> tuple[list[ChunkResult], CompressionStats, SearchStrategyStats]:
        if not hypothesis or not hypothesis.strip():
            return (
                [],
                CompressionStats(
                    original_count=0,
                    after_threshold=0,
                    after_content_dedup=0,
                    after_ngram_dedup=0,
                    after_dedup=0,
                    after_doc_limit=0,
                    clusters_merged=0,
                ),
                SearchStrategyStats(),
            )

        docs_root = self._documents_path
        query_context = self._create_query_execution_context()

        from searchkernel.search.hyde import search_with_hypothesis

        vector_results = await asyncio.to_thread(
            search_with_hypothesis,
            self._vector,
            hypothesis,
            top_k,
            excluded_files,
            docs_root,
        )

        all_doc_ids = set()
        chunk_id_to_doc_id = {}

        for result in vector_results:
            chunk_id = result["chunk_id"]
            doc_id = result["doc_id"]
            all_doc_ids.add(doc_id)
            chunk_id_to_doc_id[chunk_id] = doc_id

        strategy_results: dict[str, list[tuple[str, float]]] = {
            "semantic": [(r["chunk_id"], r.get("score", 0.0)) for r in vector_results],
        }
        result_provenance = self._build_result_provenance(strategy_results)

        weights: dict[str, float] = {"semantic": 1.0}

        fused = await self._apply_score_pipeline(strategy_results, weights)
        fused = self._apply_project_uplift(
            fused,
            query_context=query_context,
            project_context=project_context,
            result_provenance=result_provenance,
        )
        fused = self._apply_project_filter(
            fused,
            query_context=query_context,
            project_filter=project_filter,
        )

        pipeline = self._get_pipeline()

        dedup_context = pipeline.run(
            SearchContext(
                query=hypothesis,
                candidates=fused,
                metadata={
                    "get_embedding": query_context.get_chunk_embedding,
                    "get_content": query_context.get_chunk_content,
                    "top_n": top_n,
                },
            )
        )
        final = dedup_context.candidates
        compression_stats = dedup_context.metadata["compression_stats"]

        # Build strategy stats (HyDE only uses semantic search)
        strategy_stats = SearchStrategyStats(
            vector_count=len(vector_results),
        )

        chunk_results = self._materialize_chunk_results(
            final,
            query_context=query_context,
            result_provenance=result_provenance,
        )

        self._last_query_execution_stats = query_context.stats.to_dict()

        return chunk_results, compression_stats, strategy_stats

    def _materialize_chunk_results(
        self,
        final: list[tuple[str, float]],
        query_context: QueryExecutionContext | None = None,
        result_provenance: dict[str, SearchResultProvenance] | None = None,
    ) -> list[ChunkResult]:
        hydrate_chunk_result = (
            query_context.hydrate_chunk_result
            if query_context is not None
            else self._chunk_hydrator.hydrate_chunk_result
        )
        metadata: dict[str, object] = {}
        if result_provenance is not None:
            metadata["result_provenance"] = result_provenance

        stage = DEFAULT_QUERY_STAGE_REGISTRY["hydrate"](
            {}, StageDeps(hydrate_chunk_result=hydrate_chunk_result)
        )
        context = stage.run(
            SearchContext(query="", candidates=final, metadata=metadata)
        )

        missing_chunk_ids = context.metadata["missing_chunk_ids"]
        if missing_chunk_ids:
            self._queue_reindex_for_chunks(
                missing_chunk_ids,
                "chunk hydration failed during result assembly",
            )

        return context.metadata["chunk_results"]
