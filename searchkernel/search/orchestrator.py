import asyncio
import logging
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from searchkernel.config import Config
from searchkernel.indexing.manager import IndexManager
from searchkernel.indices.graph import GraphStore
from searchkernel.indices.keyword import KeywordIndex
from searchkernel.indices.vector import VectorIndex
from searchkernel.models import (
    ChunkResult,
    CompressionStats,
    SearchResultProvenance,
    SearchStrategyStats,
)
from searchkernel.pipeline.default_query_spec import DEFAULT_QUERY_SPEC
from searchkernel.pipeline.executor import PipelineExecutor
from searchkernel.pipeline.registry import DEFAULT_QUERY_STAGE_REGISTRY, StageDeps
from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.dedup_rerank import DedupRerankStage
from searchkernel.pipeline.stages.seed_bookkeeping import (
    should_skip_expensive_factual_enrichments,
)
from searchkernel.search.base_orchestrator import BaseSearchOrchestrator
from searchkernel.search.chunk_hydrator import ChunkHydrator
from searchkernel.search.classifier import QueryType
from searchkernel.search.filters import normalize_project_filter
from searchkernel.search.graph_expansion import build_graph_chunk_candidates
from searchkernel.search.path_utils import extract_doc_id_from_chunk_id
from searchkernel.search.pipeline import SearchPipelineConfig
from searchkernel.search.query_execution import QueryExecutionContext
from searchkernel.search.result_cache import QueryResultCache, QueryResultCacheKey
from searchkernel.search.score_pipeline import ScorePipelineConfig
from searchkernel.search.tag_expansion import expand_query_with_tags

logger = logging.getLogger(__name__)

_RESULT_CACHE_MAX_ENTRIES = 64


def _elapsed_ms(start_time: float) -> float:
    return round((time.perf_counter() - start_time) * 1000, 3)


@dataclass
class CachedQueryResult:
    chunk_results: list[ChunkResult]
    compression_stats: CompressionStats
    strategy_stats: SearchStrategyStats
    query_execution_stats: dict[str, int | float] | None


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
        self._last_query_execution_stats: dict[str, int | float] | None = None
        self._executor = PipelineExecutor(DEFAULT_QUERY_STAGE_REGISTRY)

    @property
    def last_query_execution_stats(self) -> dict[str, int | float] | None:
        return self._last_query_execution_stats

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
        return should_skip_expensive_factual_enrichments(
            query_type, vector_results, keyword_results
        )

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

        deps = StageDeps(
            search_vector=self._search_vector,
            search_keyword=self._search_keyword,
            rank_neighbors=self._get_ranked_graph_neighbors,
            build_chunk_candidates=self._build_graph_chunk_candidates,
            expand_query_with_tags=self._run_tag_expansion,
            boost_by_community=self._graph.boost_by_community,
            get_chunk=query_context.get_vector_chunk,
            get_parent_chunk=query_context.get_parent_chunk,
            hydrate_chunk_result=query_context.hydrate_chunk_result,
        )
        context = SearchContext(
            query=query_text,
            metadata={
                "base_semantic_weight": self._config.search.semantic_weight,
                "base_keyword_weight": self._config.search.keyword_weight,
                "base_graph_weight": 1.0,
                "requested_top_k": top_k,
                "top_n": top_n,
                "project_filter": project_filter,
                "excluded_files": excluded_files,
                "docs_root": docs_root,
                "source_filter": source_filter,
                "active_project": project_context or self._config.detected_project,
            },
        )

        total_query_start = time.perf_counter()

        for stage_spec in DEFAULT_QUERY_SPEC.stages:
            name = stage_spec.name

            # dedup_rerank runs via the orchestrator's cached DedupRerankStage
            # instance rather than a fresh run_stage call, so the reranker's
            # lazily-loaded cross-encoder model is reused across queries
            # instead of reloaded every time (see _get_pipeline/_resolve_pipeline).
            if name == "dedup_rerank":
                pipeline_start = time.perf_counter()
                pipeline = self._resolve_pipeline(
                    pipeline_config,
                    disable_reranking=context.metadata["skip_tag_expansion"],
                )
                dedup_metadata = dict(context.metadata)
                dedup_metadata["get_embedding"] = query_context.get_chunk_embedding
                dedup_metadata["get_content"] = query_context.get_chunk_content
                dedup_metadata["top_n"] = top_n
                context = pipeline.run(replace(context, metadata=dedup_metadata))
                query_context.stats.pipeline_ms = _elapsed_ms(pipeline_start)
                continue

            stage_start = time.perf_counter()
            config = dict(stage_spec.config)
            if name == "fusion":
                config["strategy_weights"] = context.metadata["strategy_weights"]
            elif name == "project_uplift":
                config["uplift"] = self._config.search.project_uplift_multiplier
            context = await self._executor.run_stage(name, config, context, deps)

            # Record timing for key stages. Vector and keyword retrieval run
            # concurrently inside RetrieveStage (asyncio.gather), so there is
            # no separately-measurable per-strategy duration -- both fields
            # record the same combined wall-clock time for the stage.
            if name == "retrieve":
                retrieve_ms = _elapsed_ms(stage_start)
                query_context.stats.vector_search_ms = retrieve_ms
                query_context.stats.keyword_search_ms = retrieve_ms
            elif name == "tag_expansion":
                query_context.stats.tag_expansion_ms = _elapsed_ms(stage_start)
            elif name == "graph_expand":
                query_context.stats.graph_expansion_ms = _elapsed_ms(stage_start)
            elif name == "fusion":
                query_context.stats.fusion_ms = _elapsed_ms(stage_start)
            elif name == "parent_expansion":
                query_context.stats.parent_expansion_ms = _elapsed_ms(stage_start)
                for chunk_id in context.metadata["missing_chunk_ids"]:
                    self._queue_reindex_for_chunks(
                        [chunk_id], "docstore lookup failed during parent expansion"
                    )
                for parent_chunk_id in context.metadata["missing_parent_chunk_ids"]:
                    self._queue_reindex_for_chunks(
                        [parent_chunk_id],
                        "parent chunk lookup failed during parent expansion",
                    )
            elif name == "hydrate":
                query_context.stats.materialization_ms = _elapsed_ms(stage_start)
                missing_chunk_ids = context.metadata["missing_chunk_ids"]
                if missing_chunk_ids:
                    self._queue_reindex_for_chunks(
                        missing_chunk_ids,
                        "chunk hydration failed during result assembly",
                    )

        compression_stats = context.metadata["compression_stats"]
        chunk_results = context.metadata["chunk_results"]
        strategy_stats = SearchStrategyStats(
            vector_count=len(context.metadata["vector_results"]),
            keyword_count=len(context.metadata["keyword_results"]),
            graph_count=len(context.metadata["graph_chunk_ids"]),
            tag_expansion_count=context.metadata["tag_expansion_count"],
        )

        query_context.stats.total_query_ms = _elapsed_ms(total_query_start)
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
            {"uplift": self._config.search.project_uplift_multiplier},
            StageDeps(get_chunk=get_chunk),
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
