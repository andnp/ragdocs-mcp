import asyncio
import logging
from pathlib import Path

from src.config import Config
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.indexing.manager import IndexManager
from src.models import (
    ChunkResult,
    CompressionStats,
    SearchResultProvenance,
    SearchStrategyStats,
)
from src.search.base_orchestrator import BaseSearchOrchestrator
from src.search.chunk_hydrator import ChunkHydrator
from src.search.classifier import classify_query, get_adaptive_weights
from src.search.filters import matches_project_filter, normalize_project_filter
from src.search.graph_expansion import build_graph_chunk_candidates
from src.search.path_utils import extract_doc_id_from_chunk_id
from src.search.pipeline import SearchPipeline, SearchPipelineConfig
from src.search.query_execution import QueryExecutionContext
from src.search.score_pipeline import ScorePipeline, ScorePipelineConfig
from src.search.tag_expansion import expand_query_with_tags

logger = logging.getLogger(__name__)

_ACTIVE_PROJECT_UPLIFT = 1.2


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
        self._pipeline: SearchPipeline | None = None
        self._pending_reindex: set[str] = set()
        self._reindex_tasks: set[asyncio.Task] = set()
        self._chunk_hydrator = ChunkHydrator(
            vector_index,
            keyword_index,
            self._documents_path,
            self._queue_reindex_for_chunks,
        )
        self._last_query_execution_stats: dict[str, int] | None = None

    @property
    def documents_path(self) -> Path:
        return self._documents_path

    def _get_pipeline(self) -> SearchPipeline:
        if self._pipeline is None:
            pipeline_config = SearchPipelineConfig(
                min_confidence=self._config.search.min_confidence,
                max_chunks_per_doc=self._config.search.max_chunks_per_doc,
                dedup_threshold=self._config.search.dedup_threshold,
                reranking_enabled=self._config.search.reranking_enabled,
                rerank_top_n=self._config.search.rerank_top_n,
            )
            self._pipeline = SearchPipeline(pipeline_config)
        return self._pipeline

    async def query(
        self,
        query_text: str,
        top_k: int = 10,
        top_n: int = 5,
        pipeline_config: SearchPipelineConfig | None = None,
        excluded_files: set[str] | None = None,
        project_filter: list[str] | None = None,
        project_context: str | None = None,
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

        docs_root = self._documents_path
        query_context = self._create_query_execution_context()

        search_tasks = [
            self._search_vector(query_text, top_k, excluded_files, docs_root),
            self._search_keyword(query_text, top_k, excluded_files, docs_root),
        ]

        results = await asyncio.gather(*search_tasks)

        vector_results = results[0]
        keyword_results = results[1]

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

        graph_seed_scores = self._build_graph_seed_scores(vector_results, keyword_results)

        # Tag-based query expansion: Find related documents via tag graph traversal
        tag_expansion_count = 0
        combined_initial_results = vector_results + keyword_results
        tag_expanded_results = expand_query_with_tags(
            initial_results=combined_initial_results,
            graph=self._graph,
            vector=self._vector,
            top_k=top_k,
            max_related_tags=5,
            max_depth=2,
        )
        applied_tag_expansion_results: list[dict[str, object]] = []

        # Merge tag-expanded results into existing result sets
        for result in tag_expanded_results:
            chunk_id = result["chunk_id"]
            doc_id = result["doc_id"]
            if chunk_id not in chunk_id_to_doc_id:
                all_doc_ids.add(doc_id)
                chunk_id_to_doc_id[chunk_id] = doc_id
                vector_results.append(result)  # Add to semantic results for fusion
                applied_tag_expansion_results.append(result)
                tag_expansion_count += 1

        ranked_graph_neighbors = self._get_ranked_graph_neighbors(graph_seed_scores)
        graph_neighbor_ids = [doc_id for doc_id, _score in ranked_graph_neighbors]
        graph_doc_scores = {
            doc_id: score for doc_id, score in ranked_graph_neighbors
        }
        graph_chunk_ids = build_graph_chunk_candidates(
            graph_neighbor_ids,
            self._vector,
            top_k,
            excluded_chunk_ids=set(chunk_id_to_doc_id),
        )

        # Build strategy stats
        strategy_stats = SearchStrategyStats(
            vector_count=len(vector_results),
            keyword_count=len(keyword_results),
            graph_count=len(graph_chunk_ids),
            tag_expansion_count=tag_expansion_count,
        )

        base_semantic = self._config.search.semantic_weight
        base_keyword = self._config.search.keyword_weight
        base_graph = 1.0

        query_type = classify_query(query_text)
        semantic_w, keyword_w, graph_w = get_adaptive_weights(
            query_type, base_semantic, base_keyword, base_graph
        )

        weights: dict[str, float] = {
            "semantic": semantic_w,
            "keyword": keyword_w,
            "graph": graph_w,
        }

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

        fused = self._apply_score_pipeline(strategy_results, weights)

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

        if pipeline_config is not None:
            pipeline = SearchPipeline(pipeline_config)
        else:
            pipeline = self._get_pipeline()

        final, compression_stats = pipeline.process(
            fused,
            query_context.get_chunk_embedding,
            query_context.get_chunk_content,
            query_text,
            top_n,
        )

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

        return chunk_results, compression_stats, strategy_stats

    def _create_query_execution_context(self) -> QueryExecutionContext:
        return QueryExecutionContext(self._vector, self._keyword, self._chunk_hydrator)

    def _expand_to_parents(
        self,
        results: list[tuple[str, float]],
        query_context: QueryExecutionContext | None = None,
        result_provenance: dict[str, SearchResultProvenance] | None = None,
    ) -> list[tuple[str, float]]:
        seen_parents: set[str] = set()
        expanded: list[tuple[str, float]] = []

        for chunk_id, score in results:
            chunk_data = (
                query_context.get_vector_chunk(chunk_id)
                if query_context is not None
                else self._vector.get_chunk_by_id(chunk_id)
            )
            if not chunk_data:
                self._queue_reindex_for_chunks(
                    [chunk_id], "docstore lookup failed during parent expansion"
                )
                expanded.append((chunk_id, score))
                continue

            metadata = chunk_data.get("metadata", {})
            parent_chunk_id = (
                metadata.get("parent_chunk_id") if isinstance(metadata, dict) else None
            )
            if parent_chunk_id:
                parent_chunk_id_str = str(parent_chunk_id)
                parent_chunk = (
                    query_context.get_parent_chunk(parent_chunk_id_str)
                    if query_context is not None
                    else self._vector.get_chunk_by_id(parent_chunk_id_str)
                )
                if parent_chunk is not None:
                    if parent_chunk_id_str not in seen_parents:
                        seen_parents.add(parent_chunk_id_str)
                        if result_provenance is not None:
                            source_provenance = result_provenance.get(chunk_id)
                            if source_provenance is not None:
                                parent_provenance = source_provenance.clone()
                                parent_provenance.parent_expanded_from = chunk_id
                                result_provenance[parent_chunk_id_str] = parent_provenance
                        expanded.append((parent_chunk_id_str, score))
                else:
                    self._queue_reindex_for_chunks(
                        [parent_chunk_id_str],
                        "parent chunk lookup failed during parent expansion",
                    )
                    expanded.append((chunk_id, score))
            else:
                expanded.append((chunk_id, score))

        return expanded

    def _build_result_provenance(
        self,
        strategy_results: dict[str, list[tuple[str, float]]],
    ) -> dict[str, SearchResultProvenance]:
        result_provenance: dict[str, SearchResultProvenance] = {}

        for strategy, result_list in strategy_results.items():
            for rank, (chunk_id, raw_score) in enumerate(result_list, start=1):
                provenance = result_provenance.setdefault(
                    chunk_id,
                    SearchResultProvenance(),
                )
                provenance.add_strategy(strategy, rank, raw_score)

        return result_provenance

    def _build_score_pipeline_config(
        self, weights: dict[str, float]
    ) -> ScorePipelineConfig:
        return ScorePipelineConfig(
            strategy_weights=weights,
        )

    def _apply_score_pipeline(
        self,
        strategy_results: dict[str, list[tuple[str, float]]],
        weights: dict[str, float],
    ) -> list[tuple[str, float]]:
        config = self._build_score_pipeline_config(weights)
        pipeline = ScorePipeline(config)
        return pipeline.run(strategy_results)

    def _get_chunk_embedding(self, chunk_id: str) -> list[float] | None:
        return self._vector.get_embedding_for_chunk(chunk_id)

    def _get_chunk_content(self, chunk_id: str) -> str | None:
        return self._chunk_hydrator.get_content(chunk_id)

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
        chunk_doc_ids = []
        for chunk_id, _ in fused:
            doc_id = chunk_id_to_doc_id.get(chunk_id)
            if doc_id is None:
                doc_id = (
                    chunk_id.rsplit("_chunk_", 1)[0]
                    if "_chunk_" in chunk_id
                    else chunk_id
                )
            chunk_doc_ids.append(doc_id)

        boosts = self._graph.boost_by_community(
            chunk_doc_ids,
            seed_doc_ids,
            1.1,
        )

        boosted = []
        for (chunk_id, score), doc_id in zip(fused, chunk_doc_ids):
            boost = boosts.get(doc_id, 1.0)
            if result_provenance is not None and boost != 1.0:
                provenance = result_provenance.setdefault(
                    chunk_id,
                    SearchResultProvenance(),
                )
                provenance.community_boost = boost
            # Clamp to [0, 1] since scores are calibrated confidence values
            boosted.append((chunk_id, min(1.0, score * boost)))

        return sorted(boosted, key=lambda x: x[1], reverse=True)

    def _apply_project_uplift(
        self,
        fused: list[tuple[str, float]],
        *,
        query_context: QueryExecutionContext | None = None,
        project_context: str | None = None,
        result_provenance: dict[str, SearchResultProvenance] | None = None,
    ) -> list[tuple[str, float]]:
        active_project = project_context or self._config.detected_project
        if not active_project:
            return fused

        boosted: list[tuple[str, float]] = []
        for chunk_id, score in fused:
            chunk_data = (
                query_context.get_vector_chunk(chunk_id)
                if query_context is not None
                else self._vector.get_chunk_by_id(chunk_id)
            )
            metadata = chunk_data.get("metadata", {}) if chunk_data else {}
            project_id = (
                metadata.get("project_id") if isinstance(metadata, dict) else None
            )
            if project_id == active_project:
                if result_provenance is not None:
                    provenance = result_provenance.setdefault(
                        chunk_id,
                        SearchResultProvenance(),
                    )
                    provenance.project_uplift = _ACTIVE_PROJECT_UPLIFT
                boosted.append((chunk_id, score * _ACTIVE_PROJECT_UPLIFT))
            else:
                boosted.append((chunk_id, score))

        return sorted(boosted, key=lambda x: x[1], reverse=True)

    def _apply_project_filter(
        self,
        fused: list[tuple[str, float]],
        *,
        query_context: QueryExecutionContext | None = None,
        project_filter: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        normalized_filter = normalize_project_filter(project_filter)
        if normalized_filter is None:
            return fused

        filtered: list[tuple[str, float]] = []
        for chunk_id, score in fused:
            chunk_data = (
                query_context.get_vector_chunk(chunk_id)
                if query_context is not None
                else self._vector.get_chunk_by_id(chunk_id)
            )
            metadata = chunk_data.get("metadata", {}) if chunk_data else {}
            project_id = (
                metadata.get("project_id") if isinstance(metadata, dict) else None
            )
            if matches_project_filter(
                str(project_id) if project_id is not None else None,
                normalized_filter,
            ):
                filtered.append((chunk_id, score))

        return filtered

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

        from src.search.hyde import search_with_hypothesis

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

        fused = self._apply_score_pipeline(strategy_results, weights)
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

        final, compression_stats = pipeline.process(
            fused,
            query_context.get_chunk_embedding,
            query_context.get_chunk_content,
            hypothesis,
            top_n,
        )

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
        chunk_results: list[ChunkResult] = []
        missing_chunk_ids: list[str] = []

        for chunk_id, score in final:
            chunk_result = (
                query_context.hydrate_chunk_result(chunk_id, score)
                if query_context is not None
                else self._chunk_hydrator.hydrate_chunk_result(chunk_id, score)
            )
            if chunk_result is not None:
                if result_provenance is not None:
                    chunk_result.provenance = result_provenance.get(chunk_id)
                chunk_results.append(chunk_result)
                continue

            missing_chunk_ids.append(chunk_id)
            chunk_results.append(
                ChunkResult(
                    chunk_id=chunk_id,
                    doc_id=extract_doc_id_from_chunk_id(chunk_id),
                    score=score,
                    header_path="",
                    file_path="",
                    content="",
                    provenance=(
                        result_provenance.get(chunk_id)
                        if result_provenance is not None
                        else None
                    ),
                )
            )

        if missing_chunk_ids:
            self._queue_reindex_for_chunks(
                missing_chunk_ids,
                "chunk hydration failed during result assembly",
            )

        return chunk_results
