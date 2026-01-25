import asyncio
import logging
import time
from pathlib import Path

from src.config import Config
from src.indices.code import CodeIndex
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.indexing.manager import IndexManager
from src.models import ChunkResult, CompressionStats, SearchStrategyStats
from src.search.base_orchestrator import BaseSearchOrchestrator
from src.search.classifier import classify_query, get_adaptive_weights
from src.search.path_utils import extract_doc_id_from_chunk_id
from src.search.pipeline import SearchPipeline, SearchPipelineConfig
from src.search.score_pipeline import ScorePipeline, ScorePipelineConfig
from src.search.tag_expansion import expand_query_with_tags

logger = logging.getLogger(__name__)


class SearchOrchestrator(BaseSearchOrchestrator[ChunkResult]):
    def __init__(
        self,
        vector_index: VectorIndex,
        keyword_index: KeywordIndex,
        graph_store: GraphStore,
        config: Config,
        index_manager: IndexManager | None = None,
        code_index: CodeIndex | None = None,
        documents_path: Path | None = None,
    ):
        super().__init__(vector_index, keyword_index, graph_store, config, documents_path)
        self._documents_path = documents_path if documents_path is not None else Path(config.indexing.documents_path)
        self._index_manager = index_manager
        self._code = code_index
        self._pipeline: SearchPipeline | None = None
        self._pending_reindex: set[str] = set()
        self._reindex_tasks: set[asyncio.Task] = set()

        # Query embedding cache with LRU eviction
        self._embedding_cache: dict[str, tuple[list[float], float]] = {}
        self._cache_max_size = 100
        self._cache_ttl = 300.0  # 5 minutes

    @property
    def documents_path(self) -> Path:
        return self._documents_path

    def _get_pipeline(self) -> SearchPipeline:
        if self._pipeline is None:
            pipeline_config = SearchPipelineConfig(
                min_confidence=self._config.search.min_confidence,
                max_chunks_per_doc=self._config.search.max_chunks_per_doc,
                dedup_enabled=self._config.search.dedup_enabled,
                dedup_threshold=self._config.search.dedup_similarity_threshold,
                ngram_dedup_enabled=self._config.search.ngram_dedup_enabled,
                ngram_dedup_threshold=self._config.search.ngram_dedup_threshold,
                mmr_enabled=self._config.search.mmr_enabled,
                mmr_lambda=self._config.search.mmr_lambda,
                parent_retrieval_enabled=self._config.document_chunking.parent_retrieval_enabled,
                rerank_enabled=self._config.search.rerank_enabled,
                rerank_model=self._config.search.rerank_model,
                rerank_top_n=self._config.search.rerank_top_n,
            )
            self._pipeline = SearchPipeline(pipeline_config)
        return self._pipeline

    def _get_cached_embedding(self, query: str) -> list[float]:
        """Get query embedding with LRU cache and TTL-based expiration.

        Cache entries expire after 5 minutes to prevent stale embeddings.
        When cache is full, evicts the oldest entry (LRU policy).

        Args:
            query: Query text to embed

        Returns:
            Embedding vector for the query
        """
        current_time = time.time()

        # Check cache
        if query in self._embedding_cache:
            embedding, timestamp = self._embedding_cache[query]
            # Expire after TTL
            if current_time - timestamp < self._cache_ttl:
                logger.debug(f"Embedding cache hit for query: {query[:50]}...")
                return embedding
            else:
                # Remove expired entry
                del self._embedding_cache[query]

        # Compute new embedding
        logger.debug(f"Embedding cache miss for query: {query[:50]}...")
        embedding = self._vector.get_text_embedding(query)

        # Evict oldest if cache full (LRU policy)
        if len(self._embedding_cache) >= self._cache_max_size:
            oldest_key = min(self._embedding_cache, key=lambda k: self._embedding_cache[k][1])
            del self._embedding_cache[oldest_key]
            logger.debug(f"Evicted oldest cache entry: {oldest_key[:50]}...")

        self._embedding_cache[query] = (embedding, current_time)
        return embedding

    async def query(
        self,
        query_text: str,
        top_k: int = 10,
        top_n: int = 5,
        pipeline_config: SearchPipelineConfig | None = None,
        excluded_files: set[str] | None = None,
    ) -> tuple[list[ChunkResult], CompressionStats, SearchStrategyStats]:
        if not query_text or not query_text.strip():
            return [], CompressionStats(
                original_count=0,
                after_threshold=0,
                after_content_dedup=0,
                after_ngram_dedup=0,
                after_dedup=0,
                after_doc_limit=0,
                clusters_merged=0,
            ), SearchStrategyStats()

        docs_root = self._documents_path

        search_tasks = [
            self._search_vector(query_text, top_k, excluded_files, docs_root),
            self._search_keyword(query_text, top_k, excluded_files, docs_root),
        ]

        code_search_enabled = (
            self._config.search.code_search_enabled
            and self._code is not None
        )
        if code_search_enabled:
            search_tasks.append(self._search_code(query_text, top_k))

        results = await asyncio.gather(*search_tasks)

        vector_results = results[0]
        keyword_results = results[1]
        code_results = results[2] if code_search_enabled else []

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

        for result in code_results:
            chunk_id = result["chunk_id"]
            doc_id = result["doc_id"]
            all_doc_ids.add(doc_id)
            chunk_id_to_doc_id[chunk_id] = doc_id

        # Tag-based query expansion: Find related documents via tag graph traversal
        tag_expansion_count = 0
        if self._config.search.tag_expansion_enabled:
            combined_initial_results = vector_results + keyword_results + code_results
            tag_expanded_results = expand_query_with_tags(
                initial_results=combined_initial_results,
                graph=self._graph,
                vector=self._vector,
                top_k=top_k,
                max_related_tags=self._config.search.tag_expansion_max_tags,
                max_depth=self._config.search.tag_expansion_depth,
            )

            # Merge tag-expanded results into existing result sets
            for result in tag_expanded_results:
                chunk_id = result["chunk_id"]
                doc_id = result["doc_id"]
                if chunk_id not in chunk_id_to_doc_id:
                    all_doc_ids.add(doc_id)
                    chunk_id_to_doc_id[chunk_id] = doc_id
                    vector_results.append(result)  # Add to semantic results for fusion
                    tag_expansion_count += 1

        graph_neighbors = self._get_graph_neighbors(list(all_doc_ids))

        # Convert graph document IDs to chunk IDs
        graph_chunk_ids = []
        for doc_id in graph_neighbors:
            chunk_ids_for_doc = self._vector.get_chunk_ids_for_document(doc_id)
            graph_chunk_ids.extend(chunk_ids_for_doc)

        # Build strategy stats
        strategy_stats = SearchStrategyStats(
            vector_count=len(vector_results),
            keyword_count=len(keyword_results),
            graph_count=len(graph_chunk_ids),
            code_count=len(code_results) if code_search_enabled else None,
            tag_expansion_count=tag_expansion_count if self._config.search.tag_expansion_enabled else None,
        )

        results_dict: dict[str, list[str]] = {
             "semantic": [r["chunk_id"] for r in vector_results],
             "keyword": [r["chunk_id"] for r in keyword_results],
             "graph": graph_chunk_ids,
         }

        if code_search_enabled and code_results:
            results_dict["code"] = [r["chunk_id"] for r in code_results]

        # Collect file modified times using asyncio.to_thread to avoid blocking event loop
        # NOTE: modified_times collected for future recency boosting but not currently used
        _modified_times = await asyncio.to_thread(
            self._collect_modified_times,
            all_doc_ids | set(graph_neighbors)
        )

        base_semantic = self._config.search.semantic_weight
        base_keyword = self._config.search.keyword_weight
        base_graph = 1.0

        if self._config.search.adaptive_weights_enabled:
            query_type = classify_query(query_text)
            semantic_w, keyword_w, graph_w = get_adaptive_weights(
                query_type, base_semantic, base_keyword, base_graph
            )
        else:
            semantic_w = base_semantic
            keyword_w = base_keyword
            graph_w = base_graph

        weights: dict[str, float] = {
            "semantic": semantic_w,
            "keyword": keyword_w,
            "graph": graph_w,
        }

        if code_search_enabled:
            weights["code"] = self._config.search.code_search_weight

        # Build strategy results with scores for ScorePipeline
        strategy_results: dict[str, list[tuple[str, float]]] = {
            "semantic": [(r["chunk_id"], r.get("score", 0.0)) for r in vector_results],
            "keyword": [(r["chunk_id"], r.get("score", 0.0)) for r in keyword_results],
            "graph": [(cid, 1.0) for cid in graph_chunk_ids],
        }
        if code_search_enabled and code_results:
            strategy_results["code"] = [(r["chunk_id"], r.get("score", 0.0)) for r in code_results]

        fused = self._apply_score_pipeline(strategy_results, weights)

        if self._config.search.community_detection_enabled:
            fused = self._apply_community_boost(fused, all_doc_ids, chunk_id_to_doc_id)

        if pipeline_config is not None:
            pipeline = SearchPipeline(pipeline_config)
        else:
            pipeline = self._get_pipeline()

        query_embedding = None
        if self._config.search.mmr_enabled or (
            pipeline_config is not None and pipeline_config.mmr_enabled
        ):
            query_embedding = self._get_cached_embedding(query_text)

        final, compression_stats = pipeline.process(
            fused,
            self._get_chunk_embedding,
            self._get_chunk_content,
            query_text,
            top_n,
            query_embedding,
        )

        # Parent expansion: if enabled, expand child chunks to parent chunks
        parent_retrieval_enabled = self._config.document_chunking.parent_retrieval_enabled
        if pipeline_config is not None and pipeline_config.parent_retrieval_enabled:
            parent_retrieval_enabled = True

        if parent_retrieval_enabled:
            final = self._expand_to_parents(final)

        chunk_results = []
        missing_chunk_ids: list[str] = []
        for chunk_id, score in final:
            chunk_data = self._vector.get_chunk_by_id(chunk_id)
            if chunk_data:
                metadata = chunk_data.get("metadata", {})
                parent_chunk_id = metadata.get("parent_chunk_id") if isinstance(metadata, dict) else None
                parent_content = None
                if parent_chunk_id:
                    parent_content = self._vector.get_parent_content(parent_chunk_id)

                chunk_results.append(ChunkResult(
                    chunk_id=chunk_id,
                    doc_id=str(chunk_data.get("doc_id", "")),
                    score=score,
                    header_path=str(chunk_data.get("header_path", "")),
                    file_path=str(chunk_data.get("file_path", "")),
                    content=str(chunk_data.get("content", "")),
                    parent_chunk_id=parent_chunk_id,
                    parent_content=parent_content,
                ))
            else:
                missing_chunk_ids.append(chunk_id)
                chunk_results.append(ChunkResult(
                    chunk_id=chunk_id,
                    doc_id=extract_doc_id_from_chunk_id(chunk_id),
                    score=score,
                    header_path="",
                    file_path="",
                    content="",
                ))

        if missing_chunk_ids:
            self._queue_reindex_for_chunks(missing_chunk_ids, "docstore lookup failed")

        return chunk_results, compression_stats, strategy_stats

    def _expand_to_parents(
        self, results: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        seen_parents: set[str] = set()
        expanded: list[tuple[str, float]] = []

        for chunk_id, score in results:
            chunk_data = self._vector.get_chunk_by_id(chunk_id)
            if not chunk_data:
                self._queue_reindex_for_chunks([chunk_id], "docstore lookup failed during parent expansion")
                expanded.append((chunk_id, score))
                continue

            metadata = chunk_data.get("metadata", {})
            parent_chunk_id = metadata.get("parent_chunk_id") if isinstance(metadata, dict) else None
            if parent_chunk_id:
                if parent_chunk_id not in seen_parents:
                    seen_parents.add(parent_chunk_id)
                    expanded.append((parent_chunk_id, score))
            else:
                expanded.append((chunk_id, score))

        return expanded

    def _build_score_pipeline_config(
        self, weights: dict[str, float]
    ) -> ScorePipelineConfig:
        return ScorePipelineConfig(
            rrf_k=self._config.search.rrf_k_constant,
            strategy_weights=weights,
            use_dynamic_weights=self._config.search.dynamic_weights_enabled,
            variance_threshold=self._config.search.variance_threshold,
            min_weight_factor=self._config.search.min_weight_factor,
            calibration_threshold=self._config.search.score_calibration_threshold,
            calibration_steepness=self._config.search.score_calibration_steepness,
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
        chunk_data = self._vector.get_chunk_by_id(chunk_id)
        if chunk_data:
            content = chunk_data.get("content")
            return str(content) if content is not None else None
        self._queue_reindex_for_chunks([chunk_id], "docstore lookup failed during content fetch")
        return None

    async def _search_vector(self, query_text: str, top_k: int, excluded_files: set[str] | None, docs_root: Path):
        if self._config.search.query_expansion_enabled:
            expanded_query = self._vector.expand_query(query_text)
        else:
            expanded_query = query_text

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self._vector.search, expanded_query, top_k, excluded_files, docs_root
        )
        logger.info(f"Vector search returned {len(results)} results with chunk_ids: {[r['chunk_id'] for r in results[:3]]}")
        return results

    async def _search_keyword(self, query_text: str, top_k: int, excluded_files: set[str] | None, docs_root: Path):
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self._keyword.search, query_text, top_k, excluded_files, docs_root
        )
        logger.info(f"Keyword search returned {len(results)} results with chunk_ids: {[r['chunk_id'] for r in results[:3]]}")
        return results

    async def _search_code(self, query_text: str, top_k: int):
        if self._code is None:
            return []

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self._code.search, query_text, top_k
        )
        logger.info(f"Code search returned {len(results)} results")
        return results

    def _get_graph_neighbors(self, doc_ids: list[str]):
        neighbors = set()
        for doc_id in doc_ids:
            doc_neighbors = self._graph.get_neighbors(doc_id, depth=1)
            neighbors.update(doc_neighbors)
        logger.info(f"Graph traversal for {doc_ids[:3]} returned {len(neighbors)} neighbors: {list(neighbors)[:5]}")
        return list(neighbors)

    def _apply_community_boost(
        self,
        fused: list[tuple[str, float]],
        seed_doc_ids: set[str],
        chunk_id_to_doc_id: dict[str, str],
    ) -> list[tuple[str, float]]:
        chunk_doc_ids = []
        for chunk_id, _ in fused:
            doc_id = chunk_id_to_doc_id.get(chunk_id)
            if doc_id is None:
                doc_id = chunk_id.rsplit("_chunk_", 1)[0] if "_chunk_" in chunk_id else chunk_id
            chunk_doc_ids.append(doc_id)

        boosts = self._graph.boost_by_community(
            chunk_doc_ids,
            seed_doc_ids,
            self._config.search.community_boost_factor,
        )

        boosted = []
        for (chunk_id, score), doc_id in zip(fused, chunk_doc_ids):
            boost = boosts.get(doc_id, 1.0)
            # Clamp to [0, 1] since scores are calibrated confidence values
            boosted.append((chunk_id, min(1.0, score * boost)))

        return sorted(boosted, key=lambda x: x[1], reverse=True)

    def _collect_modified_times(self, doc_ids: set[str]):
        modified_times = {}
        docs_path = self._documents_path

        for doc_id in doc_ids:
            # doc_id is now relative path without extension (e.g., "dir/subdir/filename")
            md_file = docs_path / f"{doc_id}.md"
            if md_file.exists():
                modified_times[doc_id] = md_file.stat().st_mtime
            else:
                markdown_file = docs_path / f"{doc_id}.markdown"
                if markdown_file.exists():
                    modified_times[doc_id] = markdown_file.stat().st_mtime

        return modified_times

    def _queue_reindex_for_chunks(self, chunk_ids: list[str], reason: str):
        doc_ids = {
            extract_doc_id_from_chunk_id(chunk_id)
            for chunk_id in chunk_ids
            if chunk_id
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
                logger.info("Reindexed %d documents after missing chunk recovery", reindexed)
            except TimeoutError as e:
                logger.warning("Reindex persist skipped (lock busy): %s", e)

    def get_documents(self, doc_ids: list[str]):
        docs_path = self._documents_path
        documents = []

        for doc_id in doc_ids:
            # doc_id is now relative path without extension
            md_file = docs_path / f"{doc_id}.md"
            if not md_file.exists():
                md_file = docs_path / f"{doc_id}.markdown"

            if md_file.exists():
                try:
                    from src.parsers.dispatcher import dispatch_parser
                    parser = dispatch_parser(str(md_file), self._config)
                    document = parser.parse(str(md_file))
                    documents.append(document)
                except Exception:
                    pass

        return documents

    def _get_chunks(self, chunk_ids: list[str]) -> list[dict]:
        chunks = []
        logger.info(f"_get_chunks called with {len(chunk_ids)} chunk_ids: {chunk_ids[:3] if len(chunk_ids) > 3 else chunk_ids}")

        for chunk_id in chunk_ids:
            chunk_data = self._vector.get_chunk_by_id(chunk_id)
            if chunk_data:
                chunks.append(chunk_data)
                logger.info(f"Successfully retrieved chunk {chunk_id}")
            else:
                logger.warning(f"Failed to retrieve chunk {chunk_id}")
                self._queue_reindex_for_chunks([chunk_id], "docstore lookup failed during chunk fetch")

        logger.info(f"_get_chunks returning {len(chunks)} chunks")
        return chunks

    async def query_with_hypothesis(
        self,
        hypothesis: str,
        top_k: int = 10,
        top_n: int = 5,
        excluded_files: set[str] | None = None,
    ) -> tuple[list[ChunkResult], CompressionStats, SearchStrategyStats]:
        if not hypothesis or not hypothesis.strip():
            return [], CompressionStats(
                original_count=0,
                after_threshold=0,
                after_content_dedup=0,
                after_ngram_dedup=0,
                after_dedup=0,
                after_doc_limit=0,
                clusters_merged=0,
            ), SearchStrategyStats()

        if not self._config.search.hyde_enabled:
            logger.warning("HyDE search disabled in config, falling back to regular query")
            return await self.query(hypothesis, top_k, top_n, excluded_files=excluded_files)

        docs_root = self._documents_path

        from src.search.hyde import search_with_hypothesis
        loop = asyncio.get_event_loop()
        vector_results = await loop.run_in_executor(
            None,
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

        weights: dict[str, float] = {"semantic": 1.0}

        fused = self._apply_score_pipeline(strategy_results, weights)

        pipeline = self._get_pipeline()

        final, compression_stats = pipeline.process(
            fused,
            self._get_chunk_embedding,
            self._get_chunk_content,
            hypothesis,
            top_n,
            None,
        )

        # Build strategy stats (HyDE only uses semantic search)
        strategy_stats = SearchStrategyStats(
            vector_count=len(vector_results),
        )

        chunk_results = []
        missing_chunk_ids: list[str] = []
        for chunk_id, score in final:
            chunk_data = self._vector.get_chunk_by_id(chunk_id)
            if chunk_data:
                metadata = chunk_data.get("metadata", {})
                parent_chunk_id = metadata.get("parent_chunk_id") if isinstance(metadata, dict) else None
                parent_content = None
                if parent_chunk_id:
                    parent_content = self._vector.get_parent_content(parent_chunk_id)

                chunk_results.append(ChunkResult(
                    chunk_id=chunk_id,
                    doc_id=str(chunk_data.get("doc_id", "")),
                    score=score,
                    header_path=str(chunk_data.get("header_path", "")),
                    file_path=str(chunk_data.get("file_path", "")),
                    content=str(chunk_data.get("content", "")),
                    parent_chunk_id=parent_chunk_id,
                    parent_content=parent_content,
                ))
            else:
                missing_chunk_ids.append(chunk_id)
                chunk_results.append(ChunkResult(
                    chunk_id=chunk_id,
                    doc_id=extract_doc_id_from_chunk_id(chunk_id),
                    score=score,
                    header_path="",
                    file_path="",
                    content="",
                ))

        if missing_chunk_ids:
            self._queue_reindex_for_chunks(missing_chunk_ids, "docstore lookup failed")

        return chunk_results, compression_stats, strategy_stats
