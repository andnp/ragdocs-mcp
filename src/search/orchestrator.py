import asyncio
import logging
from pathlib import Path

from src.config import Config
from src.indices.code import CodeIndex
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.indexing.manager import IndexManager
from src.models import ChunkResult, CompressionStats
from src.search.classifier import classify_query, get_adaptive_weights
from src.search.fusion import fuse_results
from src.search.pipeline import SearchPipeline, SearchPipelineConfig

logger = logging.getLogger(__name__)


class SearchOrchestrator:
    def __init__(
        self,
        vector_index: VectorIndex,
        keyword_index: KeywordIndex,
        graph_store: GraphStore,
        config: Config,
        index_manager: IndexManager,
        code_index: CodeIndex | None = None,
    ):
        self._vector = vector_index
        self._keyword = keyword_index
        self._graph = graph_store
        self._config = config
        self._index_manager = index_manager
        self._code = code_index
        self._pipeline: SearchPipeline | None = None

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
                parent_retrieval_enabled=self._config.chunking.parent_retrieval_enabled,
                rerank_enabled=self._config.search.rerank_enabled,
                rerank_model=self._config.search.rerank_model,
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
    ) -> tuple[list[ChunkResult], CompressionStats]:
        if not query_text or not query_text.strip():
            return [], CompressionStats(
                original_count=0,
                after_threshold=0,
                after_content_dedup=0,
                after_ngram_dedup=0,
                after_dedup=0,
                after_doc_limit=0,
                clusters_merged=0,
            )

        search_tasks = [
            self._search_vector(query_text, top_k),
            self._search_keyword(query_text, top_k),
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

        graph_neighbors = self._get_graph_neighbors(list(all_doc_ids))

        # Convert graph document IDs to chunk IDs
        graph_chunk_ids = []
        for doc_id in graph_neighbors:
            chunk_ids_for_doc = self._vector.get_chunk_ids_for_document(doc_id)
            graph_chunk_ids.extend(chunk_ids_for_doc)

        results_dict: dict[str, list[str]] = {
            "semantic": [r["chunk_id"] for r in vector_results],
            "keyword": [r["chunk_id"] for r in keyword_results],
            "graph": graph_chunk_ids,
        }

        if code_search_enabled and code_results:
            results_dict["code"] = [r["chunk_id"] for r in code_results]

        modified_times = self._collect_modified_times(all_doc_ids | set(graph_neighbors))

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

        fused = fuse_results(
            results_dict,
            self._config.search.rrf_k_constant,
            weights,
            modified_times,
        )

        if pipeline_config is not None:
            pipeline = SearchPipeline(pipeline_config)
        else:
            pipeline = self._get_pipeline()

        query_embedding = None
        if self._config.search.mmr_enabled or (
            pipeline_config is not None and pipeline_config.mmr_enabled
        ):
            query_embedding = self._vector.get_text_embedding(query_text)

        final, compression_stats = pipeline.process(
            fused,
            self._get_chunk_embedding,
            self._get_chunk_content,
            query_text,
            top_n,
            query_embedding,
        )

        # Parent expansion: if enabled, expand child chunks to parent chunks
        parent_retrieval_enabled = self._config.chunking.parent_retrieval_enabled
        if pipeline_config is not None and pipeline_config.parent_retrieval_enabled:
            parent_retrieval_enabled = True

        if parent_retrieval_enabled:
            final = self._expand_to_parents(final)

        chunk_results = []
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
                chunk_results.append(ChunkResult(
                    chunk_id=chunk_id,
                    doc_id=chunk_id.rsplit("_chunk_", 1)[0] if "_chunk_" in chunk_id else "",
                    score=score,
                    header_path="",
                    file_path="",
                    content="",
                ))

        return chunk_results, compression_stats

    def _expand_to_parents(
        self, results: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        seen_parents: set[str] = set()
        expanded: list[tuple[str, float]] = []

        for chunk_id, score in results:
            chunk_data = self._vector.get_chunk_by_id(chunk_id)
            if not chunk_data:
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

    def _get_chunk_embedding(self, chunk_id: str) -> list[float] | None:
        return self._vector.get_embedding_for_chunk(chunk_id)

    def _get_chunk_content(self, chunk_id: str) -> str | None:
        chunk_data = self._vector.get_chunk_by_id(chunk_id)
        if chunk_data:
            content = chunk_data.get("content")
            return str(content) if content is not None else None
        return None

    async def _search_vector(self, query_text: str, top_k: int):
        expanded_query = self._vector.expand_query(query_text)

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self._vector.search, expanded_query, top_k
        )
        logger.info(f"Vector search returned {len(results)} results with chunk_ids: {[r['chunk_id'] for r in results[:3]]}")
        return results

    async def _search_keyword(self, query_text: str, top_k: int):
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self._keyword.search, query_text, top_k
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

    def _collect_modified_times(self, doc_ids: set[str]):
        modified_times = {}
        docs_path = Path(self._config.indexing.documents_path)

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

    def get_documents(self, doc_ids: list[str]):
        docs_path = Path(self._config.indexing.documents_path)
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

        logger.info(f"_get_chunks returning {len(chunks)} chunks")
        return chunks
