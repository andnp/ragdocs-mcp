import asyncio
import re
from collections.abc import AsyncIterator
from pathlib import Path

from src.config import Config
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.indexing.manager import IndexManager
from src.models import ChunkResult, CompressionStats
from src.search.fusion import fuse_results
from src.search.pipeline import SearchPipeline, SearchPipelineConfig


class SearchOrchestrator:
    def __init__(
        self,
        vector_index: VectorIndex,
        keyword_index: KeywordIndex,
        graph_store: GraphStore,
        config: Config,
        index_manager: IndexManager,
    ):
        self._vector = vector_index
        self._keyword = keyword_index
        self._graph = graph_store
        self._config = config
        self._index_manager = index_manager
        self._synthesizer = None
        self._pipeline: SearchPipeline | None = None

    def _get_pipeline(self) -> SearchPipeline:
        if self._pipeline is None:
            pipeline_config = SearchPipelineConfig(
                min_confidence=self._config.search.min_confidence,
                max_chunks_per_doc=self._config.search.max_chunks_per_doc,
                dedup_enabled=self._config.search.dedup_enabled,
                dedup_threshold=self._config.search.dedup_similarity_threshold,
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
                after_dedup=0,
                after_doc_limit=0,
                clusters_merged=0,
            )

        results = await asyncio.gather(
            self._search_vector(query_text, top_k),
            self._search_keyword(query_text, top_k),
        )

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

        graph_neighbors = self._get_graph_neighbors(list(all_doc_ids))

        # Convert graph document IDs to chunk IDs
        graph_chunk_ids = []
        for doc_id in graph_neighbors:
            chunk_ids_for_doc = self._vector.get_chunk_ids_for_document(doc_id)
            graph_chunk_ids.extend(chunk_ids_for_doc)

        results_dict = {
            "semantic": [r["chunk_id"] for r in vector_results],
            "keyword": [r["chunk_id"] for r in keyword_results],
            "graph": graph_chunk_ids,
        }

        modified_times = self._collect_modified_times(all_doc_ids | set(graph_neighbors))

        weights = {
            "semantic": self._config.search.semantic_weight,
            "keyword": self._config.search.keyword_weight,
            "graph": 1.0,
        }

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

        final, compression_stats = pipeline.process(
            fused,
            self._get_chunk_embedding,
            self._get_chunk_content,
            query_text,
            top_n,
        )

        chunk_results = []
        for chunk_id, score in final:
            chunk_data = self._vector.get_chunk_by_id(chunk_id)
            if chunk_data:
                chunk_results.append(ChunkResult(
                    chunk_id=chunk_id,
                    doc_id=chunk_data.get("doc_id", ""),
                    score=score,
                    header_path=chunk_data.get("header_path", ""),
                    file_path=chunk_data.get("file_path", ""),
                    content=chunk_data.get("content", ""),
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

    def _get_chunk_embedding(self, chunk_id: str) -> list[float] | None:
        return self._vector.get_embedding_for_chunk(chunk_id)

    def _get_chunk_content(self, chunk_id: str) -> str | None:
        chunk_data = self._vector.get_chunk_by_id(chunk_id)
        if chunk_data:
            return chunk_data.get("content")
        return None

    # REVIEW [LOW] Logging: Module-level logger exists but this method creates a local
    # logger import. Remove the local import and use the module-level logger instead.
    async def _search_vector(self, query_text: str, top_k: int):
        import logging
        logger = logging.getLogger(__name__)

        expanded_query = self._vector.expand_query(query_text)

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self._vector.search, expanded_query, top_k
        )
        logger.info(f"Vector search returned {len(results)} results with chunk_ids: {[r['chunk_id'] for r in results[:3]]}")
        return results

    async def _search_keyword(self, query_text: str, top_k: int):
        import logging
        logger = logging.getLogger(__name__)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self._keyword.search, query_text, top_k
        )
        logger.info(f"Keyword search returned {len(results)} results with chunk_ids: {[r['chunk_id'] for r in results[:3]]}")
        return results

    # REVIEW [LOW] Logging: Same issue - redundant local logger import.
    def _get_graph_neighbors(self, doc_ids: list[str]):
        import logging
        logger = logging.getLogger(__name__)
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

    async def synthesize_answer(self, query: str, chunk_ids: list[str]):
        from llama_index.core import Settings, get_response_synthesizer
        from llama_index.core.llms import MockLLM
        from llama_index.core.prompts import PromptTemplate
        from llama_index.core.schema import NodeWithScore, TextNode

        if not chunk_ids:
            return "No relevant documents found for your query."

        chunks = self._get_chunks(chunk_ids)

        if not chunks:
            return "Could not retrieve content for the matched results."

        TEXT_QA_TEMPLATE = PromptTemplate(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Answer the question: {query_str}\n"
        )

        nodes = [
            NodeWithScore(
                node=TextNode(
                    text=chunk["content"],
                    metadata={
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"],
                        "header_path": chunk.get("header_path", ""),
                        "file_path": chunk.get("file_path", ""),
                    },
                ),
                score=chunk.get("score", 1.0),
            )
            for chunk in chunks
        ]

        if self._synthesizer is None:
            Settings.context_window = 32768
            try:
                self._synthesizer = get_response_synthesizer(text_qa_template=TEXT_QA_TEMPLATE)
            except ImportError:
                self._synthesizer = get_response_synthesizer(llm=MockLLM(), text_qa_template=TEXT_QA_TEMPLATE)

        response = await asyncio.get_event_loop().run_in_executor(
            None, self._synthesizer.synthesize, query, nodes
        )

        answer = str(response)
        answer = self._clean_answer(answer)
        return answer

    async def synthesize_answer_stream(
        self,
        query: str,
        chunk_ids: list[str]
    ) -> AsyncIterator[dict[str, str | dict]]:
        from llama_index.core import Settings, get_response_synthesizer
        from llama_index.core.llms import MockLLM
        from llama_index.core.prompts import PromptTemplate
        from llama_index.core.schema import NodeWithScore, TextNode

        if not chunk_ids:
            yield {
                "event": "error",
                "data": {"message": "No relevant documents found for your query."}
            }
            return

        chunks = self._get_chunks(chunk_ids)

        if not chunks:
            yield {
                "event": "error",
                "data": {"message": "Could not retrieve content for the matched results."}
            }
            return

        yield {
            "event": "start",
            "data": {"chunk_count": len(chunks)}
        }

        TEXT_QA_TEMPLATE = PromptTemplate(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Answer the question: {query_str}\n"
        )

        nodes = [
            NodeWithScore(
                node=TextNode(
                    text=chunk["content"],
                    metadata={
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"],
                        "header_path": chunk.get("header_path", ""),
                        "file_path": chunk.get("file_path", ""),
                    },
                ),
                score=chunk.get("score", 1.0),
            )
            for chunk in chunks
        ]

        if self._synthesizer is None:
            Settings.context_window = 32768
            try:
                self._synthesizer = get_response_synthesizer(text_qa_template=TEXT_QA_TEMPLATE)
            except ImportError:
                self._synthesizer = get_response_synthesizer(llm=MockLLM(), text_qa_template=TEXT_QA_TEMPLATE)

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, self._synthesizer.synthesize, query, nodes
            )

            answer_text = str(response)
            answer_text = self._clean_answer(answer_text)

            yield {
                "event": "chunk",
                "data": {"text": answer_text}
            }

            yield {
                "event": "done",
                "data": {"total_length": len(answer_text)}
            }

        except Exception as e:
            yield {
                "event": "error",
                "data": {"message": f"Synthesis failed: {str(e)}"}
            }

    # REVIEW [LOW] Logging: Same issue - redundant local logger import.
    def _get_chunks(self, chunk_ids: list[str]) -> list[dict]:
        import logging
        logger = logging.getLogger(__name__)

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

    def _clean_answer(self, answer: str) -> str:
        contamination_patterns = [
            r"given the context information and not prior knowledge[,.]?",
            r"based on the context[,.]?",
            r"according to the context[,.]?",
            r"from the context[,.]?",
        ]

        cleaned = answer
        for pattern in contamination_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip()

        return cleaned
