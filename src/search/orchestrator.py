import asyncio
from pathlib import Path

from llama_index.core import get_response_synthesizer
from llama_index.core.llms import MockLLM
from llama_index.core.schema import NodeWithScore, TextNode

from src.config import Config
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.indexing.manager import IndexManager
from src.search.fusion import fuse_results


class QueryOrchestrator:
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

    async def query(self, query_text: str, top_k: int):
        results = await asyncio.gather(
            self._search_vector(query_text, top_k),
            self._search_keyword(query_text, top_k),
        )

        vector_results = results[0]
        keyword_results = results[1]

        all_doc_ids = set(vector_results) | set(keyword_results)
        graph_neighbors = self._get_graph_neighbors(list(all_doc_ids))

        results_dict = {
            "semantic": vector_results,
            "keyword": keyword_results,
            "graph": graph_neighbors,
        }

        modified_times = self._collect_modified_times(
            set(vector_results) | set(keyword_results) | set(graph_neighbors)
        )

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

        return [doc_id for doc_id, _ in fused[:top_k]]

    async def _search_vector(self, query_text: str, top_k: int):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._vector.search, query_text, top_k
        )

    async def _search_keyword(self, query_text: str, top_k: int):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._keyword.search, query_text, top_k
        )

    def _get_graph_neighbors(self, doc_ids: list[str]):
        neighbors = set()
        for doc_id in doc_ids:
            doc_neighbors = self._graph.get_neighbors(doc_id, depth=1)
            neighbors.update(doc_neighbors)
        return list(neighbors)

    def _collect_modified_times(self, doc_ids: set[str]):
        modified_times = {}
        docs_path = Path(self._config.indexing.documents_path)

        for doc_id in doc_ids:
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

    async def synthesize_answer(self, query: str, doc_ids: list[str]):
        if not doc_ids:
            return "No relevant documents found for your query."

        documents = self.get_documents(doc_ids)

        if not documents:
            return "Could not retrieve document content for the matched results."

        nodes = [
            NodeWithScore(
                node=TextNode(
                    text=doc.content,
                    metadata={"doc_id": doc.id, "file_path": doc.file_path},
                ),
                score=1.0,
            )
            for doc in documents
        ]

        if self._synthesizer is None:
            try:
                self._synthesizer = get_response_synthesizer()
            except ImportError:
                self._synthesizer = get_response_synthesizer(llm=MockLLM())

        response = await asyncio.get_event_loop().run_in_executor(
            None, self._synthesizer.synthesize, query, nodes
        )

        return str(response)
