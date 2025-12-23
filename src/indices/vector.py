from pathlib import Path
from typing import Protocol, cast

from llama_index.core import (
    Document as LlamaDocument,
)
from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from src.models import Document


class EmbeddingModel(Protocol):
    def get_text_embedding(self, text: str) -> list[float]: ...


class VectorIndex:
    def __init__(self, embedding_model_name: str = "BAAI/bge-small-en-v1.5"):
        self._embedding_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        Settings.embed_model = self._embedding_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

        self._node_parser = MarkdownNodeParser()
        self._doc_id_to_node_ids: dict[str, list[str]] = {}
        self._vector_store: FaissVectorStore | None = None
        self._index: VectorStoreIndex | None = None

    def add(self, document: Document) -> None:
        if self._index is None:
            self._initialize_index()

        assert self._index is not None

        llama_doc = LlamaDocument(
            text=document.content,
            metadata={
                "doc_id": document.id,
                "file_path": document.file_path,
                "tags": document.tags,
                "links": document.links,
            },
            id_=document.id,
        )

        nodes = self._node_parser.get_nodes_from_documents([llama_doc])

        for node in nodes:
            node.metadata["doc_id"] = document.id

        node_ids = [node.node_id for node in nodes]
        self._doc_id_to_node_ids[document.id] = node_ids

        self._index.insert_nodes(nodes)

    def remove(self, document_id: str) -> None:
        if self._index is None or document_id not in self._doc_id_to_node_ids:
            return

        del self._doc_id_to_node_ids[document_id]

    def search(self, query: str, top_k: int = 10) -> list[str]:
        if self._index is None or not query.strip():
            return []

        retriever = self._index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        seen = set()
        results = []
        for node in nodes:
            doc_id = node.metadata.get("doc_id")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                results.append(doc_id)

        return results

    def persist(self, path: Path) -> None:
        if self._index is None:
            return

        path.mkdir(parents=True, exist_ok=True)

        storage_context = self._index.storage_context
        storage_context.persist(persist_dir=str(path))

        if self._vector_store is not None:
            import faiss
            faiss_path = path / "faiss_index.bin"
            faiss.write_index(self._vector_store._faiss_index, str(faiss_path))

        import json
        mapping_file = path / "doc_id_mapping.json"
        with open(mapping_file, "w") as f:
            json.dump(self._doc_id_to_node_ids, f)

    def load(self, path: Path) -> None:
        if not path.exists():
            self._initialize_index()
            return

        import faiss

        faiss_path = path / "faiss_index.bin"
        if faiss_path.exists():
            faiss_index = faiss.read_index(str(faiss_path))
        else:
            dimension = 384
            faiss_index = faiss.IndexFlatL2(dimension)

        self._vector_store = FaissVectorStore(faiss_index=faiss_index)

        storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store,
            persist_dir=str(path),
        )

        self._index = cast(VectorStoreIndex, load_index_from_storage(storage_context))

        import json
        mapping_file = path / "doc_id_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, "r") as f:
                self._doc_id_to_node_ids = json.load(f)
        else:
            self._doc_id_to_node_ids = {}

    def _initialize_index(self) -> None:
        import faiss

        dimension = 384
        faiss_index = faiss.IndexFlatL2(dimension)
        self._vector_store = FaissVectorStore(faiss_index=faiss_index)

        storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )

        self._index = VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
        )
