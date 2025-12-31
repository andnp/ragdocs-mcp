import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from src.models import Chunk, Document

logger = logging.getLogger(__name__)


STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "he", "she", "him", "her", "his", "hers", "we", "us", "our", "you",
    "your", "i", "me", "my", "who", "what", "which", "where", "when",
    "how", "why", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "not", "only", "same", "so", "than",
    "too", "very", "just", "also", "now", "here", "there", "then",
})


def _cosine_similarity(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class EmbeddingModel(Protocol):
    def get_text_embedding(self, text: str) -> list[float]: ...


class VectorIndex:
    def __init__(self, embedding_model_name: str = "BAAI/bge-small-en-v1.5", embedding_model: EmbeddingModel | None = None):
        self._embedding_model_name = embedding_model_name
        self._embedding_model: EmbeddingModel | None = embedding_model
        self._model_lock = threading.Lock()
        self._model_loaded = embedding_model is not None

        # If embedding model was provided, set it in Settings immediately
        if embedding_model is not None:
            from llama_index.core import Settings
            Settings.embed_model = embedding_model

        self._doc_id_to_node_ids: dict[str, list[str]] = {}
        self._chunk_id_to_node_id: dict[str, str] = {}
        self._vector_store = None
        self._index = None
        self._concept_vocabulary: dict[str, list[float]] = {}
        self._term_counts: dict[str, int] = {}  # Track term frequencies for incremental updates
        self._pending_terms: set[str] = set()  # Terms that need embedding

    def _ensure_model_loaded(self) -> None:
        if self._model_loaded:
            return
        with self._model_lock:
            if self._model_loaded:
                return
            from llama_index.core import Settings
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            self._embedding_model = HuggingFaceEmbedding(model_name=self._embedding_model_name)
            Settings.embed_model = self._embedding_model
            self._model_loaded = True

    def warm_up(self) -> None:
        self._ensure_model_loaded()

    def add(self, document: Document) -> None:
        from llama_index.core import Document as LlamaDocument

        if document.chunks:
            for chunk in document.chunks:
                self.add_chunk(chunk)
            return

        self._ensure_model_loaded()
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

        node_id = llama_doc.id_
        self._doc_id_to_node_ids[document.id] = [node_id]

        self._index.insert_nodes([llama_doc])

    def add_chunk(self, chunk: Chunk) -> None:
        from llama_index.core import Document as LlamaDocument

        self._ensure_model_loaded()
        if self._index is None:
            self._initialize_index()

        assert self._index is not None

        embedding_text = f"{chunk.header_path}\n\n{chunk.content}" if chunk.header_path else chunk.content

        llama_doc = LlamaDocument(
            text=embedding_text,
            metadata={
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "header_path": chunk.header_path,
                "file_path": chunk.file_path,
                "tags": chunk.metadata.get("tags", []),
                "links": chunk.metadata.get("links", []),
            },
            id_=chunk.chunk_id,
        )

        node_id = llama_doc.id_
        self._chunk_id_to_node_id[chunk.chunk_id] = node_id

        if chunk.doc_id not in self._doc_id_to_node_ids:
            self._doc_id_to_node_ids[chunk.doc_id] = []
        self._doc_id_to_node_ids[chunk.doc_id].append(node_id)

        self._index.insert_nodes([llama_doc])

        # Register terms for incremental vocabulary update
        self.register_document_terms(embedding_text)

    def remove(self, document_id: str) -> None:
        if self._index is None or document_id not in self._doc_id_to_node_ids:
            return

        del self._doc_id_to_node_ids[document_id]

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if self._index is None or not query.strip():
            return []

        self._ensure_model_loaded()
        retriever = self._index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        results = []
        for node in nodes:
            chunk_id = node.metadata.get("chunk_id")

            node_text = node.node.get_content() if hasattr(node.node, "get_content") else getattr(node.node, "text", "")

            if chunk_id:
                results.append({
                    "chunk_id": chunk_id,
                    "doc_id": node.metadata.get("doc_id"),
                    "score": node.score if hasattr(node, "score") else 1.0,
                    "header_path": node.metadata.get("header_path", ""),
                    "file_path": node.metadata.get("file_path", ""),
                    "content": node_text,
                    "metadata": node.metadata,
                })
            else:
                doc_id = node.metadata.get("doc_id")
                if doc_id:
                    results.append({
                        "chunk_id": doc_id,
                        "doc_id": doc_id,
                        "score": node.score if hasattr(node, "score") else 1.0,
                        "header_path": "",
                        "file_path": node.metadata.get("file_path", ""),
                        "content": node_text,
                        "metadata": node.metadata,
                    })

        return results

    # REVIEW [MED] Logging: Using INFO level for routine operations like chunk retrieval.
    # These should be DEBUG level - INFO is too verbose for per-chunk logging that
    # happens many times per query. The warning for index=None is appropriate.
    def get_chunk_by_id(self, chunk_id: str):
        if self._index is None:
            logger.warning(f"get_chunk_by_id({chunk_id}): index is None")
            return None

        logger.debug(f"get_chunk_by_id({chunk_id}): attempting direct docstore lookup")

        # Try direct lookup in docstore first
        try:
            docstore = self._index.docstore
            node = docstore.get_document(chunk_id)

            if node:
                logger.debug(f"get_chunk_by_id({chunk_id}): found in docstore")
                node_text = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")

                return {
                    "chunk_id": chunk_id,
                    "doc_id": node.metadata.get("doc_id"),
                    "score": 1.0,
                    "header_path": node.metadata.get("header_path", ""),
                    "file_path": node.metadata.get("file_path", ""),
                    "content": node_text,
                    "metadata": node.metadata,
                }
            else:
                logger.warning(f"get_chunk_by_id({chunk_id}): not found in docstore")
        except Exception as e:
            logger.warning(f"get_chunk_by_id({chunk_id}): docstore lookup failed: {e}")

        logger.debug(f"get_chunk_by_id({chunk_id}): falling back to retriever search")
        # Fallback to retriever search
        retriever = self._index.as_retriever(similarity_top_k=100)
        nodes = retriever.retrieve(chunk_id)

        for node in nodes:
            if node.metadata.get("chunk_id") == chunk_id:
                node_text = node.node.get_content() if hasattr(node.node, "get_content") else getattr(node.node, "text", "")

                return {
                    "chunk_id": chunk_id,
                    "doc_id": node.metadata.get("doc_id"),
                    "score": 1.0,
                    "header_path": node.metadata.get("header_path", ""),
                    "file_path": node.metadata.get("file_path", ""),
                    "content": node_text,
                    "metadata": node.metadata,
                }

        return None

    def get_chunk_ids_for_document(self, doc_id: str) -> list[str]:
        return self._doc_id_to_node_ids.get(doc_id, [])

    def get_document_ids(self) -> list[str]:
        return list(self._doc_id_to_node_ids.keys())

    def build_concept_vocabulary(
        self,
        min_term_length: int = 3,
        max_terms: int = 10000,
    ) -> None:
        """Build concept vocabulary from scratch. Prefer update_vocabulary_incremental() for efficiency."""
        if self._index is None:
            return

        term_counts: dict[str, int] = {}
        docstore = self._index.docstore

        for chunk_id in self._chunk_id_to_node_id:
            try:
                node = docstore.get_document(chunk_id)
                if node is None:
                    continue
                text = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")
                tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*\b', text.lower())
                for token in tokens:
                    if len(token) >= min_term_length and token not in STOPWORDS:
                        term_counts[token] = term_counts.get(token, 0) + 1
            except Exception:
                continue

        # Store term counts for future incremental updates
        self._term_counts = term_counts

        sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
        top_terms = [term for term, _ in sorted_terms[:max_terms]]

        self._ensure_model_loaded()
        self._concept_vocabulary = {}
        for term in top_terms:
            try:
                assert self._embedding_model is not None
                embedding = self._embedding_model.get_text_embedding(term)
                self._concept_vocabulary[term] = embedding
            except Exception:
                continue

        self._pending_terms.clear()
        logger.info(f"Built concept vocabulary with {len(self._concept_vocabulary)} terms")

    def extract_terms_from_text(self, text: str, min_term_length: int = 3) -> dict[str, int]:
        """Extract terms and their counts from text."""
        term_counts: dict[str, int] = {}
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*\b', text.lower())
        for token in tokens:
            if len(token) >= min_term_length and token not in STOPWORDS:
                term_counts[token] = term_counts.get(token, 0) + 1
        return term_counts

    def register_document_terms(self, text: str, min_term_length: int = 3) -> None:
        """Register terms from a newly indexed document for later vocabulary update."""
        doc_terms = self.extract_terms_from_text(text, min_term_length)
        for term, count in doc_terms.items():
            self._term_counts[term] = self._term_counts.get(term, 0) + count
            # Mark as pending if not already in vocabulary
            if term not in self._concept_vocabulary:
                self._pending_terms.add(term)

    def update_vocabulary_incremental(
        self,
        max_terms: int = 10000,
        batch_size: int = 100,
    ) -> int:
        """
        Incrementally update vocabulary by embedding only new high-frequency terms.
        Returns the number of new terms embedded.
        """
        if not self._pending_terms:
            return 0

        # Get top terms by frequency that aren't in vocabulary yet
        pending_with_counts = [
            (term, self._term_counts.get(term, 0))
            for term in self._pending_terms
        ]
        pending_with_counts.sort(key=lambda x: x[1], reverse=True)

        # Only embed terms that would make it into top max_terms
        current_vocab_size = len(self._concept_vocabulary)
        if current_vocab_size >= max_terms:
            # Vocabulary is full - only add terms with higher freq than lowest in vocab
            if self._concept_vocabulary:
                min_vocab_freq = min(
                    self._term_counts.get(t, 0)
                    for t in self._concept_vocabulary
                )
                pending_with_counts = [
                    (t, c) for t, c in pending_with_counts if c > min_vocab_freq
                ]

        # Limit batch size to avoid long blocking
        terms_to_embed = [t for t, _ in pending_with_counts[:batch_size]]

        if not terms_to_embed:
            self._pending_terms.clear()
            return 0

        self._ensure_model_loaded()
        assert self._embedding_model is not None

        embedded_count = 0
        for term in terms_to_embed:
            try:
                embedding = self._embedding_model.get_text_embedding(term)
                self._concept_vocabulary[term] = embedding
                self._pending_terms.discard(term)
                embedded_count += 1
            except Exception:
                self._pending_terms.discard(term)
                continue

        if embedded_count > 0:
            logger.debug(f"Incrementally added {embedded_count} terms to vocabulary")

        return embedded_count

    def get_pending_vocabulary_count(self) -> int:
        """Return count of terms waiting to be embedded."""
        return len(self._pending_terms)

    def get_text_embedding(self, text: str) -> list[float]:
        self._ensure_model_loaded()
        assert self._embedding_model is not None
        return self._embedding_model.get_text_embedding(text)

    def is_ready(self) -> bool:
        return self._index is not None

    def model_ready(self) -> bool:
        """Check if embedding model is loaded and ready for queries."""
        return self._model_loaded

    def get_embedding_for_chunk(self, chunk_id: str) -> list[float] | None:
        if self._index is None:
            return None
        try:
            docstore = self._index.docstore
            node = docstore.get_document(chunk_id)
            if node is None:
                return None
            embedding = getattr(node, "embedding", None)
            if embedding is not None:
                return list(embedding)
            text = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")
            if not text:
                return None
            return self.get_text_embedding(text)
        except Exception:
            return None

    def expand_query(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.5,
    ) -> str:
        if not self._concept_vocabulary:
            return query

        self._ensure_model_loaded()
        assert self._embedding_model is not None
        query_embedding = np.array(
            self._embedding_model.get_text_embedding(query),
            dtype=np.float64,
        )

        similarities: list[tuple[str, float]] = []
        for term, term_emb in self._concept_vocabulary.items():
            term_vec = np.array(term_emb, dtype=np.float64)
            sim = _cosine_similarity(query_embedding, term_vec)
            if sim >= similarity_threshold:
                similarities.append((term, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        expansion_terms = [term for term, _ in similarities[:top_k]]

        query_tokens = set(query.lower().split())
        new_terms = [t for t in expansion_terms if t not in query_tokens]

        if new_terms:
            expanded = f"{query} {' '.join(new_terms)}"
            logger.debug(f"Query expanded: '{query}' -> '{expanded}'")
            return expanded
        return query

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

        mapping_file = path / "doc_id_mapping.json"
        with open(mapping_file, "w") as f:
            json.dump(self._doc_id_to_node_ids, f)

        chunk_mapping_file = path / "chunk_id_mapping.json"
        with open(chunk_mapping_file, "w") as f:
            json.dump(self._chunk_id_to_node_id, f)

        vocab_file = path / "concept_vocabulary.json"
        with open(vocab_file, "w") as f:
            json.dump(self._concept_vocabulary, f)

        term_counts_file = path / "term_counts.json"
        with open(term_counts_file, "w") as f:
            json.dump(self._term_counts, f)

    def load(self, path: Path) -> None:
        from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
        from llama_index.vector_stores.faiss import FaissVectorStore

        # Check if index exists and has required files
        docstore_path = path / "docstore.json"
        if not path.exists() or not docstore_path.exists():
            self._initialize_index()
            return

        # Ensure embedding model is loaded before calling load_index_from_storage
        # LlamaIndex requires Settings.embed_model to be set
        self._ensure_model_loaded()

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

        mapping_file = path / "doc_id_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, "r") as f:
                self._doc_id_to_node_ids = json.load(f)
        else:
            self._doc_id_to_node_ids = {}

        chunk_mapping_file = path / "chunk_id_mapping.json"
        if chunk_mapping_file.exists():
            with open(chunk_mapping_file, "r") as f:
                self._chunk_id_to_node_id = json.load(f)
        else:
            self._chunk_id_to_node_id = {}

        vocab_file = path / "concept_vocabulary.json"
        if vocab_file.exists():
            with open(vocab_file, "r") as f:
                self._concept_vocabulary = json.load(f)
        else:
            self._concept_vocabulary = {}

        term_counts_file = path / "term_counts.json"
        if term_counts_file.exists():
            with open(term_counts_file, "r") as f:
                self._term_counts = json.load(f)
        else:
            self._term_counts = {}

        self._pending_terms.clear()

    def _initialize_index(self) -> None:
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.vector_stores.faiss import FaissVectorStore

        self._ensure_model_loaded()

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

    # IndexProtocol methods

    def add_document(self, doc_id: str, content: str, metadata: dict[str, Any]) -> None:
        from llama_index.core import Document as LlamaDocument

        self._ensure_model_loaded()
        if self._index is None:
            self._initialize_index()

        assert self._index is not None

        llama_doc = LlamaDocument(
            text=content,
            metadata={"doc_id": doc_id, **metadata},
            id_=doc_id,
        )

        node_id = llama_doc.id_
        self._doc_id_to_node_ids[doc_id] = [node_id]
        self._index.insert_nodes([llama_doc])

    def remove_document(self, doc_id: str) -> None:
        self.remove(doc_id)

    def clear(self) -> None:
        self._doc_id_to_node_ids = {}
        self._chunk_id_to_node_id = {}
        self._vector_store = None
        self._index = None
        self._concept_vocabulary = {}
        self._term_counts = {}
        self._pending_terms.clear()

    def save(self, path: Path) -> None:
        self.persist(path)

    def __len__(self) -> int:
        return len(self._doc_id_to_node_ids)
