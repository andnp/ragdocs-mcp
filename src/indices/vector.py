from __future__ import annotations

import json
import logging
import re
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

if TYPE_CHECKING:
    from src.storage.db import DatabaseManager

import numpy as np

from src.models import Chunk, Document
from src.search.types import SearchResultDict
from src.utils.atomic_io import atomic_write_json, fsync_path
from src.utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpen, CircuitState

logger = logging.getLogger(__name__)

# Constants for bounded vocabulary to prevent unbounded memory growth
MAX_VOCABULARY_SIZE: Final = 10_000  # Maximum unique terms to track
MAX_PENDING_TERMS: Final = 5_000  # Maximum terms awaiting embedding

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "he",
        "she",
        "him",
        "her",
        "his",
        "hers",
        "we",
        "us",
        "our",
        "you",
        "your",
        "i",
        "me",
        "my",
        "who",
        "what",
        "which",
        "where",
        "when",
        "how",
        "why",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "not",
        "only",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "now",
        "here",
        "there",
        "then",
    }
)


class EmbeddingModel(Protocol):
    def get_text_embedding(self, text: str) -> list[float]: ...


class VectorIndex:
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        embedding_model: EmbeddingModel | None = None,
        embedding_workers: int = 4,
    ):
        self._embedding_model_name = embedding_model_name
        self._embedding_model: EmbeddingModel | None = embedding_model
        self._model_lock = threading.Lock()
        self._model_loaded = embedding_model is not None
        self._embedding_workers = max(1, embedding_workers)

        # Circuit breaker for embedding model failure protection
        self._embedding_circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            success_threshold=2,
            window_duration=60.0,
        )

        # If embedding model was provided, set it in Settings immediately
        if embedding_model is not None:
            from llama_index.core import Settings

            Settings.embed_model = embedding_model

        self._doc_id_to_node_ids: dict[str, list[str]] = {}
        self._chunk_id_to_node_id: dict[str, str] = {}
        self._vector_store = None
        self._index = None
        # Use OrderedDict for LRU-style eviction to prevent unbounded memory growth
        self._concept_vocabulary: OrderedDict[str, list[float]] = OrderedDict()
        self._term_counts: OrderedDict[str, int] = OrderedDict()
        self._pending_terms: set[str] = set()  # Terms that need embedding
        # FAISS index for fast vocabulary nearest-neighbor search
        self._vocab_faiss_index = None
        self._vocab_terms: list[
            str
        ] = []  # Ordered terms matching FAISS index positions
        # Bounded warning deduplication with LRU eviction (max 1000 entries)
        self._warned_stale_chunk_ids: OrderedDict[str, bool] = OrderedDict()
        self._max_warned_chunks = 1000
        self._tombstoned_docs: set[str] = set()
        self._index_lock = (
            threading.Lock()
        )  # Protects index operations during concurrent access

    def _ensure_model_loaded(self, timeout: float = 120.0) -> None:
        """Load the embedding model with timeout and circuit breaker protection.

        Args:
            timeout: Maximum seconds to wait for model loading (default: 120s)

        Raises:
            RuntimeError: If model loading exceeds timeout
            CircuitBreakerOpen: If circuit breaker is open (too many recent failures)
        """
        if self._model_loaded:
            return
        with self._model_lock:
            if self._model_loaded:
                return

            from concurrent.futures import (
                ThreadPoolExecutor,
                TimeoutError as FuturesTimeoutError,
            )

            def load_model():
                from llama_index.core import Settings
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding

                model = HuggingFaceEmbedding(model_name=self._embedding_model_name)
                Settings.embed_model = model
                return model

            try:
                # Wrap model loading with circuit breaker
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(load_model)
                    self._embedding_model = self._embedding_circuit_breaker.call(
                        lambda: future.result(timeout=timeout)
                    )
                    self._model_loaded = True
                    logger.info(f"Embedding model loaded: {self._embedding_model_name}")
            except CircuitBreakerOpen as e:
                error_msg = f"Embedding model circuit breaker is OPEN. Too many recent failures. {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            except FuturesTimeoutError:
                error_msg = (
                    f"Embedding model loading timed out after {timeout}s. "
                    f"Check network connection or download model manually: {self._embedding_model_name}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from None
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}", exc_info=True)
                raise

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

        if self._index is None:
            raise RuntimeError(
                "VectorIndex not initialized - call load() or add_chunks() first"
            )

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
        self._tombstoned_docs.discard(document.id)

        # Protect index operations with lock to prevent race condition during shutdown/persist
        with self._index_lock:
            self._doc_id_to_node_ids[document.id] = [node_id]
            self._index.insert_nodes([llama_doc])

    def add_chunk(self, chunk: Chunk) -> None:
        from llama_index.core import Document as LlamaDocument

        self._ensure_model_loaded()
        if self._index is None:
            self._initialize_index()

        if self._index is None:
            raise RuntimeError(
                "VectorIndex not initialized - call load() or add_chunks() first"
            )

        embedding_text = (
            f"{chunk.header_path}\n\n{chunk.content}"
            if chunk.header_path
            else chunk.content
        )

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
                "parent_chunk_id": chunk.parent_chunk_id,
                **chunk.metadata,  # Include ALL metadata from chunk
            },
            id_=chunk.chunk_id,
        )

        node_id = llama_doc.id_
        self._tombstoned_docs.discard(chunk.doc_id)

        # Protect index operations with lock to prevent race condition during shutdown/persist
        with self._index_lock:
            self._chunk_id_to_node_id[chunk.chunk_id] = node_id

            if chunk.doc_id not in self._doc_id_to_node_ids:
                self._doc_id_to_node_ids[chunk.doc_id] = []
            self._doc_id_to_node_ids[chunk.doc_id].append(node_id)

            self._index.insert_nodes([llama_doc])

        # Register terms for incremental vocabulary update
        self.register_document_terms(embedding_text)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        if self._embedding_workers <= 1:
            for chunk in chunks:
                self.add_chunk(chunk)
            return

        self._add_chunks_parallel(chunks)

    def _add_chunks_parallel(self, chunks: list[Chunk]) -> None:
        from llama_index.core import Document as LlamaDocument

        self._ensure_model_loaded()
        if self._index is None:
            self._initialize_index()

        if self._index is None:
            raise RuntimeError(
                "VectorIndex not initialized - call load() or add_chunks() first"
            )

        # Create all LlamaDocuments sequentially (trivially fast dict construction),
        # then insert in a single batch so LlamaIndex can batch-embed efficiently.
        llama_docs: list[tuple[LlamaDocument, Chunk]] = []
        for chunk in chunks:
            try:
                embedding_text = (
                    f"{chunk.header_path}\n\n{chunk.content}"
                    if chunk.header_path
                    else chunk.content
                )
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
                        "parent_chunk_id": chunk.parent_chunk_id,
                        **chunk.metadata,
                    },
                    id_=chunk.chunk_id,
                )
                llama_docs.append((llama_doc, chunk))
            except Exception as e:
                logger.error(
                    f"Failed to create document for chunk {chunk.chunk_id}: {e}",
                    exc_info=True,
                )

        if not llama_docs:
            return

        with self._index_lock:
            for llama_doc, chunk in llama_docs:
                node_id = llama_doc.id_
                self._tombstoned_docs.discard(chunk.doc_id)
                self._chunk_id_to_node_id[chunk.chunk_id] = node_id

                if chunk.doc_id not in self._doc_id_to_node_ids:
                    self._doc_id_to_node_ids[chunk.doc_id] = []
                self._doc_id_to_node_ids[chunk.doc_id].append(node_id)

            self._index.insert_nodes([doc for doc, _ in llama_docs])

        for _, chunk in llama_docs:
            embedding_text = (
                f"{chunk.header_path}\n\n{chunk.content}"
                if chunk.header_path
                else chunk.content
            )
            self.register_document_terms(embedding_text)

    def remove(self, document_id: str) -> None:
        if self._index is None:
            return

        # Protect mapping access with lock
        with self._index_lock:
            if document_id not in self._doc_id_to_node_ids:
                return
            del self._doc_id_to_node_ids[document_id]
            self._tombstoned_docs.add(document_id)

    def remove_chunk(self, chunk_id: str) -> None:
        """Remove a specific chunk from the vector index.

        Thread-safe operation that removes chunk from mappings.
        Handles missing chunks gracefully (logs warning, doesn't raise).

        Note: Due to FAISS limitations, the actual vectors remain in the index,
        but the chunk becomes inaccessible through search by removing its mapping.
        """
        if self._index is None:
            logger.warning(f"Cannot remove chunk {chunk_id}: index not initialized")
            return

        with self._index_lock:
            try:
                # FAISS doesn't support deletion, so we only remove mappings
                # The vectors remain in the index but become inaccessible
                # This is consistent with the existing prune_document() behavior

                # Remove from chunk_id mapping
                self._chunk_id_to_node_id.pop(chunk_id, None)

                # Remove from doc_id mapping
                for doc_id, node_ids in list(self._doc_id_to_node_ids.items()):
                    if chunk_id in node_ids:
                        node_ids.remove(chunk_id)
                        if not node_ids:
                            del self._doc_id_to_node_ids[doc_id]
                        break

                logger.debug(f"Removed chunk {chunk_id} from vector index mappings")
            except Exception as e:
                logger.warning(
                    f"Failed to remove chunk {chunk_id} from vector index: {e}",
                    exc_info=True,
                )

    def update_chunk_path(
        self, old_chunk_id: str, new_chunk_id: str, new_metadata: dict
    ) -> bool:
        """Update chunk by re-adding with new path (avoids parse/chunking overhead).

        Note: Due to FAISS limitations, this re-computes the embedding but reuses the content.
        The main speedup comes from avoiding file parsing and chunking, not embedding reuse.

        Returns:
            True if update successful, False otherwise
        """
        if self._index is None:
            logger.warning("Cannot update chunk path: index not initialized")
            return False

        with self._index_lock:
            try:
                from llama_index.core import Document as LlamaDocument

                # Get old node content
                docstore = self._index.docstore
                old_node = docstore.get_document(old_chunk_id)
                if old_node is None:
                    logger.debug(f"Node {old_chunk_id} not found in docstore")
                    return False

                # Extract content
                text_content = (
                    old_node.get_content()
                    if hasattr(old_node, "get_content")
                    else getattr(old_node, "text", "")
                )
                if not text_content:
                    logger.warning(f"No content for {old_chunk_id}")
                    return False

                # Create new node with same content but new metadata
                # Let LlamaIndex handle embedding (still faster than full parse/chunk)
                new_node = LlamaDocument(
                    text=text_content,
                    metadata=new_metadata,
                    id_=new_chunk_id,
                )

                # Add new node (will compute embedding)
                self._index.insert_nodes([new_node])

                # Update internal mappings
                new_node_id = new_node.id_
                self._chunk_id_to_node_id[new_chunk_id] = new_node_id

                # Update doc_id -> node_ids mapping
                old_doc_id = (
                    old_chunk_id.split("_chunk_")[0]
                    if "_chunk_" in old_chunk_id
                    else old_chunk_id.split("#")[0]
                )
                new_doc_id = new_metadata.get(
                    "doc_id",
                    new_chunk_id.split("_chunk_")[0]
                    if "_chunk_" in new_chunk_id
                    else new_chunk_id.split("#")[0],
                )

                if old_doc_id in self._doc_id_to_node_ids:
                    if old_doc_id != new_doc_id:
                        old_node_id = self._chunk_id_to_node_id.get(old_chunk_id)
                        if old_node_id:
                            try:
                                self._doc_id_to_node_ids[old_doc_id].remove(old_node_id)
                                if not self._doc_id_to_node_ids[old_doc_id]:
                                    del self._doc_id_to_node_ids[old_doc_id]
                            except ValueError:
                                pass

                if new_doc_id not in self._doc_id_to_node_ids:
                    self._doc_id_to_node_ids[new_doc_id] = []
                if new_node_id not in self._doc_id_to_node_ids[new_doc_id]:
                    self._doc_id_to_node_ids[new_doc_id].append(new_node_id)

                # Remove old chunk mappings and docstore entry
                try:
                    docstore.delete_document(old_chunk_id)
                except Exception as e:
                    logger.debug(
                        f"Could not delete old docstore entry {old_chunk_id}: {e}"
                    )

                self._chunk_id_to_node_id.pop(old_chunk_id, None)

                logger.debug(
                    f"Moved chunk (re-embedded): {old_chunk_id} -> {new_chunk_id}"
                )
                return True

            except Exception as e:
                logger.warning(
                    f"Failed to update chunk path {old_chunk_id} -> {new_chunk_id}: {e}",
                    exc_info=True,
                )
                return False

    def prune_document(self, doc_id: str):
        if self._index is None:
            return 0

        self._tombstoned_docs.add(doc_id)
        removed = 0
        chunk_ids = list(self._doc_id_to_node_ids.get(doc_id, []))
        for chunk_id in chunk_ids:
            try:
                docstore = self._index.docstore
                if hasattr(docstore, "delete_document"):
                    docstore.delete_document(chunk_id)
                    removed += 1
                elif hasattr(docstore, "delete"):
                    docstore.delete(chunk_id)  # pyright: ignore[reportAttributeAccessIssue]
                    removed += 1
            except Exception:
                logger.debug(
                    "Docstore delete failed during prune, continuing", exc_info=True
                )

            try:
                if hasattr(self._index, "index_store"):
                    index_store = self._index.index_store  # pyright: ignore[reportAttributeAccessIssue]
                    if hasattr(index_store, "delete"):
                        index_store.delete(chunk_id)  # type: ignore[call-non-callable]
            except Exception:
                logger.debug(
                    "Index store delete failed during prune, continuing", exc_info=True
                )

            try:
                if self._vector_store is not None and hasattr(
                    self._vector_store, "delete"
                ):
                    self._vector_store.delete(chunk_id)
            except Exception:
                logger.debug(
                    "Vector store delete failed during prune, continuing", exc_info=True
                )

            self._chunk_id_to_node_id.pop(chunk_id, None)

        try:
            if hasattr(self._index, "delete_ref_doc"):
                self._index.delete_ref_doc(doc_id, delete_from_docstore=True)
        except Exception:
            logger.debug(
                "delete_ref_doc failed during prune, continuing", exc_info=True
            )

        self._doc_id_to_node_ids.pop(doc_id, None)
        return removed

    def search(
        self,
        query: str,
        top_k: int = 10,
        excluded_files: set[str] | None = None,
        docs_root: Path | None = None,
    ) -> list[SearchResultDict]:
        if self._index is None or not query.strip():
            return []

        # Check circuit breaker state before attempting search
        if self._embedding_circuit_breaker.state == CircuitState.OPEN:
            logger.warning("Search skipped: embedding circuit breaker is open")
            return []

        self._ensure_model_loaded()

        fetch_k = top_k * 2 if excluded_files else top_k
        retriever = self._index.as_retriever(similarity_top_k=fetch_k)
        nodes = retriever.retrieve(query)

        results = []
        for node in nodes:
            if excluded_files and docs_root:
                file_path = node.metadata.get("file_path", "")
                if file_path:
                    from src.search.path_utils import matches_any_excluded

                    if matches_any_excluded(file_path, excluded_files, docs_root):
                        continue

            chunk_id = node.metadata.get("chunk_id")
            doc_id = node.metadata.get("doc_id")
            if doc_id and doc_id in self._tombstoned_docs:
                continue

            node_text = (
                node.node.get_content()
                if hasattr(node.node, "get_content")
                else getattr(node.node, "text", "")
            )

            if chunk_id:
                results.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "score": node.score if hasattr(node, "score") else 1.0,
                        "header_path": node.metadata.get("header_path", ""),
                        "file_path": node.metadata.get("file_path", ""),
                        "content": node_text,
                        "metadata": node.metadata,
                    }
                )
            else:
                if doc_id:
                    results.append(
                        {
                            "chunk_id": doc_id,
                            "doc_id": doc_id,
                            "score": node.score if hasattr(node, "score") else 1.0,
                            "header_path": "",
                            "file_path": node.metadata.get("file_path", ""),
                            "content": node_text,
                            "metadata": node.metadata,
                        }
                    )

            if len(results) >= top_k:
                break

        return results

    def get_chunk_by_id(self, chunk_id: str):
        if self._index is None:
            if chunk_id not in self._warned_stale_chunk_ids:
                logger.warning(f"get_chunk_by_id({chunk_id}): index is None")
                self._log_stale_warning(chunk_id)
            return None

        doc_id = chunk_id.split("_chunk_", 1)[0] if "_chunk_" in chunk_id else chunk_id
        if doc_id in self._tombstoned_docs:
            logger.debug(f"get_chunk_by_id({chunk_id}): doc_id tombstoned")
            return None

        logger.debug(f"get_chunk_by_id({chunk_id}): attempting direct docstore lookup")

        try:
            docstore = self._index.docstore
            node = docstore.get_document(chunk_id)

            if node:
                logger.debug(f"get_chunk_by_id({chunk_id}): found in docstore")
                node_text = (
                    node.get_content()
                    if hasattr(node, "get_content")
                    else getattr(node, "text", "")
                )
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
                if chunk_id not in self._warned_stale_chunk_ids:
                    logger.warning(
                        f"get_chunk_by_id({chunk_id}): not found in docstore"
                    )
                    self._log_stale_warning(chunk_id)
                self._cleanup_stale_reference(chunk_id)
        except Exception as e:
            if chunk_id not in self._warned_stale_chunk_ids:
                logger.warning(
                    f"get_chunk_by_id({chunk_id}): docstore lookup failed: {e}"
                )
                self._log_stale_warning(chunk_id)
            self._cleanup_stale_reference(chunk_id)

        # Skip the expensive retriever fallback - if not in docstore, it won't help
        return None

    def _log_stale_warning(self, chunk_id: str) -> None:
        """Record that we've warned about this chunk_id with bounded memory usage.

        Uses LRU eviction when cache exceeds max size (1000 entries).
        """
        # Evict oldest if at capacity
        if len(self._warned_stale_chunk_ids) >= self._max_warned_chunks:
            self._warned_stale_chunk_ids.popitem(last=False)  # Remove oldest (FIFO)

        self._warned_stale_chunk_ids[chunk_id] = True

    def get_chunk_ids_for_document(self, doc_id: str) -> list[str]:
        with self._index_lock:
            return list(self._doc_id_to_node_ids.get(doc_id, []))

    def get_document_ids(self) -> list[str]:
        with self._index_lock:
            return list(self._doc_id_to_node_ids.keys())

    def get_parent_content(self, parent_chunk_id: str) -> str | None:
        chunk_data = self.get_chunk_by_id(parent_chunk_id)
        if chunk_data:
            return chunk_data.get("content")
        return None

    def build_concept_vocabulary(
        self,
        min_term_length: int = 3,
        max_terms: int = 5000,
        min_frequency: int = 2,
    ) -> None:
        """Build concept vocabulary from scratch. Prefer update_vocabulary_incremental() for efficiency."""
        if self._index is None:
            return

        term_counts: dict[str, int] = {}
        docstore = self._index.docstore

        # Snapshot to avoid race condition with concurrent indexing
        chunk_ids_snapshot = list(self._chunk_id_to_node_id.keys())
        for chunk_id in chunk_ids_snapshot:
            try:
                node = docstore.get_document(chunk_id)
                if node is None:
                    continue
                text = (
                    node.get_content()
                    if hasattr(node, "get_content")
                    else getattr(node, "text", "")
                )
                tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b", text.lower())
                for token in tokens:
                    if len(token) >= min_term_length and token not in STOPWORDS:
                        term_counts[token] = term_counts.get(token, 0) + 1
            except Exception:
                logger.debug(
                    "Failed to extract terms from chunk %s, continuing",
                    chunk_id,
                    exc_info=True,
                )
                continue

        # Store term counts for future incremental updates (convert to OrderedDict)
        self._term_counts = OrderedDict(term_counts)

        # Filter by minimum frequency to reduce vocabulary size
        sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
        top_terms = [term for term, count in sorted_terms if count >= min_frequency][
            :max_terms
        ]

        self._ensure_model_loaded()
        if self._embedding_model is None:
            raise RuntimeError("Embedding model not initialized - call warm_up() first")
        self._concept_vocabulary = OrderedDict()

        # Batch embed terms using thread pool for parallelism
        def embed_term(term: str) -> tuple[str, list[float] | None]:
            try:
                return term, self._protected_embed(term)
            except CircuitBreakerOpen:
                logger.debug(f"Circuit breaker open, skipping vocabulary term: {term}")
                return term, None
            except Exception:
                return term, None

        with ThreadPoolExecutor(max_workers=self._embedding_workers) as executor:
            results = list(executor.map(embed_term, top_terms))

        for term, embedding in results:
            if embedding is not None:
                self._concept_vocabulary[term] = embedding

        self._pending_terms.clear()
        logger.info(
            f"Built concept vocabulary with {len(self._concept_vocabulary)} terms"
        )
        self._rebuild_vocab_index()

    def extract_terms_from_text(
        self, text: str, min_term_length: int = 3
    ) -> dict[str, int]:
        """Extract terms and their counts from text."""
        term_counts: dict[str, int] = {}
        tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b", text.lower())
        for token in tokens:
            if len(token) >= min_term_length and token not in STOPWORDS:
                term_counts[token] = term_counts.get(token, 0) + 1
        return term_counts

    def register_document_terms(self, text: str, min_term_length: int = 3) -> None:
        """Register terms from a newly indexed document for later vocabulary update.

        Implements bounded collections with LRU eviction to prevent memory leaks.
        """
        doc_terms = self.extract_terms_from_text(text, min_term_length)
        for term, count in doc_terms.items():
            # Update or add term count
            self._term_counts[term] = self._term_counts.get(term, 0) + count
            # Move to end (most recently used) for LRU tracking
            self._term_counts.move_to_end(term)

            # Mark as pending if not already in vocabulary
            if term not in self._concept_vocabulary:
                self._pending_terms.add(term)

        # Evict least recently used terms if exceeding limits
        if len(self._term_counts) > MAX_VOCABULARY_SIZE:
            num_to_remove = len(self._term_counts) - MAX_VOCABULARY_SIZE
            for _ in range(num_to_remove):
                # Remove oldest (least recently used) term
                old_term = next(iter(self._term_counts))
                self._term_counts.pop(old_term)
                self._concept_vocabulary.pop(old_term, None)
                self._pending_terms.discard(old_term)
            logger.debug(
                f"Evicted {num_to_remove} LRU terms from vocabulary (limit: {MAX_VOCABULARY_SIZE})"
            )

        # Limit pending terms by keeping most frequent
        if len(self._pending_terms) > MAX_PENDING_TERMS:
            # Sort pending by frequency and keep top N
            sorted_pending = sorted(
                self._pending_terms,
                key=lambda t: self._term_counts.get(t, 0),
                reverse=True,
            )
            self._pending_terms = set(sorted_pending[:MAX_PENDING_TERMS])
            logger.debug(f"Trimmed pending terms to {MAX_PENDING_TERMS} most frequent")

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

        # Snapshot to avoid race condition with concurrent indexing
        pending_snapshot = set(self._pending_terms)
        term_counts_snapshot = dict(self._term_counts)

        # Get top terms by frequency that aren't in vocabulary yet
        pending_with_counts = [
            (term, term_counts_snapshot.get(term, 0)) for term in pending_snapshot
        ]
        pending_with_counts.sort(key=lambda x: x[1], reverse=True)

        # Only embed terms that would make it into top max_terms
        vocab_snapshot = list(self._concept_vocabulary.keys())
        current_vocab_size = len(vocab_snapshot)
        if current_vocab_size >= max_terms:
            # Vocabulary is full - only add terms with higher freq than lowest in vocab
            if vocab_snapshot:
                min_vocab_freq = min(
                    term_counts_snapshot.get(t, 0) for t in vocab_snapshot
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
        if self._embedding_model is None:
            raise RuntimeError("Embedding model not initialized - call warm_up() first")

        embedded_count = 0
        for term in terms_to_embed:
            try:
                embedding = self._protected_embed(term)
                self._concept_vocabulary[term] = embedding
                self._pending_terms.discard(term)
                embedded_count += 1
            except CircuitBreakerOpen:
                logger.debug(f"Circuit breaker open, skipping vocabulary term: {term}")
                self._pending_terms.discard(term)
                break  # Stop processing if circuit is open
            except Exception:
                logger.debug(
                    "Failed to embed term '%s', continuing", term, exc_info=True
                )
                self._pending_terms.discard(term)
                continue

        if embedded_count > 0:
            logger.debug(f"Incrementally added {embedded_count} terms to vocabulary")
            self._rebuild_vocab_index()

        return embedded_count

    def _rebuild_vocab_index(self) -> None:
        """Rebuild FAISS inner-product index over vocabulary embeddings.

        Enables O(1) approximate nearest-neighbor lookup in expand_query()
        instead of O(V×D) linear scan over all vocabulary terms.
        """
        if not self._concept_vocabulary:
            self._vocab_faiss_index = None
            self._vocab_terms = []
            return

        import faiss

        terms = list(self._concept_vocabulary.keys())
        embeddings = np.array(
            [self._concept_vocabulary[t] for t in terms], dtype=np.float32
        )
        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])  # type: ignore[attr-defined]
        index.add(embeddings)  # type: ignore[union-attr]

        self._vocab_faiss_index = index
        self._vocab_terms = terms
        logger.debug(f"Built vocabulary FAISS index with {len(terms)} terms")

    def get_pending_vocabulary_count(self) -> int:
        """Return count of terms waiting to be embedded."""
        return len(self._pending_terms)

    def get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for text with circuit breaker protection.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            CircuitBreakerOpen: If circuit breaker is open (too many recent failures)
            RuntimeError: If embedding model not initialized
        """
        return self._protected_embed(text)

    def is_ready(self) -> bool:
        return self._index is not None

    def model_ready(self) -> bool:
        """Check if embedding model is loaded and ready for queries."""
        return self._model_loaded

    @property
    def circuit_state(self) -> CircuitState:
        """Current circuit breaker state for health checks."""
        return self._embedding_circuit_breaker.state

    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker to allow retry.

        Use this when you've resolved the underlying issue (e.g., network restored,
        model downloaded) and want to allow embedding operations to proceed.
        """
        self._embedding_circuit_breaker.reset()
        logger.info("Embedding circuit breaker manually reset")

    def _protected_embed(self, text: str) -> list[float]:
        """Get embedding with circuit breaker protection.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            CircuitBreakerOpen: If circuit breaker is open (too many recent failures)
            RuntimeError: If embedding model not initialized
        """
        self._ensure_model_loaded()
        if self._embedding_model is None:
            raise RuntimeError("Embedding model not initialized - call warm_up() first")
        return self._embedding_circuit_breaker.call(
            lambda: self._embedding_model.get_text_embedding(text)  # type: ignore[union-attr]
        )

    def get_circuit_breaker_status(self) -> dict:
        """Get circuit breaker status for health check.

        Returns:
            dict with keys:
                - state: Current circuit state ("closed", "open", "half_open")
                - recent_failures: Count of failures in current window
                - open_until: Timestamp when circuit will transition to half-open (if open)
                - failure_threshold: Number of failures that triggers circuit open
                - recovery_timeout: Seconds before allowing recovery attempt
        """
        breaker = self._embedding_circuit_breaker
        open_until = None
        if breaker.state == CircuitState.OPEN:
            open_until = breaker._open_time + breaker.config.recovery_timeout

        return {
            "state": breaker.state.value,
            "recent_failures": len(breaker._failure_timestamps),
            "open_until": open_until,
            "failure_threshold": breaker.config.failure_threshold,
            "recovery_timeout": breaker.config.recovery_timeout,
        }

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
            text = (
                node.get_content()
                if hasattr(node, "get_content")
                else getattr(node, "text", "")
            )
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
        if top_k <= 0:
            return query

        if not self._concept_vocabulary:
            return query

        # Lazily rebuild FAISS index if needed (e.g., after incremental update)
        if self._vocab_faiss_index is None or not self._vocab_terms:
            self._rebuild_vocab_index()
            if self._vocab_faiss_index is None:
                return query

        try:
            import faiss

            query_embedding = np.array(
                self._protected_embed(query), dtype=np.float32
            ).reshape(1, -1)
            faiss.normalize_L2(query_embedding)

            scores, indices = self._vocab_faiss_index.search(query_embedding, top_k)  # type: ignore[union-attr]

            expansion_terms = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score >= similarity_threshold:
                    expansion_terms.append(self._vocab_terms[idx])
        except CircuitBreakerOpen:
            logger.warning("Circuit breaker open, returning unexpanded query")
            return query

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

        # Acquire lock to prevent concurrent add_chunk operations during serialization
        with self._index_lock:
            storage_context = self._index.storage_context
            storage_context.persist(persist_dir=str(path))

            if self._vector_store is not None:
                import faiss

                faiss_path = path / "faiss_index.bin"
                faiss.write_index(self._vector_store._faiss_index, str(faiss_path))  # type: ignore[attr-defined]
                fsync_path(faiss_path)

            mapping_file = path / "doc_id_mapping.json"
            atomic_write_json(mapping_file, self._doc_id_to_node_ids)

            chunk_mapping_file = path / "chunk_id_mapping.json"
            atomic_write_json(chunk_mapping_file, self._chunk_id_to_node_id)

            vocab_file = path / "concept_vocabulary.json"
            atomic_write_json(vocab_file, self._concept_vocabulary)

            term_counts_file = path / "term_counts.json"
            atomic_write_json(term_counts_file, self._term_counts)

        tombstone_file = path / "tombstones.json"
        atomic_write_json(tombstone_file, sorted(self._tombstoned_docs))

    def persist_to(self, snapshot_dir: Path) -> None:
        self.persist(snapshot_dir)

    def load_from(self, snapshot_dir: Path) -> bool:
        if not snapshot_dir.exists():
            return False

        docstore_path = snapshot_dir / "docstore.json"
        if not docstore_path.exists():
            return False

        self.load(snapshot_dir)
        return True

    def load(self, path: Path) -> None:
        from llama_index.core import (
            StorageContext,
            VectorStoreIndex,
            load_index_from_storage,
        )
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

        try:
            faiss_path = path / "faiss_index.bin"
            if faiss_path.exists():
                faiss_index = faiss.read_index(str(faiss_path))  # type: ignore[attr-defined]
            else:
                dimension = 384
                faiss_index = faiss.IndexFlatL2(dimension)  # type: ignore[attr-defined]

            self._vector_store = FaissVectorStore(faiss_index=faiss_index)

            storage_context = StorageContext.from_defaults(
                vector_store=self._vector_store,
                persist_dir=str(path),
            )

            self._index = cast(
                VectorStoreIndex, load_index_from_storage(storage_context)
            )
        except Exception as e:
            logger.warning(
                "Vector index at %s is corrupted (%s: %s); clearing and reinitializing.",
                path,
                type(e).__name__,
                e,
                exc_info=True,
            )
            self._clear_index_dir(path)
            self._initialize_index()
            return

        def load_json_file(filepath: Path) -> dict | list | None:
            if not filepath.exists():
                return None
            try:
                with open(filepath, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"Failed to load {filepath.name} (corrupted JSON): {e}")
                return None

        def load_json_ordered(filepath: Path) -> OrderedDict | None:
            if not filepath.exists():
                return None
            try:
                with open(filepath, "r") as f:
                    return json.load(f, object_pairs_hook=OrderedDict)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"Failed to load {filepath.name} (corrupted JSON): {e}")
                return None

        files_to_load = {
            "doc_id_mapping": path / "doc_id_mapping.json",
            "chunk_id_mapping": path / "chunk_id_mapping.json",
            "tombstones": path / "tombstones.json",
        }
        ordered_files = {
            "concept_vocabulary": path / "concept_vocabulary.json",
            "term_counts": path / "term_counts.json",
        }

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                name: executor.submit(load_json_file, fp)
                for name, fp in files_to_load.items()
            }
            ordered_futures = {
                name: executor.submit(load_json_ordered, fp)
                for name, fp in ordered_files.items()
            }
            results = {name: future.result() for name, future in futures.items()}
            ordered_results = {
                name: future.result() for name, future in ordered_futures.items()
            }

        doc_mapping = results["doc_id_mapping"]
        if isinstance(doc_mapping, dict):
            self._doc_id_to_node_ids = doc_mapping
        else:
            self._doc_id_to_node_ids = {}

        chunk_mapping = results["chunk_id_mapping"]
        if isinstance(chunk_mapping, dict):
            self._chunk_id_to_node_id = chunk_mapping
        else:
            self._chunk_id_to_node_id = {}

        if ordered_results["concept_vocabulary"] is not None:
            self._concept_vocabulary = ordered_results["concept_vocabulary"]
        else:
            self._concept_vocabulary = OrderedDict()

        if ordered_results["term_counts"] is not None:
            self._term_counts = ordered_results["term_counts"]
        else:
            self._term_counts = OrderedDict()

        tombstones = results["tombstones"]
        if tombstones is not None and isinstance(tombstones, list):
            self._tombstoned_docs = set(tombstones)
        else:
            self._tombstoned_docs = set()

        self._pending_terms.clear()
        self._warned_stale_chunk_ids.clear()
        self._rebuild_vocab_index()

    def _clear_index_dir(self, path: Path) -> None:
        """Remove all files from an index directory so it can be rebuilt cleanly."""
        import shutil

        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    def _initialize_index(self) -> None:
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.vector_stores.faiss import FaissVectorStore

        self._ensure_model_loaded()

        import faiss

        dimension = 384
        faiss_index = faiss.IndexFlatL2(dimension)  # type: ignore[attr-defined]
        self._vector_store = FaissVectorStore(faiss_index=faiss_index)

        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)

        self._index = VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
        )

    def _cleanup_stale_reference(self, chunk_id: str) -> None:
        """Remove a stale chunk reference from internal mappings.

        Note: This is called from get_chunk_by_id which doesn't hold the lock,
        so we acquire it here.
        """
        with self._index_lock:
            if chunk_id in self._chunk_id_to_node_id:
                del self._chunk_id_to_node_id[chunk_id]

            for doc_id, node_ids in list(self._doc_id_to_node_ids.items()):
                if chunk_id in node_ids:
                    node_ids.remove(chunk_id)
                    if not node_ids:
                        del self._doc_id_to_node_ids[doc_id]
                    break

    def reconcile_mappings(self) -> int:
        """
        Reconcile mappings by removing stale references.

        Returns:
            Count of stale references removed.
        """
        if self._index is None:
            return 0

        removed = 0
        docstore = self._index.docstore

        # Take snapshot of chunk IDs to check (avoid holding lock during docstore lookups)
        with self._index_lock:
            chunk_ids_snapshot = list(self._chunk_id_to_node_id.keys())

        for chunk_id in chunk_ids_snapshot:
            try:
                node = docstore.get_document(chunk_id)
                if node is None:
                    self._cleanup_stale_reference(chunk_id)
                    removed += 1
            except Exception:
                self._cleanup_stale_reference(chunk_id)
                removed += 1

        if removed > 0:
            logger.info(f"Reconciled vector index: removed {removed} stale references")

        return removed

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
        self._tombstoned_docs.discard(doc_id)
        self._doc_id_to_node_ids[doc_id] = [node_id]
        self._index.insert_nodes([llama_doc])

    def remove_document(self, doc_id: str) -> None:
        self.remove(doc_id)

    def clear(self) -> None:
        self._doc_id_to_node_ids = {}
        self._chunk_id_to_node_id = {}
        self._vector_store = None
        self._index = None
        self._concept_vocabulary = OrderedDict()
        self._term_counts = OrderedDict()
        self._pending_terms.clear()
        self._warned_stale_chunk_ids.clear()
        self._tombstoned_docs.clear()
        self._vocab_faiss_index = None
        self._vocab_terms = []

    def save(self, path: Path) -> None:
        self.persist(path)

    def save_to_db(self, db_manager: "DatabaseManager") -> None:
        """Persist VectorIndex state to SQLite via DatabaseManager.

        Stores:
        - Individual chunk vectors in the ``chunks`` table
        - FAISS binary, docstore, and JSON mappings in ``kv_store``
        """
        import base64

        import faiss

        if self._index is None:
            return

        conn = db_manager.get_connection()

        with self._index_lock:
            # --- a) Store each chunk's vector in the chunks table ---
            docstore = self._index.docstore
            for chunk_id in list(self._chunk_id_to_node_id.keys()):
                try:
                    node = docstore.get_document(chunk_id)
                    if node is None:
                        continue
                    text = (
                        node.get_content()
                        if hasattr(node, "get_content")
                        else getattr(node, "text", "")
                    )
                    metadata = dict(node.metadata) if hasattr(node, "metadata") else {}
                    doc_id = metadata.get("doc_id", "")
                    embedding = getattr(node, "embedding", None)
                    vector_blob: bytes | None = None
                    if embedding is not None:
                        vector_blob = np.array(embedding, dtype=np.float32).tobytes()

                    conn.execute(
                        """INSERT OR REPLACE INTO chunks
                           (chunk_id, doc_id, content, metadata, vector, indexed_at)
                           VALUES (?, ?, ?, ?, ?, strftime('%%s', 'now'))""",
                        (chunk_id, doc_id, text, json.dumps(metadata), vector_blob),
                    )
                except Exception:
                    logger.warning(
                        "Failed to save chunk %s to DB", chunk_id, exc_info=True
                    )

            # --- b) Store FAISS index binary in kv_store ---
            if self._vector_store is not None:
                faiss_bytes = faiss.serialize_index(self._vector_store._faiss_index)
                faiss_b64 = base64.b64encode(faiss_bytes.tobytes()).decode("ascii")
                conn.execute(
                    "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
                    ("vector_index:faiss_binary", faiss_b64),
                )

            # --- c) Store JSON mappings in kv_store ---
            mappings: dict[str, str] = {
                "vector_index:doc_id_mapping": json.dumps(self._doc_id_to_node_ids),
                "vector_index:chunk_id_mapping": json.dumps(self._chunk_id_to_node_id),
                "vector_index:tombstones": json.dumps(sorted(self._tombstoned_docs)),
                "vector_index:concept_vocabulary": json.dumps(self._concept_vocabulary),
                "vector_index:term_counts": json.dumps(self._term_counts),
            }
            for key, value in mappings.items():
                conn.execute(
                    "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
                    (key, value),
                )

            # --- d) Store LlamaIndex docstore and index_store in kv_store ---
            docstore_dict = self._index.storage_context.docstore.to_dict()  # type: ignore[attr-defined]
            conn.execute(
                "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
                ("vector_index:docstore", json.dumps(docstore_dict)),
            )
            index_store_dict = self._index.storage_context.index_store.to_dict()  # type: ignore[attr-defined]
            conn.execute(
                "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
                ("vector_index:index_store", json.dumps(index_store_dict)),
            )

        conn.commit()
        logger.info(
            "VectorIndex saved to SQLite (%d chunks)", len(self._chunk_id_to_node_id)
        )

    def load_from_db(self, db_manager: "DatabaseManager") -> None:
        """Restore VectorIndex state from SQLite via DatabaseManager.

        Loads FAISS binary, docstore, and all JSON mappings from the database.
        Falls back to a fresh index if no data is found.
        """
        import base64

        import faiss
        from llama_index.core import StorageContext, load_index_from_storage
        from llama_index.core.storage.docstore import SimpleDocumentStore
        from llama_index.core.storage.index_store import SimpleIndexStore
        from llama_index.vector_stores.faiss import FaissVectorStore

        conn = db_manager.get_connection()

        # Check if data exists
        row = conn.execute(
            "SELECT value FROM kv_store WHERE key = ?", ("vector_index:faiss_binary",)
        ).fetchone()
        if row is None:
            self._initialize_index()
            return

        self._ensure_model_loaded()

        try:
            # --- a) Load FAISS index ---
            faiss_b64: str = row[0] if isinstance(row, (tuple, list)) else row["value"]
            faiss_bytes = np.frombuffer(base64.b64decode(faiss_b64), dtype=np.uint8)
            faiss_index = faiss.deserialize_index(faiss_bytes)
            self._vector_store = FaissVectorStore(faiss_index=faiss_index)

            # --- b) Load docstore ---
            ds_row = conn.execute(
                "SELECT value FROM kv_store WHERE key = ?", ("vector_index:docstore",)
            ).fetchone()
            if ds_row is not None:
                ds_json = (
                    ds_row[0] if isinstance(ds_row, (tuple, list)) else ds_row["value"]
                )
                docstore = SimpleDocumentStore.from_dict(json.loads(ds_json))
            else:
                docstore = SimpleDocumentStore()

            # --- c) Reconstruct VectorStoreIndex ---
            is_row = conn.execute(
                "SELECT value FROM kv_store WHERE key = ?",
                ("vector_index:index_store",),
            ).fetchone()
            if is_row is not None:
                is_json = (
                    is_row[0] if isinstance(is_row, (tuple, list)) else is_row["value"]
                )
                index_store = SimpleIndexStore.from_dict(json.loads(is_json))
            else:
                index_store = SimpleIndexStore()

            storage_context = StorageContext.from_defaults(
                vector_store=self._vector_store,
                docstore=docstore,
                index_store=index_store,
            )
            self._index = load_index_from_storage(storage_context)

            # --- d) Load JSON mappings ---
            def _load_kv(key: str) -> str | None:
                r = conn.execute(
                    "SELECT value FROM kv_store WHERE key = ?", (key,)
                ).fetchone()
                if r is None:
                    return None
                return r[0] if isinstance(r, (tuple, list)) else r["value"]

            doc_mapping_raw = _load_kv("vector_index:doc_id_mapping")
            self._doc_id_to_node_ids = (
                json.loads(doc_mapping_raw) if doc_mapping_raw else {}
            )

            chunk_mapping_raw = _load_kv("vector_index:chunk_id_mapping")
            self._chunk_id_to_node_id = (
                json.loads(chunk_mapping_raw) if chunk_mapping_raw else {}
            )

            tombstones_raw = _load_kv("vector_index:tombstones")
            self._tombstoned_docs = (
                set(json.loads(tombstones_raw)) if tombstones_raw else set()
            )

            vocab_raw = _load_kv("vector_index:concept_vocabulary")
            self._concept_vocabulary = (
                OrderedDict(json.loads(vocab_raw)) if vocab_raw else OrderedDict()
            )

            term_counts_raw = _load_kv("vector_index:term_counts")
            self._term_counts = (
                OrderedDict(json.loads(term_counts_raw))
                if term_counts_raw
                else OrderedDict()
            )

            self._pending_terms.clear()
            self._warned_stale_chunk_ids.clear()
            self._rebuild_vocab_index()

            logger.info(
                "VectorIndex loaded from SQLite (%d chunks, %d docs)",
                len(self._chunk_id_to_node_id),
                len(self._doc_id_to_node_ids),
            )
        except Exception:
            logger.warning(
                "Failed to load VectorIndex from SQLite, initializing fresh",
                exc_info=True,
            )
            self._initialize_index()

    def __len__(self) -> int:
        return len(self._doc_id_to_node_ids)
