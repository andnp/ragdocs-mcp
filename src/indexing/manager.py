import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from src.chunking.factory import get_chunker
from src.config import Config
from src.coordination import IndexLock
from src.indices.code import CodeIndex
from src.indices.graph import GraphStore
from src.indices.hash_store import ChunkHashStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.indexing.implicit_graph import ImplicitGraphBuilder
from src.indexing.manifest import load_manifest, save_manifest
from src.models import Chunk
from src.parsers.dispatcher import dispatch_parser
from src.search.edge_types import infer_edge_type
from src.search.path_utils import compute_doc_id, resolve_doc_path

logger = logging.getLogger(__name__)


@dataclass
class FailedFile:
    path: str
    error: str
    timestamp: str


class IndexManager:
    def __init__(
        self,
        config: Config,
        vector: VectorIndex,
        keyword: KeywordIndex,
        graph: GraphStore,
        code: CodeIndex | None = None,
    ):
        self._config = config
        self.vector = vector
        self.keyword = keyword
        self.graph = graph
        self.code = code
        self._failed_files: list[FailedFile] = []
        self._chunker = get_chunker(config.document_chunking)

        # Initialize hash store for delta indexing
        index_path = Path(config.indexing.index_path)
        hash_store_path = index_path / "chunk_hashes.json"
        self._hash_store = ChunkHashStore(hash_store_path)

        logger.info(
            f"IndexManager initialized with embedding_workers={config.indexing.embedding_workers} "
            f"(mode: {'parallel' if config.indexing.embedding_workers > 1 else 'sequential'}), "
            f"delta_indexing={'enabled' if config.indexing.enable_delta_indexing else 'disabled'}"
        )

    def _get_parser_suffixes(self, fallback_suffixes: list[str] | None = None):
        suffixes: set[str] = set()
        for pattern in self._config.parsers.keys():
            suffix = Path(pattern).suffix
            if suffix:
                suffixes.add(suffix)

        if not suffixes:
            if fallback_suffixes is None:
                return []
            suffixes.update(fallback_suffixes)

        return sorted(suffixes)

    def reindex_document(self, doc_id: str, reason: str | None = None):
        docs_path = Path(self._config.indexing.documents_path)
        suffixes = self._get_parser_suffixes([".md", ".markdown", ".txt"])
        resolved_path = resolve_doc_path(doc_id, docs_path, suffixes)
        if not resolved_path:
            self.prune_document(doc_id, reason=reason)
            if reason:
                logger.warning(
                    "Reindex skipped for %s (reason: %s): file not found",
                    doc_id,
                    reason,
                )
            else:
                logger.warning("Reindex skipped for %s: file not found", doc_id)
            return False

        try:
            self.remove_document(doc_id)
            self.index_document(str(resolved_path), force=True)  # Force full re-index
            if reason:
                logger.info(
                    "Reindexed %s from %s (reason: %s)", doc_id, resolved_path, reason
                )
            else:
                logger.info("Reindexed %s from %s", doc_id, resolved_path)
            return True
        except Exception as e:
            logger.error("Failed to reindex %s: %s", doc_id, e, exc_info=True)
            return False

    def prune_document(self, doc_id: str, reason: str | None = None):
        try:
            removed_chunks = self.vector.prune_document(doc_id)
            self.keyword.remove(doc_id)
            self.graph.remove_node(doc_id)
            if self.code is not None:
                self.code.remove_by_doc_id(doc_id)

            index_path = Path(self._config.indexing.index_path)
            manifest = load_manifest(index_path)
            if manifest and manifest.indexed_files and doc_id in manifest.indexed_files:
                manifest.indexed_files.pop(doc_id, None)
                save_manifest(index_path, manifest)

            if reason:
                logger.info(
                    "Pruned %s from indices (%d chunks removed, reason: %s)",
                    doc_id,
                    removed_chunks,
                    reason,
                )
            else:
                logger.info(
                    "Pruned %s from indices (%d chunks removed)", doc_id, removed_chunks
                )
            return True
        except Exception as e:
            logger.error("Failed to prune %s: %s", doc_id, e, exc_info=True)
            return False

    def _detect_changed_chunks(self, chunks: list[Chunk]) -> tuple[list[Chunk], list[str]]:
        """Identify chunks with changed content.

        Returns:
            (changed_chunks, unchanged_chunk_ids)
        """
        if not self._config.indexing.enable_delta_indexing:
            # Delta indexing disabled, all chunks are "changed"
            return chunks, []

        changed = []
        unchanged = []

        for chunk in chunks:
            if self._hash_store.has_changed(chunk):
                changed.append(chunk)
            else:
                unchanged.append(chunk.chunk_id)

        return changed, unchanged

    def _should_use_delta_indexing(
        self,
        changed_chunks: list[Chunk],
        total_chunks: int,
    ) -> bool:
        """Decide whether to use delta or full re-index based on change ratio."""
        if total_chunks == 0:
            return True

        change_ratio = len(changed_chunks) / total_chunks
        threshold = self._config.indexing.delta_full_reindex_threshold

        if change_ratio > threshold:
            logger.info(
                f"Change ratio {change_ratio:.1%} exceeds threshold {threshold:.1%}, "
                "using full re-index"
            )
            return False

        return True

    def _update_chunks(self, doc_id: str, chunks: list[Chunk]) -> None:
        """Update specific chunks in all indices (remove old → add new)."""
        if not chunks:
            return

        # Remove old versions
        for chunk in chunks:
            self.vector.remove_chunk(chunk.chunk_id)
            self.keyword.remove_chunk(chunk.chunk_id)
            self.graph.remove_chunk(chunk.chunk_id)

        # Add new versions
        self.vector.add_chunks(chunks)
        self.keyword.add_chunks(chunks)
        for chunk in chunks:
            self.graph.add_node(chunk.chunk_id, chunk.metadata)

        logger.debug(f"Updated {len(chunks)} chunks for {doc_id}")

    def _full_reindex_document(self, doc_id: str, chunks: list[Chunk]) -> None:
        """Full re-index of document (remove all old chunks, add all new)."""
        # Remove all old chunks
        self.vector.remove(doc_id)
        self.keyword.remove(doc_id)
        self.graph.remove_node(doc_id)

        # Add all new chunks
        self.vector.add_chunks(chunks)
        self.keyword.add_chunks(chunks)
        for chunk in chunks:
            self.graph.add_node(chunk.chunk_id, chunk.metadata)

        # Update hash store (clear old hashes first)
        if self._config.indexing.enable_delta_indexing:
            self._hash_store.remove_document(doc_id)
            for chunk in chunks:
                self._hash_store.set_hash(chunk.chunk_id, chunk.content_hash)
            self._hash_store.persist()

        logger.debug(f"Full re-indexed {doc_id} with {len(chunks)} chunks")

    def _detect_file_moves(
        self,
        removed_docs: set[str],
        added_docs: dict[str, list[Chunk]],
    ) -> dict[str, str]:
        """Detect file moves by comparing content hashes.

        Args:
            removed_docs: Set of doc_ids that appear to be removed
            added_docs: Dict of doc_id -> chunks for newly added docs

        Returns:
            Dict mapping old_doc_id -> new_doc_id for detected moves
        """
        if not self._config.indexing.enable_move_detection:
            return {}

        if not self._config.indexing.enable_delta_indexing:
            logger.debug("Move detection requires delta indexing to be enabled")
            return {}

        moves: dict[str, str] = {}
        threshold = self._config.indexing.move_detection_threshold

        for new_doc_id, new_chunks in added_docs.items():
            if not new_chunks:
                continue

            # Build hash set for new document
            new_hashes = {chunk.content_hash for chunk in new_chunks}

            # Compare with each removed document
            best_match_doc = None
            best_match_ratio = 0.0

            for old_doc_id in removed_docs:
                old_chunk_data = self._hash_store.get_chunks_by_document(old_doc_id)
                if not old_chunk_data:
                    continue

                old_hashes = {hash_val for _, hash_val in old_chunk_data}

                # Calculate overlap ratio
                if not old_hashes or not new_hashes:
                    continue

                matching_hashes = new_hashes & old_hashes
                match_ratio = len(matching_hashes) / max(len(old_hashes), len(new_hashes))

                if match_ratio > best_match_ratio:
                    best_match_ratio = match_ratio
                    best_match_doc = old_doc_id

            # If match ratio exceeds threshold, it's a move
            if best_match_doc and best_match_ratio >= threshold:
                moves[best_match_doc] = new_doc_id
                logger.info(
                    f"Detected file move: {best_match_doc} -> {new_doc_id} "
                    f"(match ratio: {best_match_ratio:.1%})"
                )

        return moves

    def _apply_file_move(
        self,
        old_doc_id: str,
        new_doc_id: str,
        new_chunks: list[Chunk],
    ) -> bool:
        """Apply file move by updating indices without re-embedding.

        Updates vector index metadata, copies keyword index documents,
        and renames graph nodes.

        Returns:
            True if move successful, False if fallback to re-index needed
        """
        try:
            # Get old chunks for mapping
            old_chunk_data = self._hash_store.get_chunks_by_document(old_doc_id)
            if not old_chunk_data:
                logger.debug(f"No old chunks found for {old_doc_id}, using full reindex")
                return False

            # Build hash -> old_chunk_id mapping
            old_hash_to_chunk: dict[str, str] = {
                hash_val: chunk_id for chunk_id, hash_val in old_chunk_data
            }

            # Process each new chunk
            moved_count = 0
            failed_moves = []

            for new_chunk in new_chunks:
                old_chunk_id = old_hash_to_chunk.get(new_chunk.content_hash)
                if not old_chunk_id:
                    # Content changed, need full re-index
                    failed_moves.append(new_chunk.chunk_id)
                    continue

                # Update indices with new IDs/metadata
                new_metadata = {
                    "doc_id": new_chunk.doc_id,
                    "chunk_id": new_chunk.chunk_id,
                    "file_path": new_chunk.file_path,
                    "header_path": new_chunk.header_path,
                    **new_chunk.metadata,
                }

                # Update vector index
                if not self.vector.update_chunk_path(old_chunk_id, new_chunk.chunk_id, new_metadata):
                    failed_moves.append(new_chunk.chunk_id)
                    continue

                # Update keyword index
                if not self.keyword.move_chunk(old_chunk_id, new_chunk):
                    failed_moves.append(new_chunk.chunk_id)
                    continue

                # Update graph
                if not self.graph.rename_node(old_chunk_id, new_chunk.chunk_id):
                    # Graph update is optional, just log
                    logger.debug(f"Graph node rename failed for {old_chunk_id}")

                moved_count += 1

            # Update hash store
            self._hash_store.remove_document(old_doc_id)
            for chunk in new_chunks:
                self._hash_store.set_hash(chunk.chunk_id, chunk.content_hash)
            self._hash_store.persist()

            # If too many chunks failed, fall back to full re-index
            if failed_moves:
                failure_ratio = len(failed_moves) / len(new_chunks)
                if failure_ratio > (1.0 - self._config.indexing.move_detection_threshold):
                    logger.info(
                        f"Move operation had {failure_ratio:.1%} failures, "
                        "falling back to full re-index"
                    )
                    return False

            logger.info(
                f"Successfully moved {moved_count}/{len(new_chunks)} chunks "
                f"from {old_doc_id} to {new_doc_id}"
            )
            return True

        except Exception as e:
            logger.warning(
                f"Move operation failed for {old_doc_id} -> {new_doc_id}: {e}. "
                "Falling back to full re-index.",
                exc_info=True,
            )
            return False

    def index_document(self, file_path: str, force: bool = False):
        try:
            parser = dispatch_parser(file_path, self._config)
            document = parser.parse(file_path)

            docs_path = Path(self._config.indexing.documents_path)
            document.id = compute_doc_id(Path(file_path).resolve(), docs_path.resolve())

            chunks = self._chunker.chunk_document(document)
            document.chunks = chunks

            # Delta indexing logic
            if force or not self._config.indexing.enable_delta_indexing:
                # Force full re-index or delta disabled
                self._full_reindex_document(document.id, chunks)
            else:
                # Delta detection
                changed_chunks, unchanged_chunk_ids = self._detect_changed_chunks(chunks)

                if not changed_chunks:
                    logger.info(f"No changes in {file_path}, skipping re-index")
                    return

                # Decide: delta or full re-index?
                if not self._should_use_delta_indexing(changed_chunks, len(chunks)):
                    self._full_reindex_document(document.id, chunks)
                else:
                    logger.info(
                        f"Delta index {file_path}: "
                        f"{len(changed_chunks)} changed, {len(unchanged_chunk_ids)} unchanged"
                    )

                    # Update only changed chunks
                    self._update_chunks(document.id, changed_chunks)

                    # Update hash store for all chunks (even unchanged, to handle ID shifts)
                    for chunk in chunks:
                        self._hash_store.set_hash(chunk.chunk_id, chunk.content_hash)
                    self._hash_store.persist()

            # Add document node to graph (for links/metadata)
            # Pass full metadata including tags and file_path to the graph
            # This is crucial for ImplicitGraphBuilder to work correctly
            graph_metadata = {
                **document.metadata,
                "tags": document.tags,
                "file_path": document.file_path,
            }
            self.graph.add_node(document.id, graph_metadata)

            from src.parsers.markdown import MarkdownParser

            if isinstance(parser, MarkdownParser):
                links_with_context = parser.extract_links_with_context(file_path)
                for link_info in links_with_context:
                    edge_type = infer_edge_type(
                        link_info.header_context, link_info.target
                    )
                    self.graph.add_edge(
                        document.id,
                        link_info.target,
                        edge_type=edge_type.value,
                        edge_context=link_info.header_context,
                    )
            else:
                for link in document.links:
                    self.graph.add_edge(document.id, link, edge_type="links_to")

            if self.code is not None and self._config.search.code_search_enabled:
                from src.parsers.markdown import MarkdownParser

                if isinstance(parser, MarkdownParser):
                    code_blocks = parser.extract_code_blocks(file_path, document.id)
                    for code_block in code_blocks:
                        self.code.add_code_block(code_block)

            self._failed_files = [f for f in self._failed_files if f.path != file_path]

        except UnicodeDecodeError as e:
            # Try alternative encodings for files with encoding issues
            logger.warning(
                f"UTF-8 decode failed for {file_path}, trying alternative encodings: {e}"
            )
            for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    logger.info(
                        f"Attempting to read {file_path} with {encoding} encoding"
                    )
                    parser = dispatch_parser(file_path, self._config)
                    # This will re-attempt parsing with different encoding
                    # Note: We'd need to modify parsers to accept encoding parameter
                    # For now, just skip the file and log it
                    logger.warning(f"Skipping file with encoding issues: {file_path}")
                    break
                except Exception:
                    continue

            failed = FailedFile(
                path=file_path,
                error=f"Encoding error: {str(e)}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self._failed_files = [
                f for f in self._failed_files if f.path != file_path
            ] + [failed]
            # Don't raise - continue indexing other files
            logger.info(
                f"Continuing with remaining files after encoding error in {file_path}"
            )
        except Exception as e:
            logger.error(f"Failed to index document {file_path}: {e}", exc_info=True)
            failed = FailedFile(
                path=file_path,
                error=str(e),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self._failed_files = [
                f for f in self._failed_files if f.path != file_path
            ] + [failed]
            raise

    def remove_document(self, doc_id: str):
        errors: list[tuple[str, Exception]] = []

        for index_name, remove_fn in [
            ("vector", lambda: self.vector.remove(doc_id)),
            ("keyword", lambda: self.keyword.remove(doc_id)),
            ("graph", lambda: self.graph.remove_node(doc_id)),
        ]:
            try:
                remove_fn()
            except Exception as e:
                logger.error(
                    f"Failed to remove {doc_id} from {index_name}: {e}", exc_info=True
                )
                errors.append((index_name, e))

        if self.code is not None:
            try:
                self.code.remove_by_doc_id(doc_id)
            except Exception as e:
                logger.error(
                    f"Failed to remove {doc_id} from code index: {e}", exc_info=True
                )
                errors.append(("code", e))

        if errors:
            logger.warning(
                f"Document {doc_id} removal completed with {len(errors)} index failures"
            )

        # Remove from hash store
        if self._config.indexing.enable_delta_indexing:
            self._hash_store.remove_document(doc_id)
            self._hash_store.persist()

    def persist(self):
        """Persist all indices with retry logic for transient failures.

        Implements exponential backoff with 3 retry attempts to handle
        transient failures like disk full, NFS timeout, or permission errors.
        """
        index_path = Path(self._config.indexing.index_path)

        coordination_mode_str = self._config.indexing.coordination_mode.lower()

        if coordination_mode_str == "file_lock":
            lock = IndexLock(index_path, self._config.indexing.lock_timeout_seconds)
            lock.acquire_exclusive()
            try:
                self._persist_indices_with_retry(index_path)
            finally:
                lock.release()
        else:
            self._persist_indices_with_retry(index_path)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _persist_indices_with_retry(self, index_path: Path):
        """Execute persistence with automatic retry on failure.

        Retries up to 3 times with exponential backoff (1s, 2s, 4s).
        Raises the original exception after all retries are exhausted.
        """
        self._persist_indices(index_path)

    def _persist_indices(self, index_path: Path):
        try:
            # Build implicit graph edges (directory siblings, shared tags)
            # to improve community detection density
            implicit_builder = ImplicitGraphBuilder(self.graph)
            implicit_builder.build_implicit_edges()

            self.vector.persist(index_path / "vector")
            self.keyword.persist(index_path / "keyword")
            self.graph.persist(index_path / "graph")
            if self.code is not None:
                self.code.persist(index_path / "code")

            # Persist hash store for delta indexing
            if self._config.indexing.enable_delta_indexing:
                self._hash_store.persist()
        except Exception as e:
            logger.error(f"Failed to persist indices: {e}", exc_info=True)
            raise

    def load(self):
        index_path = Path(self._config.indexing.index_path)

        coordination_mode_str = self._config.indexing.coordination_mode.lower()

        if coordination_mode_str == "file_lock":
            lock = IndexLock(index_path, self._config.indexing.lock_timeout_seconds)
            lock.acquire_shared()
            try:
                self._load_indices(index_path)
            finally:
                lock.release()
        else:
            self._load_indices(index_path)

    def _load_indices(self, index_path: Path):
        try:
            self.vector.load(index_path / "vector")
            self.keyword.load(index_path / "keyword")
            self.graph.load(index_path / "graph")
            if self.code is not None:
                self.code.load(index_path / "code")
        except Exception as e:
            logger.error(f"Failed to load indices: {e}", exc_info=True)
            raise

    def get_document_count(self) -> int:
        return len(self.vector.get_document_ids())

    def is_ready(self) -> bool:
        """Check if all indices are loaded and ready for queries."""
        return self.vector.is_ready() and self.vector.model_ready()

    def get_failed_files(self) -> list[dict[str, str]]:
        return [
            {"path": f.path, "error": f.error, "timestamp": f.timestamp}
            for f in self._failed_files
        ]

    def reconcile_indices(
        self,
        discovered_files: list[str],
        docs_path: Path,
    ):
        """Reconcile indices with filesystem state, detecting file moves.

        Returns:
            ReconciliationResult with counts of operations performed
        """
        from src.indexing.manifest import load_manifest
        from src.indexing.reconciler import reconcile_indices
        from src.models import ReconciliationResult
        from src.parsers.dispatcher import dispatch_parser

        index_path = Path(self._config.indexing.index_path)
        saved_manifest = load_manifest(index_path)

        if not saved_manifest:
            logger.warning("No manifest found during reconciliation")
            return ReconciliationResult()

        files_to_add, doc_ids_to_remove, _ = reconcile_indices(
            discovered_files,
            saved_manifest,
            docs_path,
            include_patterns=self._config.indexing.include,
            exclude_patterns=self._config.indexing.exclude,
            exclude_hidden_dirs=self._config.indexing.exclude_hidden_dirs,
        )

        result = ReconciliationResult()

        # Detect file moves if enabled
        moved_files: dict[str, str] = {}
        if (
            self._config.indexing.enable_move_detection
            and self._config.indexing.enable_delta_indexing
            and files_to_add
            and doc_ids_to_remove
        ):
            logger.info(
                f"Move detection: Comparing {len(doc_ids_to_remove)} removed "
                f"and {len(files_to_add)} added files"
            )

            # Parse new files to get chunks for comparison
            added_docs: dict[str, list[Chunk]] = {}
            for file_path in files_to_add:
                try:
                    parser = dispatch_parser(file_path, self._config)
                    document = parser.parse(file_path)
                    doc_id = compute_doc_id(Path(file_path).resolve(), docs_path.resolve())
                    document.id = doc_id
                    chunks = self._chunker.chunk_document(document)
                    added_docs[doc_id] = chunks
                except Exception as e:
                    logger.warning(f"Failed to parse {file_path} for move detection: {e}")
                    continue

            # Detect moves
            moved_files = self._detect_file_moves(
                set(doc_ids_to_remove),
                added_docs,
            )

            logger.info(
                f"Detected {len(moved_files)} file moves "
                f"(threshold: {self._config.indexing.move_detection_threshold})"
            )

            # Apply moves
            for old_doc_id, new_doc_id in moved_files.items():
                logger.info(f"Applying file move: {old_doc_id} → {new_doc_id}")

                new_chunks = added_docs.get(new_doc_id, [])
                if not new_chunks:
                    logger.warning(f"No chunks found for moved file {new_doc_id}, skipping")
                    continue

                success = self._apply_file_move(old_doc_id, new_doc_id, new_chunks)

                if success:
                    # Remove from to_add and to_remove sets
                    if old_doc_id in doc_ids_to_remove:
                        doc_ids_to_remove.remove(old_doc_id)

                    # Find and remove the file path from files_to_add
                    for file_path in list(files_to_add):
                        if compute_doc_id(Path(file_path).resolve(), docs_path.resolve()) == new_doc_id:
                            files_to_add.remove(file_path)
                            break

                    result.moved_count += 1
                else:
                    logger.info(f"Move operation failed for {old_doc_id} → {new_doc_id}, using full reindex")

        # Process remaining removals
        for doc_id in doc_ids_to_remove:
            try:
                self.remove_document(doc_id)
                result.removed_count += 1
            except Exception as e:
                logger.error(f"Failed to remove {doc_id}: {e}", exc_info=True)
                result.failed_count += 1

        # Process remaining additions
        for file_path in files_to_add:
            try:
                self.index_document(file_path)
                result.added_count += 1
            except Exception as e:
                logger.error(f"Failed to index {file_path}: {e}", exc_info=True)
                result.failed_count += 1

        logger.info(
            f"Reconciliation complete: "
            f"added={result.added_count}, "
            f"removed={result.removed_count}, "
            f"moved={result.moved_count}, "
            f"failed={result.failed_count}"
        )

        return result
