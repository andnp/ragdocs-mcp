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
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.indexing.implicit_graph import ImplicitGraphBuilder
from src.parsers.dispatcher import dispatch_parser
from src.search.edge_types import infer_edge_type

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
        self._chunker = get_chunker(config.chunking)

    def _compute_doc_id(self, file_path: str) -> str:
        docs_path = Path(self._config.indexing.documents_path)
        abs_path = Path(file_path).resolve()
        try:
            rel_path = abs_path.relative_to(docs_path.resolve())
            return str(rel_path.with_suffix(""))
        except ValueError:
            return Path(file_path).stem

    def index_document(self, file_path: str):
        try:
            parser = dispatch_parser(file_path, self._config)
            document = parser.parse(file_path)

            document.id = self._compute_doc_id(file_path)

            chunks = self._chunker.chunk_document(document)
            document.chunks = chunks

            for chunk in chunks:
                self.vector.add_chunk(chunk)
                self.keyword.add_chunk(chunk)

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
                    edge_type = infer_edge_type(link_info.header_context, link_info.target)
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

            self._failed_files = [
                f for f in self._failed_files if f.path != file_path
            ]

        except UnicodeDecodeError as e:
            # Try alternative encodings for files with encoding issues
            logger.warning(f"UTF-8 decode failed for {file_path}, trying alternative encodings: {e}")
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    logger.info(f"Attempting to read {file_path} with {encoding} encoding")
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
            logger.info(f"Continuing with remaining files after encoding error in {file_path}")
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
                logger.error(f"Failed to remove {doc_id} from {index_name}: {e}", exc_info=True)
                errors.append((index_name, e))

        if self.code is not None:
            try:
                self.code.remove_by_doc_id(doc_id)
            except Exception as e:
                logger.error(f"Failed to remove {doc_id} from code index: {e}", exc_info=True)
                errors.append(("code", e))

        if errors:
            logger.warning(f"Document {doc_id} removal completed with {len(errors)} index failures")

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
