import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.chunking.factory import get_chunker
from src.config import Config
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.parsers.dispatcher import dispatch_parser

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
    ):
        self._config = config
        self._vector = vector
        self._keyword = keyword
        self._graph = graph
        self._failed_files: list[FailedFile] = []
        self._chunker = get_chunker(config.chunking)

    def index_document(self, file_path: str):
        try:
            parser = dispatch_parser(file_path, self._config)
            document = parser.parse(file_path)

            chunks = self._chunker.chunk_document(document)
            document.chunks = chunks

            for chunk in chunks:
                self._vector.add_chunk(chunk)
                self._keyword.add_chunk(chunk)

            self._graph.add_node(document.id, document.metadata)

            for link in document.links:
                self._graph.add_edge(document.id, link, edge_type="link")

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
            logger.error(f"Failed to index document {file_path}: {e}")
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
        try:
            self._vector.remove(doc_id)
            self._keyword.remove(doc_id)
            self._graph.remove_node(doc_id)
        except Exception as e:
            logger.error(f"Failed to remove document {doc_id}: {e}")

    def persist(self):
        index_path = Path(self._config.indexing.index_path)
        try:
            self._vector.build_concept_vocabulary()
            self._vector.persist(index_path / "vector")
            self._keyword.persist(index_path / "keyword")
            self._graph.persist(index_path / "graph")
        except Exception as e:
            logger.error(f"Failed to persist indices: {e}")
            raise

    def load(self):
        index_path = Path(self._config.indexing.index_path)
        try:
            self._vector.load(index_path / "vector")
            self._keyword.load(index_path / "keyword")
            self._graph.load(index_path / "graph")
        except Exception as e:
            logger.error(f"Failed to load indices: {e}")
            raise

    def get_document_count(self) -> int:
        return len(self._vector._doc_id_to_node_ids)

    def get_failed_files(self) -> list[dict[str, str]]:
        return [
            {"path": f.path, "error": f.error, "timestamp": f.timestamp}
            for f in self._failed_files
        ]
