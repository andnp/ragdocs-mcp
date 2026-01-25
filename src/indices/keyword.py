import atexit
import logging
import shutil
import time
from pathlib import Path
from threading import Lock
from typing import Any, cast

from whoosh import index as whoosh_index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, KEYWORD, TEXT, Schema
from whoosh.index import LockError
from whoosh.qparser import MultifieldParser
from whoosh.scoring import BM25F

from src.models import Chunk, Document

logger = logging.getLogger(__name__)

WRITER_TIMEOUT = 5.0
WRITER_MAX_RETRIES = 3
WRITER_RETRY_DELAY = 0.5

_temp_dirs: set[Path] = set()


def _cleanup_temp_dirs() -> None:
    for temp_dir in list(_temp_dirs):
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
    _temp_dirs.clear()


atexit.register(_cleanup_temp_dirs)


STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "its",
        "itself",
        "them",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "any",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "just",
        "should",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
        "ain",
        "aren",
        "couldn",
        "didn",
        "doesn",
        "hadn",
        "hasn",
        "haven",
        "isn",
        "ma",
        "mightn",
        "mustn",
        "needn",
        "shan",
        "shouldn",
        "wasn",
        "weren",
        "won",
        "wouldn",
        "do",
        "does",
        "did",
        "doing",
        "has",
        "have",
        "had",
        "having",
        "been",
        "being",
        "would",
        "could",
        "might",
        "must",
        "shall",
        "am",
        "were",
        "about",
        "above",
        "after",
        "again",
        "against",
        "before",
        "below",
        "between",
        "during",
        "from",
        "further",
        "here",
        "once",
        "out",
        "over",
        "under",
        "until",
        "up",
        "while",
    ]
)


class KeywordIndex:
    def __init__(self):
        stem_analyzer = StemmingAnalyzer(stoplist=cast(Any, STOPWORDS), minsize=2)

        self._schema = Schema(
            id=ID(stored=True, unique=True),
            doc_id=ID(stored=True),
            content=TEXT(stored=False, analyzer=stem_analyzer),
            title=TEXT(stored=False, analyzer=stem_analyzer, field_boost=3.0),
            headers=TEXT(stored=False, analyzer=stem_analyzer, field_boost=2.5),
            description=TEXT(stored=False, analyzer=stem_analyzer, field_boost=2.0),
            keywords=TEXT(stored=False, analyzer=stem_analyzer, field_boost=2.5),
            aliases=TEXT(stored=False, analyzer=stem_analyzer, field_boost=1.5),
            tags=KEYWORD(stored=False, commas=True, field_boost=2.0),
            author=TEXT(stored=False, analyzer=stem_analyzer, field_boost=1.0),
            category=KEYWORD(stored=False),
        )
        self._index = None
        self._index_path = None
        self._lock = Lock()

    def _get_writer(self):
        assert self._index is not None
        for attempt in range(WRITER_MAX_RETRIES):
            try:
                return self._index.writer(timeout=WRITER_TIMEOUT)
            except LockError:
                if attempt == WRITER_MAX_RETRIES - 1:
                    raise
                logger.warning(
                    f"Whoosh writer lock contention, retrying ({attempt + 1}/{WRITER_MAX_RETRIES})..."
                )
                time.sleep(WRITER_RETRY_DELAY * (attempt + 1))
        raise LockError("Failed to acquire writer lock after retries")

    def add(self, document: Document) -> None:
        with self._lock:
            if self._index is None:
                self._initialize_index()

            assert self._index is not None

            aliases = document.metadata.get("aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            elif not isinstance(aliases, list):
                aliases = []

            aliases_text = " ".join(str(a) for a in aliases)
            tags_text = ",".join(document.tags) if document.tags else ""
            title = str(document.metadata.get("title", ""))
            description = str(
                document.metadata.get("description", "")
                or document.metadata.get("summary", "")
            )
            keywords_list = document.metadata.get("keywords", [])
            keywords_text = (
                " ".join(keywords_list)
                if isinstance(keywords_list, list)
                else str(keywords_list)
            )
            author = str(document.metadata.get("author", ""))
            category = str(document.metadata.get("category", ""))

            writer = self._get_writer()
            try:
                writer.update_document(
                    id=document.id,
                    doc_id=document.id,
                    content=document.content,
                    title=title,
                    headers="",
                    description=description,
                    keywords=keywords_text,
                    aliases=aliases_text,
                    tags=tags_text,
                    author=author,
                    category=category,
                )
                writer.commit()
            except Exception:
                writer.cancel()
                raise

    def add_chunk(self, chunk: Chunk) -> None:
        with self._lock:
            if self._index is None:
                self._initialize_index()

            assert self._index is not None

            metadata = chunk.metadata
            tags_text = ",".join(metadata.get("tags", []))
            title = str(metadata.get("title", ""))
            headers = chunk.header_path or ""
            description = str(
                metadata.get("description", "") or metadata.get("summary", "")
            )
            keywords_list = metadata.get("keywords", [])
            keywords_text = (
                " ".join(keywords_list)
                if isinstance(keywords_list, list)
                else str(keywords_list)
            )
            aliases_list = metadata.get("aliases", [])
            aliases_text = (
                " ".join(str(a) for a in aliases_list)
                if isinstance(aliases_list, list)
                else str(aliases_list)
            )
            author = str(metadata.get("author", ""))
            category = str(metadata.get("category", ""))

            writer = self._get_writer()
            try:
                writer.update_document(
                    id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=chunk.content,
                    title=title,
                    headers=headers,
                    description=description,
                    keywords=keywords_text,
                    aliases=aliases_text,
                    tags=tags_text,
                    author=author,
                    category=category,
                )
                writer.commit()
            except Exception:
                writer.cancel()
                raise

    def remove(self, document_id: str) -> None:
        with self._lock:
            if self._index is None:
                return

            try:
                writer = self._get_writer()
                try:
                    writer.delete_by_term("id", document_id)
                    writer.delete_by_term("doc_id", document_id)
                    writer.commit()
                except Exception:
                    writer.cancel()
                    raise
            except (FileNotFoundError, OSError) as e:
                logger.warning(
                    f"Keyword index corruption detected during remove({document_id}): {e}. "
                    "Reinitializing index."
                )
                self._reinitialize_after_corruption()

    def remove_chunk(self, chunk_id: str) -> None:
        """Remove a specific chunk from keyword index.

        Thread-safe operation. Handles missing chunks and corruption gracefully.
        """
        with self._lock:
            if self._index is None:
                logger.warning(f"Cannot remove chunk {chunk_id}: index not initialized")
                return

            try:
                writer = self._get_writer()
                try:
                    writer.delete_by_term("id", chunk_id)
                    writer.commit()
                    logger.debug(f"Removed chunk {chunk_id} from keyword index")
                except Exception:
                    writer.cancel()
                    raise
            except (FileNotFoundError, OSError) as e:
                logger.warning(
                    f"Keyword index corruption detected during remove_chunk({chunk_id}): {e}. "
                    "Reinitializing index.",
                    exc_info=True,
                )
                self._reinitialize_after_corruption()
            except Exception as e:
                logger.warning(
                    f"Failed to remove chunk {chunk_id} from keyword index: {e}",
                    exc_info=True,
                )

    def move_chunk(self, old_chunk_id: str, new_chunk: Chunk) -> bool:
        """Move chunk to new ID with updated metadata (for file moves).

        Copies document with new ID and updated content/metadata, then deletes old document.
        More efficient than full re-indexing since content is not stored in index.

        Args:
            old_chunk_id: Chunk ID to move from
            new_chunk: New chunk object with updated path and metadata

        Returns:
            True if move successful, False otherwise
        """
        with self._lock:
            if self._index is None:
                logger.warning("Cannot move chunk: index not initialized")
                return False

            try:
                # Check if old document exists
                searcher = self._index.searcher()
                try:
                    old_exists = any(searcher.documents(id=old_chunk_id))
                finally:
                    searcher.close()

                if not old_exists:
                    logger.debug(f"Chunk {old_chunk_id} not found in keyword index")
                    return False

                # Extract metadata from new chunk
                metadata = new_chunk.metadata
                tags_text = ",".join(metadata.get("tags", []))
                title = str(metadata.get("title", ""))
                headers = new_chunk.header_path or ""
                description = str(
                    metadata.get("description", "") or metadata.get("summary", "")
                )
                keywords_list = metadata.get("keywords", [])
                keywords_text = (
                    " ".join(keywords_list)
                    if isinstance(keywords_list, list)
                    else str(keywords_list)
                )
                aliases_list = metadata.get("aliases", [])
                aliases_text = (
                    " ".join(str(a) for a in aliases_list)
                    if isinstance(aliases_list, list)
                    else str(aliases_list)
                )
                author = str(metadata.get("author", ""))
                category = str(metadata.get("category", ""))

                # Add new document first and commit
                writer = self._get_writer()
                try:
                    writer.add_document(
                        id=new_chunk.chunk_id,
                        doc_id=new_chunk.doc_id,
                        content=new_chunk.content,
                        title=title,
                        headers=headers,
                        description=description,
                        keywords=keywords_text,
                        aliases=aliases_text,
                        tags=tags_text,
                        author=author,
                        category=category,
                    )
                    writer.commit()
                except Exception:
                    writer.cancel()
                    raise

                # Delete old document in separate transaction
                writer = self._get_writer()
                try:
                    writer.delete_by_term("id", old_chunk_id)
                    writer.commit()
                    logger.debug(f"Moved chunk in keyword index: {old_chunk_id} -> {new_chunk.chunk_id}")
                    return True
                except Exception:
                    writer.cancel()
                    raise

            except (FileNotFoundError, OSError) as e:
                logger.warning(
                    f"Keyword index corruption detected during move_chunk: {e}. "
                    "Reinitializing index.",
                    exc_info=True,
                )
                self._reinitialize_after_corruption()
                return False
            except Exception as e:
                logger.warning(
                    f"Failed to move chunk {old_chunk_id} -> {new_chunk.chunk_id}: {e}",
                    exc_info=True,
                )
                return False

    def search(
        self,
        query: str,
        top_k: int = 10,
        excluded_files: set[str] | None = None,
        docs_root: Path | None = None,
    ) -> list[dict]:
        with self._lock:
            if self._index is None or not query.strip():
                return []

            try:
                searcher = self._index.searcher(weighting=BM25F())
            except (FileNotFoundError, OSError) as e:
                logger.warning(
                    f"Keyword index corruption detected during search: {e}. "
                    "Reinitializing index and returning empty results."
                )
                self._reinitialize_after_corruption()
                return []

            parser = MultifieldParser(
                [
                    "content",
                    "title",
                    "headers",
                    "description",
                    "keywords",
                    "aliases",
                    "tags",
                    "author",
                ],
                schema=self._schema,
            )

            try:
                parsed_query = parser.parse(query)

                fetch_k = top_k * 2 if excluded_files else top_k
                results = searcher.search(parsed_query, limit=fetch_k)

                chunk_results = []
                for hit in results:
                    if excluded_files and docs_root:
                        doc_id = hit.get("doc_id", hit["id"])
                        from src.search.path_utils import normalize_path

                        normalized_doc_id = normalize_path(doc_id, docs_root)

                        if normalized_doc_id in excluded_files:
                            continue

                        from pathlib import Path as PathLib

                        filename = PathLib(normalized_doc_id).name
                        if filename in excluded_files:
                            continue

                    chunk_results.append(
                        {
                            "chunk_id": hit["id"],
                            "doc_id": hit.get("doc_id", hit["id"]),
                            "score": hit.score,
                        }
                    )

                    if len(chunk_results) >= top_k:
                        break

                return chunk_results
            finally:
                searcher.close()

    def persist(self, path: Path) -> None:
        with self._lock:
            if self._index is None:
                return

            path.mkdir(parents=True, exist_ok=True)

            if self._index_path != path:
                import shutil

                if self._index_path and self._index_path.exists():
                    for item in self._index_path.iterdir():
                        dest = path / item.name
                        if item.is_file():
                            shutil.copy2(item, dest)

    def persist_to(self, snapshot_dir: Path) -> None:
        self.persist(snapshot_dir)

    def load_from(self, snapshot_dir: Path) -> bool:
        if not snapshot_dir.exists():
            return False

        try:
            with self._lock:
                existing_index = whoosh_index.open_dir(str(snapshot_dir))
                existing_fields = set(existing_index.schema.names())
                expected_fields = set(self._schema.names())

                if existing_fields != expected_fields:
                    existing_index.close()
                    return False

                self._index = existing_index
                self._index_path = snapshot_dir
                return True
        except (whoosh_index.EmptyIndexError, FileNotFoundError):
            return False

    def load(self, path: Path) -> None:
        with self._lock:
            if not path.exists():
                self._initialize_index()
                return

            try:
                existing_index = whoosh_index.open_dir(str(path))
                existing_fields = set(existing_index.schema.names())
                expected_fields = set(self._schema.names())

                if existing_fields != expected_fields:
                    import logging

                    logger = logging.getLogger(__name__)
                    missing = expected_fields - existing_fields
                    extra = existing_fields - expected_fields
                    logger.warning(
                        f"Keyword index schema mismatch. Missing: {missing}, Extra: {extra}. "
                        "Rebuilding index."
                    )
                    existing_index.close()
                    self._initialize_index()
                    return

                self._index = existing_index
                self._index_path = path
            except (whoosh_index.EmptyIndexError, FileNotFoundError):
                self._initialize_index()

    def _initialize_index(self) -> None:
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="whoosh_"))
        _temp_dirs.add(temp_dir)
        self._index = whoosh_index.create_in(str(temp_dir), self._schema)
        self._index_path = temp_dir

    def _reinitialize_after_corruption(self):
        if self._index_path and self._index_path in _temp_dirs:
            shutil.rmtree(self._index_path, ignore_errors=True)
            _temp_dirs.discard(self._index_path)
        self._index = None
        self._index_path = None
        self._initialize_index()

    # IndexProtocol methods

    def add_document(self, doc_id: str, content: str, metadata: dict[str, Any]) -> None:
        with self._lock:
            if self._index is None:
                self._initialize_index()

            assert self._index is not None

            tags = metadata.get("tags", [])
            tags_text = ",".join(tags) if isinstance(tags, list) else str(tags)
            title = str(metadata.get("title", ""))
            description = str(metadata.get("description", ""))
            keywords_list = metadata.get("keywords", [])
            keywords_text = (
                " ".join(keywords_list)
                if isinstance(keywords_list, list)
                else str(keywords_list)
            )
            aliases_list = metadata.get("aliases", [])
            aliases_text = (
                " ".join(str(a) for a in aliases_list)
                if isinstance(aliases_list, list)
                else str(aliases_list)
            )

            writer = self._get_writer()
            try:
                writer.update_document(
                    id=doc_id,
                    doc_id=doc_id,
                    content=content,
                    title=title,
                    description=description,
                    keywords=keywords_text,
                    aliases=aliases_text,
                    tags=tags_text,
                )
                writer.commit()
            except Exception:
                writer.cancel()
                raise

    def remove_document(self, doc_id: str) -> None:
        self.remove(doc_id)

    def clear(self) -> None:
        with self._lock:
            if self._index_path and self._index_path in _temp_dirs:
                shutil.rmtree(self._index_path, ignore_errors=True)
                _temp_dirs.discard(self._index_path)
            self._index = None
            self._index_path = None

    def save(self, path: Path) -> None:
        self.persist(path)

    def __len__(self) -> int:
        with self._lock:
            if self._index is None:
                return 0
            return self._index.doc_count()
