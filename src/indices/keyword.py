import atexit
import shutil
from pathlib import Path
from threading import Lock
from typing import Any, cast

from whoosh import index as whoosh_index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, KEYWORD, TEXT, Schema
from whoosh.qparser import MultifieldParser
from whoosh.scoring import BM25F

from src.models import Chunk, Document


_temp_dirs: set[Path] = set()


def _cleanup_temp_dirs() -> None:
    for temp_dir in list(_temp_dirs):
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
    _temp_dirs.clear()


atexit.register(_cleanup_temp_dirs)


STOPWORDS: frozenset[str] = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "for", "if", "in", "into", "is", "it", "no", "not", "of",
    "on", "or", "such", "that", "the", "their", "then", "there",
    "these", "they", "this", "to", "was", "will", "with",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "its", "itself", "them", "themselves",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "all", "each", "both", "few", "more", "most", "other", "some",
    "any", "only", "own", "same", "so", "than", "too", "very",
    "can", "just", "should", "now", "d", "ll", "m", "o", "re", "ve",
    "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn",
    "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn",
    "wasn", "weren", "won", "wouldn", "do", "does", "did", "doing",
    "has", "have", "had", "having", "been", "being", "would", "could",
    "might", "must", "shall", "am", "were", "about", "above", "after",
    "again", "against", "before", "below", "between", "during", "from",
    "further", "here", "once", "out", "over", "under", "until", "up",
    "while",
])


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

            writer = self._index.writer()
            try:
                writer.update_document(
                    id=document.id,
                    content=document.content,
                    aliases=aliases_text,
                    tags=tags_text,
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
            description = str(metadata.get("description", "") or metadata.get("summary", ""))
            keywords_list = metadata.get("keywords", [])
            keywords_text = " ".join(keywords_list) if isinstance(keywords_list, list) else str(keywords_list)
            aliases_list = metadata.get("aliases", [])
            aliases_text = " ".join(str(a) for a in aliases_list) if isinstance(aliases_list, list) else str(aliases_list)
            author = str(metadata.get("author", ""))
            category = str(metadata.get("category", ""))

            writer = self._index.writer()
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

            writer = self._index.writer()
            try:
                writer.delete_by_term("id", document_id)
                writer.commit()
            except Exception:
                writer.cancel()
                raise

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        with self._lock:
            if self._index is None or not query.strip():
                return []

            searcher = self._index.searcher(weighting=BM25F())
            parser = MultifieldParser(
                ["content", "title", "headers", "description", "keywords", "aliases", "tags", "author"],
                schema=self._schema
            )

            try:
                parsed_query = parser.parse(query)
                results = searcher.search(parsed_query, limit=top_k)

                chunk_results = []
                for hit in results:
                    chunk_results.append({
                        "chunk_id": hit["id"],
                        "doc_id": hit.get("doc_id", hit["id"]),
                        "score": hit.score,
                    })
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
            keywords_text = " ".join(keywords_list) if isinstance(keywords_list, list) else str(keywords_list)
            aliases_list = metadata.get("aliases", [])
            aliases_text = " ".join(str(a) for a in aliases_list) if isinstance(aliases_list, list) else str(aliases_list)

            writer = self._index.writer()
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
