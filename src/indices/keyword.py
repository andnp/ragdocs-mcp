from pathlib import Path
from threading import Lock
from typing import Any

from whoosh import index as whoosh_index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, KEYWORD, TEXT, Schema
from whoosh.qparser import MultifieldParser
from whoosh.scoring import BM25F

from src.models import Chunk, Document


STOPWORDS: Any = frozenset([
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
        stem_analyzer = StemmingAnalyzer(stoplist=STOPWORDS, minsize=2)

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
                self._index = whoosh_index.open_dir(str(path))
                self._index_path = path
            except (whoosh_index.EmptyIndexError, FileNotFoundError):
                self._initialize_index()

    def _initialize_index(self) -> None:
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="whoosh_"))
        self._index = whoosh_index.create_in(str(temp_dir), self._schema)
        self._index_path = temp_dir
