from pathlib import Path
from threading import Lock

from whoosh import index as whoosh_index
from whoosh.fields import ID, KEYWORD, TEXT, Schema
from whoosh.qparser import MultifieldParser
from whoosh.scoring import BM25F

from src.models import Document


class KeywordIndex:
    def __init__(self):
        self._schema = Schema(
            id=ID(stored=True, unique=True),
            content=TEXT(stored=False),
            aliases=TEXT(stored=False),
            tags=KEYWORD(stored=False, commas=True),
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

    def search(self, query: str, top_k: int = 10):
        with self._lock:
            if self._index is None or not query.strip():
                return []

            searcher = self._index.searcher(weighting=BM25F())
            parser = MultifieldParser(
                ["content", "aliases", "tags"], schema=self._schema
            )

            try:
                parsed_query = parser.parse(query)
                results = searcher.search(parsed_query, limit=top_k)
                doc_ids = [hit["id"] for hit in results]
                return doc_ids
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
