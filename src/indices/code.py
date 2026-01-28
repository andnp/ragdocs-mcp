import atexit
import logging
import re
import shutil
from collections.abc import Generator
from pathlib import Path
from threading import Lock
from typing import Any

from whoosh import index as whoosh_index
from whoosh.analysis import Filter, RegexTokenizer
from whoosh.fields import ID, TEXT, Schema
from whoosh.index import IndexError as WhooshIndexError
from whoosh.qparser import MultifieldParser
from whoosh.scoring import BM25F

from src.models import CodeBlock
from src.search.types import CodeSearchResultDict


_temp_dirs: set[Path] = set()


def _cleanup_temp_dirs() -> None:
    for temp_dir in list(_temp_dirs):
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
    _temp_dirs.clear()


atexit.register(_cleanup_temp_dirs)


logger = logging.getLogger(__name__)


class CamelCaseSplitter(Filter):
    _camel_split_pattern = re.compile(
        r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])'
    )

    def __call__(self, tokens: Any) -> Generator[Any, None, None]:
        for token in tokens:
            text = token.text
            parts = self._camel_split_pattern.split(text)
            if len(parts) > 1:
                yield token
                for part in parts:
                    if part:
                        new_token = token.copy()
                        new_token.text = part.lower()
                        yield new_token
            else:
                yield token


class SnakeCaseSplitter(Filter):
    def __call__(self, tokens: Any) -> Generator[Any, None, None]:
        for token in tokens:
            text = token.text
            if '_' in text:
                parts = text.split('_')
                yield token
                for part in parts:
                    if part:
                        new_token = token.copy()
                        new_token.text = part.lower()
                        yield new_token
            else:
                yield token


_CODE_TOKEN_PATTERN = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+')


def _create_code_analyzer():
    tokenizer = RegexTokenizer(expression=_CODE_TOKEN_PATTERN)
    return tokenizer | CamelCaseSplitter() | SnakeCaseSplitter()


class CodeIndex:
    def __init__(self):
        code_analyzer = _create_code_analyzer()

        self._schema = Schema(
            id=ID(stored=True, unique=True),
            doc_id=ID(stored=True),
            chunk_id=ID(stored=True),
            content=TEXT(stored=True, analyzer=code_analyzer),
            language=ID(stored=True),
        )
        self._index = None
        self._index_path: Path | None = None
        self._lock = Lock()

    def add_code_block(self, code_block: CodeBlock) -> None:
        with self._lock:
            if self._index is None:
                self._initialize_index()

            assert self._index is not None

            writer = self._index.writer()
            try:
                writer.update_document(
                    id=code_block.id,
                    doc_id=code_block.doc_id,
                    chunk_id=code_block.chunk_id,
                    content=code_block.content,
                    language=code_block.language,
                )
                writer.commit()
            except Exception as e:
                logger.warning("Writer operation failed in add_code_block(), cancelling: %s", e, exc_info=True)
                writer.cancel()
                raise

    def remove_by_doc_id(self, doc_id: str) -> None:
        with self._lock:
            if self._index is None:
                return

            try:
                writer = self._index.writer()
                try:
                    writer.delete_by_term("doc_id", doc_id)
                    writer.commit()
                except Exception as e:
                    logger.warning("Writer operation failed in remove_by_doc_id(), cancelling: %s", e, exc_info=True)
                    writer.cancel()
                    raise
            except (FileNotFoundError, OSError) as e:
                logger.warning(
                    f"Code index corruption detected during remove_by_doc_id({doc_id}): {e}. "
                    "Reinitializing index.",
                    exc_info=True,
                )
                self._reinitialize_after_corruption()

    def search(self, query: str, top_k: int = 10) -> list[CodeSearchResultDict]:
        with self._lock:
            if self._index is None or not query.strip():
                return []

            try:
                searcher = self._index.searcher(weighting=BM25F())
            except (FileNotFoundError, OSError) as e:
                logger.warning(
                    f"Code index corruption detected during search: {e}. "
                    "Reinitializing index and returning empty results.",
                    exc_info=True,
                )
                self._reinitialize_after_corruption()
                return []

            parser = MultifieldParser(["content"], schema=self._schema)

            try:
                parsed_query = parser.parse(query)
                results = searcher.search(parsed_query, limit=top_k)

                code_results = []
                for hit in results:
                    code_results.append({
                        "id": hit["id"],
                        "chunk_id": hit["chunk_id"],
                        "doc_id": hit["doc_id"],
                        "language": hit.get("language", ""),
                        "content": hit.get("content", ""),
                        "score": hit.score,
                    })
                return code_results
            finally:
                searcher.close()

    def persist(self, path: Path) -> None:
        with self._lock:
            if self._index is None:
                return

            path.mkdir(parents=True, exist_ok=True)

            if self._index_path != path and self._index_path and self._index_path.exists():
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
                        f"Code index schema mismatch. Missing: {missing}, Extra: {extra}. "
                        "Rebuilding index."
                    )
                    existing_index.close()
                    self._initialize_index()
                    return

                self._index = existing_index
                self._index_path = path
            except (whoosh_index.EmptyIndexError, WhooshIndexError, FileNotFoundError):
                self._initialize_index()

    def _initialize_index(self) -> None:
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="whoosh_code_"))
        _temp_dirs.add(temp_dir)
        self._index = whoosh_index.create_in(str(temp_dir), self._schema)
        self._index_path = temp_dir

    def _reinitialize_after_corruption(self) -> None:
        if self._index_path and self._index_path in _temp_dirs:
            shutil.rmtree(self._index_path, ignore_errors=True)
            _temp_dirs.discard(self._index_path)
        self._index = None
        self._index_path = None
        self._initialize_index()

    def clear(self) -> None:
        with self._lock:
            if self._index_path and self._index_path in _temp_dirs:
                shutil.rmtree(self._index_path, ignore_errors=True)
                _temp_dirs.discard(self._index_path)
            self._index = None
            self._index_path = None

    def __len__(self) -> int:
        with self._lock:
            if self._index is None:
                return 0
            return self._index.doc_count()
