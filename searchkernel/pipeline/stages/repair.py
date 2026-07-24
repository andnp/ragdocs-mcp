"""RepairStage: reindex-path-resolution ingestion stage.

Lifted from `IndexManager.reindex_document`'s doc_id -> path
resolution branch, the re-embed/repair phase of the ingestion path
(discover -> chunk -> embed -> index -> dedup/canonicalize ->
re-embed/repair). Re-embedding itself is `remove_document` followed by
`index_document(force=True)`, already covered by
`ChunkStage`/`IndexStage` -- this stage is only the
multi-root-with-single-root-fallback resolution that decides whether a
repair is even possible (`None` means the file is gone and the caller
should prune instead of reindex).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.path_utils import resolve_doc_path, resolve_doc_path_multi_root

_DOC_ID_KEY = "doc_id"
_DOCS_PATH_KEY = "docs_path"
_DOCUMENTS_ROOTS_KEY = "documents_roots"
_SUFFIXES_KEY = "suffixes"
_RESOLVED_PATH_KEY = "resolved_path"


class RepairStage:
    """Resolve a stale doc_id back to its on-disk file path for reindexing.

    Expects `context.metadata["doc_id"]` (str), `["docs_path"]`
    (`Path`), `["documents_roots"]` (`list[Path]`) and `["suffixes"]`
    (`list[str]`). Writes `context.metadata["resolved_path"]`
    (`Path | None`).
    """

    name = "repair"

    def run(self, context: SearchContext) -> SearchContext:
        doc_id: str = context.metadata[_DOC_ID_KEY]
        docs_path: Path = context.metadata[_DOCS_PATH_KEY]
        documents_roots: list[Path] = context.metadata[_DOCUMENTS_ROOTS_KEY]
        suffixes: list[str] = context.metadata[_SUFFIXES_KEY]

        resolved_path = resolve_doc_path_multi_root(doc_id, documents_roots, suffixes)
        if resolved_path is None:
            resolved_path = resolve_doc_path(doc_id, docs_path, suffixes)

        metadata = dict(context.metadata)
        metadata[_RESOLVED_PATH_KEY] = resolved_path
        return replace(context, metadata=metadata)
