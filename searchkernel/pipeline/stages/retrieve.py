"""RetrieveStage: concurrent vector+keyword retrieval query stage.

Lifted from SearchOrchestrator.query's asyncio.gather over
_search_vector/_search_keyword. Retrieval is I/O-bound (index lookups run
in a thread), so this is the first stage in the query toolkit that must
await -- it implements AsyncSearchStage rather than SearchStage.

Parameterized over the two async searcher callables (not concrete index
objects) so the same stage composes over the live orchestrator's
_search_vector/_search_keyword -- monkeypatchable in tests exactly as
before -- or any other async
`(query, top_k, excluded_files, docs_root) -> list[SearchResultDict]`
callable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path

from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.types import SearchResultDict

Searcher = Callable[
    [str, int, "set[str] | None", Path], Awaitable[list[SearchResultDict]]
]

_TOP_K_KEY = "top_k"
_EXCLUDED_FILES_KEY = "excluded_files"
_DOCS_ROOT_KEY = "docs_root"
_VECTOR_RESULTS_KEY = "vector_results"
_KEYWORD_RESULTS_KEY = "keyword_results"


class RetrieveStage:
    """Run vector + keyword retrieval concurrently.

    Expects `context.metadata` to carry `top_k` (int), `excluded_files`
    (set[str] | None) and `docs_root` (Path). Writes the raw per-strategy
    result lists to `context.metadata["vector_results"]` /
    `["keyword_results"]` -- the same dict shape
    (`chunk_id`/`doc_id`/`score`) callers consumed directly before.
    """

    name = "retrieve"

    def __init__(self, search_vector: Searcher, search_keyword: Searcher):
        self._search_vector = search_vector
        self._search_keyword = search_keyword

    async def run(self, context: SearchContext) -> SearchContext:
        top_k = context.metadata[_TOP_K_KEY]
        excluded_files = context.metadata.get(_EXCLUDED_FILES_KEY)
        docs_root = context.metadata[_DOCS_ROOT_KEY]

        vector_results, keyword_results = await asyncio.gather(
            self._search_vector(context.query, top_k, excluded_files, docs_root),
            self._search_keyword(context.query, top_k, excluded_files, docs_root),
        )

        metadata = dict(context.metadata)
        metadata[_VECTOR_RESULTS_KEY] = vector_results
        metadata[_KEYWORD_RESULTS_KEY] = keyword_results
        return replace(context, metadata=metadata)
