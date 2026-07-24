"""ContentHashDedupStage: the exact-text-match dedup query stage.

One of the per-concern stages the dedup toolkit decomposes into (see the
W4a plan). Delegates straight to deduplicate_by_content_hash -- same
inputs, same outputs -- so it is a pure extraction with no behavior
change.
"""

from __future__ import annotations

from dataclasses import replace

from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.dedup import deduplicate_by_content_hash

_GET_CONTENT_KEY = "get_content"


class ContentHashDedupStage:
    """Drop candidates whose content hashes to an already-seen value.

    Expects `context.metadata["get_content"]` (`Callable[[str], str | None]`).
    """

    name = "dedup_content_hash"

    def run(self, context: SearchContext) -> SearchContext:
        get_content = context.metadata[_GET_CONTENT_KEY]
        deduped, _removed = deduplicate_by_content_hash(context.candidates, get_content)
        return replace(context, candidates=deduped)
