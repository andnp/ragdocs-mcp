"""TagExpansionStage: tag-graph query expansion merged into result bookkeeping.

Lifted from the inline tag-expansion block in `SearchOrchestrator.query` --
expands the initial vector+keyword results via tag-graph traversal, then
merges newly-discovered chunks into the running `vector_results` list and
the `chunk_id_to_doc_id`/`all_doc_ids` bookkeeping the rest of the query
pipeline threads along, exactly as before extraction. Skippable via
`skip_tag_expansion` for the factual-query fast path, mirroring `query()`'s
existing short-circuit.

Also republishes the post-merge bookkeeping under the key names
`GraphExpandStage` (`excluded_chunk_ids`) and `CommunityBoostStage`
(`seed_doc_ids`) expect, since both stages run after tag expansion and
`query()`'s inline glue built these two views (a chunk-id set, a doc-id
set) from the exact same merged `chunk_id_to_doc_id`/`all_doc_ids`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from searchkernel.pipeline.stage import SearchContext

ExpandQueryWithTags = Callable[[list[dict[str, Any]], int], list[dict[str, Any]]]

_VECTOR_RESULTS_KEY = "vector_results"
_KEYWORD_RESULTS_KEY = "keyword_results"
_CHUNK_ID_TO_DOC_ID_KEY = "chunk_id_to_doc_id"
_ALL_DOC_IDS_KEY = "all_doc_ids"
_TOP_K_KEY = "top_k"
_SKIP_KEY = "skip_tag_expansion"
_TAG_EXPANSION_COUNT_KEY = "tag_expansion_count"
_APPLIED_TAG_EXPANSION_RESULTS_KEY = "applied_tag_expansion_results"
_SEED_DOC_IDS_KEY = "seed_doc_ids"
_EXCLUDED_CHUNK_IDS_KEY = "excluded_chunk_ids"


class TagExpansionStage:
    """Expand results via tag-graph traversal, merging new chunks into bookkeeping.

    Expects `context.metadata["vector_results"]`/`["keyword_results"]`
    (`list[dict]`), `["chunk_id_to_doc_id"]` (`dict[str, str]`),
    `["all_doc_ids"]` (`set[str]`), `["top_k"]` (`int`) and optionally
    `["skip_tag_expansion"]` (`bool`, default `False`). Writes updated
    `["vector_results"]`/`["chunk_id_to_doc_id"]`/`["all_doc_ids"]` (newly
    discovered chunks merged in), `["tag_expansion_count"]` (`int`),
    `["applied_tag_expansion_results"]` (`list[dict]`, the merged chunks
    only), `["seed_doc_ids"]` (`set[str]`, alias of the merged
    `all_doc_ids`) and `["excluded_chunk_ids"]` (`set[str]`, the merged
    `chunk_id_to_doc_id`'s keys).
    """

    name = "tag_expansion"

    def __init__(self, expand_query_with_tags: ExpandQueryWithTags):
        self._expand_query_with_tags = expand_query_with_tags

    def run(self, context: SearchContext) -> SearchContext:
        vector_results = list(context.metadata[_VECTOR_RESULTS_KEY])
        keyword_results = context.metadata[_KEYWORD_RESULTS_KEY]
        chunk_id_to_doc_id = dict(context.metadata[_CHUNK_ID_TO_DOC_ID_KEY])
        all_doc_ids = set(context.metadata[_ALL_DOC_IDS_KEY])
        top_k = context.metadata[_TOP_K_KEY]
        skip = context.metadata.get(_SKIP_KEY, False)

        if skip:
            tag_expanded_results: list[dict[str, Any]] = []
        else:
            tag_expanded_results = self._expand_query_with_tags(
                vector_results + keyword_results, top_k
            )

        applied: list[dict[str, Any]] = []
        tag_expansion_count = 0
        for result in tag_expanded_results:
            chunk_id = result["chunk_id"]
            doc_id = result["doc_id"]
            if chunk_id not in chunk_id_to_doc_id:
                all_doc_ids.add(doc_id)
                chunk_id_to_doc_id[chunk_id] = doc_id
                vector_results.append(result)
                applied.append(result)
                tag_expansion_count += 1

        metadata = dict(context.metadata)
        metadata[_VECTOR_RESULTS_KEY] = vector_results
        metadata[_CHUNK_ID_TO_DOC_ID_KEY] = chunk_id_to_doc_id
        metadata[_ALL_DOC_IDS_KEY] = all_doc_ids
        metadata[_TAG_EXPANSION_COUNT_KEY] = tag_expansion_count
        metadata[_APPLIED_TAG_EXPANSION_RESULTS_KEY] = applied
        metadata[_SEED_DOC_IDS_KEY] = all_doc_ids
        metadata[_EXCLUDED_CHUNK_IDS_KEY] = set(chunk_id_to_doc_id)
        return replace(context, metadata=metadata)
