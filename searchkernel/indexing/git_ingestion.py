"""Wires GitContentSource into the live IndexManager.

Ingests git commits as Records through the same chunking/indexing path as
documents, so they land in the shared vector/keyword/graph store and become
discoverable via SearchOrchestrator.query(source_filter=["git_commit"]).
"""

import logging

from searchkernel.adapters.sources.git import GitContentSource
from searchkernel.domain import Cursor
from searchkernel.indexing.manager import IndexManager

logger = logging.getLogger(__name__)


def ingest_git_source(
    index_manager: IndexManager,
    source: GitContentSource,
    since: Cursor | None = None,
) -> int:
    """Ingest every record yielded by a GitContentSource into index_manager.

    Returns the number of records ingested.
    """
    count = 0
    for record in source.iter_records(since):
        index_manager.index_record(record)
        count += 1

    if count:
        logger.info(
            "Ingested %d git commit(s) from %s into the live index",
            count,
            source.repo_path,
        )

    return count
