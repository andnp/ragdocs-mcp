"""TimestampSelector: domain policy for choosing a Record's recency timestamp.

Pure data/config -- no I/O, no adapters -- so "which timestamp counts as
this record's recency for time-based score boosting" is a policy
decision rather than repeated ad-hoc field access at each call site.

Default policy prefers `updated_at` over `created_at` -- `Record`'s own
docstring names `updated_at` as "the watermark for incremental sync", so
an edited-but-old record should still read as recent. Not wired into any
live scoring path yet (`RecencyBoostStage`'s time-scoring mode is never
enabled in the live config today); this exists so that integration can
consume one policy object instead of reinventing timestamp selection --
see the W4a plan's domain-policy-providers item.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from searchkernel.domain.models import Record


@dataclass(frozen=True)
class TimestampSelector:
    """Picks which of a `Record`'s timestamp fields represents recency."""

    prefer_updated_at: bool = True

    def select(self, record: Record) -> datetime:
        if self.prefer_updated_at:
            return record.updated_at
        return record.created_at


DEFAULT_TIMESTAMP_SELECTOR = TimestampSelector()
