"""StatusMultiplierPolicy: domain policy mapping `RecordStatus` to a score multiplier.

Pure data/config -- no I/O, no adapters -- so lifecycle-status-based
score adjustment (e.g. down-weighting stale/archived records) is a
policy edit, not hardcoded scoring logic. `Record.status` already
exists; nothing in the live query path reads it yet (no stage applies a
status multiplier today), so this ships as scaffolding future scoring
work can consume -- see the W4a plan's domain-policy-providers item.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from searchkernel.domain.models import RecordStatus


@dataclass(frozen=True)
class StatusMultiplierPolicy:
    """Score multiplier per `RecordStatus`."""

    multipliers: dict[RecordStatus, float] = field(
        default_factory=lambda: {
            RecordStatus.ACTIVE: 1.0,
            RecordStatus.STALE: 0.8,
            RecordStatus.ARCHIVED: 0.5,
        }
    )

    def multiplier_for(self, status: RecordStatus) -> float:
        return self.multipliers.get(status, 1.0)


DEFAULT_STATUS_MULTIPLIER_POLICY = StatusMultiplierPolicy()
