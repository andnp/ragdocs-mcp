from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, cast


type VocabularyLifecycleStatus = Literal[
    "absent",
    "stale",
    "building",
    "ready",
    "failed",
]


@dataclass
class VocabularyLifecycleState:
    status: VocabularyLifecycleStatus = "absent"
    authoritative_revision: int = 0
    materialized_revision: int = 0
    last_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "authoritative_revision": self.authoritative_revision,
            "materialized_revision": self.materialized_revision,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object] | None,
        *,
        has_authoritative_terms: bool,
        has_materialized_terms: bool,
    ) -> VocabularyLifecycleState:
        if payload is None:
            return cls(
                status=(
                    "ready"
                    if has_materialized_terms
                    else "stale" if has_authoritative_terms else "absent"
                ),
                authoritative_revision=1 if has_authoritative_terms else 0,
                materialized_revision=1 if has_materialized_terms else 0,
            )

        status = payload.get("status")
        state = cls(
            status=(
                cast(VocabularyLifecycleStatus, status)
                if isinstance(status, str)
                and status in {"absent", "stale", "building", "ready", "failed"}
                else "absent"
            ),
            authoritative_revision=(
                cast(int, payload.get("authoritative_revision"))
                if isinstance(payload.get("authoritative_revision"), int)
                else 0
            ),
            materialized_revision=(
                cast(int, payload.get("materialized_revision"))
                if isinstance(payload.get("materialized_revision"), int)
                else 0
            ),
            last_error=cast(str | None, payload.get("last_error"))
            if isinstance(payload.get("last_error"), str)
            else None,
        )
        state.reconcile_loaded_state(
            has_authoritative_terms=has_authoritative_terms,
            has_materialized_terms=has_materialized_terms,
        )
        return state

    def reconcile_loaded_state(
        self,
        *,
        has_authoritative_terms: bool,
        has_materialized_terms: bool,
    ) -> None:
        if has_authoritative_terms and self.authoritative_revision == 0:
            self.authoritative_revision = 1
        if has_materialized_terms and self.materialized_revision == 0:
            self.materialized_revision = 1
        if self.authoritative_revision < self.materialized_revision:
            self.authoritative_revision = self.materialized_revision

        if self.status == "building":
            self.status = "stale"

        if has_materialized_terms and self.materialized_revision >= self.authoritative_revision:
            self.status = "ready"
            self.last_error = None
            return

        if has_authoritative_terms or has_materialized_terms:
            self.status = "stale"
            return

        self.status = "absent"
        self.authoritative_revision = 0
        self.materialized_revision = 0
        self.last_error = None

    def mark_authoritative_mutation(self) -> None:
        self.authoritative_revision += 1
        self.status = "stale"
        self.last_error = None

    def begin_build(self) -> int:
        self.status = "building"
        self.last_error = None
        return self.authoritative_revision

    def finish_build(self, *, build_revision: int, caught_up: bool) -> None:
        if caught_up:
            self.materialized_revision = max(self.materialized_revision, build_revision)

        if caught_up and build_revision == self.authoritative_revision:
            self.status = "ready"
            self.last_error = None
            return

        if self.authoritative_revision == 0 and self.materialized_revision == 0:
            self.status = "absent"
            self.last_error = None
            return

        self.status = "stale"
        self.last_error = None

    def mark_failed(self, error: str) -> None:
        self.status = "failed"
        self.last_error = error

    def needs_catch_up(self) -> bool:
        if self.status == "failed":
            return True
        return self.authoritative_revision > self.materialized_revision

    def ready_for_query_expansion(self) -> bool:
        return (
            self.status == "ready"
            and self.materialized_revision >= self.authoritative_revision
        )
