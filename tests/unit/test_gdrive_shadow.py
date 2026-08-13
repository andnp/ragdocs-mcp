"""Tests for deterministic Google Drive shadow comparison artifacts."""

from pathlib import Path
from typing import cast

from searchkernel.api import Record

from mcp_markdown_ragdocs.gdrive.shadow import (
    ShadowArtifactStore,
    ShadowComparisonPolicy,
    ShadowObservation,
    compare_shadow_results,
    persist_index_then_shadow_artifact,
)


def test_comparison_is_source_keyed_and_normalizes_allowed_differences() -> None:
    """
    Stable source IDs keep result order irrelevant and drift categories canonical.
    """
    comparison = compare_shadow_results(
        "drive query",
        [ShadowObservation("b", 2, "owner-b", 4), ShadowObservation("a", 1, "a", 4)],
        [ShadowObservation("a", 3, "shadow-a", 5), ShadowObservation("b", 1, "b", 5)],
    )

    assert list(comparison.entries) == ["a", "b"]
    assert comparison.entries["a"] == {
        "differences": ["chunk", "index_epoch", "ranking"],
        "allowed": ["chunk", "index_epoch", "ranking"],
        "unexpected": [],
    }
    assert comparison.mismatches == 0


def test_comparison_reports_missing_source_and_disallowed_drift() -> None:
    """
    Missing records and policy-disallowed changes remain actionable mismatches.
    """
    comparison = compare_shadow_results(
        "drive query",
        [ShadowObservation("a", 1, "chunk-a", 1), ShadowObservation("gone")],
        [ShadowObservation("a", 2, "chunk-b", 2)],
        policy=ShadowComparisonPolicy(allow_chunk=False),
    )

    assert comparison.entries["a"]["unexpected"] == ["chunk"]
    assert comparison.entries["gone"]["unexpected"] == ["missing_source"]
    assert comparison.mismatches == 2


def test_barrier_persists_artifact_after_index_and_persist(tmp_path: Path) -> None:
    """
    The artifact cannot precede the durable index write barrier.
    """
    events: list[str] = []

    class Writer:
        def index_records(self, records: list[Record]) -> bool:
            events.append("index")
            return True

        def persist(self) -> None:
            events.append("persist")

    class Store(ShadowArtifactStore):
        def save(self, comparison) -> None:
            events.append("artifact")
            super().save(comparison)

    comparison = compare_shadow_results("q", [], [])
    persist_index_then_shadow_artifact(
        cast(object, Writer()), (), comparison, Store(tmp_path)
    )

    assert events == ["index", "persist", "artifact"]
    assert (tmp_path / "gdrive-shadow-comparison.json").exists()
