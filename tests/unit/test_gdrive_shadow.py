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


def test_comparison_rejects_identity_status_and_acl_drift() -> None:
    """
    Source identity, lifecycle status, and ACL changes are never normalized.
    """
    comparison = compare_shadow_results(
        "drive query",
        [
            ShadowObservation(
                "drive-a",
                1,
                "chunk-a",
                1,
                "gdrive",
                "workspace-a",
                "active",
                "acl-a",
            )
        ],
        [
            ShadowObservation(
                "drive-a",
                2,
                "chunk-b",
                2,
                "gdrive",
                "workspace-b",
                "deleted",
                "acl-b",
            )
        ],
    )

    assert comparison.entries["drive-a"] == {
        "differences": [
            "acl",
            "chunk",
            "identity",
            "index_epoch",
            "ranking",
            "status",
        ],
        "allowed": ["chunk", "index_epoch", "ranking"],
        "unexpected": ["acl", "identity", "status"],
    }
    assert comparison.mismatches == 1


def test_barrier_persists_comparison_after_index_and_persist(tmp_path: Path) -> None:
    """
    The artifact cannot precede the durable index write barrier.
    """
    events: list[str] = []
    durable = False

    class Writer:
        def index_records(self, records: list[Record]) -> bool:
            events.append("index")
            return True

        def persist(self) -> None:
            nonlocal durable
            events.append("persist")
            durable = True

    class Store(ShadowArtifactStore):
        def save(self, comparison) -> None:
            assert durable
            events.append("artifact")
            super().save(comparison)

    comparison = compare_shadow_results("q", [], [])
    persist_index_then_shadow_artifact(
        cast(object, Writer()), (), comparison, Store(tmp_path)
    )

    assert events == ["index", "persist", "artifact"]
    assert (tmp_path / "gdrive-shadow-comparison.json").exists()


def test_barrier_does_not_publish_comparison_when_index_write_fails(
    tmp_path: Path,
) -> None:
    """
    A failed index write leaves no comparison artifact to audit.
    """
    class Writer:
        def index_records(self, records: list[Record]) -> bool:
            return False

        def persist(self) -> None:
            raise AssertionError("persist must follow a successful index write")

    comparison = compare_shadow_results("q", [], [])
    try:
        persist_index_then_shadow_artifact(
            cast(object, Writer()), (), comparison, ShadowArtifactStore(tmp_path)
        )
    except RuntimeError as error:
        assert str(error) == "Google Drive shadow index write failed"
    else:
        raise AssertionError("failed index writes must not publish artifacts")

    assert not (tmp_path / "gdrive-shadow-comparison.json").exists()
