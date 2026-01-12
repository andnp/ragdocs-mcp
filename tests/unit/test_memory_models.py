"""
Unit tests for Memory Management System data models.

Tests the core data structures: MemoryFrontmatter, ExtractedLink, and MemoryDocument.
These models define the schema for memory files and their metadata.
"""

from datetime import datetime, timezone

import pytest

from src.memory.models import (
    ExtractedLink,
    LinkedMemoryResult,
    MemoryDocument,
    MemoryFrontmatter,
    MemorySearchResult,
)


# ============================================================================
# MemoryFrontmatter Tests
# ============================================================================


class TestMemoryFrontmatterDefaults:

    def test_default_type_is_journal(self):
        """
        Verify MemoryFrontmatter defaults to 'journal' type when not specified.

        This is the most common memory type for ad-hoc notes.
        """
        frontmatter = MemoryFrontmatter()
        assert frontmatter.type == "journal"

    def test_default_status_is_active(self):
        """
        Verify MemoryFrontmatter defaults to 'active' status when not specified.

        New memories should be active by default.
        """
        frontmatter = MemoryFrontmatter()
        assert frontmatter.status == "active"

    def test_default_tags_is_empty_list(self):
        """
        Verify MemoryFrontmatter defaults to empty tags list.

        Tags are optional; empty list is a sensible default.
        """
        frontmatter = MemoryFrontmatter()
        assert frontmatter.tags == []

    def test_default_created_at_is_none(self):
        """
        Verify MemoryFrontmatter defaults to None for created_at.

        Created time is set during file creation, not model initialization.
        """
        frontmatter = MemoryFrontmatter()
        assert frontmatter.created_at is None


class TestMemoryFrontmatterValidTypes:

    @pytest.mark.parametrize("valid_type", ["plan", "journal", "fact", "observation", "reflection"])
    def test_accepts_valid_types(self, valid_type: str):
        """
        Verify MemoryFrontmatter accepts all valid type values.

        The spec defines 5 valid types: plan, journal, fact, observation, reflection.
        """
        frontmatter = MemoryFrontmatter(type=valid_type)
        assert frontmatter.type == valid_type

    def test_invalid_type_falls_back_to_journal(self):
        """
        Verify invalid type values fall back to 'journal'.

        This provides graceful handling of corrupt/invalid frontmatter.
        """
        frontmatter = MemoryFrontmatter(type="invalid_type")
        assert frontmatter.type == "journal"

    def test_empty_type_falls_back_to_journal(self):
        """
        Verify empty string type falls back to 'journal'.
        """
        frontmatter = MemoryFrontmatter(type="")
        assert frontmatter.type == "journal"


class TestMemoryFrontmatterValidStatuses:

    @pytest.mark.parametrize("valid_status", ["active", "archived"])
    def test_accepts_valid_statuses(self, valid_status: str):
        """
        Verify MemoryFrontmatter accepts all valid status values.

        The spec defines 2 valid statuses: active, archived.
        """
        frontmatter = MemoryFrontmatter(status=valid_status)
        assert frontmatter.status == valid_status

    def test_invalid_status_falls_back_to_active(self):
        """
        Verify invalid status values fall back to 'active'.

        This provides graceful handling of corrupt/invalid frontmatter.
        """
        frontmatter = MemoryFrontmatter(status="invalid_status")
        assert frontmatter.status == "active"


class TestMemoryFrontmatterWithData:

    def test_full_frontmatter_construction(self):
        """
        Verify MemoryFrontmatter correctly stores all provided fields.
        """
        created = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        frontmatter = MemoryFrontmatter(
            type="plan",
            status="archived",
            tags=["refactor", "auth"],
            created_at=created,
        )

        assert frontmatter.type == "plan"
        assert frontmatter.status == "archived"
        assert frontmatter.tags == ["refactor", "auth"]
        assert frontmatter.created_at == created

    def test_tags_can_be_modified_after_construction(self):
        """
        Verify tags list is mutable after construction.

        Dataclass uses default_factory, so each instance gets its own list.
        """
        fm1 = MemoryFrontmatter()
        fm2 = MemoryFrontmatter()

        fm1.tags.append("test")

        assert "test" in fm1.tags
        assert "test" not in fm2.tags


# ============================================================================
# ExtractedLink Tests
# ============================================================================


class TestExtractedLink:

    def test_basic_link_construction(self):
        """
        Verify ExtractedLink stores all required fields.
        """
        link = ExtractedLink(
            target="src/server.py",
            edge_type="mentions",
            anchor_context="...see [[src/server.py]] for details...",
            position=42,
        )

        assert link.target == "src/server.py"
        assert link.edge_type == "mentions"
        assert link.anchor_context == "...see [[src/server.py]] for details..."
        assert link.position == 42

    def test_link_with_related_to_default_type(self):
        """
        Verify ExtractedLink can store the 'related_to' default edge type.
        """
        link = ExtractedLink(
            target="docs/api.md",
            edge_type="related_to",
            anchor_context="Check [[docs/api.md]]",
            position=0,
        )

        assert link.edge_type == "related_to"

    @pytest.mark.parametrize("edge_type", ["refactors", "plans", "debugs", "mentions", "related_to"])
    def test_all_edge_types(self, edge_type: str):
        """
        Verify ExtractedLink accepts all spec-defined edge types.
        """
        link = ExtractedLink(
            target="target.md",
            edge_type=edge_type,
            anchor_context="context",
            position=0,
        )

        assert link.edge_type == edge_type


# ============================================================================
# MemoryDocument Tests
# ============================================================================


class TestMemoryDocument:

    def test_document_construction(self):
        """
        Verify MemoryDocument stores all required fields correctly.
        """
        modified = datetime(2025, 1, 10, 14, 30, 0, tzinfo=timezone.utc)
        frontmatter = MemoryFrontmatter(type="fact", tags=["important"])
        links = [
            ExtractedLink(
                target="src/config.py",
                edge_type="mentions",
                anchor_context="config at [[src/config.py]]",
                position=10,
            )
        ]

        doc = MemoryDocument(
            id="memory:my-note",
            content="This is the memory content.",
            frontmatter=frontmatter,
            links=links,
            file_path="/path/to/my-note.md",
            modified_time=modified,
        )

        assert doc.id == "memory:my-note"
        assert doc.content == "This is the memory content."
        assert doc.frontmatter.type == "fact"
        assert doc.frontmatter.tags == ["important"]
        assert len(doc.links) == 1
        assert doc.links[0].target == "src/config.py"
        assert doc.file_path == "/path/to/my-note.md"
        assert doc.modified_time == modified

    def test_document_with_empty_links(self):
        """
        Verify MemoryDocument handles empty links list.

        Memories don't require links to other documents.
        """
        doc = MemoryDocument(
            id="memory:standalone",
            content="No links here",
            frontmatter=MemoryFrontmatter(),
            links=[],
            file_path="/path/to/standalone.md",
            modified_time=datetime.now(timezone.utc),
        )

        assert doc.links == []

    def test_document_with_multiple_links(self):
        """
        Verify MemoryDocument handles multiple links.

        Memories can reference many other files.
        """
        links = [
            ExtractedLink(target="file1.py", edge_type="refactors", anchor_context="", position=0),
            ExtractedLink(target="file2.py", edge_type="plans", anchor_context="", position=50),
            ExtractedLink(target="file3.md", edge_type="debugs", anchor_context="", position=100),
        ]

        doc = MemoryDocument(
            id="memory:multi-link",
            content="Content with many links",
            frontmatter=MemoryFrontmatter(),
            links=links,
            file_path="/path/to/multi-link.md",
            modified_time=datetime.now(timezone.utc),
        )

        assert len(doc.links) == 3
        assert doc.links[0].edge_type == "refactors"
        assert doc.links[1].edge_type == "plans"
        assert doc.links[2].edge_type == "debugs"


# ============================================================================
# MemorySearchResult Tests
# ============================================================================


class TestMemorySearchResult:

    def test_search_result_construction(self):
        """
        Verify MemorySearchResult stores search result data.
        """
        frontmatter = MemoryFrontmatter(type="plan", tags=["auth"])

        result = MemorySearchResult(
            memory_id="memory:auth-plan",
            score=0.85,
            content="Authentication refactoring plan",
            frontmatter=frontmatter,
            file_path="/memories/auth-plan.md",
            header_path="## Overview",
        )

        assert result.memory_id == "memory:auth-plan"
        assert result.score == 0.85
        assert result.content == "Authentication refactoring plan"
        assert result.frontmatter.type == "plan"
        assert result.file_path == "/memories/auth-plan.md"
        assert result.header_path == "## Overview"

    def test_search_result_default_header_path(self):
        """
        Verify MemorySearchResult defaults to empty header_path.
        """
        result = MemorySearchResult(
            memory_id="memory:test",
            score=0.5,
            content="Test",
            frontmatter=MemoryFrontmatter(),
            file_path="/memories/test.md",
        )

        assert result.header_path == ""


# ============================================================================
# LinkedMemoryResult Tests
# ============================================================================


class TestLinkedMemoryResult:

    def test_linked_result_construction(self):
        """
        Verify LinkedMemoryResult stores linked memory search data.

        This result type includes anchor context and edge type from
        the graph traversal.
        """
        result = LinkedMemoryResult(
            memory_id="memory:bug-fix",
            score=0.92,
            content="Fixed the auth bug in server.py",
            anchor_context="...will fix the bug in [[src/server.py]]...",
            edge_type="debugs",
            file_path="/memories/bug-fix.md",
        )

        assert result.memory_id == "memory:bug-fix"
        assert result.score == 0.92
        assert result.content == "Fixed the auth bug in server.py"
        assert result.anchor_context == "...will fix the bug in [[src/server.py]]..."
        assert result.edge_type == "debugs"
        assert result.file_path == "/memories/bug-fix.md"
