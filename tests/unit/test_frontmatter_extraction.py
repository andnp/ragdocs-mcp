"""
Unit tests for extended frontmatter extraction in markdown parser.
"""

import tempfile
from pathlib import Path

import pytest

from src.parsers.markdown import MarkdownParser, INDEXED_FRONTMATTER_FIELDS


@pytest.fixture
def parser():
    """Create a MarkdownParser instance."""
    return MarkdownParser()


@pytest.fixture
def temp_markdown_file():
    """Factory fixture for creating temporary markdown files."""
    created_files = []

    def _create(content: str) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            created_files.append(f.name)
            return f.name

    yield _create

    # Cleanup
    for file_path in created_files:
        Path(file_path).unlink(missing_ok=True)


class TestIndexedFrontmatterFields:
    """Test that INDEXED_FRONTMATTER_FIELDS constant is correct."""

    def test_indexed_fields_list(self):
        """Verify all expected fields are in INDEXED_FRONTMATTER_FIELDS."""
        expected = [
            "title", "description", "summary", "keywords",
            "author", "category", "type", "related"
        ]
        assert INDEXED_FRONTMATTER_FIELDS == expected


class TestTitleDescriptionExtraction:
    """Tests for title and description frontmatter extraction."""

    def test_extracts_title(self, parser, temp_markdown_file):
        """Title field is extracted to metadata."""
        content = """---
title: My Document Title
---

# Content
Some text here.
"""
        file_path = temp_markdown_file(content)
        doc = parser.parse(file_path)

        assert doc.metadata.get("title") == "My Document Title"

    def test_extracts_description(self, parser, temp_markdown_file):
        """Description field is extracted to metadata."""
        content = """---
description: A brief description of the document
---

# Content
"""
        file_path = temp_markdown_file(content)
        doc = parser.parse(file_path)

        assert doc.metadata.get("description") == "A brief description of the document"

    def test_extracts_summary(self, parser, temp_markdown_file):
        """Summary field is extracted to metadata."""
        content = """---
summary: TL;DR summary of the document
---

# Content
"""
        file_path = temp_markdown_file(content)
        doc = parser.parse(file_path)

        assert doc.metadata.get("summary") == "TL;DR summary of the document"


class TestKeywordsExtraction:
    """Tests for keywords frontmatter extraction."""

    def test_extracts_keywords_as_list(self, parser, temp_markdown_file):
        """Keywords as list are extracted to metadata."""
        content = """---
keywords:
  - python
  - markdown
  - search
---

# Content
"""
        file_path = temp_markdown_file(content)
        doc = parser.parse(file_path)

        assert doc.metadata.get("keywords") == ["python", "markdown", "search"]

    def test_extracts_keywords_as_string(self, parser, temp_markdown_file):
        """Keywords as string are converted and extracted."""
        content = """---
keywords: python, markdown, search
---

# Content
"""
        file_path = temp_markdown_file(content)
        doc = parser.parse(file_path)

        # Stored as string when provided as string
        assert doc.metadata.get("keywords") == "python, markdown, search"


class TestAuthorCategoryExtraction:
    """Tests for author and category frontmatter extraction."""

    def test_extracts_author(self, parser, temp_markdown_file):
        """Author field is extracted to metadata."""
        content = """---
author: John Doe
---

# Content
"""
        file_path = temp_markdown_file(content)
        doc = parser.parse(file_path)

        assert doc.metadata.get("author") == "John Doe"

    def test_extracts_category(self, parser, temp_markdown_file):
        """Category field is extracted to metadata."""
        content = """---
category: tutorials
---

# Content
"""
        file_path = temp_markdown_file(content)
        doc = parser.parse(file_path)

        assert doc.metadata.get("category") == "tutorials"

    def test_extracts_type(self, parser, temp_markdown_file):
        """Type field is extracted to metadata."""
        content = """---
type: reference
---

# Content
"""
        file_path = temp_markdown_file(content)
        doc = parser.parse(file_path)

        assert doc.metadata.get("type") == "reference"


class TestRelatedFieldExtraction:
    """Tests for related field extraction and graph integration."""

    def test_related_string_adds_to_links(self, parser, temp_markdown_file):
        """Related field as string is added to links for graph."""
        content = """---
related: other-document
---

# Content
Some text with [[existing-link]].
"""
        file_path = temp_markdown_file(content)
        doc = parser.parse(file_path)

        assert "other-document" in doc.links
        assert "existing-link" in doc.links

    def test_related_list_adds_to_links(self, parser, temp_markdown_file):
        """Related field as list is added to links for graph."""
        content = """---
related:
  - doc-one
  - doc-two
  - doc-three
---

# Content
"""
        file_path = temp_markdown_file(content)
        doc = parser.parse(file_path)

        assert "doc-one" in doc.links
        assert "doc-two" in doc.links
        assert "doc-three" in doc.links

    def test_related_merged_with_wikilinks(self, parser, temp_markdown_file):
        """Related links are merged with wikilinks, deduped."""
        content = """---
related:
  - related-doc
  - shared-link
---

# Content
Check out [[shared-link]] and [[wikilink-only]].
"""
        file_path = temp_markdown_file(content)
        doc = parser.parse(file_path)

        # All links present, no duplicates
        assert "related-doc" in doc.links
        assert "shared-link" in doc.links
        assert "wikilink-only" in doc.links
        # Verify no duplicates (shared-link appears once)
        assert doc.links.count("shared-link") == 1

    def test_related_stored_in_metadata(self, parser, temp_markdown_file):
        """Related field is also stored in metadata."""
        content = """---
related:
  - doc-one
  - doc-two
---

# Content
"""
        file_path = temp_markdown_file(content)
        doc = parser.parse(file_path)

        assert doc.metadata.get("related") == ["doc-one", "doc-two"]


class TestMissingFieldsHandling:
    """Tests for graceful handling of missing fields."""

    def test_missing_fields_not_in_metadata(self, parser, temp_markdown_file):
        """Missing fields don't appear in metadata."""
        content = """---
title: Only Title
---

# Content
"""
        file_path = temp_markdown_file(content)
        doc = parser.parse(file_path)

        assert doc.metadata.get("title") == "Only Title"
        assert "description" not in doc.metadata
        assert "keywords" not in doc.metadata
        assert "author" not in doc.metadata
        assert "category" not in doc.metadata
        assert "related" not in doc.metadata

    def test_no_frontmatter(self, parser, temp_markdown_file):
        """Document without frontmatter has empty metadata for indexed fields."""
        content = """# No Frontmatter

Just content here.
"""
        file_path = temp_markdown_file(content)
        doc = parser.parse(file_path)

        for field in INDEXED_FRONTMATTER_FIELDS:
            assert field not in doc.metadata

    def test_empty_frontmatter(self, parser, temp_markdown_file):
        """Empty frontmatter is handled gracefully."""
        content = """---
---

# Content
"""
        file_path = temp_markdown_file(content)
        doc = parser.parse(file_path)

        for field in INDEXED_FRONTMATTER_FIELDS:
            assert field not in doc.metadata


class TestComplexFrontmatter:
    """Tests for complex frontmatter scenarios."""

    def test_all_indexed_fields_together(self, parser, temp_markdown_file):
        """All indexed fields extracted correctly together."""
        content = """---
title: Complete Document
description: A comprehensive test document
summary: Test all fields
keywords:
  - testing
  - complete
author: Test Author
category: testing
type: test-doc
related:
  - related-one
  - related-two
---

# Content
"""
        file_path = temp_markdown_file(content)
        doc = parser.parse(file_path)

        assert doc.metadata.get("title") == "Complete Document"
        assert doc.metadata.get("description") == "A comprehensive test document"
        assert doc.metadata.get("summary") == "Test all fields"
        assert doc.metadata.get("keywords") == ["testing", "complete"]
        assert doc.metadata.get("author") == "Test Author"
        assert doc.metadata.get("category") == "testing"
        assert doc.metadata.get("type") == "test-doc"
        assert doc.metadata.get("related") == ["related-one", "related-two"]

        # Related also in links
        assert "related-one" in doc.links
        assert "related-two" in doc.links

    def test_numeric_values_converted_to_string(self, parser, temp_markdown_file):
        """Numeric values in frontmatter are converted to strings."""
        content = """---
title: 12345
category: 42
---

# Content
"""
        file_path = temp_markdown_file(content)
        doc = parser.parse(file_path)

        assert doc.metadata.get("title") == "12345"
        assert doc.metadata.get("category") == "42"

    def test_non_indexed_fields_preserved(self, parser, temp_markdown_file):
        """Non-indexed frontmatter fields are still in metadata."""
        content = """---
title: Doc Title
custom_field: custom value
version: 1.0
---

# Content
"""
        file_path = temp_markdown_file(content)
        doc = parser.parse(file_path)

        assert doc.metadata.get("title") == "Doc Title"
        assert doc.metadata.get("custom_field") == "custom value"
        assert doc.metadata.get("version") == 1.0
