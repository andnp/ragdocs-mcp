from datetime import datetime

import pytest

from src.models import Document
from src.parsers.markdown import MarkdownParser


def test_parse_simple_markdown(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Hello World\n\nThis is a test.")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert isinstance(doc, Document)
    assert doc.id == "test"
    assert "Hello World" in doc.content
    assert "This is a test" in doc.content
    assert doc.file_path == str(md_file)
    assert isinstance(doc.modified_time, datetime)


def test_parse_with_yaml_frontmatter(tmp_path):
    md_file = tmp_path / "note.md"
    md_file.write_text("""---
title: My Note
author: Test User
aliases:
  - alias1
  - alias2
tags:
  - tag1
  - tag2
---

# Content

Body text here.
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.id == "note"
    assert "# Content" in doc.content
    assert "Body text here" in doc.content
    assert "---" not in doc.content
    assert doc.metadata.get("title") == "My Note"
    assert doc.metadata.get("author") == "Test User"
    assert doc.tags == ["tag1", "tag2"]


def test_parse_aliases_from_frontmatter(tmp_path):
    md_file = tmp_path / "doc.md"
    md_file.write_text("""---
aliases:
  - first-alias
  - second-alias
---

Content here.
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.id == "doc"


def test_parse_malformed_yaml_frontmatter(tmp_path):
    md_file = tmp_path / "bad.md"
    md_file.write_text("""---
title: Unclosed String
bad_yaml: [this is not valid
---

Content here.
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.id == "bad"
    assert doc.metadata == {}
    assert "Content here" in doc.content


def test_parse_missing_file():
    parser = MarkdownParser()

    with pytest.raises(FileNotFoundError, match="File not found"):
        parser.parse("/nonexistent/file.md")


def test_parse_no_frontmatter(tmp_path):
    md_file = tmp_path / "plain.md"
    md_file.write_text("Just plain markdown content.\n\nNo frontmatter here.")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.id == "plain"
    assert doc.metadata == {}
    assert "Just plain markdown content" in doc.content
    assert "No frontmatter here" in doc.content


def test_parse_empty_frontmatter(tmp_path):
    md_file = tmp_path / "empty.md"
    md_file.write_text("""---
---

Content after empty frontmatter.
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.id == "empty"
    assert doc.metadata == {}
    assert "Content after empty frontmatter" in doc.content


def test_parse_single_alias_as_string(tmp_path):
    md_file = tmp_path / "single.md"
    md_file.write_text("""---
aliases: single-alias
---

Content.
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.id == "single"


def test_parse_single_tag_as_string(tmp_path):
    md_file = tmp_path / "tag.md"
    md_file.write_text("""---
tags: single-tag
---

Content.
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.id == "tag"
    assert doc.tags == ["single-tag"]


def test_modified_time_reflects_file_stat(tmp_path):
    md_file = tmp_path / "time.md"
    md_file.write_text("Content")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    file_mtime = datetime.fromtimestamp(md_file.stat().st_mtime)

    assert abs((doc.modified_time - file_mtime).total_seconds()) < 1


def test_parse_empty_file(tmp_path):
    """
    Verify parser handles empty (0-byte) markdown files gracefully.
    Prevents crashes when indexing directories with placeholder files.
    """
    md_file = tmp_path / "empty.md"
    md_file.write_text("")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.id == "empty"
    assert doc.content == ""
    assert doc.metadata == {}
    assert doc.tags == []


def test_parse_unicode_content(tmp_path):
    """
    Verify parser correctly handles unicode characters (emoji, CJK, accents).
    Ensures international content and modern markdown (emoji) are preserved.
    """
    md_file = tmp_path / "unicode.md"
    md_file.write_text("""---
title: Résumé 📝
author: 中文用户
---

# Hello 世界 🌍

Content with émojis 🎉 and àccénts.
""", encoding="utf-8")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.id == "unicode"
    assert "世界" in doc.content
    assert "🌍" in doc.content
    assert "🎉" in doc.content
    assert "àccénts" in doc.content
    assert doc.metadata["title"] == "Résumé 📝"
    assert doc.metadata["author"] == "中文用户"


def test_extract_wikilinks_both_forms(tmp_path):
    """
    Verify parser extracts wikilinks in both [[Note]] and [[Note|Display]] forms.
    Ensures graph edges are created for all link types while excluding transclusions.
    """
    md_file = tmp_path / "links.md"
    md_file.write_text("""# Document with Links

Standard link: [[Target Note]]
Link with display text: [[Another Note|Custom Display]]
Transclusion should be ignored: ![[Embedded Note]]
Multiple references: [[Target Note]] and [[Third Note]]
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.id == "links"
    assert set(doc.links) == {"Target Note", "Another Note", "Third Note"}
    assert "Embedded Note" not in doc.links


def test_extract_transclusions(tmp_path):
    """
    Verify parser extracts transclusions (![[Note]]) as separate metadata.
    Ensures embedded content relationships are tracked in graph without duplication.
    """
    md_file = tmp_path / "transclusions.md"
    md_file.write_text("""# Document with Transclusions

Embed this: ![[Template Note]]
Regular link: [[Reference Note]]
Another embed: ![[Snippet]]
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.id == "transclusions"
    transclusions = doc.metadata.get("transclusions", [])
    assert isinstance(transclusions, list)
    assert set(transclusions) == {"Template Note", "Snippet"}
    assert "Template Note" not in doc.links
    assert "Reference Note" in doc.links


def test_extract_inline_tags(tmp_path):
    """
    Verify parser extracts inline tags (#tag) from markdown content.
    Ensures content-based categorization works alongside frontmatter tags.
    """
    md_file = tmp_path / "inline_tags.md"
    md_file.write_text("""# Document with Tags

This is about #programming and #python specifically.
Also covers #data-science topics.

#machine-learning is mentioned at line start.
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.id == "inline_tags"
    assert set(doc.tags) == {"data-science", "machine-learning", "programming", "python"}


def test_code_block_exclusion_for_links_and_tags(tmp_path):
    """
    Verify parser excludes links and tags inside code blocks from extraction.
    Prevents false positives from code examples containing markdown-like syntax.
    """
    md_file = tmp_path / "code_blocks.md"
    md_file.write_text("""# Code Examples

This has a real link: [[Actual Note]]

```markdown
This is an example: [[Fake Link]]
Use #fake-tag in examples.
```

Real inline tag: #real-tag

Inline code: `[[Not A Link]]` and `#not-a-tag`
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.id == "code_blocks"
    assert doc.links == ["Actual Note"]
    assert "Fake Link" not in doc.links
    assert "Not A Link" not in doc.links
    assert doc.tags == ["real-tag"]
    assert "fake-tag" not in doc.tags
    assert "not-a-tag" not in doc.tags


def test_frontmatter_and_inline_tags_combination(tmp_path):
    """
    Verify parser merges frontmatter and inline tags into unified sorted list.
    Ensures complete tag coverage from both metadata and content sources.
    """
    md_file = tmp_path / "combined_tags.md"
    md_file.write_text("""---
tags:
  - yaml-tag
  - metadata-tag
---

# Content

This mentions #content-tag and #inline-tag.
Reference to #yaml-tag again (should not duplicate).
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.id == "combined_tags"
    assert doc.tags == ["content-tag", "inline-tag", "metadata-tag", "yaml-tag"]
    assert len(doc.tags) == 4


def test_malformed_wikilinks_edge_cases(tmp_path):
    """
    Verify parser handles malformed wikilinks gracefully without crashes.
    Ensures robustness against user typos and incomplete markdown syntax.
    """
    md_file = tmp_path / "malformed.md"
    md_file.write_text("""# Malformed Links

Unclosed: [[Unclosed Link
Empty: [[]]
Nested start: [[[Note]]]
Valid: [[Proper Link]]
Missing second bracket: [[Missing]
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.id == "malformed"
    assert "Proper Link" in doc.links
    assert len([link for link in doc.links if link.strip()]) >= 1
