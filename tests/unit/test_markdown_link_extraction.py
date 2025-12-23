from src.parsers.markdown import MarkdownParser


def test_extract_wikilinks_basic(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
# Test Note

This links to [[Another Note]] and [[Yet Another Note]].
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert sorted(doc.links) == ["Another Note", "Yet Another Note"]


def test_extract_wikilinks_with_display_text(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
# Test Note

This links to [[Target Note|Custom Display Text]].
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.links == ["Target Note"]


def test_extract_wikilinks_ignores_code_blocks(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
# Test Note

[[Valid Link]]

```python
# This should be ignored: [[Code Block Link]]
```

Also ignore `[[Inline Code Link]]`.
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.links == ["Valid Link"]


def test_extract_wikilinks_ignores_transclusions(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
# Test Note

[[Regular Link]]

![[This is a transclusion]]
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.links == ["Regular Link"]
    assert doc.metadata.get("transclusions") == ["This is a transclusion"]


def test_extract_transclusions_basic(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
# Test Note

![[Embedded Note]]

![[Another Embedded Note]]
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    transclusions = doc.metadata.get("transclusions", [])
    assert isinstance(transclusions, list)
    assert sorted(transclusions) == ["Another Embedded Note", "Embedded Note"]


def test_extract_transclusions_ignores_code_blocks(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
# Test Note

![[Valid Transclusion]]

```
![[Code Block Transclusion]]
```

Not this one: `![[Inline Code Transclusion]]`
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.metadata.get("transclusions") == ["Valid Transclusion"]


def test_extract_tags_basic(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
# Test Note #header-tag

This note has #tag1 and #tag2 in the content.
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert sorted(doc.tags) == ["header-tag", "tag1", "tag2"]


def test_extract_tags_with_hyphens_underscores(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
# Test Note

Tags: #my-tag #another_tag #mixed-tag_here
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert sorted(doc.tags) == ["another_tag", "mixed-tag_here", "my-tag"]


def test_extract_tags_ignores_code_blocks(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
# Test Note

Valid tag: #valid

```python
# Should ignore #code-tag
```

Also ignore `#inline-code-tag`.
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.tags == ["valid"]


def test_extract_tags_combines_frontmatter_and_inline(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""---
tags:
  - frontmatter-tag1
  - frontmatter-tag2
---

# Content

This has #inline-tag in the text.
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert sorted(doc.tags) == ["frontmatter-tag1", "frontmatter-tag2", "inline-tag"]


def test_extract_tags_deduplicates(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""---
tags:
  - duplicate-tag
---

# Content

This has #duplicate-tag and #unique-tag.
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert sorted(doc.tags) == ["duplicate-tag", "unique-tag"]


def test_extract_tags_at_line_start(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
#start-of-line

Middle #middle-tag here.
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert sorted(doc.tags) == ["middle-tag", "start-of-line"]


def test_no_links_tags_transclusions(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
# Plain Note

Just plain markdown content with no special syntax.
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.links == []
    assert doc.tags == []
    assert "transclusions" not in doc.metadata


def test_multiple_wikilinks_deduplication(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
# Test Note

[[Duplicate Link]] appears here and [[Duplicate Link]] appears again.
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert doc.links == ["Duplicate Link"]


def test_complex_mixed_syntax(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""---
tags:
  - yaml-tag
---

# Test Note #heading-tag

This has [[Link One]] and [[Link Two|Display]].

It also has ![[Embedded Note]].

Tags: #inline-tag1 #inline-tag2

```
Ignore [[Code Link]] and #code-tag and ![[Code Embed]]
```

More content with `[[Inline Code Link]]` and `#inline-code-tag`.
""")

    parser = MarkdownParser()
    doc = parser.parse(str(md_file))

    assert sorted(doc.links) == ["Link One", "Link Two"]
    assert sorted(doc.tags) == ["heading-tag", "inline-tag1", "inline-tag2", "yaml-tag"]
    assert doc.metadata.get("transclusions") == ["Embedded Note"]
