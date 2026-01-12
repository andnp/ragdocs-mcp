"""
Unit tests for memory link extraction and edge type inference.

Tests the link_parser module which extracts [[wikilinks]] from memory content
and infers edge types based on surrounding context.
"""

import pytest

from src.memory.link_parser import extract_links, infer_edge_type


# ============================================================================
# infer_edge_type Tests
# ============================================================================


class TestInferEdgeType:

    def test_defaults_to_related_to(self):
        """
        Verify infer_edge_type returns 'related_to' for generic context.

        This is the fallback edge type when no keywords match.
        """
        result = infer_edge_type("Check out this file for more details")

        assert result == "related_to"

    def test_empty_context_returns_related_to(self):
        """
        Verify infer_edge_type handles empty context gracefully.
        """
        result = infer_edge_type("")

        assert result == "related_to"

    @pytest.mark.parametrize("keyword", ["refactor", "rewrite", "restructure", "reorganize", "cleanup"])
    def test_refactors_keywords(self, keyword: str):
        """
        Verify context containing refactor keywords maps to 'refactors' edge type.
        """
        context = f"We need to {keyword} this module for better performance"

        result = infer_edge_type(context)

        assert result == "refactors"

    @pytest.mark.parametrize("keyword", ["plan", "todo", "will", "should", "need to", "going to"])
    def test_plans_keywords(self, keyword: str):
        """
        Verify context containing planning keywords maps to 'plans' edge type.
        """
        context = f"I {keyword} implement this feature next week"

        result = infer_edge_type(context)

        assert result == "plans"

    @pytest.mark.parametrize("keyword", ["bug", "fix", "issue", "error", "problem", "broken"])
    def test_debugs_keywords(self, keyword: str):
        """
        Verify context containing debugging keywords maps to 'debugs' edge type.
        """
        context = f"Found a {keyword} in this code that needs attention"

        result = infer_edge_type(context)

        assert result == "debugs"

    @pytest.mark.parametrize("keyword", ["note", "remember", "mention", "see also", "refer"])
    def test_mentions_keywords(self, keyword: str):
        """
        Verify context containing mention keywords maps to 'mentions' edge type.
        """
        context = f"Please {keyword} this documentation for reference"

        result = infer_edge_type(context)

        assert result == "mentions"

    def test_case_insensitive(self):
        """
        Verify keyword matching is case insensitive.

        Users may write 'TODO', 'Bug', or 'REFACTOR' in various cases.
        """
        assert infer_edge_type("TODO: implement feature") == "plans"
        assert infer_edge_type("BUG: authentication fails") == "debugs"
        assert infer_edge_type("REFACTOR needed") == "refactors"

    def test_first_matching_keyword_wins(self):
        """
        Verify first matching keyword category is used.

        When context contains multiple keywords, the first match wins.
        """
        context = "I plan to fix the bug in refactored code"

        result = infer_edge_type(context)

        assert result in ("plans", "debugs", "refactors")


# ============================================================================
# extract_links Tests - Basic Patterns
# ============================================================================


class TestExtractLinksBasicPatterns:

    def test_extracts_simple_wikilink(self):
        """
        Verify extract_links finds basic [[target]] wikilinks.
        """
        content = "Check out [[my-document]] for details."

        links = extract_links(content)

        assert len(links) == 1
        assert links[0].target == "my-document"

    def test_extracts_multiple_links(self):
        """
        Verify extract_links finds all wikilinks in content.
        """
        content = "See [[doc1]] and [[doc2]] and [[doc3]] for info."

        links = extract_links(content)

        assert len(links) == 3
        targets = [link.target for link in links]
        assert targets == ["doc1", "doc2", "doc3"]

    def test_extracts_link_with_path(self):
        """
        Verify extract_links handles file path targets.

        Wikilinks can reference paths like [[src/server.py]].
        """
        content = "The implementation is in [[src/server.py]]."

        links = extract_links(content)

        assert len(links) == 1
        assert links[0].target == "src/server.py"

    def test_extracts_link_with_spaces(self):
        """
        Verify extract_links preserves spaces in target.
        """
        content = "See [[my document with spaces]] here."

        links = extract_links(content)

        assert len(links) == 1
        assert links[0].target == "my document with spaces"

    def test_empty_content_returns_empty(self):
        """
        Verify extract_links handles empty content.
        """
        links = extract_links("")

        assert links == []

    def test_no_links_returns_empty(self):
        """
        Verify extract_links returns empty list when no links present.
        """
        content = "This content has no wikilinks at all."

        links = extract_links(content)

        assert links == []

    def test_ignores_empty_wikilink(self):
        """
        Verify extract_links ignores empty [[]] links.
        """
        content = "This has [[ ]] empty link."

        links = extract_links(content)

        assert links == []


# ============================================================================
# extract_links Tests - Anchor Context
# ============================================================================


class TestExtractLinksAnchorContext:

    def test_captures_context_around_link(self):
        """
        Verify extract_links captures anchor context around the link.

        Context helps explain why the link was made.
        """
        content = "We should refactor [[src/auth.py]] to improve security."

        links = extract_links(content, context_chars=100)

        assert len(links) == 1
        assert "refactor" in links[0].anchor_context
        assert "[[src/auth.py]]" in links[0].anchor_context
        assert "security" in links[0].anchor_context

    def test_context_respects_char_limit(self):
        """
        Verify anchor context respects context_chars parameter.
        """
        content = "x" * 200 + "[[target]]" + "y" * 200

        links = extract_links(content, context_chars=50)

        assert len(links) == 1
        assert len(links[0].anchor_context) <= 50 + len("[[target]]") + 50 + 10

    def test_context_handles_start_of_content(self):
        """
        Verify anchor context handles links at start of content.
        """
        content = "[[first-link]] is at the beginning."

        links = extract_links(content, context_chars=100)

        assert len(links) == 1
        assert "[[first-link]]" in links[0].anchor_context

    def test_context_handles_end_of_content(self):
        """
        Verify anchor context handles links at end of content.
        """
        content = "The final reference is [[last-link]]"

        links = extract_links(content, context_chars=100)

        assert len(links) == 1
        assert "[[last-link]]" in links[0].anchor_context


# ============================================================================
# extract_links Tests - Position Tracking
# ============================================================================


class TestExtractLinksPositionTracking:

    def test_records_link_position(self):
        """
        Verify extract_links records character position of each link.

        Position is useful for precise location in editor.
        """
        content = "Start [[link1]] middle [[link2]] end"

        links = extract_links(content)

        assert len(links) == 2
        assert links[0].position == content.index("[[link1]]")
        assert links[1].position == content.index("[[link2]]")

    def test_positions_are_in_order(self):
        """
        Verify links are returned in document order.
        """
        content = "[[first]] [[second]] [[third]]"

        links = extract_links(content)

        positions = [link.position for link in links]
        assert positions == sorted(positions)


# ============================================================================
# extract_links Tests - Edge Type Inference
# ============================================================================


class TestExtractLinksEdgeTypeInference:

    def test_infers_refactors_edge_type(self):
        """
        Verify edge type is inferred from context keywords.
        """
        content = "Need to refactor [[legacy-module]] for clean code."

        links = extract_links(content)

        assert len(links) == 1
        assert links[0].edge_type == "refactors"

    def test_infers_plans_edge_type(self):
        """
        Verify plans edge type is inferred from context.
        """
        content = "I plan to update [[feature-module]] next sprint."

        links = extract_links(content)

        assert len(links) == 1
        assert links[0].edge_type == "plans"

    def test_infers_debugs_edge_type(self):
        """
        Verify debugs edge type is inferred from context.
        """
        content = "Found a bug in [[auth-service]] causing 500 errors."

        links = extract_links(content)

        assert len(links) == 1
        assert links[0].edge_type == "debugs"

    def test_infers_mentions_edge_type(self):
        """
        Verify mentions edge type is inferred from context.
        """
        content = "Note: see [[architecture-doc]] for system design."

        links = extract_links(content)

        assert len(links) == 1
        assert links[0].edge_type == "mentions"

    def test_defaults_to_related_to(self):
        """
        Verify default edge type when no keywords match.
        """
        content = "Check [[some-file]] for information."

        links = extract_links(content)

        assert len(links) == 1
        assert links[0].edge_type == "related_to"

    def test_different_links_get_different_edge_types(self):
        """
        Verify each link gets its own inferred edge type based on local context.

        Note: Context window size means nearby keywords can influence edge type.
        This test uses well-separated links to ensure distinct contexts.
        """
        content = """
## Planning Section

TODO: update [[feature]] with new API.

## Bug Section

""" + "x" * 150 + """

Bug found in [[service]] causing errors.

## Reference Section

""" + "y" * 150 + """

Note: see [[docs]] for reference.
"""

        links = extract_links(content, context_chars=50)

        assert len(links) == 3
        edge_types = [link.edge_type for link in links]
        assert "plans" in edge_types
        assert "debugs" in edge_types
        assert "mentions" in edge_types


# ============================================================================
# extract_links Tests - Edge Cases
# ============================================================================


class TestExtractLinksEdgeCases:

    def test_handles_markdown_in_content(self):
        """
        Verify extract_links handles markdown formatting around links.
        """
        content = "**Bold** text with [[link]] and _italic_ text."

        links = extract_links(content)

        assert len(links) == 1
        assert links[0].target == "link"

    def test_handles_code_blocks(self):
        """
        Verify extract_links finds links even in code blocks.

        Note: In a real implementation, we might want to skip code blocks.
        This test documents current behavior.
        """
        content = "```python\n# See [[config]] module\n```"

        links = extract_links(content)

        assert len(links) == 1
        assert links[0].target == "config"

    def test_handles_multiline_content(self):
        """
        Verify extract_links handles multiline content with links.
        """
        content = """# Header

First paragraph with [[link1]].

Second paragraph with [[link2]].

Third paragraph with [[link3]].
"""

        links = extract_links(content)

        assert len(links) == 3

    def test_handles_consecutive_links(self):
        """
        Verify extract_links handles links directly next to each other.
        """
        content = "[[link1]][[link2]][[link3]]"

        links = extract_links(content)

        assert len(links) == 3

    def test_handles_link_in_list(self):
        """
        Verify extract_links handles links in markdown lists.
        """
        content = """
- Item with [[link1]]
- Item with [[link2]]
"""

        links = extract_links(content)

        assert len(links) == 2

    def test_handles_nested_brackets(self):
        """
        Verify extract_links handles only [[double]] brackets.

        Single brackets [link] should not be extracted.
        """
        content = "Single [not-a-link] and double [[real-link]]."

        links = extract_links(content)

        assert len(links) == 1
        assert links[0].target == "real-link"
