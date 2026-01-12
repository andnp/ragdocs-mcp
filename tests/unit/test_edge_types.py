"""
Unit tests for EdgeType enum and edge type inference.

Tests cover:
- EdgeType enum values and string representations
- Header context-based edge type inference
- Target path-based inference (test files)
- Fallback to LINKS_TO for unknown contexts
"""

from src.search.edge_types import EdgeType, HEADER_TO_EDGE_TYPE, infer_edge_type


# ============================================================================
# EdgeType Enum Tests
# ============================================================================


class TestEdgeTypeEnum:
    """Tests for EdgeType enum definition and values."""

    def test_edge_type_has_expected_values(self):
        """
        Verifies that EdgeType enum contains all four required edge types
        as defined in the specification.
        """
        assert EdgeType.LINKS_TO.value == "links_to"
        assert EdgeType.IMPLEMENTS.value == "implements"
        assert EdgeType.TESTS.value == "tests"
        assert EdgeType.RELATED.value == "related"

    def test_edge_type_count(self):
        """
        Ensures exactly 4 edge types exist as per specification.
        Guards against accidental additions or removals.
        """
        assert len(EdgeType) == 4

    def test_edge_type_string_conversion(self):
        """
        Verifies edge types can be converted to strings for serialization.
        Important for JSON persistence of graph edges.
        """
        assert str(EdgeType.LINKS_TO.value) == "links_to"
        assert str(EdgeType.TESTS.value) == "tests"


class TestHeaderToEdgeTypeMapping:
    """Tests for the HEADER_TO_EDGE_TYPE mapping dictionary."""

    def test_testing_keywords_map_to_tests(self):
        """
        Verifies that testing-related header keywords correctly map to TESTS edge type.
        Covers common variations: 'testing', 'tests', 'test'.
        """
        assert HEADER_TO_EDGE_TYPE["testing"] == EdgeType.TESTS
        assert HEADER_TO_EDGE_TYPE["tests"] == EdgeType.TESTS
        assert HEADER_TO_EDGE_TYPE["test"] == EdgeType.TESTS

    def test_implementation_keywords_map_to_implements(self):
        """
        Verifies implementation-related header keywords map to IMPLEMENTS edge type.
        Covers: 'implementation', 'implements', 'code'.
        """
        assert HEADER_TO_EDGE_TYPE["implementation"] == EdgeType.IMPLEMENTS
        assert HEADER_TO_EDGE_TYPE["implements"] == EdgeType.IMPLEMENTS
        assert HEADER_TO_EDGE_TYPE["code"] == EdgeType.IMPLEMENTS

    def test_related_keywords_map_to_related(self):
        """
        Verifies related/reference header keywords map to RELATED edge type.
        Covers: 'related', 'see also', 'references'.
        """
        assert HEADER_TO_EDGE_TYPE["related"] == EdgeType.RELATED
        assert HEADER_TO_EDGE_TYPE["see also"] == EdgeType.RELATED
        assert HEADER_TO_EDGE_TYPE["references"] == EdgeType.RELATED


# ============================================================================
# Edge Type Inference Tests
# ============================================================================


class TestInferEdgeTypeFromHeaderContext:
    """Tests for inferring edge type from header context strings."""

    def test_infer_tests_from_testing_header(self):
        """
        Links under '# Testing' header should infer TESTS edge type.
        Case-insensitive matching ensures robustness.
        """
        assert infer_edge_type("Testing", "any_target") == EdgeType.TESTS
        assert infer_edge_type("## Testing", "any_target") == EdgeType.TESTS
        assert infer_edge_type("testing", "any_target") == EdgeType.TESTS
        assert infer_edge_type("TESTING", "any_target") == EdgeType.TESTS

    def test_infer_tests_from_tests_header(self):
        """
        Links under '# Tests' header should infer TESTS edge type.
        """
        assert infer_edge_type("Tests", "any_target") == EdgeType.TESTS
        assert infer_edge_type("Unit Tests", "any_target") == EdgeType.TESTS
        assert infer_edge_type("Integration Tests", "any_target") == EdgeType.TESTS

    def test_infer_implements_from_implementation_header(self):
        """
        Links under '# Implementation' header should infer IMPLEMENTS edge type.
        """
        assert infer_edge_type("Implementation", "any_target") == EdgeType.IMPLEMENTS
        assert infer_edge_type("## Implementation Details", "any_target") == EdgeType.IMPLEMENTS
        assert infer_edge_type("implementation", "any_target") == EdgeType.IMPLEMENTS

    def test_infer_implements_from_code_header(self):
        """
        Links under '# Code' header should infer IMPLEMENTS edge type.
        Useful for documentation that uses 'Code' instead of 'Implementation'.
        """
        assert infer_edge_type("Code", "any_target") == EdgeType.IMPLEMENTS
        assert infer_edge_type("Source Code", "any_target") == EdgeType.IMPLEMENTS
        assert infer_edge_type("Code Examples", "any_target") == EdgeType.IMPLEMENTS

    def test_infer_related_from_related_header(self):
        """
        Links under '# Related' or '# See Also' headers should infer RELATED edge type.
        """
        assert infer_edge_type("Related", "any_target") == EdgeType.RELATED
        assert infer_edge_type("## Related Topics", "any_target") == EdgeType.RELATED
        assert infer_edge_type("related", "any_target") == EdgeType.RELATED

    def test_infer_related_from_see_also_header(self):
        """
        'See also' and 'References' headers should infer RELATED edge type.
        """
        assert infer_edge_type("See Also", "any_target") == EdgeType.RELATED
        assert infer_edge_type("see also", "any_target") == EdgeType.RELATED
        assert infer_edge_type("References", "any_target") == EdgeType.RELATED


class TestInferEdgeTypeFromTarget:
    """Tests for inferring edge type from target path when header context is insufficient."""

    def test_infer_tests_from_tests_directory_with_context(self):
        """
        Links to files in tests/ directory should infer TESTS edge type
        when header context exists but doesn't match any keyword.

        Note: Target path inference only triggers when header_context is non-empty
        but doesn't match any keywords. With empty context, LINKS_TO is returned.
        """
        assert infer_edge_type("Overview", "tests/test_foo.py") == EdgeType.TESTS
        assert infer_edge_type("Introduction", "tests/unit/test_bar.py") == EdgeType.TESTS

    def test_infer_tests_from_test_prefix_with_context(self):
        """
        Links to files with test_ prefix should infer TESTS edge type
        when header context exists but doesn't match keywords.
        """
        assert infer_edge_type("Overview", "test_foo.py") == EdgeType.TESTS
        assert infer_edge_type("Introduction", "test_bar") == EdgeType.TESTS

    def test_empty_context_returns_links_to_regardless_of_target(self):
        """
        Empty header context returns LINKS_TO without target path inference.
        Target inference only applies when context exists but doesn't match keywords.
        """
        assert infer_edge_type("", "tests/test_foo.py") == EdgeType.LINKS_TO
        assert infer_edge_type("", "test_bar.py") == EdgeType.LINKS_TO

    def test_header_context_takes_precedence_over_target(self):
        """
        Header context should take precedence over target path inference.
        A test file linked under Implementation should be IMPLEMENTS.
        """
        assert infer_edge_type("Implementation", "tests/test_foo.py") == EdgeType.IMPLEMENTS
        assert infer_edge_type("Related", "test_bar.py") == EdgeType.RELATED


class TestInferEdgeTypeDefault:
    """Tests for default fallback behavior."""

    def test_empty_header_context_defaults_to_links_to(self):
        """
        Empty or missing header context should default to LINKS_TO.
        """
        assert infer_edge_type("", "some_file.py") == EdgeType.LINKS_TO
        assert infer_edge_type("", "docs/readme.md") == EdgeType.LINKS_TO

    def test_unknown_header_defaults_to_links_to(self):
        """
        Unknown header keywords should default to LINKS_TO.
        Conservative default ensures no false positives.
        """
        assert infer_edge_type("Introduction", "any_target") == EdgeType.LINKS_TO
        assert infer_edge_type("Overview", "any_target") == EdgeType.LINKS_TO
        assert infer_edge_type("Getting Started", "any_target") == EdgeType.LINKS_TO
        assert infer_edge_type("Configuration", "any_target") == EdgeType.LINKS_TO

    def test_partial_keyword_match_may_match(self):
        """
        The implementation uses substring matching ('in' operator).
        'Contested' contains 'test' so it matches TESTS edge type.
        This is the current behavior - not necessarily ideal but documented.
        """
        # 'Contested' contains 'test' - this matches due to substring
        assert infer_edge_type("Contested Ideas", "any_target") == EdgeType.TESTS

        # Words that don't contain any keyword substrings default to LINKS_TO
        assert infer_edge_type("Overview", "any_target") == EdgeType.LINKS_TO
        assert infer_edge_type("Introduction", "any_target") == EdgeType.LINKS_TO


class TestInferEdgeTypeEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_none_like_empty_string(self):
        """
        Empty string header context should be handled gracefully.
        """
        result = infer_edge_type("", "target.md")
        assert result == EdgeType.LINKS_TO

    def test_whitespace_only_header(self):
        """
        Whitespace-only header should be treated as empty.
        """
        result = infer_edge_type("   ", "target.md")
        assert result == EdgeType.LINKS_TO

    def test_mixed_case_headers(self):
        """
        Mixed case headers should be normalized for matching.
        """
        assert infer_edge_type("TeStiNg", "any") == EdgeType.TESTS
        assert infer_edge_type("IMPLEMENTATION", "any") == EdgeType.IMPLEMENTS
        assert infer_edge_type("SeE AlSo", "any") == EdgeType.RELATED

    def test_header_with_special_characters(self):
        """
        Headers with markdown or special characters should still match.
        """
        assert infer_edge_type("## Testing:", "any") == EdgeType.TESTS
        assert infer_edge_type("# Implementation (v2)", "any") == EdgeType.IMPLEMENTS
        assert infer_edge_type("References & Links", "any") == EdgeType.RELATED
