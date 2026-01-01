
from src.search.utils import classify_query_type, truncate_content


class TestQueryClassification:
    def test_factual_query_what_is(self):
        """
        Verify 'what is' queries are classified as factual.
        """
        assert classify_query_type("what is a vector index?") == "factual"

    def test_factual_query_command(self):
        """
        Verify command-related queries are classified as factual.
        """
        assert classify_query_type("command to install") == "factual"

    def test_conceptual_query_why(self):
        """
        Verify 'why' queries are classified as conceptual.
        """
        assert classify_query_type("why use hybrid search?") == "conceptual"

    def test_conceptual_query_explain(self):
        """
        Verify 'explain' queries are classified as conceptual.
        """
        assert classify_query_type("explain the architecture") == "conceptual"

    def test_conceptual_query_getting_started(self):
        """
        Verify 'getting started' queries are classified as conceptual.
        """
        assert classify_query_type("getting started with authentication") == "conceptual"

    def test_conceptual_query_how_to(self):
        """
        Verify 'how to' queries are classified as conceptual.
        """
        assert classify_query_type("how to implement search") == "conceptual"

    def test_conceptual_query_with_question_mark(self):
        """
        Verify queries ending with '?' are classified as conceptual.
        """
        assert classify_query_type("should I use this?") == "conceptual"

    def test_factual_query_default(self):
        """
        Verify unknown query patterns default to factual.
        """
        assert classify_query_type("vector similarity") == "factual"


class TestContentTruncation:
    def test_truncate_short_content(self):
        """
        Verify short content is not truncated.
        """
        content = "This is short content."
        assert truncate_content(content, 200) == content

    def test_truncate_long_content(self):
        """
        Verify long content is truncated with ellipsis.
        """
        content = "a" * 300
        truncated = truncate_content(content, 200)
        assert truncated is not None
        assert len(truncated) <= 203
        assert truncated.endswith("...")

    def test_truncate_strips_trailing_space(self):
        """
        Verify truncation strips trailing whitespace before adding ellipsis.
        """
        content = "a" * 195 + "     " + "b" * 100
        truncated = truncate_content(content, 200)
        assert truncated == ("a" * 195) + "..."

    def test_truncate_empty_content(self):
        """
        Verify empty content is returned as-is.
        """
        assert truncate_content("", 200) == ""

    def test_truncate_none_content(self):
        """
        Verify None content is returned as-is.
        """
        assert truncate_content(None, 200) is None
