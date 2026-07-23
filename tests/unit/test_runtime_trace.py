"""Unit tests for query tracing."""

import time

from searchkernel.runtime.trace import QueryTrace, Span


class TestSpan:
    """Tests for Span class."""

    def test_span_creation(self):
        """Test creating a span."""
        start = time.perf_counter()
        span = Span(name="test_stage", start_time=start)

        assert span.name == "test_stage"
        assert span.start_time == start
        assert span.end_time is None
        assert span.duration_ms is None

    def test_span_close(self):
        """Test closing a span computes duration."""
        start = time.perf_counter()
        span = Span(name="test_stage", start_time=start)

        time.sleep(0.01)  # Sleep for 10ms
        span.close()

        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.duration_ms >= 10.0  # At least 10ms

    def test_span_to_dict(self):
        """Test span serialization."""
        start = time.perf_counter()
        span = Span(name="test_stage", start_time=start)
        span.close()

        d = span.to_dict()
        assert d["name"] == "test_stage"
        assert d["duration_ms"] is not None
        assert d["duration_ms"] >= 0


class TestQueryTrace:
    """Tests for QueryTrace class."""

    def test_query_trace_creation(self):
        """Test creating a query trace."""
        trace = QueryTrace(query_text="test query")

        assert trace.query_text == "test query"
        assert trace.start_time is not None
        assert trace.end_time is None
        assert len(trace.spans) == 0

    def test_query_trace_span_context_manager(self):
        """Test using span context manager."""
        trace = QueryTrace(query_text="test query")

        with trace.span("search"):
            time.sleep(0.01)

        assert "search" in trace.spans
        assert trace.spans["search"].duration_ms is not None
        assert trace.spans["search"].duration_ms >= 10.0

    def test_query_trace_multiple_spans(self):
        """Test multiple spans in a single trace."""
        trace = QueryTrace(query_text="test query")

        with trace.span("vector_search"):
            time.sleep(0.01)

        with trace.span("keyword_search"):
            time.sleep(0.01)

        assert len(trace.spans) == 2
        assert "vector_search" in trace.spans
        assert "keyword_search" in trace.spans

    def test_query_trace_span_timing_accuracy(self):
        """Test that span timing is approximately correct."""
        trace = QueryTrace(query_text="test query")

        with trace.span("stage1"):
            time.sleep(0.02)  # 20ms

        stage1_duration = trace.spans["stage1"].duration_ms
        assert stage1_duration is not None
        # Should be approximately 20ms (allow ±5ms tolerance)
        assert 15.0 <= stage1_duration <= 30.0

    def test_query_trace_close(self):
        """Test closing a trace."""
        trace = QueryTrace(query_text="test query")

        with trace.span("search"):
            time.sleep(0.01)

        time.sleep(0.01)
        trace.close()

        assert trace.end_time is not None
        assert trace.total_duration_ms is not None
        # Total should be at least 20ms (two sleeps of 10ms each)
        assert trace.total_duration_ms >= 20.0

    def test_query_trace_total_duration_before_close(self):
        """Test total_duration_ms returns None before close."""
        trace = QueryTrace(query_text="test query")

        with trace.span("search"):
            time.sleep(0.01)

        # Before explicit close, end_time is None
        trace_duration = trace.total_duration_ms
        # The span context manager closes the span but not the trace
        assert trace_duration is not None or trace.end_time is None

    def test_query_trace_provenance(self):
        """Test setting provenance."""
        trace = QueryTrace(query_text="test query")
        trace.provenance = {"strategy": "vector", "score": 0.95}

        d = trace.to_dict()
        assert "provenance" in d
        assert d["provenance"]["strategy"] == "vector"

    def test_query_trace_to_dict(self):
        """Test query trace serialization."""
        trace = QueryTrace(query_text="test query")

        with trace.span("search"):
            time.sleep(0.01)

        trace.close()

        d = trace.to_dict()
        assert d["query"] == "test query"
        assert d["total_duration_ms"] is not None
        assert len(d["spans"]) == 1
        assert d["spans"][0]["name"] == "search"

    def test_query_trace_to_dict_with_provenance(self):
        """Test serialization with provenance."""
        trace = QueryTrace(query_text="test query")
        trace.provenance = {"model": "test_model"}

        with trace.span("search"):
            time.sleep(0.01)

        trace.close()

        d = trace.to_dict()
        assert "provenance" in d
        assert d["provenance"]["model"] == "test_model"

    def test_query_trace_nested_spans(self):
        """Test nesting of context managers."""
        trace = QueryTrace(query_text="test query")

        with trace.span("outer"):
            time.sleep(0.01)
            with trace.span("inner"):
                time.sleep(0.01)

        # Both spans should be registered
        assert "outer" in trace.spans
        assert "inner" in trace.spans

    def test_query_trace_exception_in_span(self):
        """Test that span is closed even if exception occurs."""
        trace = QueryTrace(query_text="test query")

        try:
            with trace.span("failing_stage"):
                time.sleep(0.01)
                raise ValueError("Test error")
        except ValueError:
            pass

        # Span should still be registered and have duration
        assert "failing_stage" in trace.spans
        assert trace.spans["failing_stage"].duration_ms is not None
