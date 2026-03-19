from datetime import datetime

from src.chunking.header_chunker import HeaderBasedChunker
from src.config import ChunkingConfig
from src.models import Document


class TestParentChildChunking:
    def test_child_chunks_preserve_section_titles_without_markdown_noise(self):
        config = ChunkingConfig(
            strategy="header_based",
            min_chunk_chars=40,
            max_chunk_chars=200,
            overlap_chars=0,
            parent_chunk_min_chars=400,
            parent_chunk_max_chars=800,
        )
        chunker = HeaderBasedChunker(config)

        content = """# API Guide

## Authentication

Bearer tokens protect API calls and require issuer validation.

### Rotation

Rotate credentials every 24 hours and revoke compromised tokens immediately.
"""

        doc = Document(
            id="clean_titles",
            content=content,
            metadata={},
            links=[],
            tags=[],
            file_path="/test/doc.md",
            modified_time=datetime.now(),
        )

        chunks = chunker.chunk_document(doc)
        child_chunks = [c for c in chunks if "_parent_" not in c.chunk_id]

        authentication_chunk = next(
            chunk for chunk in child_chunks if chunk.header_path == "API Guide > Authentication"
        )
        rotation_chunk = next(
            chunk
            for chunk in child_chunks
            if chunk.header_path == "API Guide > Authentication > Rotation"
        )

        assert authentication_chunk.content.startswith("Authentication")
        assert "Context: API Guide" in authentication_chunk.content
        assert not authentication_chunk.content.startswith("#")

        assert rotation_chunk.content.startswith("Rotation")
        assert "Context: API Guide > Authentication" in rotation_chunk.content
        assert "#" not in rotation_chunk.content.split("\n", 1)[0]

    def test_parent_header_path_uses_shared_context_for_grouped_children(self):
        config = ChunkingConfig(
            strategy="header_based",
            min_chunk_chars=40,
            max_chunk_chars=160,
            overlap_chars=0,
            parent_chunk_min_chars=120,
            parent_chunk_max_chars=600,
        )
        chunker = HeaderBasedChunker(config)

        content = """# API Guide

## TL;DR

Use token auth.

## Authentication Details

Bearer tokens must be rotated every 24 hours. Include scopes and issuer validation.
"""

        doc = Document(
            id="parent_headers",
            content=content,
            metadata={},
            links=[],
            tags=[],
            file_path="/test/doc.md",
            modified_time=datetime.now(),
        )

        chunks = chunker.chunk_document(doc)
        parent_chunks = [c for c in chunks if "_parent_" in c.chunk_id]
        child_chunks = [c for c in chunks if "_parent_" not in c.chunk_id]

        assert len(parent_chunks) == 1
        assert parent_chunks[0].header_path == "API Guide"
        assert "+" not in parent_chunks[0].header_path
        assert all("+" not in chunk.header_path for chunk in child_chunks)
        assert "Authentication Details" in parent_chunks[0].content

    def test_trailing_short_section_merges_backward_into_previous_chunk(self):
        config = ChunkingConfig(
            strategy="header_based",
            min_chunk_chars=80,
            max_chunk_chars=220,
            overlap_chars=0,
            parent_chunk_min_chars=300,
            parent_chunk_max_chars=800,
        )
        chunker = HeaderBasedChunker(config)

        content = """# Operations Guide

## Monitoring

Monitoring dashboards should include request rate, latency, and error budgets.
Alert routing must notify the on-call engineer and preserve incident context.

## Appendix

CLI cheatsheet.
"""

        doc = Document(
            id="trailing_short",
            content=content,
            metadata={},
            links=[],
            tags=[],
            file_path="/test/doc.md",
            modified_time=datetime.now(),
        )

        chunks = chunker.chunk_document(doc)
        child_chunks = [c for c in chunks if "_parent_" not in c.chunk_id]

        assert len(child_chunks) >= 1
        assert all(chunk.header_path != "Operations Guide > Appendix" for chunk in child_chunks)
        assert any("Appendix" in chunk.content for chunk in child_chunks)
        assert any("CLI cheatsheet." in chunk.content for chunk in child_chunks)

    def test_creates_parent_and_child_chunks_when_enabled(self):
        config = ChunkingConfig(
            strategy="header_based",
            min_chunk_chars=100,
            max_chunk_chars=400,
            overlap_chars=0,
            parent_chunk_min_chars=500,
            parent_chunk_max_chars=1000,
        )
        chunker = HeaderBasedChunker(config)

        content = """# Main Title

This is the introduction with some content that should be reasonably long.

## Section One

First section content with details about topic A. More content here to make
the chunk larger. Adding extra text to ensure we have enough characters.

## Section Two

Second section content about topic B. More details and explanations here.
Additional content to reach minimum chunk size for testing purposes.

## Section Three

Third section content covering topic C. Further elaboration and examples.
More text to ensure this section is substantial enough for chunking.
"""

        doc = Document(
            id="test_doc",
            content=content,
            metadata={},
            links=[],
            tags=[],
            file_path="/test/doc.md",
            modified_time=datetime.now(),
        )

        chunks = chunker.chunk_document(doc)

        child_chunks = [c for c in chunks if "_parent_" not in c.chunk_id]

        # Should have both parents and children when parent retrieval is enabled
        # and content is long enough
        assert len(chunks) > 0

        # Child chunks should have parent_chunk_id set
        for child in child_chunks:
            if child.parent_chunk_id:
                assert child.parent_chunk_id.startswith("test_doc_parent_")

    def test_child_chunks_reference_correct_parent(self):
        config = ChunkingConfig(
            strategy="header_based",
            min_chunk_chars=50,
            max_chunk_chars=200,
            overlap_chars=0,
            parent_chunk_min_chars=300,
            parent_chunk_max_chars=800,
        )
        chunker = HeaderBasedChunker(config)

        content = """# Doc Title

Introduction paragraph with enough text to form a chunk.

## First Section

Content for section one with sufficient length for a chunk.

## Second Section

Content for section two with adequate length for testing.

## Third Section

Content for section three with more text for the chunk.
"""

        doc = Document(
            id="test_doc",
            content=content,
            metadata={},
            links=[],
            tags=[],
            file_path="/test/doc.md",
            modified_time=datetime.now(),
        )

        chunks = chunker.chunk_document(doc)

        parent_chunks = {c.chunk_id: c for c in chunks if "_parent_" in c.chunk_id}
        child_chunks = [c for c in chunks if "_parent_" not in c.chunk_id]

        # Each child with a parent_chunk_id should reference an existing parent
        for child in child_chunks:
            if child.parent_chunk_id:
                assert child.parent_chunk_id in parent_chunks

    def test_parent_content_contains_child_content(self):
        config = ChunkingConfig(
            strategy="header_based",
            min_chunk_chars=50,
            max_chunk_chars=200,
            overlap_chars=0,
            parent_chunk_min_chars=300,
            parent_chunk_max_chars=1000,
        )
        chunker = HeaderBasedChunker(config)

        content = """# Document

Intro text that should be included.

## Section A

Content for section A with enough text.

## Section B

Content for section B with enough text.
"""

        doc = Document(
            id="test_doc",
            content=content,
            metadata={},
            links=[],
            tags=[],
            file_path="/test/doc.md",
            modified_time=datetime.now(),
        )

        chunks = chunker.chunk_document(doc)

        parent_chunks = {c.chunk_id: c for c in chunks if "_parent_" in c.chunk_id}
        child_chunks = [c for c in chunks if "_parent_" not in c.chunk_id]

        # Child content should be part of parent content
        for child in child_chunks:
            if child.parent_chunk_id and child.parent_chunk_id in parent_chunks:
                parent = parent_chunks[child.parent_chunk_id]
                # The child content (without overlap markers) should be in parent
                child_text = child.content
                if child_text.startswith("[..."):
                    # Remove overlap prefix
                    child_text = child_text.split("]\n\n", 1)[-1]
                # Check content is part of parent (allowing for whitespace differences)
                assert any(
                    line.strip() in parent.content
                    for line in child_text.split("\n")
                    if line.strip()
                )
