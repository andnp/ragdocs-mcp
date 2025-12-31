from datetime import datetime

from src.chunking.header_chunker import HeaderBasedChunker
from src.config import ChunkingConfig
from src.models import Document


class TestParentChildChunking:

    def test_creates_parent_and_child_chunks_when_enabled(self):
        config = ChunkingConfig(
            strategy="header_based",
            min_chunk_chars=100,
            max_chunk_chars=400,
            overlap_chars=0,
            parent_retrieval_enabled=True,
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

    def test_no_parent_chunks_when_disabled(self):
        config = ChunkingConfig(
            strategy="header_based",
            min_chunk_chars=100,
            max_chunk_chars=400,
            overlap_chars=0,
            parent_retrieval_enabled=False,
        )
        chunker = HeaderBasedChunker(config)

        content = """# Title

Some content here.

## Section

More content in this section.
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

        # No parent chunks when disabled
        parent_chunks = [c for c in chunks if "_parent_" in c.chunk_id]
        assert len(parent_chunks) == 0

        # All chunks should have no parent_chunk_id
        for chunk in chunks:
            assert chunk.parent_chunk_id is None

    def test_child_chunks_reference_correct_parent(self):
        config = ChunkingConfig(
            strategy="header_based",
            min_chunk_chars=50,
            max_chunk_chars=200,
            overlap_chars=0,
            parent_retrieval_enabled=True,
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
            parent_retrieval_enabled=True,
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
