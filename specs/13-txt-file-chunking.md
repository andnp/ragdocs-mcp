# 13. Plain Text (.txt) File Chunking

## Executive Summary

This spec proposes adding `.txt` file indexing support to mcp-markdown-ragdocs. Unlike markdown files with header-based structure, plain text lacks semantic boundaries. Four chunking strategies were evaluated: paragraph-based (current fallback), sentence-based, semantic boundary detection, and sliding window. **Paragraph-based chunking** (Strategy 1) is selected as the optimal approach due to its simplicity (15 LOC, reuses existing logic), zero dependencies, and proven effectiveness in the current `_chunk_plain_text()` fallback. Implementation requires only parser registration and configuration—no new chunking code needed.

**Timeline:** 1-2 hours. **Risk:** Low (existing code path).

---

## 1. Goals & Non-Goals

### Goals
- Index `.txt` files with same search quality as markdown
- Reuse existing chunking infrastructure (minimal code duplication)
- Maintain configuration consistency (same min/max/overlap parameters)
- Support all current features (semantic search, keyword search, graph traversal)

### Non-Goals
- Natural language processing (NLP) for sentence boundaries (avoid spaCy/NLTK dependencies)
- Language-specific tokenization (assume UTF-8 text)
- Custom chunking strategies per file extension (single strategy for all `.txt`)
- Retroactive re-chunking of existing markdown files

---

## 2. Current State Analysis

### 2.1. Architecture Overview

**Chunking Pipeline:**
```
Parser → Chunker → IndexManager → [VectorIndex, KeywordIndex, GraphStore]
```

**Key Components:**

| Component | File | Role |
|-----------|------|------|
| `ChunkingStrategy` | [src/chunking/base.py](../src/chunking/base.py) | Abstract protocol defining `chunk_document()` |
| `HeaderBasedChunker` | [src/chunking/header_chunker.py](../src/chunking/header_chunker.py) | Markdown-specific implementation (292 LOC) |
| `get_chunker()` | [src/chunking/factory.py](../src/chunking/factory.py) | Factory function returning strategy based on config |
| `IndexManager` | [src/indexing/manager.py](../src/indexing/manager.py) | Coordinates parsing, chunking, and index updates |
| `dispatch_parser()` | [src/parsers/dispatcher.py](../src/parsers/dispatcher.py) | Routes file extensions to parsers |

**Configuration:**
```python
@dataclass
class ChunkingConfig:
    strategy: str = "header_based"
    min_chunk_chars: int = 200
    max_chunk_chars: int = 2000
    overlap_chars: int = 100
    include_parent_headers: bool = True
    parent_retrieval_enabled: bool = False
    parent_chunk_min_chars: int = 1500
    parent_chunk_max_chars: int = 2000
```

### 2.2. Existing Plain Text Handling

**Fallback Behavior:**
`HeaderBasedChunker` already handles plain text via `_chunk_plain_text()` (lines 274-347):
- Splits at paragraph boundaries (`\n\n+`)
- Respects `min_chunk_chars` and `max_chunk_chars`
- Tracks `start_pos` and `end_pos` for each chunk
- Sets `header_path = ""` (no header hierarchy)

**Limitations:**
- Only triggered when markdown has **no headers**
- No explicit support for `.txt` files (dispatcher only registers markdown)

**Opportunity:**
`.txt` files can reuse `_chunk_plain_text()` with **zero code changes**. Only parser registration needed.

---

## 3. Alternative Chunking Strategies

### Strategy 1: Paragraph-Based (RECOMMENDED) ✅

**Description:**
Split text at double newlines (`\n\n+`). Combine paragraphs until `max_chunk_chars` reached.

**Implementation:**
Reuse existing `HeaderBasedChunker._chunk_plain_text()`:
```python
# src/chunking/header_chunker.py (lines 274-347)
paragraphs = re.split(r'\n\n+', content)
```

**Pros:**
- ✅ **Zero new code** (reuses existing logic)
- ✅ Natural semantic boundaries (paragraphs = coherent units)
- ✅ Respects size constraints (min/max/overlap)
- ✅ No dependencies (stdlib `re` module)
- ✅ Already tested via markdown fallback path

**Cons:**
- ⚠️ Poor for dense text without paragraphs (e.g., logs)
- ⚠️ Fails on non-paragraph structured text (lists, tables)

**Search Quality Impact:** **High** (proven in production for headerless markdown)

**Implementation Complexity:** **15 LOC** (parser + config only)

**Impact/Cost Ratio:** **5.0** (best)

---

### Strategy 2: Sentence-Based

**Description:**
Split text into sentences using regex or NLP library (spaCy/NLTK). Group sentences into chunks.

**Implementation:**
```python
def _chunk_by_sentences(self, document: Document) -> list[Chunk]:
    # Sentence boundary detection
    sentences = re.split(r'(?<=[.!?])\s+', document.content)

    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= self.config.max_chunk_chars:
            current += " " + sent
        else:
            chunks.append(current)
            current = sent
    return chunks
```

**Pros:**
- ✅ Finer-grained chunks (better for QA)
- ✅ Natural linguistic boundaries

**Cons:**
- ❌ Regex sentence splitting unreliable (abbreviations, URLs, code)
- ❌ NLP libraries add heavy dependencies (spaCy: 100MB+)
- ❌ Slower processing (tokenization overhead)
- ❌ Poor for non-prose text (logs, config files)

**Search Quality Impact:** **Medium** (marginal improvement over paragraphs)

**Implementation Complexity:** **80 LOC** (sentence detection + chunking)

**Impact/Cost Ratio:** **0.5**

---

### Strategy 3: Semantic Boundary Detection

**Description:**
Use embeddings to detect topic shifts. Split when cosine similarity between adjacent paragraphs drops below threshold.

**Implementation:**
```python
def _chunk_by_semantic_boundaries(self, document: Document) -> list[Chunk]:
    paragraphs = re.split(r'\n\n+', document.content)
    embeddings = [self.model.embed(p) for p in paragraphs]

    chunks = []
    current_chunk = [paragraphs[0]]

    for i in range(1, len(paragraphs)):
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])
        if similarity < 0.7:  # Topic shift threshold
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [paragraphs[i]]
        else:
            current_chunk.append(paragraphs[i])

    return chunks
```

**Pros:**
- ✅ Semantically coherent chunks
- ✅ Adapts to content structure

**Cons:**
- ❌ Requires embedding model at index time (2x indexing time)
- ❌ Threshold tuning needed per corpus
- ❌ Poor for small documents (not enough paragraphs)
- ❌ Computationally expensive (O(n) embeddings per document)

**Search Quality Impact:** **High** (optimal coherence)

**Implementation Complexity:** **150 LOC** (embedding pipeline + boundary detection)

**Impact/Cost Ratio:** **0.3**

---

### Strategy 4: Sliding Window

**Description:**
Fixed-size chunks with configurable overlap. No semantic awareness.

**Implementation:**
```python
def _chunk_by_sliding_window(self, document: Document) -> list[Chunk]:
    content = document.content
    chunk_size = self.config.max_chunk_chars
    overlap = self.config.overlap_chars

    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        chunks.append(content[start:end])
        start += (chunk_size - overlap)

    return chunks
```

**Pros:**
- ✅ Simplest implementation (20 LOC)
- ✅ Predictable chunk sizes
- ✅ No dependencies

**Cons:**
- ❌ Breaks mid-sentence (poor readability)
- ❌ Ignores paragraph boundaries (low coherence)
- ❌ Suboptimal for search (random splits)

**Search Quality Impact:** **Low** (degrades relevance)

**Implementation Complexity:** **20 LOC**

**Impact/Cost Ratio:** **0.2**

---

## 4. Decision Matrix

| Strategy | Search Quality | Complexity (LOC) | Dependencies | Speed | Impact/Cost |
|----------|----------------|------------------|--------------|-------|-------------|
| **1. Paragraph-Based** ✅ | ⭐⭐⭐⭐ High | **15** | None | Fast | **5.0** |
| 2. Sentence-Based | ⭐⭐⭐ Medium | 80 | spaCy/NLTK | Slow | 0.5 |
| 3. Semantic Boundaries | ⭐⭐⭐⭐⭐ Highest | 150 | Embedding model | Very Slow | 0.3 |
| 4. Sliding Window | ⭐⭐ Low | 20 | None | Fast | 0.2 |

**Selected Strategy:** **Paragraph-Based** (Strategy 1)

**Rationale:**
1. **Proven reliability:** Already deployed in `_chunk_plain_text()` fallback
2. **Minimal risk:** Zero new chunking logic (only parser registration)
3. **Cost efficiency:** 15 LOC vs. 80-150 LOC for alternatives
4. **No dependencies:** Stdlib only (no spaCy/NLTK overhead)
5. **User preference alignment:** "simplicity, real implementation over complexity"

---

## 5. Proposed Solution

### 5.1. High-Level Design

**Approach:** Register a `PlainTextParser` that outputs `Document` objects, then reuse `HeaderBasedChunker._chunk_plain_text()` via existing fallback path.

**Key Insight:** `.txt` files are semantically equivalent to "markdown with no headers"—the chunking logic is identical.

### 5.2. Component Changes

#### 5.2.1. New Parser (NEW)

```python
# src/parsers/plaintext.py (~30 LOC)
from pathlib import Path
from datetime import datetime, timezone
from src.models import Document
from src.parsers.base import DocumentParser


class PlainTextParser(DocumentParser):
    """Parser for plain text (.txt) files."""

    def parse(self, file_path: str) -> Document:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try UTF-8 first, fall back to common encodings
        content = None
        for encoding in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
            try:
                content = path.read_text(encoding=encoding, errors="strict")
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content is None:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Cannot decode {file_path}")

        modified_time = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

        return Document(
            content=content,
            metadata={"source": str(path)},
            links=[],  # No link extraction for plain text
            tags=[],   # No tag extraction
            file_path=str(path),
            modified_time=modified_time,
        )
```

**Design Notes:**
- Follows `MarkdownParser` encoding fallback pattern (lines 34-48)
- No frontmatter parsing (plain text has no YAML headers)
- No link extraction (no markdown syntax)
- Minimal metadata (source file path only)

#### 5.2.2. Parser Registration

```python
# src/parsers/__init__.py
from src.parsers.markdown import MarkdownParser
from src.parsers.plaintext import PlainTextParser

__all__ = ["MarkdownParser", "PlainTextParser"]
```

```python
# src/parsers/dispatcher.py (modify dispatch_parser function)
from src.parsers.plaintext import PlainTextParser

def dispatch_parser(file_path: str, config: Config):
    path = PurePosixPath(file_path)

    for pattern in config.parsers.keys():
        parser_name = config.parsers[pattern]

        # Match against pattern
        if path.match(pattern) or (pattern.startswith("**/") and path.match(pattern.replace("**/", ""))):
            if parser_name == "MarkdownParser":
                return MarkdownParser()
            elif parser_name == "PlainTextParser":
                return PlainTextParser()

    raise ValueError(f"No parser registered for file: {file_path}")
```

#### 5.2.3. Configuration

```toml
# config.toml (add to [parsers] section)
[parsers]
"**/*.md" = "MarkdownParser"
"**/*.txt" = "PlainTextParser"  # NEW
```

**Rebuild Trigger:**
Adding `"**/*.txt"` to parsers config will trigger index rebuild (manifest version mismatch).

#### 5.2.4. Chunking Integration

**No code changes needed.** `HeaderBasedChunker.chunk_document()` already handles documents without headers:

```python
# src/chunking/header_chunker.py (lines 28-44, EXISTING CODE)
def chunk_document(self, document: Document) -> list[Chunk]:
    content_bytes = bytes(document.content, "utf8")
    tree = self.parser.parse(content_bytes)  # Parses as markdown
    root_node = tree.root_node

    headers = self._extract_headers(root_node, content_bytes)

    if not headers:  # ← Plain text triggers this path
        return self._chunk_plain_text(document)  # ← Reuses existing logic

    # ... markdown chunking ...
```

**Behavior:**
1. `PlainTextParser` returns `Document` with plain text content
2. `HeaderBasedChunker` attempts to parse as markdown (tree-sitter)
3. No headers found (plain text is valid markdown with no headers)
4. Falls back to `_chunk_plain_text()` (paragraph-based chunking)

**Validation:**
Plain text follows identical chunking path as headerless markdown—this code path is already tested and deployed.

---

## 6. Implementation Plan

### Phase 1: Parser Implementation (30 min)

**Tasks:**
- [ ] Create `src/parsers/plaintext.py`
- [ ] Implement `PlainTextParser.parse()`
- [ ] Add UTF-8 + fallback encoding support
- [ ] Add to `src/parsers/__init__.py` exports

**Acceptance Criteria:**
- `PlainTextParser().parse("test.txt")` returns valid `Document`
- Handles UTF-8 and latin-1 encodings
- Modified time preserved from file metadata

### Phase 2: Parser Registration (15 min)

**Tasks:**
- [ ] Modify `src/parsers/dispatcher.py` to handle parser name lookup
- [ ] Add `PlainTextParser` import and dispatch logic
- [ ] Update default config template with `"**/*.txt"` pattern

**Acceptance Criteria:**
- `dispatch_parser("doc.txt", config)` returns `PlainTextParser`
- `dispatch_parser("doc.md", config)` still returns `MarkdownParser`

### Phase 3: Testing (30 min)

**Tasks:**
- [ ] Unit test: `test_plaintext_parser()` (parse, encoding, errors)
- [ ] Integration test: `test_txt_file_indexing()` (end-to-end)
- [ ] Test chunk sizes match config (min/max/overlap)
- [ ] Test search retrieval from `.txt` chunks

**Test Cases:**

| Test | Input | Expected Output |
|------|-------|-----------------|
| `test_parse_utf8` | `hello.txt` (UTF-8) | `Document(content="hello world", ...)` |
| `test_parse_latin1` | `legacy.txt` (latin-1) | `Document(...)` with fallback encoding |
| `test_chunk_paragraphs` | Multi-paragraph `.txt` | Chunks split at `\n\n`, sizes within bounds |
| `test_search_txt_chunks` | Query + indexed `.txt` | Returns relevant chunks with scores |

### Phase 4: Documentation (15 min)

**Tasks:**
- [ ] Update `docs/configuration.md` with `.txt` parser example
- [ ] Add `.txt` support to `README.md` features list
- [ ] Document encoding handling in `docs/development.md`

---

## 7. Testing Strategy

### 7.1. Unit Tests

**File:** `tests/unit/test_plaintext_parser.py` (~50 LOC)

```python
import pytest
from pathlib import Path
from src.parsers.plaintext import PlainTextParser

def test_parse_simple_text(tmp_path):
    file = tmp_path / "test.txt"
    file.write_text("Hello world\n\nSecond paragraph")

    parser = PlainTextParser()
    doc = parser.parse(str(file))

    assert doc.content == "Hello world\n\nSecond paragraph"
    assert doc.metadata["source"] == str(file)
    assert doc.links == []
    assert doc.tags == []

def test_parse_encoding_fallback(tmp_path):
    file = tmp_path / "latin.txt"
    file.write_bytes(b"Caf\xe9")  # Latin-1 encoded "Café"

    parser = PlainTextParser()
    doc = parser.parse(str(file))

    assert "Caf" in doc.content  # Successfully decoded

def test_parse_nonexistent_file():
    parser = PlainTextParser()
    with pytest.raises(FileNotFoundError):
        parser.parse("/nonexistent/file.txt")
```

### 7.2. Integration Tests

**File:** `tests/integration/test_txt_indexing.py` (~80 LOC)

```python
def test_index_and_search_txt_file(tmp_path, index_manager, config):
    # Create test .txt file
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text(
        "Machine Learning Notes\n\n"
        "Neural networks are computational models.\n\n"
        "Training requires labeled data."
    )

    # Index document
    index_manager.index_document(str(txt_file))

    # Query
    results = index_manager.vector.search("neural networks", top_k=5)

    # Verify chunk retrieval
    assert len(results) > 0
    chunk = results[0]
    assert "neural networks" in chunk.content.lower()
    assert chunk.header_path == ""  # No headers in plain text

def test_txt_chunking_respects_size_limits(tmp_path, config):
    # Create large plain text file
    txt_file = tmp_path / "large.txt"
    paragraphs = [f"Paragraph {i} content." * 50 for i in range(100)]
    txt_file.write_text("\n\n".join(paragraphs))

    # Parse and chunk
    parser = PlainTextParser()
    doc = parser.parse(str(txt_file))

    chunker = HeaderBasedChunker(config.chunking)
    chunks = chunker.chunk_document(doc)

    # Verify constraints
    for chunk in chunks:
        assert len(chunk.content) >= config.chunking.min_chunk_chars
        assert len(chunk.content) <= config.chunking.max_chunk_chars
```

### 7.3. Performance Validation

**Benchmark:** Index 100 `.txt` files (1-10KB each), measure:
- Indexing time (should match markdown ±10%)
- Query latency (no degradation)
- Memory usage (no significant increase)

---

## 8. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Encoding issues with non-UTF-8 files | Medium | Medium | Fallback encoding sequence (UTF-8 → latin-1 → cp1252) |
| Poor chunking for dense text (logs) | Low | Low | Document as limitation; consider sliding window in future |
| Index rebuild required | High | Low | Automatic on config change (existing mechanism) |
| User confusion (why rebuild?) | Medium | Low | Log clear message: "New parser added, rebuilding index..." |

---

## 9. File Manifest

### Files to Create
- [x] `src/parsers/plaintext.py` (~30 LOC)
- [x] `tests/unit/test_plaintext_parser.py` (~50 LOC)
- [x] `tests/integration/test_txt_indexing.py` (~80 LOC)

### Files to Modify
- [x] `src/parsers/__init__.py` (+2 LOC: import + export)
- [x] `src/parsers/dispatcher.py` (+8 LOC: parser name dispatch)
- [x] `config.toml` (+1 LOC: `.txt` pattern)
- [x] `docs/configuration.md` (+15 LOC: parser example)
- [x] `docs/development.md` (+10 LOC: encoding notes)
- [x] `README.md` (+1 LOC: features list)

**Total LOC:** ~180 (90% tests, 10% production)

---

## 10. Future Enhancements (Out of Scope)

### 10.1. File Type-Specific Strategies
Allow different chunking strategies per file type:
```toml
[chunking.strategies]
"**/*.md" = "header_based"
"**/*.txt" = "paragraph_based"
"**/*.log" = "sliding_window"
```

**Complexity:** High (requires refactoring `get_chunker()` and config schema)

**Value:** Medium (most `.txt` files are prose; logs are rare)

### 10.2. Sentence-Based Chunking
Add optional sentence-level chunking for QA workloads:
```toml
[chunking]
strategy = "sentence_based"  # Requires spaCy
```

**Complexity:** High (80 LOC + spaCy dependency)

**Value:** Low (marginal search quality improvement)

### 10.3. Metadata Extraction
Extract dates, URLs, or email addresses from plain text:
```python
metadata = {
    "dates": ["2025-01-08"],
    "urls": ["https://example.com"],
    "emails": ["user@example.com"]
}
```

**Complexity:** Medium (regex patterns + extraction logic)

**Value:** Medium (useful for certain corpora, e.g., logs or emails)

---

## 11. Success Metrics

### 11.1. Functional Metrics
- [ ] `.txt` files indexed without errors
- [ ] Chunks sizes within configured bounds (200-2000 chars)
- [ ] Search queries return relevant `.txt` chunks
- [ ] Encoding fallback handles non-UTF-8 files

### 11.2. Performance Metrics
- [ ] Indexing time for `.txt` matches markdown (±10%)
- [ ] Query latency unchanged (<5ms regression)
- [ ] Memory usage stable (<5% increase)

### 11.3. Quality Metrics
- [ ] Manual review: top-3 results for 10 queries include relevant `.txt` chunks
- [ ] No chunk boundary splits mid-sentence (manual inspection of 20 random chunks)

---

## 12. Appendix: Code Examples

### A. Complete PlainTextParser Implementation

```python
# src/parsers/plaintext.py
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.models import Document
from src.parsers.base import DocumentParser

logger = logging.getLogger(__name__)


class PlainTextParser(DocumentParser):
    """
    Parser for plain text (.txt) files.

    Supports UTF-8 and common fallback encodings (latin-1, cp1252, iso-8859-1).
    Returns Document with minimal metadata (no links, tags, or frontmatter).
    """

    def parse(self, file_path: str) -> Document:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try UTF-8 first, fall back to common encodings
        content = None
        encoding_used = None
        for encoding in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
            try:
                content = path.read_text(encoding=encoding, errors="strict")
                encoding_used = encoding
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content is None:
            raise UnicodeDecodeError(
                "utf-8", b"", 0, 1,
                f"Could not decode {file_path} with any supported encoding"
            )

        if encoding_used != "utf-8":
            logger.warning(f"File {file_path} decoded with {encoding_used} encoding")

        modified_time = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

        return Document(
            content=content,
            metadata={
                "source": str(path),
                "encoding": encoding_used,
            },
            links=[],  # No link extraction for plain text
            tags=[],   # No frontmatter tags
            file_path=str(path),
            modified_time=modified_time,
        )
```

### B. Updated Dispatcher

```python
# src/parsers/dispatcher.py
from pathlib import PurePosixPath

from src.config import Config
from src.parsers.markdown import MarkdownParser
from src.parsers.plaintext import PlainTextParser


def dispatch_parser(file_path: str, config: Config):
    """
    Route file to appropriate parser based on config patterns.

    Supports:
    - MarkdownParser: **/*.md
    - PlainTextParser: **/*.txt
    """
    path = PurePosixPath(file_path)

    for pattern in config.parsers.keys():
        parser_name = config.parsers[pattern]

        # Try direct match
        try:
            if path.match(pattern):
                return _instantiate_parser(parser_name, file_path)
        except ValueError:
            pass

        # Try simplified pattern (strip **/)
        if pattern.startswith("**/"):
            simple_pattern = pattern.replace("**/", "")
            try:
                if path.match(simple_pattern):
                    return _instantiate_parser(parser_name, file_path)
            except ValueError:
                pass

    raise ValueError(f"No parser registered for file: {file_path}")


def _instantiate_parser(parser_name: str, file_path: str):
    """Factory function to create parser instances."""
    if parser_name == "MarkdownParser":
        return MarkdownParser()
    elif parser_name == "PlainTextParser":
        return PlainTextParser()
    else:
        raise ValueError(f"Unknown parser: {parser_name} for {file_path}")
```

---

## 13. References

### Internal Documents
- [specs/08-document-chunking.md](./08-document-chunking.md) - Chunking architecture
- [docs/development.md](../docs/development.md) - Adding new parsers
- [src/chunking/header_chunker.py](../src/chunking/header_chunker.py) - Existing chunking logic

### External Resources
- Python `re` module: https://docs.python.org/3/library/re.html
- Unicode encodings: https://docs.python.org/3/howto/unicode.html
- tree-sitter markdown: https://github.com/tree-sitter-grammars/tree-sitter-markdown

---

**Document Version:** 1.0
**Author:** GitHub Copilot (Planner Mode)
**Last Updated:** 2026-01-08
**Status:** Ready for Implementation
