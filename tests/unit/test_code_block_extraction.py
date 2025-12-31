import tempfile

from src.parsers.markdown import MarkdownParser


class TestExtractCodeBlocks:
    def test_extract_single_code_block(self):
        parser = MarkdownParser()

        content = """# Title

```python
def hello():
    print("world")
```
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(content)
            f.flush()

            blocks = parser.extract_code_blocks(f.name, "test_doc")

        assert len(blocks) == 1
        assert blocks[0].language == "python"
        assert 'def hello():' in blocks[0].content
        assert blocks[0].doc_id == "test_doc"

    def test_extract_multiple_code_blocks(self):
        parser = MarkdownParser()

        content = """# Title

```python
def one():
    pass
```

Some text here.

```javascript
function two() {}
```

```rust
fn three() {}
```
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(content)
            f.flush()

            blocks = parser.extract_code_blocks(f.name, "test_doc")

        assert len(blocks) == 3
        assert blocks[0].language == "python"
        assert blocks[1].language == "javascript"
        assert blocks[2].language == "rust"
        assert blocks[0].id == "test_doc_code_0"
        assert blocks[1].id == "test_doc_code_1"
        assert blocks[2].id == "test_doc_code_2"

    def test_extract_code_block_without_language(self):
        parser = MarkdownParser()

        content = """# Title

```
plain code here
```
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(content)
            f.flush()

            blocks = parser.extract_code_blocks(f.name, "test_doc")

        assert len(blocks) == 1
        assert blocks[0].language == ""
        assert "plain code here" in blocks[0].content

    def test_extract_no_code_blocks(self):
        parser = MarkdownParser()

        content = """# Title

Just regular text with no code blocks.
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(content)
            f.flush()

            blocks = parser.extract_code_blocks(f.name, "test_doc")

        assert len(blocks) == 0

    def test_extract_empty_code_block_skipped(self):
        parser = MarkdownParser()

        content = """# Title

```python
```

```javascript
actual_code();
```
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(content)
            f.flush()

            blocks = parser.extract_code_blocks(f.name, "test_doc")

        assert len(blocks) == 1
        assert blocks[0].language == "javascript"

    def test_nonexistent_file_returns_empty(self):
        parser = MarkdownParser()

        blocks = parser.extract_code_blocks("/nonexistent/file.md", "test_doc")

        assert blocks == []
