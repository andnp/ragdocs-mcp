import pytest

from src.config import Config, IndexingConfig
from src.parsers.dispatcher import dispatch_parser
from src.parsers.markdown import MarkdownParser


def test_dispatch_parser_with_config():
    config = Config(
        indexing=IndexingConfig(),
        parsers={
            "**/*.md": "MarkdownParser",
            "**/*.markdown": "MarkdownParser"
        }
    )

    parser = dispatch_parser("docs/test.md", config)
    assert isinstance(parser, MarkdownParser)


def test_dispatch_parser_markdown_extension():
    config = Config(
        indexing=IndexingConfig(),
        parsers={
            "**/*.md": "MarkdownParser",
            "**/*.markdown": "MarkdownParser"
        }
    )

    parser = dispatch_parser("notes.markdown", config)
    assert isinstance(parser, MarkdownParser)


def test_dispatch_parser_no_match():
    config = Config(
        indexing=IndexingConfig(),
        parsers={
            "**/*.md": "MarkdownParser"
        }
    )

    with pytest.raises(ValueError, match="No parser registered for file"):
        dispatch_parser("document.txt", config)


def test_dispatch_parser_nested_paths():
    config = Config(
        indexing=IndexingConfig(),
        parsers={
            "**/*.md": "MarkdownParser"
        }
    )

    parser = dispatch_parser("a/b/c/d/note.md", config)
    assert isinstance(parser, MarkdownParser)


def test_dispatch_parser_absolute_paths():
    config = Config(
        indexing=IndexingConfig(),
        parsers={
            "**/*.md": "MarkdownParser"
        }
    )

    parser = dispatch_parser("/home/user/docs/note.md", config)
    assert isinstance(parser, MarkdownParser)


def test_dispatch_parser_simple_patterns():
    config = Config(
        indexing=IndexingConfig(),
        parsers={
            "*.md": "MarkdownParser",
            "*.markdown": "MarkdownParser"
        }
    )

    parser = dispatch_parser("file.md", config)
    assert isinstance(parser, MarkdownParser)

    parser = dispatch_parser("file.markdown", config)
    assert isinstance(parser, MarkdownParser)
