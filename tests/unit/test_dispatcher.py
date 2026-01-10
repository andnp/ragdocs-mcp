import pytest

from src.config import Config, IndexingConfig
from src.parsers.dispatcher import dispatch_parser
from src.parsers.markdown import MarkdownParser
from src.parsers.plaintext import PlainTextParser


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


def test_dispatch_parser_txt():
    config = Config(
        indexing=IndexingConfig(),
        parsers={
            "**/*.txt": "PlainTextParser"
        }
    )

    parser = dispatch_parser("notes.txt", config)
    assert isinstance(parser, PlainTextParser)


def test_dispatch_parser_txt_nested():
    config = Config(
        indexing=IndexingConfig(),
        parsers={
            "**/*.txt": "PlainTextParser"
        }
    )

    parser = dispatch_parser("docs/notes/file.txt", config)
    assert isinstance(parser, PlainTextParser)


def test_dispatch_parser_mixed_types():
    config = Config(
        indexing=IndexingConfig(),
        parsers={
            "**/*.md": "MarkdownParser",
            "**/*.txt": "PlainTextParser"
        }
    )

    md_parser = dispatch_parser("file.md", config)
    assert isinstance(md_parser, MarkdownParser)

    txt_parser = dispatch_parser("file.txt", config)
    assert isinstance(txt_parser, PlainTextParser)


def test_dispatch_parser_unknown_parser():
    config = Config(
        indexing=IndexingConfig(),
        parsers={
            "**/*.xyz": "UnknownParser"
        }
    )

    with pytest.raises(ValueError, match="Unknown parser"):
        dispatch_parser("file.xyz", config)
