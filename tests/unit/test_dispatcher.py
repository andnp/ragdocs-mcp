import pytest

from src.config import Config, IndexingConfig
from src.parsers.dispatcher import ParserRegistry, dispatch_parser
from src.parsers.markdown import MarkdownParser


def test_parser_registry_register_and_get():
    registry = ParserRegistry()
    registry.register("*.md", MarkdownParser)

    parser_class = registry.get_parser("test.md")
    assert parser_class == MarkdownParser


def test_parser_registry_glob_patterns():
    registry = ParserRegistry()
    registry.register("**/*.md", MarkdownParser)
    registry.register("*.md", MarkdownParser)

    parser_class = registry.get_parser("docs/notes/test.md")
    assert parser_class == MarkdownParser

    parser_class = registry.get_parser("test.md")
    assert parser_class == MarkdownParser


def test_parser_registry_no_match_raises_error():
    registry = ParserRegistry()
    registry.register("*.md", MarkdownParser)

    with pytest.raises(ValueError, match="No parser registered for file"):
        registry.get_parser("test.txt")


def test_parser_registry_priority_first_match_wins():
    registry = ParserRegistry()
    registry.register("*.md", MarkdownParser)
    registry.register("**/*.md", MarkdownParser)

    parser_class = registry.get_parser("test.md")
    assert parser_class == MarkdownParser


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


def test_dispatch_parser_unknown_parser_name():
    config = Config(
        indexing=IndexingConfig(),
        parsers={
            "**/*.txt": "UnknownParser"
        }
    )

    with pytest.raises(ValueError, match="Unknown parser"):
        dispatch_parser("file.txt", config)


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


def test_parser_registry_multiple_patterns():
    registry = ParserRegistry()
    registry.register("*.md", MarkdownParser)
    registry.register("*.markdown", MarkdownParser)

    parser_class = registry.get_parser("file.md")
    assert parser_class == MarkdownParser

    parser_class = registry.get_parser("file.markdown")
    assert parser_class == MarkdownParser
