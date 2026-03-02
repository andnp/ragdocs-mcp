import pytest

from src.parsers.dispatcher import dispatch_parser
from src.parsers.markdown import MarkdownParser
from src.parsers.plaintext import PlainTextParser


def test_dispatch_parser_md():
    parser = dispatch_parser("docs/test.md")
    assert isinstance(parser, MarkdownParser)


def test_dispatch_parser_markdown_extension():
    parser = dispatch_parser("notes.markdown")
    assert isinstance(parser, MarkdownParser)


def test_dispatch_parser_txt():
    parser = dispatch_parser("notes.txt")
    assert isinstance(parser, PlainTextParser)


def test_dispatch_parser_no_match():
    with pytest.raises(ValueError, match="No parser registered for file"):
        dispatch_parser("document.xyz")


def test_dispatch_parser_nested_paths():
    parser = dispatch_parser("a/b/c/d/note.md")
    assert isinstance(parser, MarkdownParser)


def test_dispatch_parser_absolute_paths():
    parser = dispatch_parser("/home/user/docs/note.md")
    assert isinstance(parser, MarkdownParser)


def test_dispatch_parser_txt_nested():
    parser = dispatch_parser("docs/notes/file.txt")
    assert isinstance(parser, PlainTextParser)


def test_dispatch_parser_mixed_types():
    md_parser = dispatch_parser("file.md")
    assert isinstance(md_parser, MarkdownParser)

    txt_parser = dispatch_parser("file.txt")
    assert isinstance(txt_parser, PlainTextParser)


def test_dispatch_parser_case_insensitive_extension():
    parser = dispatch_parser("FILE.MD")
    assert isinstance(parser, MarkdownParser)
