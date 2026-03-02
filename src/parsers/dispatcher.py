from pathlib import Path

from src.parsers.base import DocumentParser
from src.parsers.markdown import MarkdownParser
from src.parsers.plaintext import PlainTextParser

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".md", ".markdown", ".txt"})

_EXTENSION_TO_PARSER: dict[str, str] = {
    ".md": "MarkdownParser",
    ".markdown": "MarkdownParser",
    ".txt": "PlainTextParser",
}


def dispatch_parser(file_path: str) -> DocumentParser:
    suffix = Path(file_path).suffix.lower()
    parser_name = _EXTENSION_TO_PARSER.get(suffix)
    if parser_name is None:
        raise ValueError(f"No parser registered for file: {file_path}")
    return _instantiate_parser(parser_name, file_path)


def _instantiate_parser(parser_name: str, file_path: str) -> DocumentParser:
    if parser_name == "MarkdownParser":
        return MarkdownParser()
    elif parser_name == "PlainTextParser":
        return PlainTextParser()
    else:
        raise ValueError(f"Unknown parser: {parser_name} for {file_path}")
