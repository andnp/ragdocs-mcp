from pathlib import PurePosixPath
from typing import Type

from src.config import Config
from src.parsers.base import DocumentParser
from src.parsers.markdown import MarkdownParser


class ParserRegistry:
    def __init__(self):
        self._registry: dict[str, Type[DocumentParser]] = {}

    def register(self, pattern: str, parser_class: Type[DocumentParser]) -> None:
        self._registry[pattern] = parser_class

    def get_parser(self, file_path: str) -> Type[DocumentParser]:
        path = PurePosixPath(file_path)

        for pattern, parser_class in self._registry.items():
            try:
                if path.match(pattern):
                    return parser_class
            except ValueError:
                pass

        raise ValueError(f"No parser registered for file: {file_path}")


_default_registry = ParserRegistry()
_default_registry.register("*.md", MarkdownParser)
_default_registry.register("*.markdown", MarkdownParser)
_default_registry.register("**/*.md", MarkdownParser)
_default_registry.register("**/*.markdown", MarkdownParser)


def dispatch_parser(file_path: str, config: Config):
    registry = ParserRegistry()

    for pattern, parser_name in config.parsers.items():
        if parser_name == "MarkdownParser":
            registry.register(pattern, MarkdownParser)
            if pattern.startswith("**/"):
                simple_pattern = pattern.replace("**/", "")
                registry.register(simple_pattern, MarkdownParser)
        else:
            raise ValueError(f"Unknown parser: {parser_name}")

    parser_class = registry.get_parser(file_path)
    return parser_class()
