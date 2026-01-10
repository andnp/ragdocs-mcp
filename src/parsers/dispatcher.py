from pathlib import PurePosixPath

from src.config import Config
from src.parsers.markdown import MarkdownParser
from src.parsers.plaintext import PlainTextParser


def dispatch_parser(file_path: str, config: Config):
    path = PurePosixPath(file_path)

    for pattern in config.parsers.keys():
        parser_name = config.parsers[pattern]

        matched = False
        try:
            if path.match(pattern):
                matched = True
        except ValueError:
            pass

        if not matched and pattern.startswith("**/"):
            simple_pattern = pattern.replace("**/", "")
            try:
                if path.match(simple_pattern):
                    matched = True
            except ValueError:
                pass

        if matched:
            return _instantiate_parser(parser_name, file_path)

    raise ValueError(f"No parser registered for file: {file_path}")


def _instantiate_parser(parser_name: str, file_path: str):
    if parser_name == "MarkdownParser":
        return MarkdownParser()
    elif parser_name == "PlainTextParser":
        return PlainTextParser()
    else:
        raise ValueError(f"Unknown parser: {parser_name} for {file_path}")
