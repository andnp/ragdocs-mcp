from pathlib import PurePosixPath

from src.config import Config
from src.parsers.markdown import MarkdownParser


def dispatch_parser(file_path: str, config: Config):
    path = PurePosixPath(file_path)

    for pattern in config.parsers.keys():
        try:
            if path.match(pattern):
                return MarkdownParser()
        except ValueError:
            pass

        # Handle simplified pattern matching for **/ prefix
        if pattern.startswith("**/"):
            simple_pattern = pattern.replace("**/", "")
            try:
                if path.match(simple_pattern):
                    return MarkdownParser()
            except ValueError:
                pass

    raise ValueError(f"No parser registered for file: {file_path}")
