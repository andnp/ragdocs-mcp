import logging
from datetime import datetime, timezone
from pathlib import Path

from src.models import Document
from src.parsers.base import DocumentParser

logger = logging.getLogger(__name__)


class PlainTextParser(DocumentParser):
    def parse(self, file_path: str):
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

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

        metadata: dict[str, str | list[str] | int | float | bool] = {"source": str(path)}
        if encoding_used and encoding_used != "utf-8":
            metadata["encoding"] = encoding_used

        return Document(
            id=path.stem,
            content=content,
            metadata=metadata,
            links=[],
            tags=[],
            file_path=str(path),
            modified_time=modified_time,
        )
