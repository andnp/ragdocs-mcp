import logging
from datetime import UTC, datetime
from pathlib import Path

from mcp_markdown_ragdocs.models import Document, DocumentMetadataValue
from mcp_markdown_ragdocs.parsers.base import DocumentParser
from mcp_markdown_ragdocs.parsers.encoding import read_text_with_encoding_fallback

logger = logging.getLogger(__name__)


class PlainTextParser(DocumentParser):
    def parse(self, file_path: str):
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content, encoding_used = read_text_with_encoding_fallback(file_path)

        if encoding_used != "utf-8":
            logger.warning(f"File {file_path} decoded with {encoding_used} encoding")

        modified_time = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)

        metadata: dict[str, DocumentMetadataValue] = {
            "source": str(path)
        }
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
