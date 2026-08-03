import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import yaml
from tree_sitter import Language, Node, Parser
from tree_sitter_markdown import language

from mcp_markdown_ragdocs.models import Document, DocumentMetadataValue
from mcp_markdown_ragdocs.parsers.base import DocumentParser
from mcp_markdown_ragdocs.parsers.encoding import read_text_with_encoding_fallback

logger = logging.getLogger(__name__)


INDEXED_FRONTMATTER_FIELDS = [
    "title",
    "description",
    "summary",
    "keywords",
    "author",
    "category",
    "type",
    "related",
]

@dataclass
class LinkWithContext:
    target: str
    header_context: str


def _normalize_frontmatter_value(value: object) -> object:
    """Convert YAML-loaded metadata into JSON-serializable values.

    PyYAML may coerce scalars such as dates and datetimes into native Python
    objects. Those values later flow into chunk metadata and LlamaIndex's
    docstore persistence, which expects JSON-serializable metadata.
    """

    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, list | tuple):
        return [_normalize_frontmatter_value(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _normalize_frontmatter_value(item)
            for key, item in value.items()
        }

    return str(value)


def _document_metadata_value(value: object) -> DocumentMetadataValue:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if value is None:
        return None
    if isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


class MarkdownParser(DocumentParser):
    def __init__(self):
        self.parser = Parser(Language(language()))

    def parse(self, file_path: str) -> Document:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        content, encoding = read_text_with_encoding_fallback(file_path)
        if encoding != "utf-8":
            logger.warning(f"File {file_path} decoded with {encoding} encoding")

        content_bytes = bytes(content, "utf8")
        tree = self.parser.parse(content_bytes)
        root_node = tree.root_node

        frontmatter_metadata = self._extract_frontmatter(content)

        text_content = self._extract_text_content(content, root_node)

        frontmatter_tags = _string_list(frontmatter_metadata.get("tags"))

        wikilinks = self._extract_wikilinks(root_node, content_bytes)
        transclusions = self._extract_transclusions(root_node, content_bytes)
        inline_tags = self._extract_tags(root_node, content_bytes)

        all_tags = sorted(set(frontmatter_tags + inline_tags))

        file_stat = os.stat(file_path)
        modified_time = datetime.fromtimestamp(file_stat.st_mtime, tz=UTC)

        doc_id = Path(file_path).stem

        metadata: dict[str, DocumentMetadataValue] = {
            key: _document_metadata_value(value)
            for key, value in frontmatter_metadata.items()
        }
        metadata.pop("aliases", None)
        metadata.pop("tags", None)
        if transclusions:
            metadata["transclusions"] = transclusions

        for field in INDEXED_FRONTMATTER_FIELDS:
            if field in frontmatter_metadata:
                value = frontmatter_metadata[field]
                if isinstance(value, list):
                    metadata[field] = [str(item) for item in value]
                else:
                    metadata[field] = str(value)

        related = _string_list(frontmatter_metadata.get("related"))
        wikilinks = list(set(wikilinks) | set(related))

        return Document(
            id=doc_id,
            content=text_content,
            metadata=metadata,
            links=wikilinks,
            tags=all_tags,
            file_path=file_path,
            modified_time=modified_time,
        )

    def _extract_frontmatter(self, content: str) -> dict[str, object]:
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if not match:
            return {}

        yaml_content = match.group(1)

        try:
            metadata = yaml.safe_load(yaml_content)
            if metadata is None:
                return {}
            if not isinstance(metadata, dict):
                return {}
            return {
                str(key): _normalize_frontmatter_value(value)
                for key, value in metadata.items()
            }
        except yaml.YAMLError:
            return {}

    def _extract_text_content(self, content: str, root_node: Node) -> str:
        frontmatter_pattern = r"^---\s*\n.*?\n---\s*\n"
        text_without_frontmatter = re.sub(
            frontmatter_pattern, "", content, count=1, flags=re.DOTALL
        )

        return text_without_frontmatter.strip()

    def _collect_non_code_text(self, node: Node, content_bytes: bytes, parts: list):
        if node.type in ("fenced_code_block", "indented_code_block"):
            return

        if node.type == "inline":
            in_backticks = False
            last_pos = node.start_byte

            for child in node.children:
                if child.type == "`":
                    if not in_backticks:
                        if child.start_byte > last_pos:
                            text = content_bytes[last_pos : child.start_byte].decode(
                                "utf8"
                            )
                            parts.append(text)
                        in_backticks = True
                        last_pos = child.end_byte
                    else:
                        in_backticks = False
                        last_pos = child.end_byte

            if not in_backticks and last_pos < node.end_byte:
                text = content_bytes[last_pos : node.end_byte].decode("utf8")
                parts.append(text)

            parts.append(" ")
            return

        if node.children:
            for child in node.children:
                self._collect_non_code_text(child, content_bytes, parts)

    def _get_text_excluding_code(self, root_node: Node, content_bytes: bytes):
        parts = []
        self._collect_non_code_text(root_node, content_bytes, parts)
        return "".join(parts)

    def _extract_wikilinks(self, root_node: Node, content_bytes: bytes):
        wikilinks = set()
        wikilink_pattern = re.compile(r"(?<!!)\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
        markdown_link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)")

        text = self._get_text_excluding_code(root_node, content_bytes)
        matches = wikilink_pattern.findall(text)
        wikilinks.update(matches)
        wikilinks.update(markdown_link_pattern.findall(text))

        return list(wikilinks)

    def extract_links_with_context(self, file_path: str) -> list[LinkWithContext]:
        if not os.path.exists(file_path):
            return []

        try:
            content, _ = read_text_with_encoding_fallback(file_path)
        except UnicodeDecodeError:
            return []

        content_bytes = bytes(content, "utf8")
        tree = self.parser.parse(content_bytes)
        root_node = tree.root_node

        headers = self._extract_header_positions(root_node, content_bytes)
        wikilink_pattern = re.compile(r"(?<!!)\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

        text = self._get_text_excluding_code(root_node, content_bytes)

        links_with_context: list[LinkWithContext] = []
        for match in wikilink_pattern.finditer(text):
            target = match.group(1)
            position = match.start()
            header_context = self._find_header_context_at_position(headers, position)
            links_with_context.append(
                LinkWithContext(target=target, header_context=header_context)
            )

        return links_with_context

    def _extract_header_positions(
        self, root_node: Node, content_bytes: bytes
    ) -> list[tuple[int, int, str]]:
        headers: list[tuple[int, int, str]] = []

        def visit(node: Node) -> None:
            if node.type in ("atx_heading", "setext_heading"):
                text = ""
                for child in node.children:
                    if child.type == "inline":
                        text = (
                            content_bytes[child.start_byte : child.end_byte]
                            .decode("utf8")
                            .strip()
                        )
                        break

                headers.append((node.start_byte, node.end_byte, text))

            for child in node.children:
                visit(child)

        visit(root_node)
        return sorted(headers, key=lambda x: x[0])

    def _find_header_context_at_position(
        self, headers: list[tuple[int, int, str]], position: int
    ) -> str:
        current_header = ""
        for start, end, text in headers:
            if start <= position:
                current_header = text
            else:
                break
        return current_header

    def _extract_transclusions(self, root_node: Node, content_bytes: bytes):
        transclusions = set()
        transclusion_pattern = re.compile(r"!\[\[([^\]]+)\]\]")

        text = self._get_text_excluding_code(root_node, content_bytes)
        matches = transclusion_pattern.findall(text)
        transclusions.update(matches)

        return list(transclusions)

    def _extract_tags(self, root_node: Node, content_bytes: bytes):
        tags = set()
        tag_pattern = re.compile(r"(?:^|\s)#([\w-]+)", re.MULTILINE)

        text = self._get_text_excluding_code(root_node, content_bytes)
        matches = tag_pattern.findall(text)
        tags.update(matches)

        return list(tags)
