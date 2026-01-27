import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from tree_sitter import Language, Parser, Node
from tree_sitter_markdown import language

from src.models import CodeBlock, Document
from src.parsers.base import DocumentParser

logger = logging.getLogger(__name__)


INDEXED_FRONTMATTER_FIELDS = [
    "title", "description", "summary", "keywords",
    "author", "category", "type", "related"
]


@dataclass
class LinkWithContext:
    target: str
    header_context: str


class MarkdownParser(DocumentParser):
    def __init__(self):
        self.parser = Parser(Language(language()))

    def parse(self, file_path: str) -> Document:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try UTF-8 first, fall back to other encodings if needed
        content = None
        last_error = None
        for encoding in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
            try:
                with open(file_path, "r", encoding=encoding, errors="strict") as f:
                    content = f.read()
                if encoding != "utf-8":
                    logger.warning(f"File {file_path} decoded with {encoding} encoding")
                break
            except (UnicodeDecodeError, LookupError) as e:
                last_error = e
                continue

        if content is None:
            raise UnicodeDecodeError(
                "utf-8", b"", 0, 1,
                f"Could not decode {file_path} with any supported encoding. Last error: {last_error}"
            )

        content_bytes = bytes(content, "utf8")
        tree = self.parser.parse(content_bytes)
        root_node = tree.root_node

        frontmatter_metadata = self._extract_frontmatter(content)

        text_content = self._extract_text_content(content, root_node)

        aliases = frontmatter_metadata.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]

        frontmatter_tags = frontmatter_metadata.get("tags", [])
        if isinstance(frontmatter_tags, str):
            frontmatter_tags = [frontmatter_tags]

        wikilinks = self._extract_wikilinks(root_node, content_bytes)
        transclusions = self._extract_transclusions(root_node, content_bytes)
        inline_tags = self._extract_tags(root_node, content_bytes)

        all_tags = sorted(set(frontmatter_tags + inline_tags))

        file_stat = os.stat(file_path)
        modified_time = datetime.fromtimestamp(file_stat.st_mtime)

        doc_id = Path(file_path).stem

        metadata = dict(frontmatter_metadata)
        if "aliases" in metadata:
            del metadata["aliases"]
        if "tags" in metadata:
            del metadata["tags"]
        if transclusions:
            metadata["transclusions"] = transclusions

        for field in INDEXED_FRONTMATTER_FIELDS:
            if field in frontmatter_metadata:
                value = frontmatter_metadata[field]
                if isinstance(value, list):
                    metadata[field] = value
                else:
                    metadata[field] = str(value)

        related = frontmatter_metadata.get("related", [])
        if isinstance(related, str):
            related = [related]
        elif not isinstance(related, list):
            related = []
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

    def _extract_frontmatter(self, content: str):
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
            return metadata
        except yaml.YAMLError:
            return {}

    def _extract_text_content(self, content: str, root_node: Node):
        frontmatter_pattern = r"^---\s*\n.*?\n---\s*\n"
        text_without_frontmatter = re.sub(frontmatter_pattern, "", content, count=1, flags=re.DOTALL)

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
                            text = content_bytes[last_pos:child.start_byte].decode("utf8")
                            parts.append(text)
                        in_backticks = True
                        last_pos = child.end_byte
                    else:
                        in_backticks = False
                        last_pos = child.end_byte

            if not in_backticks and last_pos < node.end_byte:
                text = content_bytes[last_pos:node.end_byte].decode("utf8")
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

        text = self._get_text_excluding_code(root_node, content_bytes)
        matches = wikilink_pattern.findall(text)
        wikilinks.update(matches)

        return list(wikilinks)

    def extract_links_with_context(self, file_path: str) -> list[LinkWithContext]:
        if not os.path.exists(file_path):
            return []

        content = None
        for encoding in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
            try:
                with open(file_path, "r", encoding=encoding, errors="strict") as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content is None:
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
            links_with_context.append(LinkWithContext(target=target, header_context=header_context))

        return links_with_context

    def _extract_header_positions(self, root_node: Node, content_bytes: bytes) -> list[tuple[int, int, str]]:
        headers: list[tuple[int, int, str]] = []

        def visit(node: Node) -> None:
            if node.type in ("atx_heading", "setext_heading"):
                text = ""
                for child in node.children:
                    if child.type == "inline":
                        text = content_bytes[child.start_byte:child.end_byte].decode("utf8").strip()
                        break

                headers.append((node.start_byte, node.end_byte, text))

            for child in node.children:
                visit(child)

        visit(root_node)
        return sorted(headers, key=lambda x: x[0])

    def _find_header_context_at_position(self, headers: list[tuple[int, int, str]], position: int) -> str:
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

    def extract_code_blocks(self, file_path: str, doc_id: str) -> list[CodeBlock]:
        if not os.path.exists(file_path):
            return []

        content = None
        for encoding in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
            try:
                with open(file_path, "r", encoding=encoding, errors="strict") as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content is None:
            return []

        content_bytes = bytes(content, "utf8")
        tree = self.parser.parse(content_bytes)
        root_node = tree.root_node

        code_blocks: list[CodeBlock] = []
        block_index = 0

        self._collect_code_blocks(
            root_node, content_bytes, doc_id, code_blocks, block_index
        )

        return code_blocks

    def _collect_code_blocks(
        self,
        node: Node,
        content_bytes: bytes,
        doc_id: str,
        code_blocks: list[CodeBlock],
        block_index: int,
    ) -> int:
        if node.type == "fenced_code_block":
            language = ""
            code_content = ""

            for child in node.children:
                if child.type == "info_string":
                    language = content_bytes[child.start_byte:child.end_byte].decode("utf8").strip()
                elif child.type == "code_fence_content":
                    code_content = content_bytes[child.start_byte:child.end_byte].decode("utf8")

            if code_content.strip():
                block_id = f"{doc_id}_code_{block_index}"
                chunk_id = f"{doc_id}_chunk_0"

                code_blocks.append(CodeBlock(
                    id=block_id,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    content=code_content.strip(),
                    language=language,
                ))
                block_index += 1

        for child in node.children:
            block_index = self._collect_code_blocks(
                child, content_bytes, doc_id, code_blocks, block_index
            )

        return block_index
