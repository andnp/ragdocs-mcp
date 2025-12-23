import os
import re
from datetime import datetime
from pathlib import Path

import yaml
from tree_sitter import Language, Parser, Node
from tree_sitter_markdown import language

from src.models import Document
from src.parsers.base import DocumentParser


class MarkdownParser(DocumentParser):
    def __init__(self):
        self.parser = Parser(Language(language()))

    def parse(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

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
