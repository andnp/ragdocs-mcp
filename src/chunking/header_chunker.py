import re
from dataclasses import dataclass

from tree_sitter import Language, Parser
from tree_sitter_markdown import language

from src.chunking.base import ChunkingStrategy
from src.config import ChunkingConfig
from src.models import Chunk, Document


@dataclass
class HeaderNode:
    level: int
    text: str
    start_pos: int
    end_pos: int


class HeaderBasedChunker(ChunkingStrategy):
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.parser = Parser(Language(language()))

    def chunk_document(self, document: Document) -> list[Chunk]:
        content_bytes = bytes(document.content, "utf8")
        tree = self.parser.parse(content_bytes)
        root_node = tree.root_node

        headers = self._extract_headers(root_node, content_bytes)

        if not headers:
            return self._chunk_plain_text(document)

        initial_chunks = self._create_initial_chunks(document, headers)
        merged_chunks = self._merge_small_chunks(initial_chunks)
        split_chunks = self._split_large_chunks(merged_chunks)
        final_chunks = self._apply_overlap(split_chunks)

        final_chunks = self._create_parent_child_chunks(document, final_chunks)

        return final_chunks

    def _extract_headers(self, root_node, content_bytes: bytes) -> list[HeaderNode]:
        headers = []

        def byte_to_char_pos(byte_pos: int) -> int:
            return len(content_bytes[:byte_pos].decode("utf8"))

        def visit(node):
            if node.type in ("atx_heading", "setext_heading"):
                level = 1
                text = ""
                marker_start = node.start_byte

                for child in node.children:
                    if child.type in ("atx_h1_marker", "setext_h1_underline"):
                        level = 1
                        marker_start = child.start_byte
                    elif child.type in ("atx_h2_marker", "setext_h2_underline"):
                        level = 2
                        marker_start = child.start_byte
                    elif child.type == "atx_h3_marker":
                        level = 3
                        marker_start = child.start_byte
                    elif child.type == "atx_h4_marker":
                        level = 4
                        marker_start = child.start_byte
                    elif child.type == "atx_h5_marker":
                        level = 5
                        marker_start = child.start_byte
                    elif child.type == "atx_h6_marker":
                        level = 6
                        marker_start = child.start_byte
                    elif child.type == "inline":
                        text = (
                            content_bytes[child.start_byte : child.end_byte]
                            .decode("utf8")
                            .strip()
                        )

                headers.append(
                    HeaderNode(
                        level=level,
                        text=text,
                        start_pos=byte_to_char_pos(marker_start),
                        end_pos=byte_to_char_pos(node.end_byte),
                    )
                )

            for child in node.children:
                visit(child)

        visit(root_node)
        return headers

    def _create_initial_chunks(
        self, document: Document, headers: list[HeaderNode]
    ) -> list[Chunk]:
        chunks = []
        content = document.content

        for i, header in enumerate(headers):
            start_pos = header.start_pos
            end_pos = headers[i + 1].start_pos if i + 1 < len(headers) else len(content)

            header_path = self._build_header_path(headers, i)
            section_body = content[header.end_pos:end_pos].strip()
            chunk_content = self._compose_chunk_content(header_path, section_body)

            chunk_id = f"{document.id}_chunk_{i}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.id,
                    content=chunk_content,
                    metadata={
                        **document.metadata,
                        "tags": document.tags,
                        "links": document.links,
                    },
                    chunk_index=i,
                    header_path=header_path,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    file_path=document.file_path,
                    modified_time=document.modified_time,
                )
            )

        return chunks

    def _build_header_path(self, headers: list[HeaderNode], current_index: int) -> str:
        current_header = headers[current_index]
        current_level = current_header.level

        path_parts = [current_header.text]

        for i in range(current_index - 1, -1, -1):
            if headers[i].level < current_level:
                path_parts.insert(0, headers[i].text)
                current_level = headers[i].level

        return " > ".join(path_parts)

    def _compose_chunk_content(self, header_path: str, section_body: str) -> str:
        cleaned_body = section_body.strip()
        header_parts = self._split_header_path(header_path)
        if not header_parts:
            return cleaned_body

        display_lines = [header_parts[-1]]
        parent_context = " > ".join(header_parts[:-1])
        if parent_context:
            display_lines.append(f"Context: {parent_context}")

        display_prefix = "\n".join(display_lines)
        if cleaned_body:
            return f"{display_prefix}\n\n{cleaned_body}"
        return display_prefix

    def _split_header_path(self, header_path: str) -> list[str]:
        return [part.strip() for part in header_path.split(">") if part.strip()]

    def _shared_header_prefix(self, header_paths: list[str]) -> list[str]:
        split_paths = [self._split_header_path(path) for path in header_paths if path]
        if not split_paths:
            return []

        shared = split_paths[0]
        for parts in split_paths[1:]:
            prefix_length = 0
            for left, right in zip(shared, parts):
                if left != right:
                    break
                prefix_length += 1
            shared = shared[:prefix_length]
            if not shared:
                break

        return shared

    def _combine_header_paths(self, *header_paths: str) -> str:
        non_empty_paths = [path for path in header_paths if path]
        if not non_empty_paths:
            return ""

        shared_prefix = self._shared_header_prefix(non_empty_paths)
        shared_prefix_text = " > ".join(shared_prefix)

        suffixes: list[str] = []
        for header_path in non_empty_paths:
            parts = self._split_header_path(header_path)
            suffix = " > ".join(parts[len(shared_prefix) :])
            if suffix and suffix not in suffixes:
                suffixes.append(suffix)

        if shared_prefix_text and not suffixes:
            return shared_prefix_text
        if shared_prefix_text and len(suffixes) == 1:
            return f"{shared_prefix_text} > {suffixes[0]}"
        if shared_prefix_text and suffixes:
            return f"{shared_prefix_text} > {' / '.join(suffixes)}"

        return " / ".join(non_empty_paths)

    def _header_path_extends(self, base_path: str, candidate_path: str) -> bool:
        base_parts = self._split_header_path(base_path)
        candidate_parts = self._split_header_path(candidate_path)
        return bool(base_parts) and candidate_parts[: len(base_parts)] == base_parts

    def _merge_chunk_pair(self, left: Chunk, right: Chunk) -> Chunk:
        left_body = self._strip_structured_chunk_prefix(left.content, left.header_path)
        right_body = self._strip_structured_chunk_prefix(right.content, right.header_path)

        left_is_context_only = left.content.strip() != "" and not left_body.strip()
        right_is_context_only = right.content.strip() != "" and not right_body.strip()

        if left_is_context_only and self._header_path_extends(
            left.header_path,
            right.header_path,
        ):
            combined_content = right.content
            merged_header_path = right.header_path
        elif right_is_context_only and self._header_path_extends(
            right.header_path,
            left.header_path,
        ):
            combined_content = left.content
            merged_header_path = left.header_path
        else:
            combined_content = f"{left.content}\n\n{right.content}"
            merged_header_path = self._combine_header_paths(
                left.header_path,
                right.header_path,
            )

        return Chunk(
            chunk_id=left.chunk_id,
            doc_id=left.doc_id,
            content=combined_content,
            metadata=left.metadata,
            chunk_index=left.chunk_index,
            header_path=merged_header_path,
            start_pos=left.start_pos,
            end_pos=right.end_pos,
            file_path=left.file_path,
            modified_time=left.modified_time,
        )

    def _strip_structured_chunk_prefix(self, content: str, header_path: str) -> str:
        header_parts = self._split_header_path(header_path)
        if not header_parts:
            return content

        lines = content.splitlines()
        if not lines or lines[0].strip() != header_parts[-1]:
            return content

        index = 1
        expected_context = " > ".join(header_parts[:-1])
        if expected_context and index < len(lines):
            context_prefix = "Context: "
            if lines[index].startswith(context_prefix):
                observed_context = lines[index][len(context_prefix) :].strip()
                if observed_context == expected_context:
                    index += 1

        while index < len(lines) and not lines[index].strip():
            index += 1

        return "\n".join(lines[index:]).lstrip()

    def _select_parent_header_path(self, chunks: list[Chunk]) -> str:
        header_paths = [chunk.header_path for chunk in chunks if chunk.header_path]
        if not header_paths:
            return ""

        shared_prefix = self._shared_header_prefix(header_paths)
        if shared_prefix:
            return " > ".join(shared_prefix)

        return header_paths[0]

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return chunks

        merged = []
        i = 0

        while i < len(chunks):
            current = chunks[i]

            if len(current.content) >= self.config.min_chunk_chars:
                merged.append(current)
                i += 1
                continue

            if i + 1 < len(chunks):
                merged.append(self._merge_chunk_pair(current, chunks[i + 1]))
                i += 2
                continue

            if merged:
                merged[-1] = self._merge_chunk_pair(merged[-1], current)
            else:
                merged.append(current)
            i += 1

        return merged

    def _split_large_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        result = []

        for chunk in chunks:
            if len(chunk.content) <= self.config.max_chunk_chars:
                result.append(chunk)
                continue

            section_body = self._strip_structured_chunk_prefix(
                chunk.content,
                chunk.header_path,
            )
            paragraphs = re.split(r"\n\n+", section_body if section_body else chunk.content)
            current_content = ""
            sub_index = 0

            for para in paragraphs:
                if not current_content:
                    current_content = para
                elif (
                    len(current_content) + len(para) + 2 <= self.config.max_chunk_chars
                ):
                    current_content += "\n\n" + para
                else:
                    sub_chunk_content = self._compose_chunk_content(
                        chunk.header_path,
                        current_content,
                    )
                    sub_chunk = Chunk(
                        chunk_id=f"{chunk.chunk_id}_sub_{sub_index}",
                        doc_id=chunk.doc_id,
                        content=sub_chunk_content,
                        metadata=chunk.metadata,
                        chunk_index=chunk.chunk_index,
                        header_path=chunk.header_path,
                        start_pos=chunk.start_pos,
                        end_pos=chunk.end_pos,
                        file_path=chunk.file_path,
                        modified_time=chunk.modified_time,
                    )
                    result.append(sub_chunk)
                    current_content = para
                    sub_index += 1

            if current_content:
                sub_chunk_content = self._compose_chunk_content(
                    chunk.header_path,
                    current_content,
                )
                sub_chunk = Chunk(
                    chunk_id=f"{chunk.chunk_id}_sub_{sub_index}"
                    if sub_index > 0
                    else chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=sub_chunk_content,
                    metadata=chunk.metadata,
                    chunk_index=chunk.chunk_index,
                    header_path=chunk.header_path,
                    start_pos=chunk.start_pos,
                    end_pos=chunk.end_pos,
                    file_path=chunk.file_path,
                    modified_time=chunk.modified_time,
                )
                result.append(sub_chunk)

        return result

    def _apply_overlap(self, chunks: list[Chunk]) -> list[Chunk]:
        if len(chunks) <= 1 or self.config.overlap_chars <= 0:
            return chunks

        result = []
        for i, chunk in enumerate(chunks):
            content = chunk.content
            should_overlap = False

            if i > 0:
                prev_chunk = chunks[i - 1]

                # Only apply overlap between sub-chunks from the same parent
                # (i.e., chunks that were force-split due to max_chunk_chars)
                current_is_subchunk = "_sub_" in chunk.chunk_id
                prev_is_subchunk = "_sub_" in prev_chunk.chunk_id

                if current_is_subchunk and prev_is_subchunk:
                    # Check if they're siblings (same parent chunk)
                    current_base = chunk.chunk_id.rsplit("_sub_", 1)[0]
                    prev_base = prev_chunk.chunk_id.rsplit("_sub_", 1)[0]
                    should_overlap = current_base == prev_base

            if should_overlap:
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk.content[-self.config.overlap_chars :]
                if overlap_text:
                    content = f"[...{overlap_text}]\n\n{content}"

            overlapped_chunk = Chunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                content=content,
                metadata=chunk.metadata,
                chunk_index=chunk.chunk_index,
                header_path=chunk.header_path,
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos,
                file_path=chunk.file_path,
                modified_time=chunk.modified_time,
            )
            result.append(overlapped_chunk)

        return result

    def _chunk_plain_text(self, document: Document) -> list[Chunk]:
        content = document.content
        if len(content) <= self.config.max_chunk_chars:
            return [
                Chunk(
                    chunk_id=f"{document.id}_chunk_0",
                    doc_id=document.id,
                    content=content,
                    metadata={
                        **document.metadata,
                        "tags": document.tags,
                        "links": document.links,
                    },
                    chunk_index=0,
                    header_path="",
                    start_pos=0,
                    end_pos=len(content),
                    file_path=document.file_path,
                    modified_time=document.modified_time,
                )
            ]

        chunks = []
        paragraphs = re.split(r"\n\n+", content)
        current_content = ""
        chunk_index = 0
        start_pos = 0

        for para in paragraphs:
            if not current_content:
                current_content = para
            elif len(current_content) + len(para) + 2 <= self.config.max_chunk_chars:
                current_content += "\n\n" + para
            else:
                end_pos = start_pos + len(current_content)
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.id}_chunk_{chunk_index}",
                        doc_id=document.id,
                        content=current_content,
                        metadata={
                            **document.metadata,
                            "tags": document.tags,
                            "links": document.links,
                        },
                        chunk_index=chunk_index,
                        header_path="",
                        start_pos=start_pos,
                        end_pos=end_pos,
                        file_path=document.file_path,
                        modified_time=document.modified_time,
                    )
                )
                start_pos = end_pos
                current_content = para
                chunk_index += 1

        if current_content:
            end_pos = start_pos + len(current_content)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.id}_chunk_{chunk_index}",
                    doc_id=document.id,
                    content=current_content,
                    metadata={
                        **document.metadata,
                        "tags": document.tags,
                        "links": document.links,
                    },
                    chunk_index=chunk_index,
                    header_path="",
                    start_pos=start_pos,
                    end_pos=end_pos,
                    file_path=document.file_path,
                    modified_time=document.modified_time,
                )
            )

        return chunks

    def _create_parent_child_chunks(
        self, document: Document, chunks: list[Chunk]
    ) -> list[Chunk]:
        if not chunks:
            return chunks

        parent_min = self.config.parent_chunk_min_chars
        parent_max = self.config.parent_chunk_max_chars

        parents: list[Chunk] = []
        children: list[Chunk] = []

        current_parent_chunks: list[Chunk] = []
        current_parent_content = ""
        parent_index = 0

        for chunk in chunks:
            if not current_parent_content:
                current_parent_chunks = [chunk]
                current_parent_content = chunk.content
            elif len(current_parent_content) + len(chunk.content) + 2 <= parent_max:
                current_parent_chunks.append(chunk)
                current_parent_content += "\n\n" + chunk.content
            else:
                if len(current_parent_content) >= parent_min:
                    parent_chunk_id = f"{document.id}_parent_{parent_index}"
                    parent = Chunk(
                        chunk_id=parent_chunk_id,
                        doc_id=document.id,
                        content=current_parent_content,
                        metadata=current_parent_chunks[0].metadata,
                        chunk_index=parent_index,
                        header_path=self._select_parent_header_path(
                            current_parent_chunks
                        ),
                        start_pos=current_parent_chunks[0].start_pos,
                        end_pos=current_parent_chunks[-1].end_pos,
                        file_path=document.file_path,
                        modified_time=document.modified_time,
                    )
                    parents.append(parent)

                    for child_chunk in current_parent_chunks:
                        child = Chunk(
                            chunk_id=child_chunk.chunk_id,
                            doc_id=child_chunk.doc_id,
                            content=child_chunk.content,
                            metadata=child_chunk.metadata,
                            chunk_index=child_chunk.chunk_index,
                            header_path=child_chunk.header_path,
                            start_pos=child_chunk.start_pos,
                            end_pos=child_chunk.end_pos,
                            file_path=child_chunk.file_path,
                            modified_time=child_chunk.modified_time,
                            parent_chunk_id=parent_chunk_id,
                        )
                        children.append(child)

                    parent_index += 1
                else:
                    for child_chunk in current_parent_chunks:
                        children.append(child_chunk)

                current_parent_chunks = [chunk]
                current_parent_content = chunk.content

        if current_parent_chunks:
            if len(current_parent_content) >= parent_min:
                parent_chunk_id = f"{document.id}_parent_{parent_index}"
                parent = Chunk(
                    chunk_id=parent_chunk_id,
                    doc_id=document.id,
                    content=current_parent_content,
                    metadata=current_parent_chunks[0].metadata,
                    chunk_index=parent_index,
                    header_path=self._select_parent_header_path(current_parent_chunks),
                    start_pos=current_parent_chunks[0].start_pos,
                    end_pos=current_parent_chunks[-1].end_pos,
                    file_path=document.file_path,
                    modified_time=document.modified_time,
                )
                parents.append(parent)

                for child_chunk in current_parent_chunks:
                    child = Chunk(
                        chunk_id=child_chunk.chunk_id,
                        doc_id=child_chunk.doc_id,
                        content=child_chunk.content,
                        metadata=child_chunk.metadata,
                        chunk_index=child_chunk.chunk_index,
                        header_path=child_chunk.header_path,
                        start_pos=child_chunk.start_pos,
                        end_pos=child_chunk.end_pos,
                        file_path=child_chunk.file_path,
                        modified_time=child_chunk.modified_time,
                        parent_chunk_id=parent_chunk_id,
                    )
                    children.append(child)
            else:
                for child_chunk in current_parent_chunks:
                    children.append(child_chunk)

        return parents + children
