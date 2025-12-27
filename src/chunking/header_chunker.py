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
                        text = content_bytes[child.start_byte:child.end_byte].decode("utf8").strip()

                headers.append(HeaderNode(
                    level=level,
                    text=text,
                    start_pos=byte_to_char_pos(marker_start),
                    end_pos=byte_to_char_pos(node.end_byte)
                ))

            for child in node.children:
                visit(child)

        visit(root_node)
        return headers

    def _create_initial_chunks(self, document: Document, headers: list[HeaderNode]) -> list[Chunk]:
        chunks = []
        content = document.content

        for i, header in enumerate(headers):
            start_pos = header.start_pos
            end_pos = headers[i + 1].start_pos if i + 1 < len(headers) else len(content)

            chunk_content = content[start_pos:end_pos].strip()

            header_path = self._build_header_path(headers, i)

            chunk_id = f"{document.id}_chunk_{i}"

            chunks.append(Chunk(
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
            ))

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
                next_chunk = chunks[i + 1]
                combined_content = current.content + "\n\n" + next_chunk.content

                merged_chunk = Chunk(
                    chunk_id=current.chunk_id,
                    doc_id=current.doc_id,
                    content=combined_content,
                    metadata=current.metadata,
                    chunk_index=current.chunk_index,
                    header_path=f"{current.header_path} + {next_chunk.header_path}",
                    start_pos=current.start_pos,
                    end_pos=next_chunk.end_pos,
                    file_path=current.file_path,
                    modified_time=current.modified_time,
                )
                merged.append(merged_chunk)
                i += 2
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

            paragraphs = re.split(r'\n\n+', chunk.content)
            current_content = ""
            sub_index = 0

            for para in paragraphs:
                if not current_content:
                    current_content = para
                elif len(current_content) + len(para) + 2 <= self.config.max_chunk_chars:
                    current_content += "\n\n" + para
                else:
                    sub_chunk = Chunk(
                        chunk_id=f"{chunk.chunk_id}_sub_{sub_index}",
                        doc_id=chunk.doc_id,
                        content=current_content,
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
                sub_chunk = Chunk(
                    chunk_id=f"{chunk.chunk_id}_sub_{sub_index}" if sub_index > 0 else chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=current_content,
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
                    should_overlap = (current_base == prev_base)

            if should_overlap:
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk.content[-self.config.overlap_chars:]
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
            return [Chunk(
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
            )]

        chunks = []
        paragraphs = re.split(r'\n\n+', content)
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
                chunks.append(Chunk(
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
                ))
                start_pos = end_pos
                current_content = para
                chunk_index += 1

        if current_content:
            end_pos = start_pos + len(current_content)
            chunks.append(Chunk(
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
            ))

        return chunks
