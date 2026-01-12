from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from src.memory.storage import (
    get_memory_file_path,
    get_trash_path,
    list_memory_files,
)

if TYPE_CHECKING:
    from src.context import ApplicationContext

logger = logging.getLogger(__name__)


def _format_frontmatter(
    memory_type: str,
    tags: list[str],
    status: str = "active",
) -> str:
    created_at = datetime.now(timezone.utc).isoformat()
    tags_str = ", ".join(f'"{t}"' for t in tags) if tags else ""

    return f"""---
type: "{memory_type}"
status: "{status}"
tags: [{tags_str}]
created_at: "{created_at}"
---

"""


async def create_memory(
    ctx: ApplicationContext,
    filename: str,
    content: str,
    tags: list[str],
    memory_type: str = "journal",
) -> dict[str, str]:
    if ctx.memory_manager is None:
        return {"error": "Memory system is not enabled"}

    memory_path = ctx.memory_manager.memory_path
    file_path = get_memory_file_path(memory_path, filename)

    if file_path.exists():
        return {"error": f"Memory file already exists: {filename}"}

    frontmatter = _format_frontmatter(memory_type, tags)
    full_content = frontmatter + content

    try:
        file_path.write_text(full_content, encoding="utf-8")
        ctx.memory_manager.index_memory(str(file_path))
        ctx.memory_manager.persist()

        return {
            "status": "created",
            "filename": filename,
            "path": str(file_path),
        }
    except Exception as e:
        logger.error(f"Failed to create memory: {e}", exc_info=True)
        return {"error": str(e)}


async def append_memory(
    ctx: ApplicationContext,
    filename: str,
    content: str,
) -> dict[str, str]:
    if ctx.memory_manager is None:
        return {"error": "Memory system is not enabled"}

    memory_path = ctx.memory_manager.memory_path
    file_path = get_memory_file_path(memory_path, filename)

    if not file_path.exists():
        return {"error": f"Memory file not found: {filename}"}

    try:
        existing_content = file_path.read_text(encoding="utf-8")
        new_content = existing_content.rstrip() + "\n\n" + content
        file_path.write_text(new_content, encoding="utf-8")

        memory_id = f"memory:{Path(filename).with_suffix('')}"
        ctx.memory_manager.remove_memory(memory_id)
        ctx.memory_manager.index_memory(str(file_path))
        ctx.memory_manager.persist()

        return {
            "status": "appended",
            "filename": filename,
            "path": str(file_path),
        }
    except Exception as e:
        logger.error(f"Failed to append to memory: {e}", exc_info=True)
        return {"error": str(e)}


async def read_memory(
    ctx: ApplicationContext,
    filename: str,
) -> dict[str, str]:
    if ctx.memory_manager is None:
        return {"error": "Memory system is not enabled"}

    memory_path = ctx.memory_manager.memory_path
    file_path = get_memory_file_path(memory_path, filename)

    if not file_path.exists():
        return {"error": f"Memory file not found: {filename}"}

    try:
        content = file_path.read_text(encoding="utf-8")
        return {
            "filename": filename,
            "content": content,
            "path": str(file_path),
        }
    except Exception as e:
        logger.error(f"Failed to read memory: {e}", exc_info=True)
        return {"error": str(e)}


async def update_memory(
    ctx: ApplicationContext,
    filename: str,
    content: str,
) -> dict[str, str]:
    if ctx.memory_manager is None:
        return {"error": "Memory system is not enabled"}

    memory_path = ctx.memory_manager.memory_path
    file_path = get_memory_file_path(memory_path, filename)

    if not file_path.exists():
        return {"error": f"Memory file not found: {filename}"}

    try:
        file_path.write_text(content, encoding="utf-8")

        memory_id = f"memory:{Path(filename).with_suffix('')}"
        ctx.memory_manager.remove_memory(memory_id)
        ctx.memory_manager.index_memory(str(file_path))
        ctx.memory_manager.persist()

        return {
            "status": "updated",
            "filename": filename,
            "path": str(file_path),
        }
    except Exception as e:
        logger.error(f"Failed to update memory: {e}", exc_info=True)
        return {"error": str(e)}


async def delete_memory(
    ctx: ApplicationContext,
    filename: str,
) -> dict[str, str]:
    if ctx.memory_manager is None:
        return {"error": "Memory system is not enabled"}

    memory_path = ctx.memory_manager.memory_path
    file_path = get_memory_file_path(memory_path, filename)

    if not file_path.exists():
        return {"error": f"Memory file not found: {filename}"}

    try:
        trash_path = get_trash_path(memory_path)
        trash_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        trash_file = trash_path / f"{file_path.stem}_{timestamp}{file_path.suffix}"
        shutil.move(str(file_path), str(trash_file))

        memory_id = f"memory:{Path(filename).with_suffix('')}"
        ctx.memory_manager.remove_memory(memory_id)
        ctx.memory_manager.persist()

        return {
            "status": "deleted",
            "filename": filename,
            "moved_to": str(trash_file),
        }
    except Exception as e:
        logger.error(f"Failed to delete memory: {e}", exc_info=True)
        return {"error": str(e)}


async def search_memories(
    ctx: ApplicationContext,
    query: str,
    limit: int = 5,
    filter_tags: list[str] | None = None,
    filter_type: str | None = None,
    load_full_memory: bool = False,
) -> list[dict]:
    if ctx.memory_manager is None or ctx.memory_search is None:
        return [{"error": "Memory system is not enabled"}]

    try:
        results = await ctx.memory_search.search_memories(
            query=query,
            limit=limit,
            filter_tags=filter_tags,
            filter_type=filter_type,
            load_full_memory=load_full_memory,
        )

        return [
            {
                "memory_id": r.memory_id,
                "score": r.score,
                "content": r.content,
                "type": r.frontmatter.type,
                "status": r.frontmatter.status,
                "tags": r.frontmatter.tags,
                "file_path": r.file_path,
                "header_path": r.header_path,
            }
            for r in results
        ]
    except Exception as e:
        logger.error(f"Failed to search memories: {e}", exc_info=True)
        return [{"error": str(e)}]


async def search_linked_memories(
    ctx: ApplicationContext,
    query: str,
    target_document: str,
    limit: int = 5,
) -> list[dict]:
    if ctx.memory_manager is None or ctx.memory_search is None:
        return [{"error": "Memory system is not enabled"}]

    try:
        results = await ctx.memory_search.search_linked_memories(
            query=query,
            target_document=target_document,
            limit=limit,
        )

        return [
            {
                "memory_id": r.memory_id,
                "score": r.score,
                "content": r.content,
                "anchor_context": r.anchor_context,
                "edge_type": r.edge_type,
                "file_path": r.file_path,
            }
            for r in results
        ]
    except Exception as e:
        logger.error(f"Failed to search linked memories: {e}", exc_info=True)
        return [{"error": str(e)}]


async def get_memory_stats(ctx: ApplicationContext) -> dict:
    if ctx.memory_manager is None:
        return {"error": "Memory system is not enabled"}

    try:
        memory_files = list_memory_files(ctx.memory_manager.memory_path)
        total_size = ctx.memory_manager.get_total_size_bytes()
        tags = ctx.memory_manager.get_all_tags()
        types = ctx.memory_manager.get_all_types()

        size_str = _format_size(total_size)

        return {
            "count": len(memory_files),
            "total_size": size_str,
            "tags": tags,
            "types": types,
            "memory_path": str(ctx.memory_manager.memory_path),
        }
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}", exc_info=True)
        return {"error": str(e)}


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"


async def search_by_tag_cluster(
    ctx: ApplicationContext,
    tag: str,
    depth: int = 2,
    limit: int = 10,
) -> list[dict]:
    if ctx.memory_manager is None or ctx.memory_search is None:
        return [{"error": "Memory system is not enabled"}]

    try:
        results = await ctx.memory_search.search_by_tag_cluster(
            tag=tag,
            depth=depth,
            limit=limit,
        )

        return [
            {
                "memory_id": r.memory_id,
                "score": r.score,
                "content": r.content,
                "type": r.frontmatter.type,
                "tags": r.frontmatter.tags,
                "file_path": r.file_path,
            }
            for r in results
        ]
    except Exception as e:
        logger.error(f"Failed to search by tag cluster: {e}", exc_info=True)
        return [{"error": str(e)}]


async def get_tag_graph(ctx: ApplicationContext) -> dict:
    if ctx.memory_manager is None or ctx.memory_search is None:
        return {"error": "Memory system is not enabled"}

    try:
        tag_frequencies = ctx.memory_search.get_tag_frequency_map()

        co_occurrences: list[dict] = []
        for tag in tag_frequencies:
            related = ctx.memory_search.get_related_tags(tag)
            for related_tag, count in related[:5]:
                co_occurrences.append({
                    "tag": tag,
                    "related_tag": related_tag,
                    "count": count,
                })

        return {
            "tag_frequencies": tag_frequencies,
            "co_occurrences": co_occurrences,
        }
    except Exception as e:
        logger.error(f"Failed to get tag graph: {e}", exc_info=True)
        return {"error": str(e)}


async def suggest_related_tags(
    ctx: ApplicationContext,
    tag: str,
) -> dict:
    if ctx.memory_manager is None or ctx.memory_search is None:
        return {"error": "Memory system is not enabled"}

    try:
        related_tags = ctx.memory_search.get_related_tags(tag)

        return {
            "tag": tag,
            "related_tags": [
                {"tag": t, "count": c}
                for t, c in related_tags
            ],
        }
    except Exception as e:
        logger.error(f"Failed to suggest related tags: {e}", exc_info=True)
        return {"error": str(e)}


async def get_memory_versions(
    ctx: ApplicationContext,
    filename: str,
) -> dict:
    if ctx.memory_manager is None or ctx.memory_search is None:
        return {"error": "Memory system is not enabled"}

    try:
        memory_id = f"memory:{Path(filename).stem}"
        return ctx.memory_search.get_memory_versions(memory_id)
    except Exception as e:
        logger.error(f"Failed to get memory versions: {e}", exc_info=True)
        return {"error": str(e)}


async def get_memory_dependencies(
    ctx: ApplicationContext,
    filename: str,
) -> list[dict]:
    if ctx.memory_manager is None or ctx.memory_search is None:
        return [{"error": "Memory system is not enabled"}]

    try:
        memory_id = f"memory:{Path(filename).stem}"
        return ctx.memory_search.get_memory_dependencies(memory_id)
    except Exception as e:
        logger.error(f"Failed to get memory dependencies: {e}", exc_info=True)
        return [{"error": str(e)}]


async def detect_contradictions(
    ctx: ApplicationContext,
    filename: str,
) -> list[dict]:
    if ctx.memory_manager is None or ctx.memory_search is None:
        return [{"error": "Memory system is not enabled"}]

    try:
        memory_id = f"memory:{Path(filename).stem}"
        return ctx.memory_search.detect_contradictions(memory_id)
    except Exception as e:
        logger.error(f"Failed to detect contradictions: {e}", exc_info=True)
        return [{"error": str(e)}]


async def merge_memories(
    ctx: ApplicationContext,
    source_files: list[str],
    target_file: str,
    summary_content: str,
) -> dict[str, str]:
    if ctx.memory_manager is None:
        return {"error": "Memory system is not enabled"}

    if not source_files:
        return {"error": "No source files provided"}

    memory_path = ctx.memory_manager.memory_path
    target_path = get_memory_file_path(memory_path, target_file)

    if target_path.exists():
        return {"error": f"Target file already exists: {target_file}"}

    source_paths: list[Path] = []
    for source_file in source_files:
        source_path = get_memory_file_path(memory_path, source_file)
        if not source_path.exists():
            return {"error": f"Source file not found: {source_file}"}
        source_paths.append(source_path)

    try:
        target_path.write_text(summary_content, encoding="utf-8")

        ctx.memory_manager.index_memory(str(target_path))

        trash_path = get_trash_path(memory_path)
        trash_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        for source_path in source_paths:
            memory_id = f"memory:{source_path.stem}"
            ctx.memory_manager.remove_memory(memory_id)

            trash_file = trash_path / f"{source_path.stem}_{timestamp}{source_path.suffix}"
            shutil.move(str(source_path), str(trash_file))

        ctx.memory_manager.persist()

        return {
            "status": "merged",
            "target_file": target_file,
            "sources_merged": str(len(source_files)),
            "target_path": str(target_path),
        }
    except Exception as e:
        logger.error(f"Failed to merge memories: {e}", exc_info=True)
        return {"error": str(e)}
