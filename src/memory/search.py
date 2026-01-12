import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import Config
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.memory.manager import MemoryIndexManager
from src.memory.models import (
    LinkedMemoryResult,
    MemoryFrontmatter,
    MemorySearchResult,
)
from src.search.fusion import fuse_results
from src.search.tag_expansion import expand_query_with_tags

logger = logging.getLogger(__name__)


def apply_memory_recency_boost(
    score: float,
    created_at: datetime | None,
    boost_days: int,
    boost_factor: float,
) -> float:
    if created_at is None:
        return score

    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    age_days = (now - created_at).days

    if age_days <= boost_days:
        return score * boost_factor

    return score


class MemorySearchOrchestrator:
    def __init__(
        self,
        vector: VectorIndex,
        keyword: KeywordIndex,
        graph: GraphStore,
        config: Config,
        manager: MemoryIndexManager,
    ):
        self._vector = vector
        self._keyword = keyword
        self._graph = graph
        self._config = config
        self._manager = manager

    async def search_memories(
        self,
        query: str,
        limit: int = 5,
        filter_tags: list[str] | None = None,
        filter_type: str | None = None,
        load_full_memory: bool = False,
    ) -> list[MemorySearchResult]:
        if not query or not query.strip():
            return []

        top_k = max(20, limit * 4)

        search_tasks = [
            self._search_vector(query, top_k),
            self._search_keyword(query, top_k),
        ]

        results = await asyncio.gather(*search_tasks)
        vector_results = results[0]
        keyword_results = results[1]

        # Tag-based query expansion for memories
        if self._config.search.tag_expansion_enabled:
            combined_initial_results = vector_results + keyword_results
            tag_expanded_results = expand_query_with_tags(
                initial_results=combined_initial_results,
                graph=self._graph,
                vector=self._vector,
                top_k=top_k,
                max_related_tags=self._config.search.tag_expansion_max_tags,
                max_depth=self._config.search.tag_expansion_depth,
            )

            # Merge tag-expanded results into vector results
            existing_chunk_ids = {r["chunk_id"] for r in vector_results}
            for result in tag_expanded_results:
                if result["chunk_id"] not in existing_chunk_ids:
                    vector_results.append(result)

        results_dict: dict[str, list[str]] = {
            "semantic": [r["chunk_id"] for r in vector_results],
            "keyword": [r["chunk_id"] for r in keyword_results],
        }

        weights = {
            "semantic": self._config.search.semantic_weight,
            "keyword": self._config.search.keyword_weight,
        }

        modified_times = self._collect_modified_times(
            {r["chunk_id"] for r in vector_results} |
            {r["chunk_id"] for r in keyword_results}
        )

        fused = fuse_results(
            results_dict,
            self._config.search.rrf_k_constant,
            weights,
            modified_times,
        )

        memory_results: list[MemorySearchResult] = []

        for chunk_id, score in fused:
            chunk_data = self._vector.get_chunk_by_id(chunk_id)
            if not chunk_data:
                continue

            metadata = chunk_data.get("metadata", {})

            # Type guard: ensure metadata is a dict
            if not isinstance(metadata, dict):
                continue

            if filter_type and metadata.get("memory_type") != filter_type:
                continue

            if filter_tags:
                chunk_tags = metadata.get("memory_tags", [])
                if not any(tag in chunk_tags for tag in filter_tags):
                    continue

            created_at_str = metadata.get("memory_created_at")
            created_at = None
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                except ValueError:
                    pass

            boosted_score = apply_memory_recency_boost(
                score,
                created_at,
                self._config.memory.recency_boost_days,
                self._config.memory.recency_boost_factor,
            )

            frontmatter = MemoryFrontmatter(
                type=metadata.get("memory_type", "journal"),
                status=metadata.get("memory_status", "active"),
                tags=metadata.get("memory_tags", []),
                created_at=created_at,
            )

            content = str(chunk_data.get("content", ""))
            if load_full_memory:
                file_path = chunk_data.get("file_path")
                if file_path and isinstance(file_path, str):
                    try:
                        full_content = Path(file_path).read_text(encoding="utf-8")
                        content = full_content
                    except Exception as e:
                        logger.warning(f"Failed to load full memory from {file_path}: {e}")

            memory_results.append(MemorySearchResult(
                memory_id=str(chunk_data.get("doc_id", chunk_id)),
                score=boosted_score,
                content=content,
                frontmatter=frontmatter,
                file_path=str(chunk_data.get("file_path", "")),
                header_path=str(chunk_data.get("header_path", "")),
            ))

            if len(memory_results) >= limit:
                break

        memory_results.sort(key=lambda r: r.score, reverse=True)
        return memory_results[:limit]

    async def search_linked_memories(
        self,
        query: str,
        target_document: str,
        limit: int = 5,
    ) -> list[LinkedMemoryResult]:
        ghost_id = f"ghost:{target_document}"

        edges = self._graph.get_edges_to(ghost_id)
        if not edges:
            return []

        memory_ids = [edge["source"] for edge in edges]
        edge_map = {edge["source"]: edge for edge in edges}

        memory_chunks: list[tuple[str, str, float]] = []

        for memory_id in memory_ids:
            chunk_ids = self._vector.get_chunk_ids_for_document(memory_id)

            for chunk_id in chunk_ids:
                chunk_data = self._vector.get_chunk_by_id(chunk_id)
                if chunk_data:
                    content = str(chunk_data.get("content", ""))
                    if query.lower() in content.lower():
                        memory_chunks.append((memory_id, chunk_id, 1.0))
                    else:
                        memory_chunks.append((memory_id, chunk_id, 0.5))

        if query.strip():
            query_embedding = self._vector.get_text_embedding(query)
            if query_embedding:
                scored_chunks: list[tuple[str, str, float]] = []
                for memory_id, chunk_id, base_score in memory_chunks:
                    chunk_embedding = self._vector.get_embedding_for_chunk(chunk_id)
                    if chunk_embedding:
                        similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                        scored_chunks.append((memory_id, chunk_id, similarity))
                    else:
                        scored_chunks.append((memory_id, chunk_id, base_score))

                memory_chunks = scored_chunks

        memory_chunks.sort(key=lambda x: x[2], reverse=True)

        seen_memories: set[str] = set()
        results: list[LinkedMemoryResult] = []

        for memory_id, chunk_id, score in memory_chunks:
            if memory_id in seen_memories:
                continue

            seen_memories.add(memory_id)

            chunk_data = self._vector.get_chunk_by_id(chunk_id)
            if not chunk_data:
                continue

            edge = edge_map.get(memory_id, {})

            results.append(LinkedMemoryResult(
                memory_id=memory_id,
                score=score,
                content=str(chunk_data.get("content", "")),
                anchor_context=edge.get("edge_context", ""),
                edge_type=edge.get("edge_type", "related_to"),
                file_path=str(chunk_data.get("file_path", "")),
            ))

            if len(results) >= limit:
                break

        return results

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        if len(vec_a) != len(vec_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def _search_vector(self, query: str, top_k: int) -> list[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._vector.search, query, top_k, None, None
        )

    async def _search_keyword(self, query: str, top_k: int) -> list[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._keyword.search, query, top_k, None, None
        )

    def _collect_modified_times(self, chunk_ids: set[str]) -> dict[str, float]:
        modified_times: dict[str, float] = {}

        for chunk_id in chunk_ids:
            chunk_data = self._vector.get_chunk_by_id(chunk_id)
            if chunk_data:
                file_path = chunk_data.get("file_path")
                # Type guard: ensure file_path is a string
                if file_path and isinstance(file_path, str):
                    path = Path(file_path)
                    if path.exists():
                        modified_times[chunk_id] = path.stat().st_mtime

        return modified_times

    async def search_by_tag_cluster(
        self, tag: str, depth: int = 2, limit: int = 10
    ) -> list[MemorySearchResult]:
        from src.memory.link_parser import normalize_tag

        normalized_tag = normalize_tag(tag)
        tag_id = f"tag:{normalized_tag}"

        if not self._graph.has_node(tag_id):
            return []

        depth = min(depth, 3)

        memory_ids = set()
        current_level = {tag_id}

        for _ in range(depth):
            next_level = set()
            for node_id in current_level:
                if node_id.startswith("tag:"):
                    predecessors = [p for p in self._graph.get_neighbors(node_id, 1) if p.startswith("memory:")]
                    memory_ids.update(predecessors)

                    tag_neighbors = [p for p in self._graph.get_neighbors(node_id, 1) if p.startswith("tag:")]
                    next_level.update(tag_neighbors)

            current_level = next_level

        if not memory_ids:
            return []

        results: list[MemorySearchResult] = []

        for memory_id in list(memory_ids)[:limit]:
            chunk_ids = self._vector.get_chunk_ids_for_document(memory_id)
            if not chunk_ids:
                continue

            chunk_id = chunk_ids[0]
            chunk_data = self._vector.get_chunk_by_id(chunk_id)
            if not chunk_data:
                continue

            metadata = chunk_data.get("metadata", {})
            # Type guard: ensure metadata is a dict
            if not isinstance(metadata, dict):
                continue

            created_at_str = metadata.get("memory_created_at")
            created_at = None
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                except ValueError:
                    pass

            frontmatter = MemoryFrontmatter(
                type=metadata.get("memory_type", "journal"),
                status=metadata.get("memory_status", "active"),
                tags=metadata.get("memory_tags", []),
                created_at=created_at,
            )

            results.append(MemorySearchResult(
                memory_id=memory_id,
                score=1.0,
                content=str(chunk_data.get("content", "")),
                frontmatter=frontmatter,
                file_path=str(chunk_data.get("file_path", "")),
                header_path=str(chunk_data.get("header_path", "")),
            ))

            if len(results) >= limit:
                break

        return results

    def get_related_tags(self, tag: str) -> list[tuple[str, int]]:
        from src.memory.link_parser import normalize_tag

        normalized_tag = normalize_tag(tag)
        tag_id = f"tag:{normalized_tag}"

        if not self._graph.has_node(tag_id):
            return []

        memory_ids = [p for p in self._graph.get_neighbors(tag_id, 1) if p.startswith("memory:")]

        tag_counts: dict[str, int] = {}

        for memory_id in memory_ids:
            tag_neighbors = [p for p in self._graph.get_neighbors(memory_id, 1) if p.startswith("tag:")]
            for tag_node_id in tag_neighbors:
                if tag_node_id == tag_id:
                    continue

                tag_name = tag_node_id[4:]
                tag_counts[tag_name] = tag_counts.get(tag_name, 0) + 1

        return sorted(tag_counts.items(), key=lambda x: -x[1])

    def get_tag_frequency_map(self) -> dict[str, int]:
        tag_counts: dict[str, int] = {}

        for node_id in self._graph._graph.nodes():
            if node_id.startswith("tag:"):
                tag_name = node_id[4:]
                memory_ids = [p for p in self._graph.get_neighbors(node_id, 1) if p.startswith("memory:")]
                tag_counts[tag_name] = len(memory_ids)

        return tag_counts

    def get_memory_versions(self, memory_id: str) -> dict:
        if not self._graph.has_node(memory_id):
            return {"error": f"Memory not found: {memory_id}"}

        chain: list[dict] = []
        current_id = memory_id
        visited = set()

        while current_id and current_id not in visited:
            visited.add(current_id)

            chunk_ids = self._vector.get_chunk_ids_for_document(current_id)
            if chunk_ids:
                chunk_data = self._vector.get_chunk_by_id(chunk_ids[0])
                if chunk_data:
                    chain.append({
                        "memory_id": current_id,
                        "file_path": str(chunk_data.get("file_path", "")),
                    })

            edges = self._graph._graph.out_edges(current_id, data=True)
            supersedes_edges = [e for e in edges if e[2].get("edge_type") == "SUPERSEDES"]

            if supersedes_edges:
                current_id = supersedes_edges[0][1]
            else:
                break

        return {"version_chain": chain, "count": len(chain)}

    def get_memory_dependencies(self, memory_id: str) -> list[dict]:
        if not self._graph.has_node(memory_id):
            return []

        dependencies: list[dict] = []
        edges = self._graph._graph.out_edges(memory_id, data=True)
        depends_edges = [e for e in edges if e[2].get("edge_type") == "DEPENDS_ON"]

        for _, target_id, edge_data in depends_edges:
            chunk_ids = self._vector.get_chunk_ids_for_document(target_id)
            if chunk_ids:
                chunk_data = self._vector.get_chunk_by_id(chunk_ids[0])
                if chunk_data:
                    dependencies.append({
                        "memory_id": target_id,
                        "file_path": str(chunk_data.get("file_path", "")),
                        "context": edge_data.get("edge_context", ""),
                    })

        return dependencies

    def detect_contradictions(self, memory_id: str) -> list[dict]:
        if not self._graph.has_node(memory_id):
            return []

        contradictions: list[dict] = []
        edges = self._graph._graph.out_edges(memory_id, data=True)
        contradict_edges = [e for e in edges if e[2].get("edge_type") == "CONTRADICTS"]

        for _, target_id, edge_data in contradict_edges:
            chunk_ids = self._vector.get_chunk_ids_for_document(target_id)
            if chunk_ids:
                chunk_data = self._vector.get_chunk_by_id(chunk_ids[0])
                if chunk_data:
                    contradictions.append({
                        "memory_id": target_id,
                        "file_path": str(chunk_data.get("file_path", "")),
                        "context": edge_data.get("edge_context", ""),
                    })

        return contradictions
