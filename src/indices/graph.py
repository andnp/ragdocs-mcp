import json
import logging
import random
import sqlite3
from collections import Counter
from pathlib import Path
from threading import Lock
from typing import Any

from src.indices._corruption import is_corruption_error
from src.indices.protocol import SearchResult
from src.search.community import compute_community_boost, get_community_members
from src.storage.db import DatabaseManager

logger = logging.getLogger(__name__)


class GraphStore:
    def __init__(self, db_manager: DatabaseManager | None = None):
        if db_manager is None:
            import tempfile

            _tmp = tempfile.mkdtemp(prefix="graph_idx_")
            db_manager = DatabaseManager(Path(_tmp) / "index.db")
        self._db = db_manager
        self._lock = Lock()
        self._communities: dict[str, int] = {}
        self._last_community_node_count = 0

    def _conn(self) -> sqlite3.Connection:
        return self._db.get_connection()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, doc_id: str, metadata: dict) -> None:
        with self._lock:
            try:
                conn = self._conn()
                conn.execute(
                    "INSERT OR REPLACE INTO graph_nodes (node_id, metadata) VALUES (?, ?)",
                    (doc_id, json.dumps(metadata)),
                )
                conn.commit()
            except Exception as e:
                if is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return
                raise

    def has_node(self, doc_id: str) -> bool:
        with self._lock:
            try:
                conn = self._conn()
                row = conn.execute(
                    "SELECT 1 FROM graph_nodes WHERE node_id = ?", (doc_id,)
                ).fetchone()
                return row is not None
            except Exception as e:
                if is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return False
                raise

    def remove_node(self, doc_id: str) -> None:
        with self._lock:
            try:
                conn = self._conn()
                conn.execute("DELETE FROM graph_nodes WHERE node_id = ?", (doc_id,))
                conn.commit()
            except Exception as e:
                if is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return
                raise

    def remove_chunk(self, chunk_id: str) -> None:
        """Remove chunk node from graph.

        Thread-safe operation. Handles missing chunks gracefully.
        """
        with self._lock:
            try:
                conn = self._conn()
                row = conn.execute(
                    "SELECT 1 FROM graph_nodes WHERE node_id = ?", (chunk_id,)
                ).fetchone()
                if row:
                    conn.execute(
                        "DELETE FROM graph_nodes WHERE node_id = ?", (chunk_id,)
                    )
                    conn.commit()
                    logger.debug("Removed chunk %s from graph", chunk_id)
                else:
                    logger.debug(
                        "Chunk %s not in graph (already removed or never added)",
                        chunk_id,
                    )
            except Exception as e:
                if is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return
                raise

    def rename_node(self, old_id: str, new_id: str) -> bool:
        """Rename node preserving all edges and metadata (for file moves).

        Returns:
            True if rename successful, False if old node not found
        """
        with self._lock:
            try:
                conn = self._conn()
                row = conn.execute(
                    "SELECT metadata FROM graph_nodes WHERE node_id = ?",
                    (old_id,),
                ).fetchone()
                if not row:
                    logger.debug("Node %s not found in graph", old_id)
                    return False

                metadata = row[0]

                in_edges = conn.execute(
                    "SELECT source, edge_type, edge_context FROM graph_edges WHERE target = ?",
                    (old_id,),
                ).fetchall()
                out_edges = conn.execute(
                    "SELECT target, edge_type, edge_context FROM graph_edges WHERE source = ?",
                    (old_id,),
                ).fetchall()

                # Delete old node (CASCADE removes edges)
                conn.execute("DELETE FROM graph_nodes WHERE node_id = ?", (old_id,))

                # Insert new node
                conn.execute(
                    "INSERT OR REPLACE INTO graph_nodes (node_id, metadata) VALUES (?, ?)",
                    (new_id, metadata),
                )

                # Recreate edges
                for source, edge_type, edge_context in in_edges:
                    conn.execute(
                        "INSERT OR IGNORE INTO graph_nodes (node_id) VALUES (?)",
                        (source,),
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO graph_edges (source, target, edge_type, edge_context) VALUES (?, ?, ?, ?)",
                        (source, new_id, edge_type, edge_context),
                    )
                for target, edge_type, edge_context in out_edges:
                    conn.execute(
                        "INSERT OR IGNORE INTO graph_nodes (node_id) VALUES (?)",
                        (target,),
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO graph_edges (source, target, edge_type, edge_context) VALUES (?, ?, ?, ?)",
                        (new_id, target, edge_type, edge_context),
                    )
                conn.commit()
                logger.debug("Renamed node in graph: %s -> %s", old_id, new_id)
                return True
            except Exception as e:
                if is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return False
                raise

    def get_node_metadata(self, doc_id: str) -> dict:
        """Get metadata for a node.

        Returns:
            Metadata dict, or empty dict if node not found.
        """
        with self._lock:
            try:
                conn = self._conn()
                row = conn.execute(
                    "SELECT metadata FROM graph_nodes WHERE node_id = ?",
                    (doc_id,),
                ).fetchone()
                if row and row[0]:
                    return json.loads(row[0])
                return {}
            except Exception as e:
                if is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return {}
                raise

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        edge_context: str = "",
    ) -> None:
        with self._lock:
            try:
                conn = self._conn()
                conn.execute(
                    "INSERT OR IGNORE INTO graph_nodes (node_id) VALUES (?)",
                    (source,),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO graph_nodes (node_id) VALUES (?)",
                    (target,),
                )
                conn.execute(
                    """INSERT OR REPLACE INTO graph_edges
                       (source, target, edge_type, edge_context)
                       VALUES (?, ?, ?, ?)""",
                    (source, target, edge_type, edge_context),
                )
                conn.commit()
            except Exception as e:
                if is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return
                raise

    def get_edges_to(self, target: str) -> list[dict[str, str]]:
        with self._lock:
            try:
                conn = self._conn()
                rows = conn.execute(
                    "SELECT source, edge_type, edge_context FROM graph_edges WHERE target = ?",
                    (target,),
                ).fetchall()
                return [
                    {
                        "source": r[0],
                        "target": target,
                        "edge_type": r[1],
                        "edge_context": r[2],
                    }
                    for r in rows
                ]
            except Exception as e:
                if is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return []
                raise

    def get_edges_from(self, source: str) -> list[dict[str, str]]:
        """Get all edges originating from the given node."""
        with self._lock:
            try:
                conn = self._conn()
                rows = conn.execute(
                    "SELECT target, edge_type, edge_context FROM graph_edges WHERE source = ?",
                    (source,),
                ).fetchall()
                return [
                    {
                        "source": source,
                        "target": r[0],
                        "edge_type": r[1],
                        "edge_context": r[2],
                    }
                    for r in rows
                ]
            except Exception as e:
                if is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return []
                raise

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def get_neighbors(self, doc_id: str, depth: int = 1) -> list[str]:
        """Get neighbors up to specified depth via BFS."""
        with self._lock:
            try:
                conn = self._conn()
                row = conn.execute(
                    "SELECT 1 FROM graph_nodes WHERE node_id = ?", (doc_id,)
                ).fetchone()
                if not row:
                    return []

                visited: set[str] = {doc_id}
                current_level: set[str] = {doc_id}

                for _ in range(depth):
                    if not current_level:
                        break
                    placeholders = ",".join("?" * len(current_level))
                    params = list(current_level)

                    rows = conn.execute(
                        f"SELECT target FROM graph_edges WHERE source IN ({placeholders})",
                        params,
                    ).fetchall()
                    next_level = {r[0] for r in rows}

                    rows = conn.execute(
                        f"SELECT source FROM graph_edges WHERE target IN ({placeholders})",
                        params,
                    ).fetchall()
                    next_level.update(r[0] for r in rows)

                    next_level -= visited
                    visited.update(next_level)
                    current_level = next_level

                visited.discard(doc_id)
                return list(visited)
            except Exception as e:
                if is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return []
                raise

    def get_neighbors_batch(self, doc_ids: list[str], depth: int = 1) -> list[str]:
        """Get neighbors for multiple seed nodes in a single lock acquisition."""
        with self._lock:
            try:
                conn = self._conn()
                seed_ids: set[str] = set()
                current_level: set[str] = set()

                for doc_id in doc_ids:
                    row = conn.execute(
                        "SELECT 1 FROM graph_nodes WHERE node_id = ?", (doc_id,)
                    ).fetchone()
                    if row:
                        current_level.add(doc_id)
                        seed_ids.add(doc_id)

                if not current_level:
                    return []

                visited: set[str] = set(seed_ids)

                for _ in range(depth):
                    if not current_level:
                        break
                    placeholders = ",".join("?" * len(current_level))
                    params = list(current_level)

                    rows = conn.execute(
                        f"SELECT target FROM graph_edges WHERE source IN ({placeholders})",
                        params,
                    ).fetchall()
                    next_level = {r[0] for r in rows}

                    rows = conn.execute(
                        f"SELECT source FROM graph_edges WHERE target IN ({placeholders})",
                        params,
                    ).fetchall()
                    next_level.update(r[0] for r in rows)

                    next_level -= visited
                    visited.update(next_level)
                    current_level = next_level

                visited -= seed_ids
                return list(visited)
            except Exception as e:
                if is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return []
                raise

    # ------------------------------------------------------------------
    # Bulk access (used by ImplicitGraphBuilder and MemorySearchOrchestrator)
    # ------------------------------------------------------------------

    def get_all_nodes_with_metadata(self) -> list[tuple[str, dict]]:
        """Return all (node_id, metadata) pairs."""
        with self._lock:
            try:
                conn = self._conn()
                rows = conn.execute(
                    "SELECT node_id, metadata FROM graph_nodes"
                ).fetchall()
                result = []
                for r in rows:
                    metadata = json.loads(r[1]) if r[1] else {}
                    result.append((r[0], metadata))
                return result
            except Exception as e:
                if is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return []
                raise

    def get_all_node_ids(self) -> list[str]:
        """Return all node IDs."""
        with self._lock:
            try:
                conn = self._conn()
                rows = conn.execute("SELECT node_id FROM graph_nodes").fetchall()
                return [r[0] for r in rows]
            except Exception as e:
                if is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return []
                raise

    def get_outgoing_edges(self, source: str) -> list[tuple[str, str, dict[str, str]]]:
        """Get outgoing edges as (source, target, edge_data) tuples.

        Matches the NetworkX ``out_edges(data=True)`` interface for
        compatibility with callers like ``MemorySearchOrchestrator``.
        """
        with self._lock:
            try:
                conn = self._conn()
                rows = conn.execute(
                    "SELECT target, edge_type, edge_context FROM graph_edges WHERE source = ?",
                    (source,),
                ).fetchall()
                return [
                    (source, r[0], {"edge_type": r[1], "edge_context": r[2]})
                    for r in rows
                ]
            except Exception as e:
                if is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return []
                raise

    # ------------------------------------------------------------------
    # Community detection (label propagation, no NetworkX)
    # ------------------------------------------------------------------

    def detect_communities(self, algorithm: str = "louvain") -> dict[str, int]:
        """Detect communities using iterative label propagation.

        The ``algorithm`` parameter is accepted for API compatibility but
        always runs label propagation (NetworkX is no longer used).
        """
        with self._lock:
            try:
                conn = self._conn()
                node_count = conn.execute(
                    "SELECT COUNT(*) FROM graph_nodes"
                ).fetchone()[0]
                if node_count == 0:
                    self._communities = {}
                    return {}

                nodes = [
                    r[0]
                    for r in conn.execute("SELECT node_id FROM graph_nodes").fetchall()
                ]
                edges = conn.execute(
                    "SELECT source, target FROM graph_edges"
                ).fetchall()

                adjacency: dict[str, set[str]] = {n: set() for n in nodes}
                for source, target in edges:
                    if source in adjacency and target in adjacency:
                        adjacency[source].add(target)
                        adjacency[target].add(source)

                labels: dict[str, int] = {node: i for i, node in enumerate(nodes)}
                max_iterations = 100
                shuffled_nodes = list(nodes)

                for _ in range(max_iterations):
                    changed = False
                    random.shuffle(shuffled_nodes)
                    for node in shuffled_nodes:
                        neighbor_labels = [
                            labels[n] for n in adjacency[node] if n in labels
                        ]
                        if not neighbor_labels:
                            continue
                        label_counts = Counter(neighbor_labels)
                        most_common = label_counts.most_common(1)[0][0]
                        if labels[node] != most_common:
                            labels[node] = most_common
                            changed = True
                    if not changed:
                        break

                self._communities = labels
                self._last_community_node_count = node_count
                logger.info(
                    "Detected %d communities across %d nodes (label propagation)",
                    len(set(labels.values())),
                    len(labels),
                )
                return labels
            except Exception as e:
                if is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return {}
                raise

    def get_community(self, doc_id: str) -> int | None:
        with self._lock:
            return self._communities.get(doc_id)

    def get_community_members(self, community_id: int) -> list[str]:
        with self._lock:
            return get_community_members(self._communities, community_id)

    def boost_by_community(
        self,
        doc_ids: list[str],
        seed_doc_ids: set[str],
        boost_factor: float = 1.1,
    ) -> dict[str, float]:
        with self._lock:
            return compute_community_boost(
                doc_ids, self._communities, seed_doc_ids, boost_factor
            )

    def _should_recompute_communities(self) -> bool:
        try:
            conn = self._conn()
            current_count = conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[
                0
            ]
        except Exception:
            return False

        last_count = self._last_community_node_count
        if last_count == 0:
            return True

        change_ratio = abs(current_count - last_count) / max(last_count, 1)
        return change_ratio > 0.1

    # ------------------------------------------------------------------
    # Persistence (IndexProtocol compatibility)
    # ------------------------------------------------------------------

    def persist(self, path: Path) -> None:
        """No-op for data (lives in SQLite). Recomputes communities if needed."""
        if self._should_recompute_communities():
            self.detect_communities()

    def persist_to(self, snapshot_dir: Path) -> None:
        self.persist(snapshot_dir)

    def load_from(self, snapshot_dir: Path) -> bool:
        return True

    def load(self, path: Path) -> None:
        """No-op for data. Attempts one-time migration from legacy graph.json."""
        self._migrate_from_json(path)

    def save(self, path: Path) -> None:
        self.persist(path)

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def _migrate_from_json(self, index_path: Path) -> None:
        """One-time migration: load legacy graph.json into SQLite tables."""
        graph_file = index_path / "graph.json"
        if not graph_file.exists():
            return

        try:
            with open(graph_file) as f:
                graph_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read legacy graph.json for migration, skipping")
            return

        if not isinstance(graph_data, dict):
            logger.warning(
                "Legacy graph.json has unexpected type %s, skipping migration",
                type(graph_data).__name__,
            )
            return

        nodes = graph_data.get("nodes", [])
        links = graph_data.get("links", [])

        conn = self._conn()
        for node_entry in nodes:
            node_id = node_entry.get("id", "")
            if not node_id:
                continue
            metadata = {k: v for k, v in node_entry.items() if k != "id"}
            conn.execute(
                "INSERT OR IGNORE INTO graph_nodes (node_id, metadata) VALUES (?, ?)",
                (node_id, json.dumps(metadata)),
            )

        for link in links:
            source = link.get("source", "")
            target = link.get("target", "")
            if not source or not target:
                continue
            edge_type = link.get("edge_type", "related_to")
            edge_context = link.get("edge_context", "")
            conn.execute(
                "INSERT OR IGNORE INTO graph_nodes (node_id) VALUES (?)",
                (source,),
            )
            conn.execute(
                "INSERT OR IGNORE INTO graph_nodes (node_id) VALUES (?)",
                (target,),
            )
            conn.execute(
                "INSERT OR IGNORE INTO graph_edges (source, target, edge_type, edge_context) VALUES (?, ?, ?, ?)",
                (source, target, edge_type, edge_context),
            )

        conn.commit()

        try:
            graph_file.rename(graph_file.with_suffix(".json.bak"))
        except OSError:
            logger.warning("Could not rename graph.json to .bak")
        communities_file = index_path / "communities.json"
        if communities_file.exists():
            try:
                communities_file.rename(communities_file.with_suffix(".json.bak"))
            except OSError:
                logger.warning("Could not rename communities.json to .bak")

        logger.info(
            "Migrated graph from JSON to SQLite (%d nodes, %d edges)",
            len(nodes),
            len(links),
        )

    # ------------------------------------------------------------------
    # IndexProtocol methods
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        with self._lock:
            try:
                conn = self._conn()
                rows = conn.execute(
                    "SELECT node_id, metadata FROM graph_nodes"
                ).fetchall()
                results = []
                query_lower = query.lower()
                for r in rows:
                    node_id = r[0]
                    metadata = json.loads(r[1]) if r[1] else {}
                    content = metadata.get("content", "")
                    if (
                        query_lower in node_id.lower()
                        or query_lower in str(content).lower()
                    ):
                        results.append(
                            SearchResult(
                                doc_id=node_id,
                                score=1.0,
                                metadata=metadata,
                            )
                        )
                        if len(results) >= limit:
                            break
                return results
            except Exception as e:
                if is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return []
                raise

    def add_document(self, doc_id: str, content: str, metadata: dict[str, Any]) -> None:
        full_metadata = {"content": content, **metadata}
        self.add_node(doc_id, full_metadata)

    def remove_document(self, doc_id: str) -> None:
        self.remove_node(doc_id)

    def clear(self) -> None:
        with self._lock:
            try:
                conn = self._conn()
                conn.execute("DELETE FROM graph_edges")
                conn.execute("DELETE FROM graph_nodes")
                conn.commit()
                self._communities = {}
            except Exception as e:
                if is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return
                raise

    def _reinitialize_after_corruption(self) -> None:
        """Clear all graph data so reconciliation rebuilds from source."""
        try:
            conn = self._conn()
            conn.execute("DELETE FROM graph_edges")
            conn.execute("DELETE FROM graph_nodes")
            conn.commit()
        except Exception:
            logger.warning(
                "Could not clean graph tables during corruption recovery",
                exc_info=True,
            )
        self._communities = {}

    def __len__(self) -> int:
        with self._lock:
            try:
                conn = self._conn()
                row = conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()
                return row[0]
            except Exception as e:
                if is_corruption_error(e):
                    self._reinitialize_after_corruption()
                    return 0
                raise
