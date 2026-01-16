import logging
from pathlib import Path

from src.indices.graph import GraphStore

logger = logging.getLogger(__name__)


class ImplicitGraphBuilder:
    def __init__(self, graph: GraphStore):
        self.graph = graph

    def build_implicit_edges(self):
        """Builds all types of implicit edges."""
        logger.info("Building implicit graph edges...")
        self._build_directory_edges()
        self._build_tag_edges()

    def _build_directory_edges(self):
        """
        Connects documents that reside in the same directory.
        Edge type: 'directory_sibling'
        """
        # Group docs by directory
        dir_groups: dict[str, list[str]] = {}
        
        # We need to access the graph's nodes directly to get paths
        # This assumes the graph nodes are doc_ids and might have metadata
        with self.graph._lock:
            nodes = list(self.graph._graph.nodes(data=True))

        for doc_id, metadata in nodes:
            # Skip if it's a memory or tag node
            if doc_id.startswith("memory:") or doc_id.startswith("tag:"):
                continue

            # Try to get path from metadata, fallback to doc_id if it looks like a path
            file_path = metadata.get("file_path")
            if not file_path:
                # If doc_id is a relative path (standard in this app), use parent dir
                try:
                    parent_dir = str(Path(doc_id).parent)
                    if parent_dir == ".":
                        parent_dir = "" # Root
                except Exception:
                    continue
            else:
                try:
                    parent_dir = str(Path(file_path).parent)
                except Exception:
                    continue
            
            if parent_dir not in dir_groups:
                dir_groups[parent_dir] = []
            dir_groups[parent_dir].append(doc_id)

        # Create edges
        edge_count = 0
        for dir_path, doc_ids in dir_groups.items():
            if len(doc_ids) < 2:
                continue
            
            # Connect all in directory to a "hub" (first file) or clique?
            # For large directories, clique is too expensive (N^2).
            # Let's connect sequentially for now: A -> B -> C -> A (ring)
            # Or better: Connect all to the "index" file if it exists, otherwise ring.
            
            sorted_ids = sorted(doc_ids)
            for i in range(len(sorted_ids)):
                source = sorted_ids[i]
                target = sorted_ids[(i + 1) % len(sorted_ids)]
                
                # Bi-directional link
                if not self.graph.has_node(source) or not self.graph.has_node(target):
                    continue

                self.graph.add_edge(source, target, edge_type="directory_sibling", edge_context=f"Same directory: {dir_path}")
                self.graph.add_edge(target, source, edge_type="directory_sibling", edge_context=f"Same directory: {dir_path}")
                edge_count += 2

        logger.info(f"Added {edge_count} directory_sibling edges")

    def _build_tag_edges(self):
        """
        Connects documents that share the same tags.
        Edge type: 'shared_tag'
        """
        tag_groups: dict[str, list[str]] = {}

        with self.graph._lock:
            nodes = list(self.graph._graph.nodes(data=True))

        for doc_id, metadata in nodes:
            tags = metadata.get("tags", [])
            if not isinstance(tags, list):
                continue
            
            for tag in tags:
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(doc_id)

        edge_count = 0
        for tag, doc_ids in tag_groups.items():
            if len(doc_ids) < 2:
                continue
            
            # For tags, we might have MANY documents.
            # Clique is dangerous. Ring is safe.
            
            sorted_ids = sorted(doc_ids)
            for i in range(len(sorted_ids)):
                source = sorted_ids[i]
                target = sorted_ids[(i + 1) % len(sorted_ids)]
                
                if not self.graph.has_node(source) or not self.graph.has_node(target):
                    continue

                self.graph.add_edge(source, target, edge_type="shared_tag", edge_context=f"Shared tag: #{tag}")
                self.graph.add_edge(target, source, edge_type="shared_tag", edge_context=f"Shared tag: #{tag}")
                edge_count += 2

        logger.info(f"Added {edge_count} shared_tag edges")
