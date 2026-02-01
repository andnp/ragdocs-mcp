# Simplification Opportunities for Index System

## Observed Patterns

### 1. Index Creation Duplication (High Impact)

**Problem**: The same index creation pattern appears in 4 places:
- `ApplicationContext.create()` (lines 73-84)
- `ReadOnlyContext.create()` (lines 43-50)
- `src/memory/init.py` (lines 32-39)
- `src/worker/process.py` (lines 197-205)

```python
# Current pattern (repeated 4x)
vector = VectorIndex(
    embedding_model_name=embedding_model_name,
    embedding_workers=config.indexing.embedding_workers,
)
keyword = KeywordIndex()
graph = GraphStore()
```

**Recommendation**: Create a unified `IndexTriple` factory:

```python
# src/indices/factory.py
from dataclasses import dataclass

@dataclass
class IndexTriple:
    """Bundled indices that are always used together."""
    vector: VectorIndex
    keyword: KeywordIndex
    graph: GraphStore

def create_indices(
    config: Config,
    warm_up: bool = False,
) -> IndexTriple:
    """Create the standard index triple from configuration."""
    embedding_model = resolve_embedding_model(config)
    
    vector = VectorIndex(
        embedding_model_name=embedding_model,
        embedding_workers=config.indexing.embedding_workers,
    )
    if warm_up:
        vector.warm_up()
    
    return IndexTriple(
        vector=vector,
        keyword=KeywordIndex(),
        graph=GraphStore(),
    )
```

**Benefit**: Single place to modify index creation, reduce duplication ~60 LOC.

---

### 2. Embedding Model Resolution (Medium Impact)

**Problem**: This pattern appears everywhere:
```python
embedding_model_name = config.llm.embedding_model
if embedding_model_name == "local":
    embedding_model_name = "BAAI/bge-small-en-v1.5"
```

**Recommendation**: Add a computed property to `Config` or `LLMConfig`:

```python
# src/config.py
@dataclass
class LLMConfig:
    embedding_model: str = "local"
    
    @property
    def resolved_embedding_model(self) -> str:
        """Return actual model name, resolving 'local' to default."""
        if self.embedding_model == "local":
            return "BAAI/bge-small-en-v1.5"
        return self.embedding_model
```

**Benefit**: Self-documenting, single source of truth for resolution.

---

### 3. Path Resolution at Different Times (Medium Impact)

**Problem**: Paths are resolved at different stages:
- `resolve_index_path()` called in `ApplicationContext.create()`
- `resolve_memory_path()` called in multiple places
- `resolve_documents_path()` called in context creation

**Recommendation**: Create a `ResolvedPaths` dataclass populated once:

```python
# src/paths.py
@dataclass
class ResolvedPaths:
    index: Path
    documents: Path
    memory: Path | None
    snapshots: Path
    
    @classmethod
    def resolve(cls, config: Config, detected_project: str | None) -> ResolvedPaths:
        """Resolve all paths once based on configuration."""
        return cls(
            index=resolve_index_path(config, detected_project),
            documents=Path(resolve_documents_path(config, detected_project)),
            memory=resolve_memory_path(config, detected_project) if config.memory.enabled else None,
            snapshots=resolve_index_path(config, detected_project) / "snapshots",
        )
```

**Benefit**: All paths computed once, passed explicitly, easier to test.

---

### 4. Persist/Load Pattern Duplication (Lower Impact)

**Problem**: Both `IndexManager` and `MemoryIndexManager` have similar persist/load code:
```python
# IndexManager._persist_indices()
self.vector.persist(index_path / "vector")
self.keyword.persist(index_path / "keyword")
self.graph.persist(index_path / "graph")

# MemoryIndexManager.persist()
self._vector.persist(indices_path / "vector")
self._keyword.persist(indices_path / "keyword")
self._graph.persist(indices_path / "graph")
```

**Recommendation**: If `IndexTriple` is introduced, add persist/load methods:

```python
class IndexTriple:
    def persist(self, path: Path) -> None:
        self.vector.persist(path / "vector")
        self.keyword.persist(path / "keyword")
        self.graph.persist(path / "graph")
    
    def load(self, path: Path) -> None:
        self.vector.load(path / "vector")
        self.keyword.load(path / "keyword")
        self.graph.load(path / "graph")
```

---

## Implementation Priority

1. **Embedding Model Resolution** - Quick win, ~5 LOC, immediate benefit
2. **IndexTriple Factory** - Medium effort, removes duplication, improves testability
3. **ResolvedPaths** - Medium effort, explicit data flow, prevents path confusion
4. **Persist/Load Unification** - Lower priority, depends on IndexTriple

## Notes

- These changes would be breaking for tests that mock individual components
- Consider implementing gradually, starting with embedding model resolution
- Each change should include migration of existing code + tests
