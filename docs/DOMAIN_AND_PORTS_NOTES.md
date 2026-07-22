# Domain and Ports Implementation Notes

## W1 Task 2, 3, 6 Summary

Created pure domain types (`searchkernel/domain/`) and canonical port protocols (`searchkernel/ports/`) with inward-only dependencies enforced by import-linter.

## Existing Protocol Reconciliation

The following existing protocols were found in the codebase and do NOT cleanly map to the new canonical ports. This is noted for the composition-root refactoring work (out of scope for W1):

### 1. EmbeddingModel (searchkernel/indices/vector.py)

**Existing shape:**
```python
class EmbeddingModel(Protocol):
    def get_text_embedding(self, text: str) -> list[float]: ...
```

**New canonical shape (searchkernel/ports/embedding.py):**
```python
class EmbeddingProvider(Protocol):
    model_name: str
    dim: int
    def embed(self, texts: list[str]) -> list[Vector]: ...
```

**Mismatches:**
- Existing: single-text interface
- New: batched interface (for performance)
- Existing: no metadata (model_name, dim)
- New: model metadata required

**Action for later workstream:** W3 (provider registries) will introduce `EmbeddingProvider` implementations that wrap or replace `EmbeddingModel` usage. The existing `VectorIndex._embedding_model: EmbeddingModel` field will need to be rewired to use `EmbeddingProvider` during composition-root extraction (Task 4 of W1, done separately).

### 2. IndexProtocol (searchkernel/indices/protocol.py)

**Existing shape:**
```python
class IndexProtocol(Protocol):
    def search(self, query: str, limit: int = 10) -> list[SearchResult]: ...
    def add_document(self, doc_id: str, content: str, metadata: dict) -> None: ...
    def remove_document(self, doc_id: str) -> None: ...
    def clear(self) -> None: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
    def __len__(self) -> int: ...
```

**New canonical shapes (searchkernel/ports/stores.py):**
- `VectorStore`: `upsert(records, model_name, dim)`, `search(query_vector, k)`, `delete(ids)`, `epoch()`
- `KeywordStore`: `index(records)`, `search(query, k)`

**Mismatches:**
- Existing: generic "search" doesn't distinguish vector vs. keyword
- Existing: lacks per-model embedding isolation, epochs, filters
- Existing: includes file I/O (save/load, clear)
- New: stores manage only their domain (vectors or keywords), no I/O

**Action for later workstream:** The `IndexProtocol` appears to be an in-memory index abstraction that predates the store-port architecture. It will be deprecated during composition-root work (W1 Task 4) as the unified store (VectorStore + KeywordStore + GraphStore on Postgres) replaces it.

### 3. IndexManagerLike (searchkernel/indexing/tasks.py)

**Existing shape:**
```python
class IndexManagerLike(Protocol):
    def index_document(self, file_path: str, force: bool = False) -> None: ...
    def index_documents(self, file_paths: list[str], force: bool = False, persist: bool = False) -> None: ...
    def remove_document(self, doc_id: str) -> None: ...
    def remove_documents(self, doc_ids: list[str], persist: bool = False) -> None: ...
    def persist(self) -> None: ...
```

**Canonical port relationship:** None directly.

**Rationale:** `IndexManagerLike` is an orchestration layer for managing file-based document indexing. It is not a store port; it is a consumer of the store ports (it would ultimately call VectorStore/KeywordStore/GraphStore.upsert internally after chunking/embedding). This is a runtime concern, not a port.

**Action for later workstream:** The composition-root refactor (W1 Task 4, done separately) will preserve `IndexManager` but wire it to use the new stores internally via dependency injection.

## Import-Linter Configuration

The rule in `pyproject.toml` enforces that `searchkernel.domain` and `searchkernel.ports` import nothing from:
- `searchkernel.adapters`
- `searchkernel.runtime`
- `searchkernel.storage`
- `searchkernel.indices`
- `searchkernel.search`
- `searchkernel.indexing`
- `searchkernel.daemon`
- `searchkernel.mcp`
- `searchkernel.worker`
- `searchkernel.coordination`
- `searchkernel.git`

This enforces the hexagonal architecture and prevents the core from depending on concrete implementations.

## Adapter Methods on Document and CommitResult

Added `to_record()` and `from_record()` methods to the existing `Document` and `CommitResult` types to enable interoperability with the new `Record` type. These are purely additive (no changes to existing fields/behavior).

- **Document.to_record()**: Converts to Record with source_kind="note"
- **CommitResult.to_record()**: Converts to Record with source_kind="git_commit"

These adapters allow gradual migration of code to use the domain Record type without forcing immediate refactoring of all call sites.

## Next Steps (Later Workstreams)

- **W1 Task 4** (composition root): Extract ApplicationContext.create() and wire EmbeddingModel → EmbeddingProvider and IndexProtocol → VectorStore/KeywordStore
- **W2** (git as first ContentSource): Reimplement git_commits.db reading as a ContentSource adapter
- **W3** (provider registries): Create EmbeddingProvider implementations and registry
- **W4a** (query/ingestion stages): Build composable stages on top of the ports
