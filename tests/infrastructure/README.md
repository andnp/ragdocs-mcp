# Test Infrastructure Guide

This directory contains documentation and utilities for test infrastructure.

## Persistent Database Fixtures

The project now provides both ephemeral and persistent database fixtures for testing.

### Fixture Categories

#### 1. Ephemeral Fixtures (Default - Use `tmp_path`)
Fast, isolated fixtures that use pytest's built-in `tmp_path` for complete test isolation.

**When to use:**
- Unit tests
- Fast iteration during development
- Tests that don't need to verify persistence
- Each test needs complete isolation

**Example:**
```python
def test_index_document(tmp_path):
    """Fast isolated test with ephemeral storage."""
    config = Config(
        indexing=IndexingConfig(
            documents_path=str(tmp_path / "docs"),
            index_path=str(tmp_path / "indices"),
        ),
        # ... other config
    )
    vector = VectorIndex()
    keyword = KeywordIndex()
    graph = GraphStore()
    manager = IndexManager(config, vector, keyword, graph)

    # Test code here - all data discarded after test
```

#### 2. Session-Scoped Persistent Fixtures
Storage persists for entire test session across all tests.

**When to use:**
- Testing cross-test persistence behavior
- Simulating production-like scenarios
- Performance testing with larger datasets
- Testing manifest checking across multiple "startups"

**Available fixtures:**
- `persistent_storage_root` - Root directory for session storage
- `persistent_docs_path` - Documents directory (session-scoped)
- `persistent_index_path` - Indices directory (session-scoped)
- `persistent_config` - Configuration with persistent paths

**Example:**
```python
def test_index_persistence_across_sessions(
    persistent_config,
    persistent_manager_isolated,
):
    """Test that indices persist across manager instances."""
    # First manager instance - create and persist
    manager1 = persistent_manager_isolated
    manager1.index_document("test.md")
    manager1.persist()

    # Second manager instance - load persisted data
    vector2 = VectorIndex()
    keyword2 = KeywordIndex()
    graph2 = GraphStore()
    manager2 = IndexManager(persistent_config, vector2, keyword2, graph2)
    manager2.load()

    # Verify data was persisted
    results = manager2._vector.search("test query", top_k=5)
    assert "test" in results
```

#### 3. Module-Scoped Persistent Fixtures
Storage persists within a test module, shared across tests in that module.

**When to use:**
- Integration tests within a module
- Sharing expensive setup across related tests
- Testing sequential operations on same data
- Balancing performance with some isolation

**Available fixtures:**
- `persistent_indices_module` - Shared indices within module
- `persistent_manager_module` - Shared manager within module
- `persistent_config_module` - Module-specific persistent config

**Example:**
```python
@pytest.fixture(scope="module", autouse=True)
def setup_test_corpus(persistent_manager_module, persistent_config):
    """Load test corpus once for entire module."""
    docs_path = Path(persistent_config.indexing.documents_path)

    # Create test documents
    (docs_path / "doc1.md").write_text("# Document 1")
    (docs_path / "doc2.md").write_text("# Document 2")

    # Index all documents once
    persistent_manager_module.index_document(str(docs_path / "doc1.md"))
    persistent_manager_module.index_document(str(docs_path / "doc2.md"))
    persistent_manager_module.persist()


def test_search_document1(persistent_manager_module):
    """Test uses shared indexed corpus."""
    results = persistent_manager_module._vector.search("document 1", top_k=5)
    assert "doc1" in results


def test_search_document2(persistent_manager_module):
    """Test uses shared indexed corpus."""
    results = persistent_manager_module._vector.search("document 2", top_k=5)
    assert "doc2" in results
```

#### 4. Function-Scoped Persistent Fixtures
Fresh indices for each test but can persist to/load from disk.

**When to use:**
- Testing persist/load cycles
- Testing manifest changes
- Need isolation but want to test persistence
- Each test needs fresh indices but persistent storage

**Available fixtures:**
- `persistent_indices_isolated` - Fresh indices per test
- `persistent_manager_isolated` - Fresh manager per test
- `persistent_manager_with_module_config` - Fresh manager, module storage

**Example:**
```python
def test_manifest_version_mismatch(
    persistent_config,
    persistent_manager_isolated,
    persistent_index_path,
):
    """Test rebuild on manifest version change."""
    from src.indexing.manifest import IndexManifest, save_manifest, load_manifest, should_rebuild

    # Create initial manifest
    manifest_v1 = IndexManifest(
        spec_version="1.0.0",
        embedding_model="model-v1",
        parsers={"**/*.md": "MarkdownParser"},
    )
    save_manifest(persistent_index_path, manifest_v1)

    # Create new manifest with version change
    manifest_v2 = IndexManifest(
        spec_version="2.0.0",
        embedding_model="model-v1",
        parsers={"**/*.md": "MarkdownParser"},
    )

    # Verify rebuild is triggered
    saved = load_manifest(persistent_index_path)
    assert should_rebuild(manifest_v2, saved) is True
```

### Cleanup Fixtures

Optional cleanup fixtures for guaranteed cleanup even with persistent paths.

**Available fixtures:**
- `cleanup_persistent_indices` - Remove all index files after test
- `cleanup_persistent_docs` - Remove all documents after test

**Example:**
```python
def test_with_guaranteed_cleanup(
    persistent_manager_isolated,
    cleanup_persistent_indices,
    cleanup_persistent_docs,
):
    """Test with automatic cleanup of persistent storage."""
    # Test code here
    # Storage will be cleaned up even though paths are persistent
    pass
```

## Choosing the Right Fixture

### Decision Tree

```
Do you need to test persistence/loading?
├─ No  → Use tmp_path (ephemeral)
└─ Yes → Do you need isolation between tests?
         ├─ Yes → Use persistent_manager_isolated (function-scoped)
         └─ No  → Do you need to share data across tests?
                  ├─ Within a module → Use persistent_manager_module
                  └─ Across modules  → Use persistent_config with session fixtures
```

### Performance Considerations

- **Fastest**: `tmp_path` (ephemeral) - no persistence overhead
- **Fast**: `persistent_manager_with_module_config` - module-level sharing
- **Moderate**: `persistent_manager_module` - full session persistence
- **Realistic**: `persistent_config` + manual lifecycle - production-like

### Isolation Levels

- **Complete isolation**: `tmp_path` (each test independent)
- **Test-level isolation**: `persistent_manager_isolated` (fresh indices, persistent paths)
- **Module-level sharing**: `persistent_manager_module` (shared within module)
- **Session-level sharing**: `persistent_config` with session fixtures (shared across all)

## Example Test Files

### Unit Test (Ephemeral)
```python
# tests/unit/test_vector_index.py
def test_vector_index_add_and_search(tmp_path):
    """Fast unit test with complete isolation."""
    vector = VectorIndex()
    doc = Document(...)
    vector.add(doc)
    results = vector.search("query", top_k=5)
    assert "doc-id" in results
```

### Integration Test (Module-Scoped Persistent)
```python
# tests/integration/test_persistence.py
@pytest.fixture(scope="module")
def indexed_corpus(persistent_manager_module, persistent_config):
    """Index corpus once for module."""
    docs_path = Path(persistent_config.indexing.documents_path)
    # Create and index documents
    # ...
    persistent_manager_module.persist()
    return persistent_manager_module


def test_vector_search(indexed_corpus):
    """Test uses shared persistent corpus."""
    results = indexed_corpus._vector.search("query", top_k=5)
    assert len(results) > 0


def test_keyword_search(indexed_corpus):
    """Test uses shared persistent corpus."""
    results = indexed_corpus._keyword.search("query", top_k=5)
    assert len(results) > 0
```

### E2E Test (Session-Scoped Persistent)
```python
# tests/e2e/test_full_lifecycle.py
def test_index_rebuild_on_manifest_change(
    persistent_config,
    persistent_index_path,
):
    """Test full rebuild cycle with persistent storage."""
    # Test startup → index → persist → restart → load flow
    # ...
```

## Migration Guide

### Converting Existing Tests

**Before (tmp_path only):**
```python
def test_index_document(tmp_path):
    config = Config(
        indexing=IndexingConfig(
            documents_path=str(tmp_path / "docs"),
            index_path=str(tmp_path / "indices"),
        ),
        # ...
    )
    vector = VectorIndex()
    keyword = KeywordIndex()
    graph = GraphStore()
    manager = IndexManager(config, vector, keyword, graph)
    # test code
```

**After (with persistent option):**
```python
def test_index_document(tmp_path):
    """Keep tmp_path for fast unit tests."""
    # Same as before - no changes needed


def test_index_persistence_realistic(
    persistent_manager_isolated,
    persistent_config,
):
    """New test for persistence behavior."""
    manager = persistent_manager_isolated
    manager.index_document("test.md")
    manager.persist()

    # Create new manager to test loading
    vector2 = VectorIndex()
    keyword2 = KeywordIndex()
    graph2 = GraphStore()
    manager2 = IndexManager(persistent_config, vector2, keyword2, graph2)
    manager2.load()

    # Verify persistence
    # ...
```

## Best Practices

1. **Default to ephemeral**: Use `tmp_path` unless you specifically need persistence
2. **Document fixture choice**: Add comment explaining why persistent fixture was chosen
3. **Clean up when needed**: Use cleanup fixtures for tests that need guaranteed cleanup
4. **Share expensive setup**: Use module-scoped fixtures for expensive index creation
5. **Test both paths**: Have both ephemeral and persistent test variants for critical paths
6. **Avoid test coupling**: If using shared fixtures, ensure tests don't depend on execution order
7. **Monitor performance**: Track test suite time - persistent fixtures can be slower

## Troubleshooting

### Problem: Tests interfere with each other
**Solution**: Use function-scoped fixtures (`persistent_manager_isolated`) or add cleanup fixtures

### Problem: Tests are too slow
**Solution**: Use module-scoped fixtures for expensive setup or revert to `tmp_path`

### Problem: Persistent storage fills up
**Solution**: Use cleanup fixtures or scope fixtures appropriately

### Problem: Can't reproduce production persistence issues
**Solution**: Use session-scoped fixtures with manual lifecycle management
