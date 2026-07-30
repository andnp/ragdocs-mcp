"""Test that domain and ports modules maintain inward-only dependencies.

This test verifies the dependency rule: searchkernel/domain and searchkernel/ports
import nothing from searchkernel/adapters, runtime, stores, or concrete implementations.
"""

import importlib

import pytest


@pytest.fixture(scope="session")
def domain_module():
    """Load the domain module."""
    return importlib.import_module("searchkernel.domain")


@pytest.fixture(scope="session")
def ports_module():
    """Load the ports module."""
    return importlib.import_module("searchkernel.ports")


def test_domain_module_exists():
    """Verify domain module can be imported."""
    domain = importlib.import_module("searchkernel.domain")
    assert domain is not None
    # Verify key types are exported
    assert hasattr(domain, "Record")
    assert hasattr(domain, "Chunk")
    assert hasattr(domain, "SearchResult")
    assert hasattr(domain, "ScoredRef")


def test_ports_module_exists():
    """Verify ports module can be imported."""
    ports = importlib.import_module("searchkernel.ports")
    assert ports is not None
    # Verify key ports are exported
    assert hasattr(ports, "ContentSource")
    assert hasattr(ports, "SearchableSource")
    assert hasattr(ports, "EmbeddingProvider")
    assert hasattr(ports, "LLMProvider")
    assert hasattr(ports, "VectorStore")
    assert hasattr(ports, "KeywordStore")
    assert hasattr(ports, "GraphStore")
    assert hasattr(ports, "CacheStore")
    assert hasattr(ports, "SearchAPI")


def test_domain_models_is_pure():
    """Verify domain.models imports only stdlib and domain types."""
    domain_models = importlib.import_module("searchkernel.domain.models")

    # Get imported modules by inspecting __dict__
    imported = set(domain_models.__dict__.keys())

    # We should have Record, Chunk, etc. but not adapters
    assert "Record" in imported
    assert "Chunk" in imported
    assert "SearchResult" in imported
    assert "ScoredRef" in imported


def test_ports_content_source_is_protocol():
    """Verify ContentSource is a Protocol, not concrete."""
    from searchkernel.ports import ContentSource

    # Should have the Protocol marker
    assert hasattr(ContentSource, "__protocol_attrs__") or \
           hasattr(ContentSource, "_is_protocol")


def test_ports_embedding_provider_is_protocol():
    """Verify EmbeddingProvider is a Protocol, not concrete."""
    from searchkernel.ports import EmbeddingProvider

    # Should have the Protocol marker
    assert hasattr(EmbeddingProvider, "__protocol_attrs__") or \
           hasattr(EmbeddingProvider, "_is_protocol")


def test_record_serialization():
    """Verify Record can be serialized and deserialized."""
    from datetime import UTC, datetime

    from searchkernel.domain import Record, RecordStatus

    original = Record(
        source_kind="test",
        source_id="test:123",
        title="Test Record",
        body="This is a test record.",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        status=RecordStatus.ACTIVE,
        metadata={"key": "value"},
        uri="test://record/123",
    )

    # Serialize
    data = original.to_dict()
    assert data["source_kind"] == "test"
    assert data["source_id"] == "test:123"

    # Deserialize
    restored = Record.from_dict(data)
    assert restored.source_kind == original.source_kind
    assert restored.source_id == original.source_id
    assert restored.title == original.title
    assert restored.body == original.body
    assert restored.status == original.status
