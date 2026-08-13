"""Tests for logical content-source registration on the canonical manager."""

from types import SimpleNamespace


def test_record_manager_registers_sources_on_the_existing_search_kernel(record_manager):
    """
    Register a source through the manager's public boundary.
    Keep the source available to SearchKernel ingestion without a second index.
    """
    source = SimpleNamespace(source_kind="gdrive")

    record_manager.register_content_source(source)

    assert record_manager.get_content_source("gdrive") is source
    assert record_manager.kernel.kernel._content_sources["gdrive"] is source
