from pathlib import Path

import pytest

from scripts.check_public_searchkernel_imports import (
    find_private_imports,
    find_private_imports_in_source,
)

PACKAGE_ROOT = Path(__file__).parents[2] / "mcp_markdown_ragdocs"


def test_app_uses_public_searchkernel_imports() -> None:
    violations = find_private_imports(PACKAGE_ROOT)

    assert violations == []


@pytest.mark.parametrize(
    "source",
    [
        "from searchkernel.indices.local import LocalVectorStore",
        "from searchkernel.search.record_pipeline import RecordSearchPipeline",
        "import searchkernel.storage.db",
    ],
)
def test_private_searchkernel_imports_are_rejected(source: str) -> None:
    violations = find_private_imports_in_source(
        source,
        module_name="mcp_markdown_ragdocs.indexing.manager",
    )

    assert len(violations) == 1


@pytest.mark.parametrize(
    "source",
    [
        "from searchkernel import Record",
        "from searchkernel.api import Record",
        "from searchkernel.domain import Record",
        "from searchkernel.ports import ContentSource",
    ],
)
def test_public_searchkernel_imports_are_allowed(source: str) -> None:
    assert (
        find_private_imports_in_source(
            source,
            module_name="mcp_markdown_ragdocs.adapters.sources.local",
        )
        == []
    )


def test_private_optional_backend_import_is_rejected() -> None:
    source = "from searchkernel.adapters.stores.pgvector import PGVectorStore"

    assert (
        find_private_imports_in_source(
            source,
            module_name="mcp_markdown_ragdocs.context",
        )
        != []
    )


@pytest.mark.parametrize(
    ("prefix", "call"),
    [
        ("import importlib\n", 'importlib.import_module("searchkernel.search.record_pipeline")'),
        ("import importlib as kernel_loader\n", 'kernel_loader.import_module("searchkernel.search.record_pipeline")'),
        ("from importlib import import_module\n", 'import_module("searchkernel.search.record_pipeline")'),
        ("from importlib import import_module as kernel_loader\n", 'kernel_loader("searchkernel.adapters.stores.pgvector_index")'),
    ],
)
def test_constant_dynamic_private_imports_are_rejected(
    prefix: str,
    call: str,
) -> None:
    assert (
        find_private_imports_in_source(
            f"{prefix}{call}",
            module_name="mcp_markdown_ragdocs.adapters.pgvector",
        )
        != []
    )


def test_public_dynamic_import_is_allowed() -> None:
    assert (
        find_private_imports_in_source(
            'import importlib\nimportlib.import_module("searchkernel.api")',
            module_name="mcp_markdown_ragdocs.adapters.sources.local",
        )
        == []
    )


@pytest.mark.parametrize(
    ("prefix", "call"),
    [
        ("import importlib as kernel_loader\n", 'kernel_loader.import_module("searchkernel.api")'),
        ("from importlib import import_module as kernel_loader\n", 'kernel_loader("searchkernel.api")'),
    ],
)
def test_public_dynamic_import_aliases_are_allowed(prefix: str, call: str) -> None:
    assert (
        find_private_imports_in_source(
            f"{prefix}{call}",
            module_name="mcp_markdown_ragdocs.adapters.sources.local",
        )
        == []
    )
