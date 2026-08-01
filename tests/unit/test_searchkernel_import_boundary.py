from pathlib import Path

import pytest

from scripts.check_public_searchkernel_imports import (
    find_private_imports,
    find_private_imports_in_source,
)


PACKAGE_ROOT = Path(__file__).parents[2] / "mcp_markdown_ragdocs"


def test_app_uses_searchkernel_public_surface() -> None:
    assert find_private_imports(PACKAGE_ROOT) == []


@pytest.mark.parametrize(
    "source",
    [
        "from searchkernel.indices import VectorIndex",
        "from searchkernel.search.path_utils import normalize_path",
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
    source = "from searchkernel.adapters.stores.pgvector_index import PGVectorIndex"

    assert (
        find_private_imports_in_source(
            source,
            module_name="mcp_markdown_ragdocs.context",
        )
        != []
    )
