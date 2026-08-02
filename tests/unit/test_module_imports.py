import importlib
from pathlib import Path

import pytest


def discover_modules(package_path: Path, package_name: str) -> list[str]:
    modules = []

    for item in package_path.rglob("*.py"):
        if item.name.startswith("_") and item.name != "__init__.py":
            continue

        relative = item.relative_to(package_path.parent)
        module_path = str(relative.with_suffix("")).replace("/", ".")

        module_path = module_path.removesuffix(".__init__")

        modules.append(module_path)

    return sorted(modules)


@pytest.fixture(scope="module")
def src_modules():
    package_path = Path(__file__).parent.parent.parent / "mcp_markdown_ragdocs"
    return discover_modules(package_path, "mcp_markdown_ragdocs")


def test_all_src_modules_importable(src_modules):
    failed_imports = []

    for module_name in src_modules:
        try:
            importlib.import_module(module_name)
        except Exception as e:  # noqa: BLE001 -- module-level code can raise anything at import time
            failed_imports.append((module_name, str(e)))

    if failed_imports:
        error_msg = "\n".join([f"  {mod}: {err}" for mod, err in failed_imports])
        pytest.fail(f"Failed to import {len(failed_imports)} modules:\n{error_msg}")


def test_critical_modules_import_without_error():
    critical_modules = [
        "mcp_markdown_ragdocs.config",
        "mcp_markdown_ragdocs.context",
        "mcp_markdown_ragdocs.mcp.server",
        "mcp_markdown_ragdocs.server",
        "mcp_markdown_ragdocs.git.watcher",
        "mcp_markdown_ragdocs.indexing.manager",
        "searchkernel.api",
        "searchkernel.local",
        "searchkernel.indices.local",
        "searchkernel.search.record_pipeline",
    ]

    failed = []
    for module_name in critical_modules:
        try:
            importlib.import_module(module_name)
        except Exception as e:  # noqa: BLE001 -- module-level code can raise anything at import time
            failed.append((module_name, type(e).__name__, str(e)))

    if failed:
        error_msg = "\n".join([f"  {mod}: {typ} - {err}" for mod, typ, err in failed])
        pytest.fail(f"Critical modules failed to import:\n{error_msg}")
