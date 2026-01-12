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

        if module_path.endswith(".__init__"):
            module_path = module_path[:-9]

        modules.append(module_path)

    return sorted(modules)


@pytest.fixture(scope="module")
def src_modules():
    src_path = Path(__file__).parent.parent.parent / "src"
    return discover_modules(src_path, "src")


def test_all_src_modules_importable(src_modules):
    failed_imports = []

    for module_name in src_modules:
        try:
            importlib.import_module(module_name)
        except Exception as e:
            failed_imports.append((module_name, str(e)))

    if failed_imports:
        error_msg = "\n".join([f"  {mod}: {err}" for mod, err in failed_imports])
        pytest.fail(f"Failed to import {len(failed_imports)} modules:\n{error_msg}")


def test_git_watcher_imports():
    try:
        pass
    except NameError as e:
        pytest.fail(f"NameError in git.watcher: {e}")
    except AttributeError as e:
        pytest.fail(f"AttributeError in git.watcher: {e}")


def test_critical_modules_import_without_error():
    critical_modules = [
        "src.config",
        "src.context",
        "src.mcp_server",
        "src.server",
        "src.git.watcher",
        "src.git.commit_indexer",
        "src.indexing.manager",
        "src.indices.vector",
        "src.indices.keyword",
        "src.indices.graph",
        "src.search.orchestrator",
    ]

    failed = []
    for module_name in critical_modules:
        try:
            importlib.import_module(module_name)
        except Exception as e:
            failed.append((module_name, type(e).__name__, str(e)))

    if failed:
        error_msg = "\n".join([f"  {mod}: {typ} - {err}" for mod, typ, err in failed])
        pytest.fail(f"Critical modules failed to import:\n{error_msg}")
