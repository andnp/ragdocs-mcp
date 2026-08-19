"""Static architecture contracts for application lifecycle and task ports."""

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2] / "mcp_markdown_ragdocs"
PORTS = {
    PACKAGE_ROOT / "app" / "services.py": {"ManagerIndexingPort"},
    PACKAGE_ROOT / "indexing" / "record_ports.py": {
        "RecordIdentityCatalog",
        "SQLiteConnectionProvider",
    },
    PACKAGE_ROOT / "context.py": {"ContextIndexingPort"},
    PACKAGE_ROOT / "lifecycle.py": {"LifecycleContextPort", "GitIndexingContextPort"},
    PACKAGE_ROOT / "coordination" / "task_leases.py": {"TaskLeasePort"},
    PACKAGE_ROOT / "coordination" / "task_submission.py": {"TaskQueuePort"},
    PACKAGE_ROOT / "coordination" / "work_intents.py": {"WorkIntentPort"},
}
TASK_REGISTRATION_MODULES = (
    PACKAGE_ROOT / "indexing" / "task_registration.py",
    PACKAGE_ROOT / "indexing" / "tasks.py",
    PACKAGE_ROOT / "gdrive" / "tasks.py",
)
CONCRETE_COORDINATION_STORES = {"TaskLeaseStore", "WorkIntentStore"}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def test_application_lifecycle_context_task_ports_remain_protocols() -> None:
    """Keep application-owned lifecycle and task seams structural protocols.

    The guard checks only the declared port classes, so implementation details
    can change without turning this into a broad source snapshot.
    """
    violations: list[str] = []
    for path, expected_names in PORTS.items():
        classes = {
            node.name: node
            for node in ast.walk(_tree(path))
            if isinstance(node, ast.ClassDef)
        }
        for name in expected_names:
            if not any(_base_name(base) == "Protocol" for base in classes[name].bases):
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}:{name}")

    assert violations == []


def test_task_registration_modules_do_not_construct_sqlite_coordination_stores() -> None:
    """Keep concrete SQLite coordination construction in composition roots.

    Registration modules may consume ports, but must not import or construct
    the concrete stores that implement those ports.
    """
    violations: list[str] = []
    for path in TASK_REGISTRATION_MODULES:
        tree = _tree(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported = {alias.name for alias in node.names}
                violations.extend(
                    f"{path.relative_to(PACKAGE_ROOT)} imports {name}"
                    for name in imported & CONCRETE_COORDINATION_STORES
                )
            elif isinstance(node, ast.Import):
                violations.extend(
                    f"{path.relative_to(PACKAGE_ROOT)} imports {alias.asname or alias.name}"
                    for alias in node.names
                    if alias.name in CONCRETE_COORDINATION_STORES
                )
            elif isinstance(node, ast.Call) and _base_name(node.func) in CONCRETE_COORDINATION_STORES:
                violations.append(
                    f"{path.relative_to(PACKAGE_ROOT)} constructs {_base_name(node.func)}"
                )

    assert violations == []
