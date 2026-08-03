"""Check that the application consumes searchkernel through public modules."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PUBLIC_MODULES = frozenset(
    {
        "searchkernel",
        "searchkernel.adapters.stores",
        "searchkernel.api",
        "searchkernel.domain",
        "searchkernel.ports",
    }
)

@dataclass(frozen=True)
class ImportViolation:
    module_name: str
    imported_module: str
    line: int
    path: Path

    def __str__(self) -> str:
        return (
            f"{self.path}:{self.line}: {self.module_name} imports "
            f"private module {self.imported_module}"
        )


def _module_name(path: Path, package_root: Path) -> str:
    relative = path.relative_to(package_root.parent).with_suffix("")
    return ".".join(relative.parts).removesuffix(".__init__")


def _imported_modules(tree: ast.AST) -> Iterable[tuple[str, int]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from ((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            yield node.module, node.lineno
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "importlib"
            and node.func.attr == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            yield node.args[0].value, node.lineno


def find_private_imports_in_source(
    source: str,
    *,
    module_name: str,
    path: Path = Path("<source>"),
) -> list[ImportViolation]:
    violations = []
    for imported_module, line in _imported_modules(ast.parse(source)):
        if not imported_module.startswith("searchkernel."):
            continue
        if imported_module in PUBLIC_MODULES:
            continue
        violations.append(
            ImportViolation(module_name, imported_module, line, path)
        )
    return sorted(
        violations,
        key=lambda violation: (violation.path.as_posix(), violation.line),
    )


def find_private_imports(package_root: Path) -> list[ImportViolation]:
    violations = []
    for path in sorted(package_root.rglob("*.py")):
        module_name = _module_name(path, package_root)
        violations.extend(
            find_private_imports_in_source(
                path.read_text(encoding="utf-8"),
                module_name=module_name,
                path=path,
            )
        )
    return violations


def main() -> int:
    package_root = Path(__file__).resolve().parents[1] / "mcp_markdown_ragdocs"
    violations = find_private_imports(package_root)
    if violations:
        print("\n".join(map(str, violations)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
