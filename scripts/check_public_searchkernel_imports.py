"""Check that the application consumes searchkernel through public modules."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PUBLIC_MODULES = frozenset(
    {
        "searchkernel",
        "searchkernel.api",
        "searchkernel.domain",
        "searchkernel.ports",
    }
)

# The optional pgvector provider is intentionally composed by the application
# until the kernel publishes that provider through its facade.
ALLOWED_PRIVATE_IMPORTS = {
    "mcp_markdown_ragdocs.context": frozenset(
        {"searchkernel.adapters.stores.pgvector_index"}
    ),
}


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


def find_private_imports_in_source(
    source: str,
    *,
    module_name: str,
    path: Path = Path("<source>"),
) -> list[ImportViolation]:
    violations = []
    allowed_private = ALLOWED_PRIVATE_IMPORTS.get(module_name, frozenset())
    for imported_module, line in _imported_modules(ast.parse(source)):
        if not imported_module.startswith("searchkernel."):
            continue
        if imported_module in PUBLIC_MODULES or imported_module in allowed_private:
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
