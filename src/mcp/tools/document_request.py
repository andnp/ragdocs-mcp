"""Normalized request models for MCP document tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.mcp.handlers import MAX_TOP_N, MIN_TOP_N
from src.mcp.validation import (
    validate_boolean,
    validate_enum,
    validate_float_range,
    validate_integer_range,
    validate_optional_string,
    validate_query,
    validate_string_list,
)

ScopeMode = Literal["global", "active_project", "explicit_projects"]
UniquenessMode = Literal["allow_multiple", "one_per_document"]


@dataclass(frozen=True)
class NormalizedQueryDocumentsRequest:
    """Canonical query_documents request after validation and alias normalization."""

    query: str
    top_n: int
    min_score: float
    similarity_threshold: float
    show_stats: bool
    excluded_files_raw: tuple[str, ...]
    uniqueness_mode: UniquenessMode
    scope_mode: ScopeMode
    scope_projects: tuple[str, ...] = ()
    preferred_project: str | None = None

    @property
    def project_filter(self) -> list[str]:
        if self.scope_mode != "explicit_projects":
            return []
        return list(self.scope_projects)

    @property
    def project_context(self) -> str | None:
        return self.preferred_project

    @property
    def max_chunks_per_doc(self) -> int:
        return 1 if self.uniqueness_mode == "one_per_document" else 0

    @property
    def result_header(self) -> str:
        if self.uniqueness_mode == "one_per_document":
            return "Search Results (Unique Documents)"
        return "Search Results"


def normalize_query_documents_request(
    arguments: dict,
) -> NormalizedQueryDocumentsRequest:
    """Validate query_documents arguments and map legacy scope aliases."""

    query = validate_query(arguments, "query")
    top_n = validate_integer_range(
        arguments, "top_n", default=5, min_val=MIN_TOP_N, max_val=MAX_TOP_N
    )
    min_score = validate_float_range(
        arguments, "min_score", default=0.3, min_val=0.0, max_val=1.0
    )
    similarity_threshold = validate_float_range(
        arguments, "similarity_threshold", default=0.85, min_val=0.5, max_val=1.0
    )
    show_stats = validate_boolean(arguments, "show_stats", default=False)
    excluded_files_raw = tuple(
        validate_string_list(arguments, "excluded_files", default=[])
    )

    uniqueness_value = arguments.get("uniqueness_mode", "allow_multiple")
    if uniqueness_value not in {"allow_multiple", "one_per_document"}:
        raise ValueError(
            "Invalid uniqueness_mode: "
            f"{uniqueness_value}. Must be 'allow_multiple' or 'one_per_document'"
        )
    uniqueness_mode: UniquenessMode = uniqueness_value

    scope_mode = validate_enum(
        arguments,
        "scope_mode",
        {"global", "active_project", "explicit_projects"},
        default=None,
    )
    scope_projects = tuple(
        validate_string_list(arguments, "scope_projects", default=[])
    )
    preferred_project = validate_optional_string(arguments, "preferred_project")

    legacy_project_filter = tuple(
        validate_string_list(arguments, "project_filter", default=[])
    )
    legacy_project_context = validate_optional_string(arguments, "project_context")

    if (
        scope_mode == "explicit_projects"
        and not scope_projects
        and legacy_project_filter
    ):
        scope_projects = legacy_project_filter

    if scope_mode == "active_project" and preferred_project is None:
        preferred_project = legacy_project_context

    if scope_mode is None:
        if scope_projects or legacy_project_filter:
            scope_mode = "explicit_projects"
            if not scope_projects:
                scope_projects = legacy_project_filter
        elif preferred_project or legacy_project_context:
            scope_mode = "active_project"
            if preferred_project is None:
                preferred_project = legacy_project_context
        else:
            scope_mode = "global"

    if scope_mode == "explicit_projects" and preferred_project is None:
        preferred_project = legacy_project_context

    return NormalizedQueryDocumentsRequest(
        query=query,
        top_n=top_n,
        min_score=min_score,
        similarity_threshold=similarity_threshold,
        show_stats=show_stats,
        excluded_files_raw=excluded_files_raw,
        uniqueness_mode=uniqueness_mode,
        scope_mode=scope_mode,
        scope_projects=scope_projects,
        preferred_project=preferred_project,
    )