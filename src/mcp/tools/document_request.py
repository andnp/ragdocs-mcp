"""Normalized request models for MCP document tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.mcp.handlers import MAX_TOP_N, MIN_TOP_N
from src.mcp.validation import (
    ValidationError,
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
    """Canonical query_documents request after validation."""

    query: str
    top_n: int
    min_score: float
    similarity_threshold: float
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
        if self.scope_mode == "global":
            return None
        return self.preferred_project

    @property
    def max_chunks_per_doc(self) -> int:
        return 1 if self.uniqueness_mode == "one_per_document" else 0


def normalize_query_documents_request(
    arguments: dict,
) -> NormalizedQueryDocumentsRequest:
    """Validate canonical query_documents arguments."""

    allowed_keys = {
        "excluded_files",
        "min_score",
        "preferred_project",
        "query",
        "scope_mode",
        "scope_projects",
        "similarity_threshold",
        "top_n",
        "uniqueness_mode",
    }
    unexpected_keys = sorted(set(arguments) - allowed_keys)
    if unexpected_keys:
        raise ValidationError(
            "Unexpected parameter(s): "
            + ", ".join(unexpected_keys)
            + ". query_documents now accepts canonical scope fields only"
        )

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
    excluded_files_raw = tuple(
        validate_string_list(arguments, "excluded_files", default=[])
    )

    uniqueness_value = arguments.get("uniqueness_mode", "allow_multiple")
    if uniqueness_value not in {"allow_multiple", "one_per_document"}:
        raise ValidationError(
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

    if scope_mode is None:
        scope_mode = "global"

    if scope_mode == "explicit_projects" and not scope_projects:
        raise ValidationError(
            "scope_projects must be provided when scope_mode is 'explicit_projects'"
        )

    if scope_mode != "explicit_projects" and scope_projects:
        raise ValidationError(
            "scope_projects may only be provided when scope_mode is 'explicit_projects'"
        )

    if scope_mode == "global" and preferred_project is not None:
        raise ValidationError(
            "preferred_project may only be provided when scope_mode is not 'global'"
        )

    return NormalizedQueryDocumentsRequest(
        query=query,
        top_n=top_n,
        min_score=min_score,
        similarity_threshold=similarity_threshold,
        excluded_files_raw=excluded_files_raw,
        uniqueness_mode=uniqueness_mode,
        scope_mode=scope_mode,
        scope_projects=scope_projects,
        preferred_project=preferred_project,
    )