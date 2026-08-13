from __future__ import annotations

from collections.abc import Mapping, Sequence
from hmac import compare_digest
from pathlib import Path
from time import monotonic
from typing import Any, cast

from searchkernel.api import (
    FEDERATION_CONTRACT_VERSION,
    CallerAuthorizationContext,
    MAX_SNIPPET_LENGTH,
    MAX_TOP_K,
    FederationSourceCapabilities,
    SearchHit,
    SearchHitProvenance,
    SearchRequest,
    SearchResponse,
    SourceIdentity,
)

from mcp_markdown_ragdocs.gdrive.health import DriveSourceHealth, GDriveHealthStore

type JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

RAGDOCS_SOURCE = SourceIdentity(source_kind="ragdocs", source_id="local")
RAGDOCS_CAPABILITIES = FederationSourceCapabilities(
    supports_filters=True,
    supports_source_selection=False,
    supports_rerank_text=False,
    supports_partial_results=True,
    supports_cancellation=True,
)
GDRIVE_CAPABILITIES = FederationSourceCapabilities(
    supports_filters=True,
    supports_source_selection=False,
    supports_rerank_text=False,
    supports_partial_results=True,
    supports_cancellation=True,
)
SEARCH_READ_SCOPE = "search:read"
TRUSTED_CALLER_ID = "devkit"


def build_federation_capabilities(
    *, gdrive_workspace_id: str | None = None
) -> dict[str, object]:
    """Return the generic v1 capabilities with configured logical sources."""
    payload = RAGDOCS_CAPABILITIES.to_dict()
    sources: list[dict[str, object]] = []
    if gdrive_workspace_id is not None:
        sources.append(
            {
                "source": SourceIdentity(
                    source_kind="gdrive",
                    source_id="drive",
                    workspace_id=gdrive_workspace_id,
                ).to_dict(),
                "capabilities": GDRIVE_CAPABILITIES.to_dict(),
            }
        )
    payload["sources"] = sources
    return payload


def load_gdrive_source_health(index_path: Path, workspace_id: str) -> dict[str, object]:
    """Return the latest Drive health snapshot or a typed unavailable state."""
    stored = GDriveHealthStore(index_path).load(workspace_id)
    if stored is not None:
        return stored
    return DriveSourceHealth.evaluate(
        workspace_id,
        (),
        available=False,
    ).to_payload()


class FederationRequestError(ValueError):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def authenticate_bearer(
    authorization: str | None, *, configured_token: str | None
) -> CallerAuthorizationContext:
    if not authorization:
        raise FederationRequestError(
            "bearer authentication is required", status_code=401
        )
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise FederationRequestError(
            "bearer authentication is required", status_code=401
        )
    if not configured_token:
        raise FederationRequestError(
            "federation authentication is not configured", status_code=503
        )
    if not compare_digest(parts[1], configured_token):
        raise FederationRequestError("invalid bearer credentials", status_code=401)
    return CallerAuthorizationContext(
        caller_id=TRUSTED_CALLER_ID,
        scopes=(SEARCH_READ_SCOPE,),
    )


def _string_list(filters: Mapping[str, object], *names: str) -> list[str]:
    for name in names:
        if name not in filters:
            continue
        value = filters[name]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise FederationRequestError(
                f"{name} must be an array of strings",
                status_code=400,
            )
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise FederationRequestError(
                f"{name} must be an array of non-empty strings",
                status_code=400,
            )
        result = list(value)
        if len(set(result)) != len(result):
            raise FederationRequestError(
                f"{name} must not contain duplicates",
                status_code=400,
            )
        return result
    return []


def _authorized_projects(request: SearchRequest) -> list[str] | None:
    caller = request.caller
    if caller is None:
        raise FederationRequestError(
            "caller authorization context is required",
            status_code=401,
        )
    if SEARCH_READ_SCOPE not in caller.scopes:
        raise FederationRequestError(
            f"caller is missing required scope: {SEARCH_READ_SCOPE}",
            status_code=403,
        )

    claims = caller.claims
    for name in ("project_ids", "allowed_projects", "projects"):
        if name in claims:
            value = claims[name]
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise FederationRequestError(
                    f"caller claim {name} must be an array of strings",
                    status_code=403,
                )
            if any(not isinstance(item, str) or not item.strip() for item in value):
                raise FederationRequestError(
                    f"caller claim {name} must be an array of non-empty strings",
                    status_code=403,
                )
            return [item for item in value if isinstance(item, str)]
    return None


def _record_project_id(record: Any) -> str | None:
    workspace_id = getattr(record, "workspace_id", None)
    if workspace_id is not None:
        return workspace_id
    metadata = getattr(record, "metadata", {})
    project_id = metadata.get("project_id") if isinstance(metadata, Mapping) else None
    return project_id if isinstance(project_id, str) else None


def _citation_uri(record: Any) -> str | None:
    uri = getattr(record, "uri", None)
    if isinstance(uri, str) and uri:
        return uri
    metadata = getattr(record, "metadata", {})
    file_path = metadata.get("file_path") if isinstance(metadata, Mapping) else None
    if isinstance(file_path, str) and file_path:
        return f"file://{file_path}"
    return None


def _safe_metadata(record: Any) -> dict[str, JsonValue]:
    metadata = getattr(record, "metadata", {})
    if not isinstance(metadata, Mapping):
        return {}
    allowed = ("chunk_id", "doc_id", "file_path", "header_path", "project_id")
    return cast(
        dict[str, JsonValue],
        {
            key: metadata[key]
            for key in allowed
            if key in metadata
            and isinstance(metadata[key], (str, int, float, bool, type(None)))
        },
    )


async def execute_federation_search(
    orchestrator: Any,
    request: SearchRequest,
    *,
    request_id: str,
    elapsed_start: float | None = None,
) -> SearchResponse:
    if request.contract_version != FEDERATION_CONTRACT_VERSION:
        raise FederationRequestError(
            f"unsupported contract_version: {request.contract_version}",
            status_code=400,
        )
    allowed_projects = _authorized_projects(request)
    filters = dict(request.filters)
    project_filter = _string_list(filters, "project_ids", "project_filter")
    if allowed_projects is not None:
        if project_filter and not set(project_filter).issubset(allowed_projects):
            raise FederationRequestError(
                "requested project scope exceeds caller authorization",
                status_code=403,
            )
        project_filter = project_filter or allowed_projects

    if request.source_selection:
        raise FederationRequestError(
            "source_selection is not supported by the ragdocs source",
            status_code=400,
        )

    source_kinds = _string_list(filters, "source_kinds", "source_filter")
    native_filters: dict[str, JsonValue] = dict(filters)
    if project_filter:
        if len(project_filter) == 1:
            native_filters["workspace_id"] = project_filter[0]
            native_filters.pop("project_ids", None)
            native_filters.pop("project_filter", None)
        else:
            native_filters["project_ids"] = cast(list[JsonValue], project_filter)
            native_filters.pop("project_filter", None)
    if source_kinds:
        native_filters["source_kinds"] = cast(list[JsonValue], source_kinds)
    search_limit = min(
        MAX_TOP_K,
        max(request.top_k, request.top_k * 10 if project_filter else request.top_k),
    )
    outcome = await orchestrator.search(
        request.query,
        limit=search_limit,
        filters=native_filters,
    )

    hits: list[SearchHit] = []
    warnings: list[str] = []
    for failure in getattr(outcome, "failures", ()):
        message = str(getattr(failure, "message", failure))
        if message and message not in warnings:
            warnings.append(message[:512])

    for result in getattr(outcome, "results", ()):
        record = result.record
        project_id = _record_project_id(record)
        if project_filter and project_id not in project_filter:
            continue
        rank = len(hits) + 1
        native_provenance = getattr(result, "provenance", None)
        details: dict[str, JsonValue] = {}
        if native_provenance is not None and hasattr(native_provenance, "to_dict"):
            details["native_provenance"] = cast(JsonValue, native_provenance.to_dict())
        hit = SearchHit(
            workspace_id=getattr(record, "workspace_id", None),
            source_kind=record.source_kind,
            source_id=record.source_id,
            title=record.title,
            snippet=record.body[:MAX_SNIPPET_LENGTH],
            source_rank=rank,
            uri=_citation_uri(record),
            native_score=float(result.score),
            created_at=record.created_at,
            updated_at=record.updated_at,
            lifecycle=record.status,
            metadata=_safe_metadata(record),
            provenance=SearchHitProvenance(
                source=RAGDOCS_SOURCE,
                request_id=request_id,
                retrieval_method="ragdocs-native",
                details=details,
            ),
        )
        hits.append(hit)
        if len(hits) >= request.top_k:
            break

    elapsed_ms = (
        max(0.0, (monotonic() - elapsed_start) * 1000)
        if elapsed_start is not None
        else 0.0
    )
    return SearchResponse(
        source=RAGDOCS_SOURCE,
        hits=tuple(hits),
        elapsed_ms=elapsed_ms,
        partial=bool(getattr(outcome, "degraded", False)),
        warnings=tuple(warnings),
        capabilities=RAGDOCS_CAPABILITIES,
    )


__all__ = [
    "FEDERATION_CONTRACT_VERSION",
    "FederationRequestError",
    "RAGDOCS_CAPABILITIES",
    "RAGDOCS_SOURCE",
    "TRUSTED_CALLER_ID",
    "authenticate_bearer",
    "execute_federation_search",
]
