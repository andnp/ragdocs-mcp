from __future__ import annotations

from collections.abc import Mapping, Sequence
from hmac import compare_digest
from pathlib import Path
from time import monotonic
from typing import Protocol, runtime_checkable

from searchkernel.api import (
    FEDERATION_CONTRACT_VERSION,
    CallerAuthorizationContext,
    MAX_SNIPPET_LENGTH,
    MAX_TOP_K,
    FederationSourceCapabilities,
    Record,
    RecordSearchOutcome,
    SearchHit,
    SearchHitProvenance,
    SearchRequest,
    SearchResponse,
    SourceIdentity,
)

from mcp_markdown_ragdocs.gdrive.health import DriveSourceHealth, GDriveHealthStore

type JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


@runtime_checkable
class FederationSearchOrchestrator(Protocol):
    async def search(
        self,
        query: str,
        *,
        limit: int,
        filters: Mapping[str, object],
    ) -> RecordSearchOutcome: ...

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
DRIVE_SOURCE_KIND = "gdrive"
DRIVE_WORKSPACE_CLAIM = "drive_workspace_ids"


def build_federation_capabilities(
    *, gdrive_workspace_id: str | None = None
) -> dict[str, JsonValue]:
    """Return the generic v1 capabilities with configured logical sources."""
    payload = _json_object(RAGDOCS_CAPABILITIES.to_dict())
    sources: list[JsonValue] = []
    if gdrive_workspace_id is not None:
        sources.append(
            _json_object(
                {
                    "source": SourceIdentity(
                    source_kind="gdrive",
                    source_id="drive",
                    workspace_id=gdrive_workspace_id,
                    ).to_dict(),
                    "capabilities": GDRIVE_CAPABILITIES.to_dict(),
                }
            )
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
    authorization: str | None,
    *,
    configured_token: str | None,
    drive_workspace_ids: Sequence[object] = (),
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
    claims: dict[str, JsonValue] = {}
    if drive_workspace_ids:
        claims[DRIVE_WORKSPACE_CLAIM] = _json_value(list(drive_workspace_ids))
    return CallerAuthorizationContext(
        caller_id=TRUSTED_CALLER_ID,
        scopes=(SEARCH_READ_SCOPE,),
        claims=claims,
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


def _authorized_projects(
    request: SearchRequest, *, drive_requested: bool
) -> list[str] | None:
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
    if drive_requested:
        return None

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


def _authorized_drive_workspaces(
    request: SearchRequest,
    *,
    drive_requested: bool,
    owned_drive_workspace_ids: Sequence[str],
) -> frozenset[str]:
    claims = request.caller.claims if request.caller is not None else {}
    value = claims.get(DRIVE_WORKSPACE_CLAIM)
    if value is None:
        if drive_requested:
            raise FederationRequestError(
                f"caller claim {DRIVE_WORKSPACE_CLAIM} is required for Drive search",
                status_code=403,
            )
        return frozenset()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FederationRequestError(
            f"caller claim {DRIVE_WORKSPACE_CLAIM} must be an array of strings",
            status_code=403,
        )
    workspace_ids = [item for item in value if isinstance(item, str)]
    if len(workspace_ids) != len(value) or any(not item.strip() for item in workspace_ids):
        raise FederationRequestError(
            f"caller claim {DRIVE_WORKSPACE_CLAIM} must be an array of non-empty strings",
            status_code=403,
        )
    if not workspace_ids or len(set(workspace_ids)) != len(workspace_ids):
        raise FederationRequestError(
            f"caller claim {DRIVE_WORKSPACE_CLAIM} must contain unique workspaces",
            status_code=403,
        )
    owned = set(owned_drive_workspace_ids)
    if not set(workspace_ids).issubset(owned):
        raise FederationRequestError(
            f"caller claim {DRIVE_WORKSPACE_CLAIM} exceeds ragdocs Drive scope",
            status_code=403,
        )
    return frozenset(workspace_ids)


def _record_has_drive_scope_membership(record: Record) -> bool:
    metadata = record.metadata
    memberships = metadata.get("scope_memberships") if isinstance(metadata, Mapping) else None
    return isinstance(memberships, Sequence) and not isinstance(memberships, (str, bytes)) and bool(memberships)


def _record_is_authorized_for_drive(
    record: Record,
    *,
    authorized_workspaces: frozenset[str],
) -> bool:
    if record.source_kind != DRIVE_SOURCE_KIND:
        return True
    workspace_id = record.workspace_id
    return (
        isinstance(workspace_id, str)
        and workspace_id in authorized_workspaces
        and _record_has_drive_scope_membership(record)
    )


def _record_project_id(record: Record) -> str | None:
    workspace_id = record.workspace_id
    if workspace_id is not None:
        return workspace_id
    metadata = record.metadata
    project_id = metadata.get("project_id") if isinstance(metadata, Mapping) else None
    return project_id if isinstance(project_id, str) else None


def _citation_uri(record: Record) -> str | None:
    uri = record.uri
    if isinstance(uri, str) and uri:
        return uri
    metadata = record.metadata
    file_path = metadata.get("file_path") if isinstance(metadata, Mapping) else None
    if isinstance(file_path, str) and file_path:
        return f"file://{file_path}"
    return None


def _safe_metadata(record: Record) -> dict[str, JsonValue]:
    metadata = record.metadata
    allowed = ("chunk_id", "doc_id", "file_path", "header_path", "project_id")
    result: dict[str, JsonValue] = {}
    for key in allowed:
        value = metadata.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    return result


async def execute_federation_search(
    orchestrator: FederationSearchOrchestrator,
    request: SearchRequest,
    *,
    request_id: str,
    owned_drive_workspace_ids: Sequence[str],
    elapsed_start: float | None = None,
) -> SearchResponse:
    if request.contract_version != FEDERATION_CONTRACT_VERSION:
        raise FederationRequestError(
            f"unsupported contract_version: {request.contract_version}",
            status_code=400,
        )
    filters = dict(request.filters)
    source_kinds = _string_list(filters, "source_kinds", "source_filter")
    drive_requested = DRIVE_SOURCE_KIND in source_kinds
    allowed_projects = _authorized_projects(request, drive_requested=drive_requested)
    authorized_drive_workspaces = _authorized_drive_workspaces(
        request,
        drive_requested=drive_requested,
        owned_drive_workspace_ids=owned_drive_workspace_ids or (),
    )
    project_filter = _string_list(filters, "project_ids", "project_filter")
    if drive_requested:
        project_filter = []
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

    native_filters: dict[str, JsonValue] = dict(filters)
    if drive_requested:
        for name in (
            "workspace_id",
            "project_ids",
            "project_filter",
            "ranking_workspace_id",
            "project_context",
        ):
            native_filters.pop(name, None)
    if project_filter:
        if len(project_filter) == 1:
            native_filters["workspace_id"] = project_filter[0]
            native_filters.pop("project_ids", None)
            native_filters.pop("project_filter", None)
        else:
            native_filters["project_ids"] = _json_value(project_filter)
            native_filters.pop("project_filter", None)
    if source_kinds:
        native_filters["source_kinds"] = _json_value(source_kinds)
    if drive_requested:
        native_filters["source_scoped_filters"] = _json_value(
            {
                DRIVE_SOURCE_KIND: {
                    "workspace_ids": sorted(authorized_drive_workspaces),
                    "metadata_non_empty": ["scope_memberships"],
                }
            },
        )
    if drive_requested and len(authorized_drive_workspaces) == 1:
        native_filters["workspace_id"] = next(iter(authorized_drive_workspaces))
    search_limit = min(
        MAX_TOP_K,
        max(
            request.top_k,
            request.top_k * 10 if project_filter else request.top_k,
        ),
    )
    outcome = await orchestrator.search(
        request.query,
        limit=search_limit,
        filters=native_filters,
    )

    hits: list[SearchHit] = []
    warnings: list[str] = []
    for failure in outcome.failures:
        message = failure.message
        if message and message not in warnings:
            warnings.append(message[:512])

    for result in outcome.results:
        record = result.record
        project_id = _record_project_id(record)
        if project_filter and project_id not in project_filter:
            continue
        if not _record_is_authorized_for_drive(
            record,
            authorized_workspaces=authorized_drive_workspaces,
        ):
            continue
        rank = len(hits) + 1
        details: dict[str, JsonValue] = {}
        details["native_provenance"] = _json_value(result.provenance.to_dict())
        hit = SearchHit(
            workspace_id=record.workspace_id,
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
        partial=outcome.degraded,
        warnings=tuple(warnings),
        capabilities=RAGDOCS_CAPABILITIES,
    )


__all__ = [
    "FEDERATION_CONTRACT_VERSION",
    "FederationRequestError",
    "DRIVE_SOURCE_KIND",
    "DRIVE_WORKSPACE_CLAIM",
    "FederationSearchOrchestrator",
    "RAGDOCS_CAPABILITIES",
    "RAGDOCS_SOURCE",
    "TRUSTED_CALLER_ID",
    "authenticate_bearer",
    "execute_federation_search",
]


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            key: _json_value(item)
            for key, item in value.items()
            if isinstance(key, str)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item) for item in value]
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


def _json_object(value: Mapping[str, object]) -> dict[str, JsonValue]:
    return {
        key: _json_value(item)
        for key, item in value.items()
    }
