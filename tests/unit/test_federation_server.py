import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from fastapi.testclient import TestClient
from searchkernel.domain import Record, SearchResultProvenance
from searchkernel.ports.federation import (
    CallerAuthorizationContext,
    JsonValue,
    SearchRequest,
)

from mcp_markdown_ragdocs.config import Config
from mcp_markdown_ragdocs.federation import (
    FederationRequestError,
    TRUSTED_CALLER_ID,
    authenticate_bearer,
)
from mcp_markdown_ragdocs.gdrive.health import DriveScopeHealth, DriveSourceHealth, GDriveHealthStore
from mcp_markdown_ragdocs.server import create_app

AUTH_TOKEN = "test-deployment-token"


def _app(
    *,
    drive_workspace_id: str | None = None,
    drive_workspace_ids: tuple[object, ...] = (),
):
    app = create_app()
    app.state.config = Config()
    app.state.config.federation.deployment_token = AUTH_TOKEN
    app.state.config.federation.drive_workspace_ids = cast(Any, drive_workspace_ids)
    if drive_workspace_id is not None:
        app.state.config.gdrive.enabled = True
        app.state.config.gdrive.workspace_id = drive_workspace_id
    return app


def _request(**kwargs: object) -> dict[str, JsonValue]:
    return SearchRequest(
        "authentication",
        caller=CallerAuthorizationContext(
            caller_id="devkit",
            scopes=("search:read",),
        ),
        **cast(Any, kwargs),
    ).to_dict()


def _result(
    *,
    project_id: str = "project-a",
    score: float = 0.9,
    source_id: str = "chunk-a",
    source_kind: str = "note",
    uri: str = "file:///docs/auth.md#authentication",
):
    record = Record(
        source_kind=source_kind,
        source_id=source_id,
        title="Authentication",
        body="Use the documented authentication flow.",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
        metadata={
            "chunk_id": source_id,
            "doc_id": "doc-a",
            "file_path": "/docs/auth.md",
            "header_path": "Authentication",
            "project_id": project_id,
        },
        uri=uri,
        workspace_id=project_id,
    )
    return SimpleNamespace(
        record=record,
        score=score,
        provenance=SearchResultProvenance(strategies=("keyword",)),
    )


def _drive_result(
    *,
    workspace_id: str = "workspace",
    source_id: str = "drive-a",
    score: float = 0.9,
    scope_memberships: tuple[str, ...] = ("shared-drive",),
):
    record = Record(
        source_kind="gdrive",
        source_id=source_id,
        title="Drive document",
        body="A document from Google Drive.",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
        metadata={
            "chunk_id": f"{source_id}-chunk",
            "doc_id": source_id,
            "file_path": f"/drive/{source_id}.md",
            "header_path": "Drive document",
            "scope_memberships": list(scope_memberships),
        },
        uri=f"gdrive://{source_id}",
        workspace_id=workspace_id,
    )
    return SimpleNamespace(
        record=record,
        score=score,
        provenance=SearchResultProvenance(strategies=("keyword",)),
    )


class _Orchestrator:
    def __init__(self, result=None):
        self.result = result or _result()
        self.calls: list[dict[str, Any]] = []

    async def search(self, query, *, limit, filters):
        self.calls.append({"query": query, "limit": limit, "filters": filters})
        return SimpleNamespace(results=(self.result,), failures=(), degraded=False)


class _RankedOrchestrator:
    def __init__(self, results):
        self.results = tuple(results)
        self.calls: list[dict[str, Any]] = []

    async def search(self, query, *, limit, filters):
        self.calls.append({"query": query, "limit": limit, "filters": filters})
        return SimpleNamespace(
            results=tuple(sorted(self.results, key=lambda result: -result.score)),
            failures=(),
            degraded=False,
        )


class _SourceFilteringOrchestrator(_RankedOrchestrator):
    async def search(self, query, *, limit, filters):
        outcome = await super().search(query, limit=limit, filters=filters)
        source_kinds = filters.get("source_kinds")
        if not isinstance(source_kinds, list):
            return outcome
        return SimpleNamespace(
            results=tuple(
                result
                for result in outcome.results
                if result.record.source_kind in source_kinds
            ),
            failures=outcome.failures,
            degraded=outcome.degraded,
        )


def _client(
    orchestrator: Any,
    *,
    drive_workspace_id: str | None = None,
    drive_workspace_ids: tuple[object, ...] = (),
) -> TestClient:
    app = _app(
        drive_workspace_id=drive_workspace_id,
        drive_workspace_ids=drive_workspace_ids,
    )
    app.state.orchestrator = orchestrator
    return TestClient(app, headers={"Authorization": f"Bearer {AUTH_TOKEN}"})


def test_federation_search_success_preserves_native_provenance():
    orchestrator = _Orchestrator()
    client = _client(orchestrator)
    response = client.post(
        "/v1/search",
        json=_request(request_id="request-123", trace_id="trace-123"),
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "request-123"
    payload = response.json()
    assert payload["contract_version"] == "v1"
    assert payload["source"] == {
        "source_kind": "ragdocs",
        "source_id": "local",
        "workspace_id": None,
    }
    assert payload["hits"][0]["source_rank"] == 1
    assert payload["hits"][0]["uri"] == "file:///docs/auth.md#authentication"
    assert payload["hits"][0]["provenance"]["source"]["source_id"] == "local"
    assert payload["hits"][0]["provenance"]["request_id"] == "request-123"
    assert orchestrator.calls[0]["query"] == "authentication"


def test_federation_capabilities_and_health():
    """Preserve the v1 local capability and health contract by default."""
    client = TestClient(
        _app(), headers={"Authorization": f"Bearer {AUTH_TOKEN}"}
    )
    capabilities = client.get("/v1/search/capabilities")
    health = client.get("/v1/health")

    assert capabilities.status_code == 200
    assert capabilities.json()["contract_versions"] == ["v1"]
    assert capabilities.json()["max_top_k"] == 1000
    assert health.json()["status"] == "ok"
    assert health.json()["contract_version"] == "v1"
    assert health.json()["source"]["source_kind"] == "ragdocs"
    assert capabilities.json()["sources"] == []
    assert "source_health" not in health.json()


def test_federation_exposes_enabled_drive_capabilities_and_fresh_health(
    tmp_path: Path,
):
    """Expose Drive identity, capabilities, and persisted freshness state."""
    app = _app()
    config = app.state.config
    config.gdrive.enabled = True
    config.gdrive.workspace_id = "workspace"
    app.state.config = config
    app.state.index_path = tmp_path
    GDriveHealthStore(tmp_path).save(
        DriveSourceHealth.evaluate(
            "workspace",
            (DriveScopeHealth("shared-with-me", indexed_records=4, last_success_at=99),),
            observed_at=100,
            stale_after_seconds=10,
            watch_mode="poll",
        )
    )

    client = TestClient(app, headers={"Authorization": f"Bearer {AUTH_TOKEN}"})
    capabilities = client.get("/v1/search/capabilities").json()
    health = client.get("/v1/health").json()

    assert capabilities["contract_versions"] == ["v1"]
    assert capabilities["sources"] == [
        {
            "source": {
                "source_kind": "gdrive",
                "source_id": "drive",
                "workspace_id": "workspace",
            },
            "capabilities": {
                "contract_versions": ["v1"],
                "supports_filters": True,
                "supports_source_selection": False,
                "supports_rerank_text": False,
                "supports_partial_results": True,
                "supports_cancellation": True,
                "max_top_k": 1000,
                "max_rerank_text_length": 4096,
            },
        }
    ]
    assert health["status"] == "ok"
    assert health["source_health"]["gdrive"]["status"] == "healthy"
    assert health["source_health"]["gdrive"]["source"]["observed_at"] == 100
    assert health["source_health"]["gdrive"]["source"]["stale_after_seconds"] == 10


def test_federation_reports_unavailable_drive_without_health_snapshot(tmp_path: Path):
    """Report a typed unavailable state before Drive has completed a sync."""
    app = _app()
    config = app.state.config
    config.gdrive.enabled = True
    config.gdrive.workspace_id = "workspace"
    app.state.config = config
    app.state.index_path = tmp_path

    health = TestClient(
        app, headers={"Authorization": f"Bearer {AUTH_TOKEN}"}
    ).get("/v1/health").json()

    assert health["source_health"]["gdrive"]["status"] == "unavailable"
    assert health["source_health"]["gdrive"]["source"]["source_kind"] == "gdrive"
    assert health["source_health"]["gdrive"]["source"]["available"] is False


def test_federation_health_reports_stale_drive(tmp_path: Path):
    """
    Surface stale Drive freshness through the authenticated v1 health endpoint.
    """
    app = _app(drive_workspace_id="workspace")
    app.state.index_path = tmp_path
    GDriveHealthStore(tmp_path).save(
        DriveSourceHealth.evaluate(
            "workspace",
            (DriveScopeHealth("shared-with-me", indexed_records=2, last_success_at=80),),
            observed_at=100,
            stale_after_seconds=10,
        )
    )

    response = TestClient(
        app, headers={"Authorization": f"Bearer {AUTH_TOKEN}"}
    ).get("/v1/health")

    assert response.status_code == 200
    assert response.json()["source_health"]["gdrive"]["status"] == "stale"


def test_federation_health_reports_incomplete_drive_acl(tmp_path: Path):
    """
    Surface incomplete Drive ACL coverage before callers trust the corpus.
    """
    app = _app(drive_workspace_id="workspace")
    app.state.index_path = tmp_path
    GDriveHealthStore(tmp_path).save(
        DriveSourceHealth.evaluate(
            "workspace",
            (
                DriveScopeHealth(
                    "shared-with-me",
                    indexed_records=2,
                    acl_complete=False,
                    last_success_at=99,
                ),
            ),
            observed_at=100,
            stale_after_seconds=10,
        )
    )

    response = TestClient(
        app, headers={"Authorization": f"Bearer {AUTH_TOKEN}"}
    ).get("/v1/health")

    assert response.status_code == 200
    assert response.json()["source_health"]["gdrive"]["status"] == "acl-incomplete"


def test_federation_search_rejects_invalid_request():
    client = TestClient(
        _app(), headers={"Authorization": f"Bearer {AUTH_TOKEN}"}
    )
    response = client.post(
        "/v1/search",
        json={"query": "authentication", "contract_version": "v2"},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"


def test_bearer_authentication_maps_token_to_trusted_caller():
    """
    Resolve the deployment credential to the fixed federation caller contract.
    """
    caller = authenticate_bearer(
        f"Bearer {AUTH_TOKEN}", configured_token=AUTH_TOKEN
    )

    assert caller.caller_id == TRUSTED_CALLER_ID
    assert caller.scopes == ("search:read",)
    assert caller.claims == {}


def test_bearer_authentication_includes_drive_workspace_claims():
    """Map configured Drive scope to the authenticated caller claim shape."""
    caller = authenticate_bearer(
        f"Bearer {AUTH_TOKEN}",
        configured_token=AUTH_TOKEN,
        drive_workspace_ids=("workspace",),
    )

    assert caller.claims == {"drive_workspace_ids": ["workspace"]}


def test_bearer_authentication_rejects_missing_and_invalid_credentials():
    """
    Reject absent and incorrect credentials before parsing or searching.
    """
    for authorization in (None, "Bearer wrong-token"):
        try:
            authenticate_bearer(authorization, configured_token=AUTH_TOKEN)
        except FederationRequestError as error:
            assert error.status_code == 401
        else:
            raise AssertionError("invalid credentials were accepted")


def test_federation_endpoints_require_bearer_authentication():
    """
    Protect capabilities and health alongside the federation search endpoint.
    """
    client = TestClient(_app())

    assert client.get("/v1/search/capabilities").status_code == 401
    assert client.get("/v1/health").status_code == 401
    assert client.post("/v1/search", json=_request()).status_code == 401


def test_federation_search_rejects_invalid_deployment_token():
    """
    Reject an invalid caller token before any Drive search is attempted.
    """
    client = TestClient(
        _app(), headers={"Authorization": "Bearer expired-token"}
    )

    response = client.post("/v1/search", json=_request())

    assert response.status_code == 401
    assert response.json()["error"] == "invalid bearer credentials"


def test_request_caller_spoof_does_not_change_authenticated_identity():
    """
    Ignore an attacker-supplied caller and its legacy project claims.
    """
    orchestrator = _Orchestrator()
    client = _client(orchestrator)
    request = SearchRequest(
        "authentication",
        caller=CallerAuthorizationContext(
            caller_id="attacker",
            scopes=("search:read",),
            claims={"project_ids": ["project-a"]},
        ),
        filters={"project_ids": ["project-b"]},
    )

    response = client.post("/v1/search", json=request.to_dict())

    assert response.status_code == 200
    assert response.json()["hits"] == []


def test_federation_search_uses_bearer_identity_over_body_caller():
    orchestrator = _Orchestrator()
    client = _client(orchestrator)
    missing_caller = client.post(
        "/v1/search",
        json=SearchRequest("authentication").to_dict(),
    )
    missing_scope = client.post(
        "/v1/search",
        json=SearchRequest(
            "authentication",
            caller=CallerAuthorizationContext(caller_id="devkit"),
        ).to_dict(),
    )

    assert missing_caller.status_code == 200
    assert missing_scope.status_code == 200


def test_federation_search_enforces_authorized_project_claims():
    orchestrator = _Orchestrator()
    request = SearchRequest(
        "authentication",
        caller=CallerAuthorizationContext(
            caller_id="devkit",
            scopes=("search:read",),
            claims={"project_ids": ["project-a"]},
        ),
        filters={"project_ids": ["project-b"]},
    )
    client = _client(orchestrator)
    response = client.post("/v1/search", json=request.to_dict())

    assert response.status_code == 200


def test_federation_search_forwards_scoped_native_filters():
    orchestrator = _Orchestrator()
    client = _client(orchestrator)
    response = client.post(
        "/v1/search",
        json=_request(filters={"project_ids": ["project-a", "project-b"]}),
    )

    assert response.status_code == 200
    assert orchestrator.calls[0]["filters"]["project_ids"] == [
        "project-a",
        "project-b",
    ]


def test_federation_search_uses_workspace_scope_for_one_project():
    orchestrator = _Orchestrator()
    client = _client(orchestrator)
    response = client.post(
        "/v1/search",
        json=_request(filters={"project_ids": ["project-a"]}),
    )

    assert response.status_code == 200
    assert orchestrator.calls[0]["filters"]["workspace_id"] == "project-a"


def test_drive_search_rejects_missing_workspace_claims():
    """Reject Drive searches when the authenticated caller has no claim."""
    client = _client(
        _Orchestrator(_drive_result()),
        drive_workspace_id="workspace",
    )
    response = client.post(
        "/v1/search",
        json=_request(filters={"source_kinds": ["gdrive"]}),
    )

    assert response.status_code == 403


def test_drive_search_rejects_invalid_workspace_claims():
    """Reject non-string Drive workspace claim values before searching."""
    client = _client(
        _Orchestrator(_drive_result()),
        drive_workspace_id="workspace",
        drive_workspace_ids=(123,),
    )
    response = client.post(
        "/v1/search",
        json=_request(filters={"source_kinds": ["gdrive"]}),
    )

    assert response.status_code == 403


def test_drive_search_rejects_workspace_claims_outside_owned_scope():
    """Reject authenticated Drive claims outside ragdocs-owned scope."""
    client = _client(
        _Orchestrator(_drive_result(workspace_id="other")),
        drive_workspace_id="workspace",
        drive_workspace_ids=("other",),
    )
    response = client.post(
        "/v1/search",
        json=_request(filters={"source_kinds": ["gdrive"]}),
    )

    assert response.status_code == 403


def test_drive_search_ignores_request_workspace_context_for_authorization():
    """Authorize Drive only from bearer claims and owned scope membership."""
    orchestrator = _Orchestrator(_drive_result())
    client = _client(
        orchestrator,
        drive_workspace_id="workspace",
        drive_workspace_ids=("workspace",),
    )
    response = client.post(
        "/v1/search",
        json=_request(
            filters={
                "source_kinds": ["gdrive"],
                "workspace_id": "other",
                "project_ids": ["other"],
                "ranking_workspace_id": "other",
                "project_context": "other",
            }
        ),
    )

    assert response.status_code == 200
    assert response.json()["hits"][0]["workspace_id"] == "workspace"
    native_filters = orchestrator.calls[0]["filters"]
    assert native_filters["workspace_id"] == "workspace"
    assert "project_ids" not in native_filters
    assert "ranking_workspace_id" not in native_filters
    assert "project_context" not in native_filters


def test_drive_search_requires_scope_membership_on_each_record():
    """Filter Drive records without ragdocs-owned scope membership."""
    orchestrator = _Orchestrator(_drive_result(scope_memberships=()))
    client = _client(
        orchestrator,
        drive_workspace_id="workspace",
        drive_workspace_ids=("workspace",),
    )
    response = client.post(
        "/v1/search",
        json=_request(filters={"source_kinds": ["gdrive"]}),
    )

    assert response.status_code == 200
    assert response.json()["hits"] == []


def test_drive_acl_filters_before_top_k():
    """Do not let an unauthorized high-score Drive result consume top-k."""
    orchestrator = _RankedOrchestrator(
        (
            _drive_result(
                workspace_id="other",
                source_id="unauthorized",
                score=1.0,
            ),
            _drive_result(source_id="authorized", score=0.5),
        )
    )
    client = _client(
        orchestrator,
        drive_workspace_id="workspace",
        drive_workspace_ids=("workspace",),
    )
    response = client.post(
        "/v1/search",
        json=_request(
            top_k=1,
            filters={"source_kinds": ["gdrive"]},
        ),
    )

    assert response.status_code == 200
    assert [hit["source_id"] for hit in response.json()["hits"]] == ["authorized"]
    assert orchestrator.calls[0]["limit"] == 10


def test_authenticated_drive_only_search_preserves_identity_and_provenance():
    """
    Return an authenticated Drive hit with source identity and citations.
    """
    orchestrator = _Orchestrator(_drive_result(source_id="drive-only"))
    client = _client(
        orchestrator,
        drive_workspace_id="workspace",
        drive_workspace_ids=("workspace",),
    )

    response = client.post(
        "/v1/search",
        json=_request(
            request_id="drive-request",
            filters={"source_kinds": ["gdrive"]},
        ),
    )

    assert response.status_code == 200
    payload = response.json()
    hit = payload["hits"][0]
    assert hit["source_kind"] == "gdrive"
    assert hit["source_id"] == "drive-only"
    assert hit["workspace_id"] == "workspace"
    assert hit["uri"] == "gdrive://drive-only"
    assert hit["provenance"] == {
        "source": {
            "source_kind": "ragdocs",
            "source_id": "local",
            "workspace_id": None,
        },
        "request_id": "drive-request",
        "retrieval_method": "ragdocs-native",
        "details": {
            "native_provenance": {
                "strategies": ["keyword"],
            }
        },
    }
    assert orchestrator.calls[0]["filters"]["source_kinds"] == ["gdrive"]


def test_authenticated_drive_endpoint_accepts_authorized_request():
    """
    Accept a deterministic authenticated Drive request at the v1 boundary.
    """
    orchestrator = _Orchestrator(_drive_result(source_id="accepted-drive"))
    client = _client(
        orchestrator,
        drive_workspace_id="workspace",
        drive_workspace_ids=("workspace",),
    )

    response = client.post(
        "/v1/search",
        json=_request(
            request_id="authenticated-acceptance",
            filters={"source_kinds": ["gdrive"]},
        ),
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "authenticated-acceptance"
    assert response.json()["hits"][0]["source_id"] == "accepted-drive"


def test_authenticated_joint_search_filters_by_source_kind_at_the_boundary():
    """
    Preserve both local and Drive identities in an authenticated joint search.
    """
    orchestrator = _RankedOrchestrator(
        (
            _drive_result(source_id="drive-joint", score=0.9),
            _result(project_id="project-a", score=0.8),
        )
    )
    client = _client(
        orchestrator,
        drive_workspace_id="workspace",
        drive_workspace_ids=("workspace",),
    )

    response = client.post(
        "/v1/search",
        json=_request(
            top_k=2,
            filters={"source_kinds": ["note", "gdrive"]},
        ),
    )

    assert response.status_code == 200
    assert [hit["source_kind"] for hit in response.json()["hits"]] == [
        "gdrive",
        "note",
    ]
    assert [hit["source_id"] for hit in response.json()["hits"]] == [
        "drive-joint",
        "chunk-a",
    ]
    assert orchestrator.calls[0]["filters"]["source_kinds"] == [
        "note",
        "gdrive",
    ]


def test_drive_only_search_cannot_leak_markdown_or_git_records():
    """
    Apply the Drive source filter before results cross the v1 boundary.
    """
    orchestrator = _SourceFilteringOrchestrator(
        (
            _drive_result(source_id="drive-result", score=0.9),
            _result(source_id="markdown-result", score=0.8),
            _result(source_id="git-result", source_kind="git_commit", score=0.7),
        )
    )
    client = _client(
        orchestrator,
        drive_workspace_id="workspace",
        drive_workspace_ids=("workspace",),
    )

    response = client.post(
        "/v1/search",
        json=_request(top_k=3, filters={"source_kinds": ["gdrive"]}),
    )

    assert response.status_code == 200
    assert [hit["source_kind"] for hit in response.json()["hits"]] == ["gdrive"]
    assert orchestrator.calls[0]["filters"]["source_kinds"] == ["gdrive"]


def test_shared_source_ids_remain_distinct_across_drive_markdown_and_git():
    """
    Keep logical source kinds distinct when physical IDs overlap.
    """
    orchestrator = _SourceFilteringOrchestrator(
        (
            _drive_result(source_id="shared-id", score=0.9),
            _result(source_id="shared-id", score=0.8),
            _result(source_id="shared-id", source_kind="git_commit", score=0.7),
        )
    )
    client = _client(
        orchestrator,
        drive_workspace_id="workspace",
        drive_workspace_ids=("workspace",),
    )

    response = client.post(
        "/v1/search",
        json=_request(
            top_k=3,
            filters={"source_kinds": ["note", "git_commit", "gdrive"]},
        ),
    )

    assert response.status_code == 200
    assert {
        (hit["source_kind"], hit["source_id"])
        for hit in response.json()["hits"]
    } == {
        ("gdrive", "shared-id"),
        ("note", "shared-id"),
        ("git_commit", "shared-id"),
    }


def test_federation_search_honors_deadline():
    class SlowOrchestrator(_Orchestrator):
        async def search(self, query, *, limit, filters):
            await asyncio.sleep(0.05)
            return await super().search(query, limit=limit, filters=filters)

    client = _client(SlowOrchestrator())
    response = client.post(
        "/v1/search",
        json=_request(
            deadline_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        ),
    )

    assert response.status_code == 408


def test_federation_search_returns_gateway_timeout_for_slow_native_search():
    class SlowOrchestrator(_Orchestrator):
        async def search(self, query, *, limit, filters):
            await asyncio.sleep(0.05)
            return await super().search(query, limit=limit, filters=filters)

    client = _client(SlowOrchestrator())
    response = client.post(
        "/v1/search",
        json=_request(
            deadline_at=(datetime.now(UTC) + timedelta(seconds=0.01)).isoformat(),
        ),
    )

    assert response.status_code == 504
