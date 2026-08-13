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
from mcp_markdown_ragdocs.gdrive.health import DriveScopeHealth, DriveSourceHealth, GDriveHealthStore
from mcp_markdown_ragdocs.server import create_app


def _request(**kwargs: object) -> dict[str, JsonValue]:
    return SearchRequest(
        "authentication",
        caller=CallerAuthorizationContext(
            caller_id="devkit",
            scopes=("search:read",),
        ),
        **cast(Any, kwargs),
    ).to_dict()


def _result(*, project_id: str = "project-a", score: float = 0.9):
    record = Record(
        source_kind="note",
        source_id="chunk-a",
        title="Authentication",
        body="Use the documented authentication flow.",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
        metadata={
            "chunk_id": "chunk-a",
            "doc_id": "doc-a",
            "file_path": "/docs/auth.md",
            "header_path": "Authentication",
            "project_id": project_id,
        },
        uri="file:///docs/auth.md#authentication",
        workspace_id=project_id,
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


def _client(orchestrator: _Orchestrator) -> TestClient:
    app = create_app()
    app.state.orchestrator = orchestrator
    return TestClient(app)


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
    app = create_app()
    client = TestClient(app)
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
    app = create_app()
    config = Config()
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

    client = TestClient(app)
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
    app = create_app()
    config = Config()
    config.gdrive.enabled = True
    config.gdrive.workspace_id = "workspace"
    app.state.config = config
    app.state.index_path = tmp_path

    health = TestClient(app).get("/v1/health").json()

    assert health["source_health"]["gdrive"]["status"] == "unavailable"
    assert health["source_health"]["gdrive"]["source"]["source_kind"] == "gdrive"
    assert health["source_health"]["gdrive"]["source"]["available"] is False


def test_federation_search_rejects_invalid_request():
    app = create_app()
    client = TestClient(app)
    response = client.post(
        "/v1/search",
        json={"query": "authentication", "contract_version": "v2"},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"


def test_federation_search_enforces_authentication_and_project_scope():
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

    assert missing_caller.status_code == 401
    assert missing_scope.status_code == 403


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

    assert response.status_code == 403


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
