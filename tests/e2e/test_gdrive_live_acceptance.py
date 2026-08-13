"""Optional live acceptance boundary for the authenticated Drive federation."""

import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest


@pytest.mark.e2e
def test_live_gdrive_federation_acceptance_boundary() -> None:
    """
    Exercise a deployed v1 Drive search when explicit live credentials exist.

    Set RAGDOCS_GDRIVE_LIVE_URL to the ragdocs base URL and
    RAGDOCS_GDRIVE_LIVE_TOKEN to its federation bearer token. The test skips
    without both values so ordinary deterministic test runs never use a live
    network or implicit credentials.
    """
    base_url = os.environ.get("RAGDOCS_GDRIVE_LIVE_URL")
    token = os.environ.get("RAGDOCS_GDRIVE_LIVE_TOKEN")
    if not base_url or not token:
        pytest.skip(
            "set RAGDOCS_GDRIVE_LIVE_URL and RAGDOCS_GDRIVE_LIVE_TOKEN "
            "for the optional live Drive acceptance boundary"
        )

    payload = {
        "query": "drive",
        "top_k": 5,
        "filters": {"source_kinds": ["gdrive"]},
        "source_selection": [],
        "caller": None,
        "deadline_at": None,
        "cancellation_id": None,
        "request_id": "live-gdrive-acceptance",
        "trace_id": "",
        "contract_version": "v1",
    }
    request = Request(
        f"{base_url.rstrip('/')}/v1/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            body = json.load(response)
            status = response.status
    except URLError as error:
        pytest.fail(f"live Drive federation endpoint was unreachable: {error}")

    assert status == 200
    assert body["contract_version"] == "v1"
    assert all(hit["source_kind"] == "gdrive" for hit in body["hits"])
