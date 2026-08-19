from searchkernel.api import RecordSearchOutcome

from mcp_markdown_ragdocs.app.searchkernel_adapter import build_search_diagnostics


def test_build_search_diagnostics_exposes_degraded_outcome() -> None:
    """Expose missing records as degraded canonical-search diagnostics.

    Application transports need degradation information without receiving the
    SearchKernel outcome object itself.
    """
    outcome = RecordSearchOutcome(missing_record_ids=("note:doc:chunk",))

    diagnostics = build_search_diagnostics(outcome)

    assert diagnostics["degraded"] is True
    assert diagnostics["missing_record_ids"] == ["note:doc:chunk"]
