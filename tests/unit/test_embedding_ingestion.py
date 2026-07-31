from dataclasses import dataclass

import pytest

from mcp_markdown_ragdocs.ingestion import EmbeddingInput, embed_and_upsert


@dataclass
class _Provider:
    model_name: str = "test-model"

    def __post_init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text))] for text in texts]


class _Sink:
    def __init__(self, rejected: set[str] | None = None) -> None:
        self.rows: list[dict[str, object]] = []
        self.rejected = rejected or set()

    def upsert(self, **kwargs: object) -> bool:
        self.rows.append(kwargs)
        return str(kwargs["source_id"]) not in self.rejected


def _inputs(count: int) -> list[EmbeddingInput]:
    return [
        EmbeddingInput(
            source_kind="memory",
            source_id=f"memory-{index}",
            text=f"text-{index}",
            workspace_id="workspace",
            source_updated_at=f"version-{index}",
        )
        for index in range(count)
    ]


def test_embed_and_upsert_batches_inputs_and_preserves_source_metadata() -> None:
    provider = _Provider()
    sink = _Sink()

    result = embed_and_upsert(_inputs(5), provider=provider, sink=sink, batch_size=2)

    assert result.attempted == 5
    assert result.stored == 5
    assert result.rejected == 0
    assert result.batches == 3
    assert provider.calls == [["text-0", "text-1"], ["text-2", "text-3"], ["text-4"]]
    assert sink.rows[0] == {
        "source_kind": "memory",
        "source_id": "memory-0",
        "workspace_id": "workspace",
        "model_name": "test-model",
        "embedding": [6.0],
        "source_updated_at": "version-0",
    }


def test_embed_and_upsert_counts_rejected_writes() -> None:
    result = embed_and_upsert(
        _inputs(2),
        provider=_Provider(),
        sink=_Sink(rejected={"memory-1"}),
        batch_size=10,
    )

    assert result.stored == 1
    assert result.rejected == 1


def test_embed_and_upsert_empty_input_does_not_call_adapters() -> None:
    provider = _Provider()
    sink = _Sink()

    result = embed_and_upsert([], provider=provider, sink=sink, batch_size=2)

    assert result.attempted == 0
    assert result.stored == 0
    assert result.rejected == 0
    assert result.batches == 0
    assert provider.calls == []
    assert sink.rows == []


def test_embed_and_upsert_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        embed_and_upsert([], provider=_Provider(), sink=_Sink(), batch_size=0)


def test_embed_and_upsert_rejects_provider_count_mismatch_before_writes() -> None:
    class _ShortProvider(_Provider):
        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(texts)
            return []

    sink = _Sink()
    with pytest.raises(ValueError, match="returned 0 vectors for 2 inputs"):
        embed_and_upsert(_inputs(2), provider=_ShortProvider(), sink=sink, batch_size=2)
    assert sink.rows == []
