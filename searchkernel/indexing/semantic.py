"""Synchronous contracts and planning for semantic indexing work."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from searchkernel.domain import Chunk

SemanticTier = Literal["coarse", "fine"]


def build_embedding_text(header_path: str, content: str) -> str:
    """Return the exact text used for chunk embeddings."""
    return f"{header_path}\n\n{content}" if header_path else content


@dataclass(frozen=True, slots=True)
class EncoderFingerprint:
    """Stable model settings used to namespace reusable vectors."""

    model: str
    version: str = ""
    normalization: str = ""
    query_instruction: str = ""
    text_instruction: str = ""
    dimension: int = 0

    @property
    def namespace(self) -> str:
        payload = {
            "model": self.model,
            "version": self.version,
            "normalization": self.normalization,
            "query_instruction": self.query_instruction,
            "text_instruction": self.text_instruction,
            "dimension": self.dimension,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def embedding_identity(text: str, encoder_namespace: str) -> str:
    """Hash the model input and encoder namespace into one cache identity."""
    payload = json.dumps(
        {"namespace": encoder_namespace, "text": text},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def semantic_input_for_chunk(
    chunk: Chunk,
    encoder_namespace: str = "",
    *,
    tier: SemanticTier = "fine",
    priority: int = 0,
) -> SemanticInput:
    """Build the canonical semantic input for one chunk."""
    text = build_embedding_text(chunk.metadata.get("header_path", ""), chunk.content)
    return SemanticInput(
        source_id=chunk.chunk_id,
        text=text,
        content_hash=embedding_identity(text, encoder_namespace),
        tier=tier,
        priority=priority,
    )


@dataclass(frozen=True, slots=True)
class SemanticInput:
    source_id: str
    text: str
    content_hash: str
    tier: SemanticTier
    priority: int


@dataclass(frozen=True, slots=True)
class SemanticWorkUnit:
    content_hash: str
    text: str
    source_ids: tuple[str, ...]
    tier: SemanticTier
    priority: int


class EmbeddingCache(Protocol):
    def get_many(
        self, content_hashes: Sequence[str]
    ) -> Mapping[str, Sequence[float]]: ...

    def put_many(self, vectors: Mapping[str, Sequence[float]]) -> None: ...


class EmbeddingEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class LlamaIndexEmbeddingCacheAdapter:
    """Adapts our content-hash EmbeddingCache to llama_index's BaseKVStore shape.

    llama_index's ``BaseEmbedding.embeddings_cache`` field expects a
    ``get(key, collection)`` / ``put(key, val, collection)`` KV-store shape and
    routes both single and batched embedding calls (the same path
    ``VectorStoreIndex.insert_nodes`` uses) through it automatically. Setting
    this on an embedding model's ``embeddings_cache`` field is the only
    integration point needed - no wrapper subclass of the embedding model
    itself is required or safe, since embedding models expose more methods
    (batch, async, query variants) than a hand-written wrapper class can
    reliably proxy.
    """

    def __init__(self, cache: EmbeddingCache, encoder_namespace: str) -> None:
        self.cache = cache
        self.encoder_namespace = encoder_namespace

    def get(self, key: str, collection: str = "embeddings") -> dict[str, list[float]] | None:
        content_hash = embedding_identity(key, self.encoder_namespace)
        cached = self.cache.get_many([content_hash])
        vector = cached.get(content_hash)
        if vector is None:
            return None
        return {content_hash: list(vector)}

    def put(
        self, key: str, val: dict[str, Sequence[float]], collection: str = "embeddings"
    ) -> None:
        content_hash = embedding_identity(key, self.encoder_namespace)
        vector = next(iter(val.values()))
        self.cache.put_many({content_hash: vector})


class VectorMaterializer(Protocol):
    def materialize(
        self, source_id: str, vector: Sequence[float], semantic_input: SemanticInput
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class SemanticProgress:
    completed: int
    total: int
    cache_hits: int
    cache_misses: int


@dataclass(frozen=True, slots=True)
class SemanticPlan:
    hits: Mapping[str, Sequence[float]]
    misses: tuple[SemanticWorkUnit, ...]
    groups: Mapping[str, tuple[SemanticInput, ...]]


class SemanticWorkPlanner:
    """Plan and execute deterministic, deduplicated semantic work."""

    def __init__(self, encoder_namespace: str, dimension: int = 0):
        self.encoder_namespace = encoder_namespace
        self.dimension = dimension

    def prepare(self, inputs: Sequence[SemanticInput]) -> list[SemanticInput]:
        return [
            SemanticInput(
                source_id=item.source_id,
                text=item.text,
                content_hash=embedding_identity(item.text, self.encoder_namespace),
                tier=item.tier,
                priority=item.priority,
            )
            for item in inputs
        ]

    def group(
        self, inputs: Sequence[SemanticInput]
    ) -> dict[str, tuple[SemanticInput, ...]]:
        grouped: dict[str, list[SemanticInput]] = {}
        for item in inputs:
            grouped.setdefault(item.content_hash, []).append(item)
        return {
            key: tuple(sorted(value, key=lambda item: item.source_id))
            for key, value in grouped.items()
        }

    def plan(
        self, inputs: Sequence[SemanticInput], cache: EmbeddingCache
    ) -> SemanticPlan:
        groups = self.group(self.prepare(inputs))
        hashes = sorted(groups)
        cached = cache.get_many(hashes)
        hits: dict[str, Sequence[float]] = {}
        misses: list[SemanticWorkUnit] = []
        for content_hash in hashes:
            items = groups[content_hash]
            vector = cached.get(content_hash)
            if vector is not None and self._valid_vector(vector):
                hits[content_hash] = vector
            else:
                first = min(items, key=lambda item: (item.priority, item.source_id))
                misses.append(
                    SemanticWorkUnit(
                        content_hash=content_hash,
                        text=first.text,
                        source_ids=tuple(sorted(item.source_id for item in items)),
                        tier=min(
                            (item.tier for item in items),
                            key=lambda tier: (tier != "coarse", tier),
                        ),
                        priority=min(item.priority for item in items),
                    )
                )
        misses.sort(
            key=lambda work: (work.priority, work.tier != "coarse", work.content_hash)
        )
        return SemanticPlan(hits=hits, misses=tuple(misses), groups=groups)

    def execute(
        self,
        inputs: Sequence[SemanticInput],
        cache: EmbeddingCache,
        encoder: EmbeddingEncoder,
        materializer: VectorMaterializer,
        progress: Callable[[SemanticProgress], None] | None = None,
    ) -> SemanticProgress:
        plan = self.plan(inputs, cache)
        total = len(plan.groups)
        completed = 0

        for content_hash, vector in plan.hits.items():
            for item in plan.groups[content_hash]:
                self._materialize(materializer, item, vector)
            completed += 1
            self._report(progress, completed, total, len(plan.hits), len(plan.misses))

        if plan.misses:
            vectors = encoder.encode([work.text for work in plan.misses])
            if len(vectors) != len(plan.misses):
                raise ValueError(
                    "encoder returned a different number of vectors than texts"
                )
            to_cache: dict[str, Sequence[float]] = {}
            for work, vector in zip(plan.misses, vectors, strict=True):
                self._validate_vector(vector)
                to_cache[work.content_hash] = vector
                for item in plan.groups[work.content_hash]:
                    self._materialize(materializer, item, vector)
                completed += 1
                self._report(
                    progress, completed, total, len(plan.hits), len(plan.misses)
                )
            cache.put_many(to_cache)
        return SemanticProgress(completed, total, len(plan.hits), len(plan.misses))

    def _valid_vector(self, vector: Sequence[float]) -> bool:
        try:
            self._validate_vector(vector)
        except ValueError:
            return False
        return True

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if not vector or any(not math.isfinite(float(value)) for value in vector):
            raise ValueError("embedding vector must contain finite values")
        if self.dimension and len(vector) != self.dimension:
            raise ValueError("embedding vector has unexpected dimension")

    @staticmethod
    def _materialize(
        materializer: VectorMaterializer, item: SemanticInput, vector: Sequence[float]
    ) -> None:
        materializer.materialize(item.source_id, vector, item)

    @staticmethod
    def _report(
        callback: Callable[[SemanticProgress], None] | None,
        completed: int,
        total: int,
        cache_hits: int,
        cache_misses: int,
    ) -> None:
        if callback is not None:
            callback(SemanticProgress(completed, total, cache_hits, cache_misses))
