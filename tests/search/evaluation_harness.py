from __future__ import annotations

from dataclasses import dataclass
from math import log2
from pathlib import Path

from src.config import (
    ChunkingConfig,
    Config,
    IndexingConfig,
    LLMConfig,
    ProjectConfig,
    SearchConfig,
)
from src.indexing.manager import IndexManager
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import EmbeddingModel, VectorIndex
from src.search.orchestrator import SearchOrchestrator
from src.search.path_utils import compute_doc_id
from src.storage.db import DatabaseManager

FIXTURE_CORPUS_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "search_eval"


@dataclass(frozen=True)
class SearchEvaluationCase:
    case_id: str
    query: str
    relevant_paths: tuple[str, ...]
    expected_top1_path: str | None = None
    required_hits_at_k: tuple[tuple[int, int], ...] = ((1, 1),)
    project_context: str | None = None
    project_filter: tuple[str, ...] | None = None
    top_n: int = 5


@dataclass(frozen=True)
class DocumentRankingMetrics:
    reciprocal_rank: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    ndcg_at_3: float
    ndcg_at_5: float


@dataclass(frozen=True)
class SearchEvaluationCaseResult:
    case: SearchEvaluationCase
    relevant_doc_ids: tuple[str, ...]
    ranked_doc_ids: tuple[str, ...]
    ranked_paths: tuple[str, ...]
    metrics: DocumentRankingMetrics

    def top_path(self) -> str | None:
        if not self.ranked_paths:
            return None
        return self.ranked_paths[0]

    def hit_count_at(self, k: int) -> int:
        relevant = set(self.relevant_doc_ids)
        return sum(1 for doc_id in self.ranked_doc_ids[:k] if doc_id in relevant)


@dataclass(frozen=True)
class SearchEvaluationAggregate:
    query_count: int
    mrr: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    ndcg_at_3: float
    ndcg_at_5: float


@dataclass(frozen=True)
class SearchEvaluationReport:
    case_results: tuple[SearchEvaluationCaseResult, ...]
    aggregate: SearchEvaluationAggregate

    def expectation_failures(self) -> list[str]:
        failures: list[str] = []
        for result in self.case_results:
            expected_top1 = result.case.expected_top1_path
            observed_top1 = result.top_path()
            if expected_top1 is not None and observed_top1 != expected_top1:
                failures.append(
                    f"{result.case.case_id}: expected top-1 {expected_top1}, got {observed_top1}; ranked={list(result.ranked_paths)}"
                )

            for k, minimum_hits in result.case.required_hits_at_k:
                observed_hits = result.hit_count_at(k)
                if observed_hits < minimum_hits:
                    failures.append(
                        f"{result.case.case_id}: expected at least {minimum_hits} judged docs in top-{k}, got {observed_hits}; ranked={list(result.ranked_paths[:k])}"
                    )

        return failures

    def format_summary(self) -> str:
        aggregate = self.aggregate
        lines = [
            (
                "aggregate: "
                f"queries={aggregate.query_count}, "
                f"MRR={aggregate.mrr:.3f}, "
                f"Recall@1={aggregate.recall_at_1:.3f}, "
                f"Recall@3={aggregate.recall_at_3:.3f}, "
                f"Recall@5={aggregate.recall_at_5:.3f}, "
                f"NDCG@3={aggregate.ndcg_at_3:.3f}, "
                f"NDCG@5={aggregate.ndcg_at_5:.3f}"
            )
        ]
        for result in self.case_results:
            lines.append(
                (
                    f"- {result.case.case_id}: "
                    f"RR={result.metrics.reciprocal_rank:.3f}, "
                    f"Recall@3={result.metrics.recall_at_3:.3f}, "
                    f"Recall@5={result.metrics.recall_at_5:.3f}, "
                    f"top={list(result.ranked_paths[:3])}"
                )
            )
        return "\n".join(lines)


class SearchEvaluationHarness:
    def __init__(
        self,
        *,
        corpus_root: Path,
        orchestrator: SearchOrchestrator,
        path_to_doc_id: dict[str, str],
        doc_id_to_path: dict[str, str],
        cases: tuple[SearchEvaluationCase, ...],
    ):
        self.corpus_root = corpus_root
        self.orchestrator = orchestrator
        self.path_to_doc_id = path_to_doc_id
        self.doc_id_to_path = doc_id_to_path
        self.cases = cases

    async def evaluate(self) -> SearchEvaluationReport:
        case_results: list[SearchEvaluationCaseResult] = []

        for case in self.cases:
            chunk_results, _compression_stats, _strategy_stats = await self.orchestrator.query(
                case.query,
                top_k=max(10, case.top_n * 2),
                top_n=case.top_n,
                project_context=case.project_context,
                project_filter=list(case.project_filter)
                if case.project_filter is not None
                else None,
            )
            ranked_doc_ids = tuple(
                _dedupe_doc_ids([result.doc_id for result in chunk_results])
            )
            ranked_paths = tuple(
                self.doc_id_to_path.get(doc_id, f"<unknown:{doc_id}>")
                for doc_id in ranked_doc_ids
            )
            relevant_doc_ids = tuple(
                self.path_to_doc_id[relative_path] for relative_path in case.relevant_paths
            )
            metrics = compute_ranking_metrics(ranked_doc_ids, relevant_doc_ids)
            case_results.append(
                SearchEvaluationCaseResult(
                    case=case,
                    relevant_doc_ids=relevant_doc_ids,
                    ranked_doc_ids=ranked_doc_ids,
                    ranked_paths=ranked_paths,
                    metrics=metrics,
                )
            )

        return SearchEvaluationReport(
            case_results=tuple(case_results),
            aggregate=_aggregate_metrics(case_results),
        )


def build_search_evaluation_harness(
    tmp_path: Path,
    embedding_model: EmbeddingModel,
    *,
    corpus_root: Path = FIXTURE_CORPUS_ROOT,
) -> SearchEvaluationHarness:
    corpus_root = corpus_root.resolve()
    index_path = tmp_path / "search_eval_index"

    config = Config(
        indexing=IndexingConfig(
            documents_path=str(corpus_root),
            index_path=str(index_path),
        ),
        search=SearchConfig(),
        chunking=ChunkingConfig(),
        llm=LLMConfig(embedding_model="BAAI/bge-small-en-v1.5"),
        projects=[
            ProjectConfig(name="alpha", path=str((corpus_root / "alpha").resolve())),
            ProjectConfig(name="beta", path=str((corpus_root / "beta").resolve())),
        ],
    )

    vector = VectorIndex(
        embedding_model=embedding_model,
        embedding_workers=1,
        torch_num_threads=1,
    )
    keyword = KeywordIndex(DatabaseManager(index_path / "keyword.db"))
    graph = GraphStore(DatabaseManager(index_path / "graph.db"))
    manager = IndexManager(config, vector, keyword, graph)

    corpus_files = sorted(corpus_root.rglob("*.md"))
    path_to_doc_id: dict[str, str] = {}
    doc_id_to_path: dict[str, str] = {}

    for file_path in corpus_files:
        manager.index_document(str(file_path))
        relative_path = str(file_path.relative_to(corpus_root))
        doc_id = compute_doc_id(file_path.resolve(), corpus_root)
        path_to_doc_id[relative_path] = doc_id
        doc_id_to_path[doc_id] = relative_path

    manager.persist()
    orchestrator = SearchOrchestrator(
        vector,
        keyword,
        graph,
        config,
        manager,
        documents_path=corpus_root,
    )

    return SearchEvaluationHarness(
        corpus_root=corpus_root,
        orchestrator=orchestrator,
        path_to_doc_id=path_to_doc_id,
        doc_id_to_path=doc_id_to_path,
        cases=SEARCH_EVALUATION_CASES,
    )


def compute_ranking_metrics(
    ranked_doc_ids: tuple[str, ...],
    relevant_doc_ids: tuple[str, ...],
) -> DocumentRankingMetrics:
    relevant = set(relevant_doc_ids)
    reciprocal_rank = 0.0
    for index, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant:
            reciprocal_rank = 1.0 / index
            break

    return DocumentRankingMetrics(
        reciprocal_rank=reciprocal_rank,
        recall_at_1=_recall_at_k(ranked_doc_ids, relevant, 1),
        recall_at_3=_recall_at_k(ranked_doc_ids, relevant, 3),
        recall_at_5=_recall_at_k(ranked_doc_ids, relevant, 5),
        ndcg_at_3=_ndcg_at_k(ranked_doc_ids, relevant, 3),
        ndcg_at_5=_ndcg_at_k(ranked_doc_ids, relevant, 5),
    )


def _aggregate_metrics(
    case_results: list[SearchEvaluationCaseResult],
) -> SearchEvaluationAggregate:
    count = len(case_results)
    if count == 0:
        return SearchEvaluationAggregate(
            query_count=0,
            mrr=0.0,
            recall_at_1=0.0,
            recall_at_3=0.0,
            recall_at_5=0.0,
            ndcg_at_3=0.0,
            ndcg_at_5=0.0,
        )

    return SearchEvaluationAggregate(
        query_count=count,
        mrr=sum(result.metrics.reciprocal_rank for result in case_results) / count,
        recall_at_1=sum(result.metrics.recall_at_1 for result in case_results) / count,
        recall_at_3=sum(result.metrics.recall_at_3 for result in case_results) / count,
        recall_at_5=sum(result.metrics.recall_at_5 for result in case_results) / count,
        ndcg_at_3=sum(result.metrics.ndcg_at_3 for result in case_results) / count,
        ndcg_at_5=sum(result.metrics.ndcg_at_5 for result in case_results) / count,
    )


def _recall_at_k(ranked_doc_ids: tuple[str, ...], relevant_doc_ids: set[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0
    hits = sum(1 for doc_id in ranked_doc_ids[:k] if doc_id in relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def _ndcg_at_k(ranked_doc_ids: tuple[str, ...], relevant_doc_ids: set[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            dcg += 1.0 / log2(rank + 1)

    ideal_hits = min(k, len(relevant_doc_ids))
    if ideal_hits == 0:
        return 0.0

    ideal_dcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def _dedupe_doc_ids(doc_ids: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for doc_id in doc_ids:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        deduped.append(doc_id)
    return deduped


SEARCH_EVALUATION_CASES = (
    SearchEvaluationCase(
        case_id="exact_heading",
        query="Authentication Overview",
        relevant_paths=("alpha/docs/authentication-overview.md",),
        expected_top1_path="alpha/docs/authentication-overview.md",
    ),
    SearchEvaluationCase(
        case_id="conceptual_refresh_tokens",
        query="how long do refresh tokens last",
        relevant_paths=("alpha/docs/token-lifecycle.md",),
        expected_top1_path="alpha/docs/token-lifecycle.md",
    ),
    SearchEvaluationCase(
        case_id="graph_adjacent_api_auth",
        query="bearer tokens protected endpoints",
        relevant_paths=(
            "alpha/docs/api-authentication.md",
            "alpha/docs/authentication-overview.md",
        ),
        expected_top1_path="alpha/docs/api-authentication.md",
        required_hits_at_k=((1, 1), (5, 2)),
    ),
    SearchEvaluationCase(
        case_id="artifact_fileish",
        query="mcp_server.py list_tools call_tool",
        relevant_paths=("alpha/docs/src-mcp_server-py.md",),
        expected_top1_path="alpha/docs/src-mcp_server-py.md",
    ),
    SearchEvaluationCase(
        case_id="scoped_project_context",
        query="Project Rollout Checklist",
        relevant_paths=("alpha/docs/project-rollout-checklist.md",),
        expected_top1_path="alpha/docs/project-rollout-checklist.md",
        project_context="alpha",
    ),
)
