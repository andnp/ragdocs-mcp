---
name: dogfood-live-corpus
description: 'Use a subagent to probe the live Ragdocs corpus, discover search-quality gaps, and produce an evidence-based improvement report.'
---

# Dogfood the Live Corpus

## When to Use
- Evaluate search, indexing, ranking, graph, or searchkernel changes against real data.
- Run a recurring search-quality feedback loop without consuming the parent agent's context.
- Generate improvement hypotheses before implementation.

## Parent Agent Setup
1. Refresh the global editable install:
   ```bash
   uv tool install --editable --force /home/andy/Projects/personal/mcp-markdown-ragdocs
   ```
2. Restart the global daemon with a corpus-sized timeout:
   ```bash
   mcp-markdown-ragdocs daemon restart --timeout 120
   ```
3. Wait for readiness:
   ```bash
   mcp-markdown-ragdocs daemon status
   mcp-markdown-ragdocs index stats
   ```
   Continue only when the daemon is running and the index reports
   `Index state: ready` with `Remaining estimate: 0`.
4. Delegate the audit to one general-purpose subagent using the
   `gpt-5.6-luna` model. The subagent must work read-only and run queries
   sequentially, never concurrently while the daemon is starting or rebuilding.

## Lifecycle Validity Gates
Before the first query, capture JSON snapshots from `daemon status --json`,
`queue status --json`, and `index stats --json`. Record the daemon `pid`, `status`, and `lifecycle`,
plus index `index_state.status`, `remaining_estimate`, `pending_count`,
`running_count`, and `failed_count`.

Require the pre-query snapshot to show one ready daemon PID, a ready index,
`remaining_estimate: 0`, and no pending or running tasks. After the query
portfolio, capture the same three snapshots again. The run is invalid if the
PID changes, readiness regresses, or any new pending/running/failed work
appears without an explicitly documented corpus change. A transient query
failure does not waive this gate: capture the post-query state and report any
replacement, rebuild, or queue churn before interpreting search results.

Use `--json` for all six snapshots so PID, readiness, and queue/task values can
be compared mechanically. Preserve the snapshots with the audit report.

## Subagent Brief
Give the subagent the repository path and this complete instruction:

> You are auditing the live Ragdocs search system. Do not edit files, install
> packages, restart unrelated services, commit, or push. Confirm daemon status
> and corpus readiness before searching. Use `mcp-markdown-ragdocs query ... 
> --json` and inspect `file_path`, `header_path`, `score`, and
> `provenance.strategies`. Use `--project-filter` when a repository-scoped
> comparison is useful. Run the query categories and gap checks below. After
> the predetermined checks, invent at least three hypotheses of your own,
> choose queries that could falsify them, and record the results even when
> they do not reveal a gap. Return the structured report format below with
> concrete paths, ranks, scores, and proposed ownership.

## Required Gap Checks
Look for evidence of:
- lexical misses from stopwords, morphology, punctuation, spelling, or
  natural-language phrasing;
- semantic misses from paraphrases, synonyms, vague descriptions, or
  terminology changes;
- ranking noise, duplicate chunks, parent/child overrepresentation, and
  weak title or heading handling;
- graph failures on links, neighbors, multi-hop questions, and cross-document
  context;
- project or source-scope leakage, especially Git history outranking documents;
- poor abstention on unrelated or impossible questions;
- stale or missing records after recent file changes or daemon restarts;
- artifact/code queries involving filenames, symbols, paths, and exact tokens;
- latency, degraded-mode responses, strategy failures, or unstable results.

## Query Portfolio
Run several queries from each applicable category:
1. Exact title or heading.
2. Exact fact containing distinctive terms.
3. Natural-language question.
4. Paraphrase using different vocabulary.
5. Synonym or morphology variation.
6. Multi-hop or graph-neighbor question.
7. Filename, symbol, path, or code-shaped query.
8. Ambiguous query with multiple plausible documents.
9. Typo or punctuation variation.
10. Project-scoped and unscoped versions of the same query.
11. Recent-change or Git-history query when git indexing is enabled.
12. Unrelated no-answer query.

For each query, record the intended target or explain why no result should
match. Query the global corpus when measuring real user behavior, then repeat
important cases with an explicit project filter to separate ranking problems
from corpus-scope noise.

## Hypothesis Exploration
After the required portfolio, the subagent must:
- propose at least three novel hypotheses based on observed results;
- select one or more falsifying queries for each hypothesis;
- vary query wording, scope, and source filters where available;
- pursue at least one unexpected lead rather than only confirming the initial
  diagnosis;
- distinguish a reproducible gap from a one-off surprising result.

Do not hide failures, inconclusive probes, or cases where the intended document
exists but ranks below noise.

## Required Report
Return a concise but evidence-rich report with these sections:

1. **Environment**: checkout, installed searchkernel version, daemon lifecycle,
   corpus counts, scope, and readiness.
2. **Executive summary**: strongest wins, most important gaps, and confidence.
3. **Query evidence**: a table with query, scope, intended target, observed
   top results, target rank, scores, strategies, and pass/fail.
4. **Gap catalogue**: one entry per gap with severity, confidence,
   reproducibility, concrete evidence, and likely root cause.
5. **Hypotheses chased**: novel hypotheses, falsifying queries, and outcomes.
6. **Ownership**: classify each proposed fix as `searchkernel`, `ragdocs`, or
   `evaluation/operations`, with a short rationale.
7. **Prioritized follow-up**: quick wins, deeper changes, and new regression
   cases to add.
8. **Limitations**: missing controls, ambiguous judgments, unavailable
   request-level filters, and anything not tested.

The subagent must not propose a fix solely from a score. Tie every finding to
the returned content, rank, strategy provenance, scope, or daemon behavior.
