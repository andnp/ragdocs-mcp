# Contract: Search, Filtering, and 1.2x Project Uplift

**Status:** Not started — verified against code on 2026-03-17
**Date:** 2026-03-16
**Related:** `docs/plans/02-global-daemon-huey-and-soft-projects.md`, `docs/plans/07-global-store-and-schema-contract.md`, `docs/specs/17-search-overhaul.md`

## Executive Summary

This document defines the ranking contract for soft project semantics in Ragdocs V2. The corpus remains global-first. Active project context may apply a conservative `1.2x` uplift to matching results, but it must not silently hard-filter the corpus by default.

## Verified Implementation Status (2026-03-17)

- No verified project metadata pipeline currently stamps `project_id` onto indexed content.
- No search/ranking code currently applies the fixed $1.2\times$ uplift described here.
- Project behavior is still driven primarily by path selection and per-project index resolution in `src/config.py`, which is the opposite of this contract's intended storage/ranking split.

This contract remains future work.

## Mandatory Reference Rule

Future agents should inspect `/home/andy/Projects/personal/mcp-memory/src/mcp_memory/relational/search.py` before changing ranking semantics. The direct implementation will differ, but the ranking principle is the same: metadata-aware boost, not metadata-based partitioning.

## Goals

1. preserve global-first search
2. improve local relevance when active project context is known
3. keep scoring behavior predictable and bounded
4. avoid hidden filtering surprises

## Non-Goals

1. introduce per-project isolated search indices
2. make project context a hard filter by default
3. redesign the entire ranking stack in the same tranche

## Global-First Rule

Document search and git history search must evaluate the full indexed corpus unless the caller explicitly requests a project filter.

Project context affects **score**, not **eligibility**, by default.

## Base Ranking Model

Ragdocs already fuses strategy results via weighted scoring / RRF-like logic. V2 should preserve that base pipeline and append one additional multiplier for project match.

### Project Match Multiplier

For the first implementation pass:

$$
score_{final} = score_{base} \times 1.2
$$

when the result's `project_id` matches the active project context.

Otherwise:

$$
score_{final} = score_{base}
$$

## Ranking Invariants

1. no active project context means no uplift
2. non-matching results remain eligible
3. explicit project filters may restrict eligibility, but only when requested
4. project uplift applies after base fusion, not before candidate generation
5. uplift is bounded and fixed at `1.2x` for v1

## Filter Semantics

### Default Behavior

- no project filter
- optional project uplift
- full-corpus results remain visible

### Explicit Filter Behavior

If a caller explicitly requests a project filter, the system may restrict results to matching `project_id` values. That behavior must be visible in the request contract and never inferred silently from ambient context.

## Required Data Contract

Results participating in uplift must carry or resolve:

- `project_id`
- `doc_id`
- `chunk_id` where applicable

If `project_id` is missing, the result is treated as non-matching and receives multiplier `1.0`.

## Current Code Impact

### Likely Edit Points

- `src/search/fusion.py`
- `src/search/orchestrator.py`
- `src/search/score_pipeline.py`
- `src/indexing/manager.py`
- `src/storage/db.py`
- `src/config.py`

### Likely Config Addition

- `project_boost_multiplier: float = 1.2`

The first implementation may keep this fixed in code if that reduces risk, but the contract assumes eventual config ownership.

## Git History Search Rule

Git history search follows the same semantics:

- project context may upweight commits associated with the active project
- project context must not hard-filter commit results by default

## Evaluation Rules

At minimum, verification should show:

1. matching-project results rise relative to non-matching peers when base scores are similar
2. non-matching but strongly relevant results still appear
3. no-project queries behave the same as before the feature
4. explicit project filters behave differently and intentionally

## Failure Cases To Avoid

1. silently discarding cross-project results
2. multiplying score before fusion and distorting candidate balance
3. allowing project context to affect daemon identity or storage selection
4. applying uplift to results with ambiguous or missing project metadata as though they matched

## Acceptance Criteria

1. default search stays global-first
2. active project context produces a fixed `1.2x` uplift for matching results only
3. explicit filtering remains opt-in and transparent
4. document and git search use the same soft-project principle