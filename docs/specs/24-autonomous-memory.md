# Feature Spec 24: Autonomous Memory Consolidation

**Status:** Draft
**Related:** [16-memory-management.md](16-memory-management.md), [23-concurrency-huey.md](23-concurrency-huey.md)

## 1. Overview
The current Memory System (System 2) requires the AI agent to explicitly plan, format, and organize markdown files. This cognitive load disrupts active coding tasks and leads to memory decay over time. 

This specification introduces a **Dual-System Memory Architecture**:
- **System 1 (Cache):** A fast, append-only scratchpad for raw thoughts and facts.
- **System 2 (Deep):** The existing, structured markdown memory graph.

Crucially, it introduces an **Autonomous Background Agent** that uses local CLI AI tools (like `gemini` or `gh copilot`) to periodically consolidate System 1 into System 2, and defragment System 2 autonomously, keeping the context window sharp and organized.

## 2. System 1: The Fast Cache
We introduce a new, extremely low-friction MCP tool for active agents.

### Tool: `record_thought(text: str)`
- **Functionality**: Appends the text, along with a timestamp, to an SQLite table `system1_journal`.
- **Purpose**: Allows the active agent to quickly stash context, bug findings, or plans without deciding on filenames, tags, or structure.

## 3. The Autonomous Agent (Huey Tasks)
The heavy lifting is performed by background tasks scheduled via Huey. To execute LLM prompts without requiring API keys, we rely on a provider layer that wraps existing authenticated CLI tools on the user's machine.

### 3.1. CLI Provider Layer (`src/memory/providers.py`)
Abstract interface for AI execution:
```python
class AIAssistantProvider(Protocol):
    def generate_json(self, prompt: str, schema: dict) -> dict:
        ...
```
- **Implementations**:
  - `GeminiCLIProvider`: Executes `subprocess.run(["gemini", "ask", "--json", ...])`.
  - `CopilotCLIProvider`: Executes `gh copilot suggest ...`.

### 3.2. Fast Consolidation (The Flusher)
**Trigger**: When `system1_journal` reaches > 10 unmerged entries (checked periodically by Huey).
**Workflow**:
1.  **Extract**: Pull the unprocessed entries from System 1.
2.  **Context RAG**: Run a lightweight keyword/vector search on System 2 to find existing memory files related to the new thoughts.
3.  **Prompt the AI**: Provide the System 1 entries and the relevant System 2 context. 
4.  **Execute**: The AI returns a strict JSON payload mapping out `create`, `update`, or `delete` operations for System 2 markdown files.
5.  **Cleanup**: Mark the System 1 entries as `merged`.

### 3.3. Slow Consolidation (The Defragmenter)
**Trigger**: A Huey cron task running **every 6 hours**.
**Workflow**: To prevent context window blowouts, this task uses a "Strategy Roulette" to select a small batch of memories to review.

**Strategy Roulette**:
1.  **Semantic Cluster (50% prob)**: Queries the vector DB for clusters of highly similar memories. Goal: Merge redundant facts discovered across multiple sessions.
2.  **Random Walk (30% prob)**: Selects 5-10 random memory files. Goal: General garbage collection and finding obscure contradictions.
3.  **Temporal Decay (20% prob)**: Selects the oldest, least recently accessed memories. Goal: Prune transient, obsolete information.

**The Prompt**: Provide the selected batch of memories to the AI. Ask it to compress, merge, and delete redundant or obsolete info.

## 4. Safety Mechanisms & Escape Hatches

Because the agent runs autonomously and destructively (deleting/updating files), rigid safety nets are required.

### 4.1. The Escape Hatch
The AI must never be forced to make changes. The JSON schema explicitly supports returning an empty operations list if the memory graph is already optimal.
```json
{
  "reasoning": "The retrieved memories are concise and non-redundant. No actions required.",
  "operations": []
}
```

### 4.2. Strict JSON Schema
The CLI Provider MUST enforce a JSON schema return format. If parsing fails, the Huey task aborts and leaves the system state untouched.

### 4.3. Soft Deletes (The Trash)
When an autonomous task executes an `update` or `delete` operation via `src/memory/tools.py`, the previous version of the file is always moved to `.memories/.trash/filename.timestamp.md`. No data is permanently destroyed by the AI.

### 4.4. The Audit Log
Every consolidation cycle writes a brief summary to `system/consolidation-log.md` (e.g., *"[2026-03-01 14:00] Strategy: Semantic Cluster. Merged 'auth-bug' into 'auth-spec'. Deleted 'auth-bug'. Reason: Consolidated redundant token expiry notes."*)

## 5. Implementation Steps
1.  **Provider Layer**: Implement `AIAssistantProvider` interface and the `GeminiCLIProvider`.
2.  **System 1 Tools**: Add `record_thought` MCP tool and backing SQLite table.
3.  **Fast Task**: Create the Huey task for flushing System 1.
4.  **Slow Task**: Implement the vector clustering/random walk sampling logic and the 6-hour cron Huey task.
5.  **Integration**: Wire the execution logic to the existing memory `Manager` to ensure all graph links and indices are updated when files are autonomously modified.