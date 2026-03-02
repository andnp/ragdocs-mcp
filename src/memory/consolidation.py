"""Memory consolidation — converting journal entries into refined memories."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from src.memory.journal import JournalEntry, System1Journal

if TYPE_CHECKING:
    from src.memory.providers import AIProvider

logger = logging.getLogger(__name__)


def _slugify(text: str, max_len: int = 40) -> str:
    """Create a filesystem-safe slug from text."""
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:max_len] or "untitled"


def _generate_filename(entries: list[JournalEntry]) -> str:
    """Generate a memory filename from journal entries."""
    first_content = entries[0].content[:60] if entries else "thought"
    slug = _slugify(first_content)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{slug}-{ts}.md"


def _group_related_entries(
    entries: list[JournalEntry], threshold: float = 0.3
) -> list[list[JournalEntry]]:
    """Group entries by simple keyword overlap.

    Uses Jaccard similarity on word sets. Groups entries that share
    enough vocabulary to likely be about the same topic.
    """
    if not entries:
        return []

    # Compute word sets for each entry
    word_sets = [set(e.content.lower().split()) for e in entries]

    # Simple greedy clustering
    used: set[int] = set()
    groups: list[list[JournalEntry]] = []

    for i, entry in enumerate(entries):
        if i in used:
            continue
        group = [entry]
        used.add(i)

        for j in range(i + 1, len(entries)):
            if j in used:
                continue
            # Jaccard similarity
            intersection = len(word_sets[i] & word_sets[j])
            union = len(word_sets[i] | word_sets[j])
            if union > 0 and intersection / union >= threshold:
                group.append(entries[j])
                used.add(j)

        groups.append(group)

    return groups


class FastConsolidation:
    """Fast consolidation: batch journal entries into memories.

    Groups related pending entries and creates a memory file for
    each group. No AI required — uses simple keyword clustering.
    """

    def __init__(
        self,
        journal: System1Journal,
        memory_path: Path,
        *,
        batch_size: int = 20,
        similarity_threshold: float = 0.3,
    ) -> None:
        self._journal = journal
        self._memory_path = memory_path
        self._batch_size = batch_size
        self._similarity_threshold = similarity_threshold

    def consolidate(self) -> list[str]:
        """Process pending entries into memories.

        Returns list of created memory filenames.
        """
        entries = self._journal.get_pending(limit=self._batch_size)
        if not entries:
            logger.info("No pending journal entries to consolidate")
            return []

        groups = _group_related_entries(entries, self._similarity_threshold)
        created_files: list[str] = []
        processed_ids: list[int] = []

        for group in groups:
            filename = _generate_filename(group)
            content = self._format_group(group)

            file_path = self._memory_path / filename
            try:
                self._write_memory(file_path, content, group)
                created_files.append(filename)
                processed_ids.extend(e.id for e in group)
            except Exception:
                logger.error("Failed to create memory %s", filename, exc_info=True)

        if processed_ids:
            self._journal.mark_processed(processed_ids)
            logger.info(
                "Fast consolidation: %d entries → %d memories",
                len(processed_ids),
                len(created_files),
            )

        return created_files

    def _format_group(self, group: list[JournalEntry]) -> str:
        """Format a group of entries into memory content."""
        lines: list[str] = []
        for entry in group:
            ts = datetime.fromtimestamp(entry.timestamp, tz=timezone.utc)
            lines.append(f"- [{ts.strftime('%Y-%m-%d %H:%M')}] {entry.content}")
        return "\n".join(lines)

    def _write_memory(
        self,
        file_path: Path,
        content: str,
        entries: list[JournalEntry],
    ) -> None:
        """Write a memory file with frontmatter."""
        now = datetime.now(timezone.utc)
        tags = ["auto-consolidated", "system1"]

        frontmatter = (
            f"---\n"
            f"type: observation\n"
            f"status: active\n"
            f"tags: [{', '.join(tags)}]\n"
            f"created_at: {now.isoformat()}\n"
            f"source_entries: {[e.id for e in entries]}\n"
            f"---\n\n"
        )

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(frontmatter + content, encoding="utf-8")


class SlowConsolidation:
    """AI-assisted consolidation using Strategy Roulette.

    Asks an AI provider to analyze pending entries and propose
    actions (create, merge, ignore). Falls back to fast consolidation
    if the AI is unavailable.
    """

    def __init__(
        self,
        journal: System1Journal,
        memory_path: Path,
        provider: AIProvider,
        *,
        batch_size: int = 10,
    ) -> None:
        self._journal = journal
        self._memory_path = memory_path
        self._provider = provider
        self._batch_size = batch_size
        self._fast = FastConsolidation(journal, memory_path, batch_size=batch_size)

    async def consolidate(self) -> list[str]:
        """Process pending entries using AI analysis.

        Returns list of created/modified memory filenames.
        Falls back to fast consolidation on AI failure.
        """
        entries = self._journal.get_pending(limit=self._batch_size)
        if not entries:
            return []

        try:
            actions = await self._analyze(entries)
            return self._execute_actions(actions, entries)
        except Exception:
            logger.warning(
                "AI consolidation failed, falling back to fast mode",
                exc_info=True,
            )
            return self._fast.consolidate()

    async def _analyze(self, entries: list[JournalEntry]) -> list[dict]:
        """Ask AI to analyze entries and propose actions."""
        entry_text = "\n".join(
            f"[{i}] {e.content}" for i, e in enumerate(entries)
        )

        prompt = (
            "Analyze these journal entries and propose consolidation actions.\n"
            "Each action should be one of:\n"
            "- create: Make a new memory from entries\n"
            "- ignore: Skip entries that aren't worth keeping\n\n"
            'Return JSON: {"actions": [{"type": "create"|"ignore", '
            '"entry_indices": [0,1,...], "title": "...", "content": "..."}]}\n\n'
            f"Entries:\n{entry_text}"
        )

        response = await self._provider.ask(prompt)
        actions = response.get("actions", [])
        if not isinstance(actions, list):
            raise ValueError(f"Expected actions list, got {type(actions).__name__}")
        return actions

    def _execute_actions(
        self,
        actions: list[dict],
        entries: list[JournalEntry],
    ) -> list[str]:
        """Execute AI-proposed actions."""
        created_files: list[str] = []
        processed_ids: list[int] = []

        for action in actions:
            action_type = action.get("type", "")
            indices = action.get("entry_indices", [])

            # Validate indices
            valid_entries = [
                entries[i]
                for i in indices
                if isinstance(i, int) and 0 <= i < len(entries)
            ]

            if not valid_entries:
                continue

            if action_type == "create":
                title = action.get("title", "")
                content = action.get("content", "")
                if not content:
                    content = self._fast._format_group(valid_entries)

                filename = _generate_filename(valid_entries)
                if title:
                    slug = _slugify(title)
                    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                    filename = f"{slug}-{ts}.md"

                file_path = self._memory_path / filename
                try:
                    self._fast._write_memory(file_path, content, valid_entries)
                    created_files.append(filename)
                    processed_ids.extend(e.id for e in valid_entries)
                except Exception:
                    logger.error("Failed to create memory %s", filename, exc_info=True)

            elif action_type == "ignore":
                processed_ids.extend(e.id for e in valid_entries)

        if processed_ids:
            self._journal.mark_processed(processed_ids)

        return created_files
