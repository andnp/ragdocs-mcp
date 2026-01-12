import re
from pathlib import Path

from src.memory.models import ExtractedLink


LINK_PATTERN = re.compile(r'\[\[([^\]]+)\]\]')

EDGE_TYPE_KEYWORDS: dict[str, list[str]] = {
    "refactors": ["refactor", "rewrite", "restructure", "reorganize", "cleanup"],
    "plans": ["plan", "todo", "will", "should", "need to", "going to"],
    "debugs": ["bug", "fix", "issue", "error", "problem", "broken"],
    "mentions": ["note", "remember", "mention", "see also", "refer"],
}

MEMORY_RELATIONSHIP_KEYWORDS: dict[str, list[str]] = {
    "SUPERSEDES": ["supersedes", "replaces", "obsoletes", "deprecates"],
    "EXTENDS": ["extends", "builds on", "elaborates", "expands on"],
    "CONTRADICTS": ["contradicts", "conflicts with", "disagrees with"],
    "DEPENDS_ON": ["depends on", "requires", "needs", "builds on"],
}


def normalize_tag(tag: str) -> str:
    return tag.strip().lower()


def infer_edge_type(context: str) -> str:
    context_lower = context.lower()

    for edge_type, keywords in EDGE_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in context_lower:
                return edge_type

    return "related_to"


def infer_memory_relationship(context: str) -> str:
    context_lower = context.lower()

    for edge_type, keywords in MEMORY_RELATIONSHIP_KEYWORDS.items():
        for keyword in keywords:
            if keyword in context_lower:
                return edge_type

    return "RELATED_TO"


def parse_link_target(target: str) -> tuple[str, bool]:
    if target.startswith("memory:"):
        memory_name = target[7:].strip()
        memory_id = f"memory:{Path(memory_name).stem}"
        return memory_id, True
    return target, False


def extract_links(content: str, context_chars: int = 100) -> list[ExtractedLink]:
    links: list[ExtractedLink] = []

    for match in LINK_PATTERN.finditer(content):
        target = match.group(1).strip()
        if not target:
            continue

        start_pos = match.start()
        end_pos = match.end()

        context_start = max(0, start_pos - context_chars)
        context_end = min(len(content), end_pos + context_chars)

        before_text = content[context_start:start_pos]
        after_text = content[end_pos:context_end]
        anchor_context = f"{before_text}[[{target}]]{after_text}"

        resolved_target, is_memory = parse_link_target(target)

        if is_memory:
            edge_type = infer_memory_relationship(anchor_context)
        else:
            edge_type = infer_edge_type(anchor_context)

        links.append(ExtractedLink(
            target=resolved_target,
            edge_type=edge_type,
            anchor_context=anchor_context.strip(),
            position=start_pos,
            is_memory_link=is_memory,
        ))

    return links
