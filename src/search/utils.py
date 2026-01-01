def classify_query_type(query: str):
    query_lower = query.lower().strip()

    factual_keywords = [
        "what is", "define", "syntax", "example", "command",
        "configure", "config", "setup", "install", "enable",
        "disable", "usage",
    ]

    conceptual_keywords = [
        "why", "explain", "architecture", "design", "compare",
        "difference", "overview", "concept", "understand", "background",
        "getting started", "how to", "steps to", "guide",
    ]

    for keyword in conceptual_keywords:
        if keyword in query_lower:
            return "conceptual"

    for keyword in factual_keywords:
        if keyword in query_lower:
            return "factual"

    if "?" in query:
        return "conceptual"

    return "factual"


def truncate_content(content: str | None, max_chars: int = 200):
    if not content or len(content) <= max_chars:
        return content

    truncated = content[:max_chars].rstrip()
    return f"{truncated}..."
