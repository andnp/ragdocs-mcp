"""LLMProvider port: adapters for LLM-assisted operations.

Pluggable LLM providers with tiered fallback. Supports both unstructured text
generation and structured output (JSON Schema).
"""

from typing import Any, Protocol, runtime_checkable

from searchkernel.domain import Tier


@runtime_checkable
class LLMProvider(Protocol):
    """Generates text or structured output via an LLM.

    Used for LLM-assisted stages (reranking judgments, query rewriting,
    chunk contextualization, etc.) with tiered fallback (FAST → SMART).
    """

    async def complete(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any] | None = None,
        tier: Tier = Tier.FAST,
    ) -> str | dict[str, Any]:
        """
        Generate a completion in response to a prompt.

        Args:
            prompt: The input prompt string.
            response_format: Optional JSON Schema dict for structured output.
                           If provided, the output is parsed as JSON.
            tier: Performance tier (FAST for SLM/local, SMART for cloud LLMs).

        Returns:
            If response_format is None: a string (free-text completion).
            If response_format is set: a dict (JSON-parsed structured output).
        """
        ...
