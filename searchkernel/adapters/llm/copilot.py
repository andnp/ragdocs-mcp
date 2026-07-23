"""Copilot CLI LLMProvider adapter.

Shells out to the GitHub Copilot CLI (copilot command) for non-interactive
completions. Defaults to model gpt-5.6-luna. The CLI is invoked with -p for
the prompt and --model for the model specification.

This is an ADDITIVE port implementation; no other LLM adapters are modified.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from typing import Any

from searchkernel.domain import Tier


class CopilotLLMProvider:
    """LLMProvider backed by the GitHub Copilot CLI.

    Invokes `copilot -p <prompt> --model <model_name> --allow-all` to
    generate completions. The CLI must be installed and available in PATH.
    Errors are deferred until first use (not on import).
    """

    def __init__(self, model_name: str = "gpt-5.6-luna"):
        """Initialize the Copilot LLM provider.

        Args:
            model_name: The model to use (default: gpt-5.6-luna).
        """
        self.model_name = model_name
        self._copilot_path: str | None = None
        self._checked_copilot = False

    def _ensure_copilot_available(self) -> None:
        """Check if copilot CLI is available. Raises on first error, not import."""
        if self._checked_copilot:
            return
        self._checked_copilot = True

        self._copilot_path = shutil.which("copilot")
        if self._copilot_path is None:
            raise RuntimeError(
                "copilot CLI not found in PATH. "
                "Install GitHub Copilot CLI (https://github.com/github/copilot.vim) "
                "or ensure it is available as `copilot` command."
            )

    async def complete(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any] | None = None,
        tier: Tier = Tier.FAST,
    ) -> str | dict[str, Any]:
        """Generate a completion via the Copilot CLI.

        Args:
            prompt: The input prompt string.
            response_format: Optional JSON Schema dict for structured output.
                           If provided, output is parsed as JSON.
            tier: Performance tier (FAST/SMART); both use the same model.

        Returns:
            If response_format is None: the raw completion string.
            If response_format is set: JSON-parsed dict.

        Raises:
            RuntimeError: If copilot CLI is not available or subprocess fails.
        """
        self._ensure_copilot_available()

        # Run copilot in non-interactive mode with the specified model.
        cmd = [
            self._copilot_path,
            "-p",
            prompt,
            "--model",
            self.model_name,
            "--allow-all",
            "-s",  # silent mode: output only the response
        ]

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2-minute timeout for completion
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"copilot completion timed out after 120 seconds for model {self.model_name}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"copilot subprocess failed: {e}") from e

        if result.returncode != 0:
            raise RuntimeError(
                f"copilot exited with code {result.returncode}: {result.stderr}"
            )

        completion_text = result.stdout.strip()

        # If JSON response format is requested, parse and validate.
        if response_format is not None:
            try:
                parsed = json.loads(completion_text)
                return parsed
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"copilot output is not valid JSON: {completion_text[:200]}"
                ) from e

        return completion_text
