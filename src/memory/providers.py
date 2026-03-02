"""AI provider abstraction for memory consolidation tasks."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


class AIProvider(Protocol):
    """Protocol for AI providers used in memory consolidation."""

    async def ask(self, prompt: str) -> dict:
        """Send a prompt and return a parsed JSON response."""
        ...


@dataclass
class AIResponse:
    """Structured response from an AI provider."""

    raw_text: str
    parsed: dict | None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.parsed is not None and self.error is None


class GeminiCLIProvider:
    """AI provider using the Gemini CLI (`gemini` command).

    Calls `gemini ask --json <prompt>` as a subprocess and parses
    the JSON response. Handles subprocess failures and invalid JSON
    gracefully.
    """

    def __init__(
        self,
        *,
        command: str = "gemini",
        timeout: float = 60.0,
        max_retries: int = 1,
    ) -> None:
        self._command = command
        self._timeout = timeout
        self._max_retries = max_retries

    async def ask(self, prompt: str) -> dict:
        """Send prompt to Gemini CLI and return parsed JSON response.

        Raises:
            RuntimeError: If the CLI fails or returns invalid JSON after retries.
        """
        last_error: str | None = None

        for attempt in range(1 + self._max_retries):
            response = await self._execute(prompt)
            if response.success:
                assert response.parsed is not None
                return response.parsed
            last_error = response.error
            if attempt < self._max_retries:
                logger.warning(
                    "Gemini CLI attempt %d failed: %s. Retrying...",
                    attempt + 1,
                    last_error,
                )

        raise RuntimeError(
            f"Gemini CLI failed after {1 + self._max_retries} attempts: {last_error}"
        )

    async def _execute(self, prompt: str) -> AIResponse:
        """Execute a single CLI call."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self._command,
                "ask",
                "--json",
                prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return AIResponse(raw_text="", parsed=None, error="Command timed out")

            stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0:
                return AIResponse(
                    raw_text=stdout_text,
                    parsed=None,
                    error=f"Exit code {proc.returncode}: {stderr_text or stdout_text}",
                )

            return self._parse_response(stdout_text)

        except FileNotFoundError:
            return AIResponse(
                raw_text="",
                parsed=None,
                error=f"Command not found: {self._command}",
            )
        except OSError as e:
            return AIResponse(
                raw_text="",
                parsed=None,
                error=f"OS error: {e}",
            )

    def _parse_response(self, text: str) -> AIResponse:
        """Parse JSON from CLI output."""
        if not text:
            return AIResponse(raw_text=text, parsed=None, error="Empty response")

        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                return AIResponse(
                    raw_text=text,
                    parsed=None,
                    error=f"Expected JSON object, got {type(parsed).__name__}",
                )
            return AIResponse(raw_text=text, parsed=parsed)
        except json.JSONDecodeError as e:
            # Try to find JSON embedded in text (common with CLI tools)
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                try:
                    parsed = json.loads(text[json_start:json_end])
                    if isinstance(parsed, dict):
                        return AIResponse(raw_text=text, parsed=parsed)
                except json.JSONDecodeError:
                    pass
            return AIResponse(
                raw_text=text,
                parsed=None,
                error=f"Invalid JSON: {e}",
            )
