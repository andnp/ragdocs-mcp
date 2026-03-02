"""
Unit tests for AI provider layer.

Commit 4.2: Verifies JSON parsing and error handling for AI CLI providers.
"""

from __future__ import annotations

import json

import pytest

from src.memory.providers import AIResponse, GeminiCLIProvider


class TestAIResponse:
    """Tests for AIResponse dataclass."""

    def test_success_response(self) -> None:
        """Successful response has parsed dict and no error."""
        resp = AIResponse(raw_text='{"action": "create"}', parsed={"action": "create"})
        assert resp.success
        assert resp.parsed == {"action": "create"}

    def test_error_response(self) -> None:
        """Error response has error string and no parsed data."""
        resp = AIResponse(raw_text="", parsed=None, error="Command not found")
        assert not resp.success
        assert resp.error == "Command not found"


class TestParseResponse:
    """Test GeminiCLIProvider._parse_response() directly."""

    @pytest.fixture()
    def provider(self) -> GeminiCLIProvider:
        """Create a GeminiCLIProvider instance for parsing tests."""
        return GeminiCLIProvider()

    def test_valid_json_object(self, provider: GeminiCLIProvider) -> None:
        """Valid JSON object is parsed correctly."""
        resp = provider._parse_response('{"action": "merge", "target": "memory-1"}')
        assert resp.success
        assert resp.parsed == {"action": "merge", "target": "memory-1"}

    def test_empty_response(self, provider: GeminiCLIProvider) -> None:
        """Empty string returns error."""
        resp = provider._parse_response("")
        assert not resp.success
        assert resp.error == "Empty response"

    def test_invalid_json(self, provider: GeminiCLIProvider) -> None:
        """Malformed JSON returns error."""
        resp = provider._parse_response("not json at all")
        assert not resp.success
        assert "Invalid JSON" in (resp.error or "")

    def test_json_array_rejected(self, provider: GeminiCLIProvider) -> None:
        """JSON array (not object) is rejected."""
        resp = provider._parse_response("[1, 2, 3]")
        assert not resp.success
        assert "Expected JSON object" in (resp.error or "")

    def test_embedded_json_extraction(self, provider: GeminiCLIProvider) -> None:
        """JSON embedded in surrounding text is extracted."""
        text = 'Here is the result: {"action": "create", "content": "test"} done.'
        resp = provider._parse_response(text)
        assert resp.success
        assert resp.parsed == {"action": "create", "content": "test"}

    def test_nested_json(self, provider: GeminiCLIProvider) -> None:
        """Nested JSON structures are handled."""
        data = {"action": "merge", "sources": ["a", "b"], "metadata": {"score": 0.9}}
        resp = provider._parse_response(json.dumps(data))
        assert resp.success
        assert resp.parsed == data

    def test_unicode_content(self, provider: GeminiCLIProvider) -> None:
        """Unicode content in JSON is preserved."""
        data = {"content": "Résumé with émojis 🎉"}
        resp = provider._parse_response(json.dumps(data))
        assert resp.success
        assert resp.parsed is not None
        assert resp.parsed["content"] == "Résumé with émojis 🎉"


class TestExecute:
    """Test _execute with a real subprocess (using system commands)."""

    @pytest.mark.asyncio
    async def test_command_not_found(self) -> None:
        """Non-existent command returns error."""
        provider = GeminiCLIProvider(command="nonexistent_command_12345")
        resp = await provider._execute("test prompt")
        assert not resp.success
        assert "not found" in (resp.error or "").lower()

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self) -> None:
        """Non-zero exit code returns error."""
        # Use 'false' command which always exits 1
        provider = GeminiCLIProvider(command="false")
        resp = await provider._execute("test")
        assert not resp.success
        assert "Exit code" in (resp.error or "")


class TestAsk:
    """Test the ask() method error handling."""

    @pytest.mark.asyncio
    async def test_ask_raises_on_persistent_failure(self) -> None:
        """ask() raises RuntimeError after all retries fail."""
        provider = GeminiCLIProvider(
            command="nonexistent_command_12345",
            max_retries=0,
        )
        with pytest.raises(RuntimeError, match="failed after"):
            await provider.ask("test prompt")
