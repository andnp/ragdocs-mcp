"""Unit tests for the Copilot LLMProvider adapter.

These tests MOCK the subprocess call to avoid requiring the actual
copilot CLI. The real CLI is verified separately if needed.
"""

import json
from unittest import mock

import pytest

from searchkernel.adapters.llm import CopilotLLMProvider
from searchkernel.domain import Tier
from searchkernel.ports.llm import LLMProvider


def test_satisfies_port():
    """CopilotLLMProvider must conform to the LLMProvider protocol."""
    provider = CopilotLLMProvider()
    assert isinstance(provider, LLMProvider)


def test_default_model_name():
    """Default model should be gpt-5.6-luna."""
    provider = CopilotLLMProvider()
    assert provider.model_name == "gpt-5.6-luna"


def test_custom_model_name():
    """Should accept custom model names."""
    provider = CopilotLLMProvider(model_name="gpt-4.0")
    assert provider.model_name == "gpt-4.0"


@pytest.mark.asyncio
async def test_complete_text_response_mocked():
    """Test text completion with mocked subprocess."""
    provider = CopilotLLMProvider()
    provider._copilot_path = "/usr/bin/copilot"  # Mock path

    expected_response = "This is a test completion."

    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0,
            stdout=expected_response,
            stderr="",
        )

        result = await provider.complete("Test prompt", tier=Tier.FAST)

        assert result == expected_response
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "copilot" in call_args[0]
        assert "-p" in call_args
        assert "Test prompt" in call_args
        assert "--model" in call_args
        assert "gpt-5.6-luna" in call_args
        assert "--allow-all" in call_args


@pytest.mark.asyncio
async def test_complete_json_response_mocked():
    """Test JSON-formatted completion with mocked subprocess."""
    provider = CopilotLLMProvider()
    provider._copilot_path = "/usr/bin/copilot"

    json_response = {"reasoning": "Test", "conclusion": "OK"}
    json_str = json.dumps(json_response)

    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0,
            stdout=json_str,
            stderr="",
        )

        response_format = {"type": "object", "properties": {}}
        result = await provider.complete(
            "Test prompt",
            response_format=response_format,
            tier=Tier.SMART,
        )

        assert result == json_response


@pytest.mark.asyncio
async def test_copilot_not_found_on_first_use():
    """Error should be raised on first use, not on import."""
    provider = CopilotLLMProvider()
    # _copilot_path and _checked_copilot are initially None/False

    with (
        mock.patch("shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="copilot CLI not found"),
    ):
        await provider.complete("Test prompt")


@pytest.mark.asyncio
async def test_subprocess_error_propagated():
    """Subprocess errors should be raised as RuntimeError."""
    provider = CopilotLLMProvider()
    provider._copilot_path = "/usr/bin/copilot"

    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=1,
            stdout="",
            stderr="Authentication failed",
        )

        with pytest.raises(RuntimeError, match="exited with code 1"):
            await provider.complete("Test prompt")


@pytest.mark.asyncio
async def test_subprocess_timeout():
    """Timeout should be handled gracefully."""
    import subprocess

    provider = CopilotLLMProvider()
    provider._copilot_path = "/usr/bin/copilot"

    with mock.patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired("copilot", 120)

        with pytest.raises(RuntimeError, match="timed out"):
            await provider.complete("Test prompt")


@pytest.mark.asyncio
async def test_json_parse_error_on_invalid_json():
    """Invalid JSON should raise RuntimeError when response_format is set."""
    provider = CopilotLLMProvider()
    provider._copilot_path = "/usr/bin/copilot"

    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0,
            stdout="This is not valid JSON {",
            stderr="",
        )

        response_format = {"type": "object"}
        with pytest.raises(RuntimeError, match="not valid JSON"):
            await provider.complete(
                "Test prompt",
                response_format=response_format,
            )


@pytest.mark.asyncio
async def test_tier_fast_and_smart_use_same_model():
    """Both FAST and SMART tiers should use the specified model."""
    provider = CopilotLLMProvider(model_name="gpt-5.6-luna")
    provider._copilot_path = "/usr/bin/copilot"

    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0,
            stdout="Response",
            stderr="",
        )

        # Test FAST tier
        await provider.complete("Prompt", tier=Tier.FAST)
        fast_call = mock_run.call_args[0][0]
        assert "gpt-5.6-luna" in fast_call

        mock_run.reset_mock()

        # Test SMART tier
        await provider.complete("Prompt", tier=Tier.SMART)
        smart_call = mock_run.call_args[0][0]
        assert "gpt-5.6-luna" in smart_call
