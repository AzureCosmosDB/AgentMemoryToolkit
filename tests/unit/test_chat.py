"""Unit tests for agent_memory_toolkit.chat.ChatClient (sync)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_memory_toolkit.chat import ChatClient
from agent_memory_toolkit.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_chat_client_init_defaults():
    client = ChatClient(endpoint="https://test.openai.azure.com", api_key="test-key")
    assert client._model == "gpt-4o-mini"
    assert client._endpoint == "https://test.openai.azure.com"
    assert client._api_key == "test-key"
    assert client._client is None  # lazy init


def test_chat_client_custom_model():
    client = ChatClient(
        endpoint="https://test.openai.azure.com",
        api_key="key",
        model="gpt-4o",
    )
    assert client._model == "gpt-4o"


def test_chat_client_no_params():
    client = ChatClient()
    assert client._endpoint is None
    assert client._api_key is None
    assert client._credential is None


# ---------------------------------------------------------------------------
# generate() – configuration errors
# ---------------------------------------------------------------------------


def test_chat_client_no_endpoint_raises():
    client = ChatClient()
    with pytest.raises(ConfigurationError, match="endpoint"):
        client.generate([{"role": "user", "content": "test"}])


def test_chat_client_no_credentials_raises():
    client = ChatClient(endpoint="https://test.openai.azure.com")
    with pytest.raises(ConfigurationError, match="api_key or a TokenCredential"):
        client.generate([{"role": "user", "content": "test"}])


# ---------------------------------------------------------------------------
# generate() – success path
# ---------------------------------------------------------------------------


def test_generate_returns_content():
    client = ChatClient(endpoint="https://test.openai.azure.com", api_key="test-key")

    mock_choice = MagicMock()
    mock_choice.message.content = "Hello, world!"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = mock_response
    client._client = mock_openai_client

    result = client.generate([{"role": "user", "content": "Hi"}])
    assert result == "Hello, world!"


def test_generate_forces_temperature_to_one():
    """Temperature is hardcoded to 1.0; caller-supplied values are ignored.

    Rationale: newer Azure OpenAI models (gpt-5.x, o-series) only accept
    the default temperature value; older models accept 1.0 as a valid
    value. Forcing 1.0 universally keeps behavior uniform and removes
    the need for per-model sampling-knob carve-outs."""
    client = ChatClient(endpoint="https://test.openai.azure.com", api_key="test-key")

    mock_choice = MagicMock()
    mock_choice.message.content = "response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = mock_response
    client._client = mock_openai_client

    client.generate(
        [{"role": "user", "content": "Hi"}],
        temperature=0.5,
    )
    call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
    assert call_kwargs["temperature"] == 1.0


def test_generate_passes_response_format():
    client = ChatClient(endpoint="https://test.openai.azure.com", api_key="test-key")

    mock_choice = MagicMock()
    mock_choice.message.content = '{"key": "value"}'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = mock_response
    client._client = mock_openai_client

    fmt = {"type": "json_object"}
    client.generate(
        [{"role": "user", "content": "Hi"}],
        response_format=fmt,
    )
    call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
    assert call_kwargs["response_format"] == fmt


# ---------------------------------------------------------------------------
# generate() – retry on rate limit
# ---------------------------------------------------------------------------


def test_generate_retries_on_rate_limit():
    import openai

    client = ChatClient(endpoint="https://test.openai.azure.com", api_key="test-key")

    mock_choice = MagicMock()
    mock_choice.message.content = "recovered"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    mock_openai_client = MagicMock()
    rate_err = openai.RateLimitError(
        message="rate limited",
        response=MagicMock(status_code=429),
        body=None,
    )
    mock_openai_client.chat.completions.create.side_effect = [
        rate_err,
        mock_response,
    ]
    client._client = mock_openai_client

    result = client.generate(
        [{"role": "user", "content": "test"}],
        max_retries=2,
        base_delay=0.01,
    )
    assert result == "recovered"
    assert mock_openai_client.chat.completions.create.call_count == 2


def test_generate_exhausts_retries_on_rate_limit():
    import openai

    client = ChatClient(endpoint="https://test.openai.azure.com", api_key="test-key")

    mock_openai_client = MagicMock()
    rate_err = openai.RateLimitError(
        message="rate limited",
        response=MagicMock(status_code=429),
        body=None,
    )
    mock_openai_client.chat.completions.create.side_effect = [rate_err] * 3
    client._client = mock_openai_client

    with pytest.raises(openai.RateLimitError, match="rate limited"):
        client.generate(
            [{"role": "user", "content": "test"}],
            max_retries=3,
            base_delay=0.01,
        )


# ---------------------------------------------------------------------------
# generate() – non-retryable errors
# ---------------------------------------------------------------------------


def test_generate_propagates_generic_exception():
    client = ChatClient(endpoint="https://test.openai.azure.com", api_key="test-key")

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.side_effect = RuntimeError("boom")
    client._client = mock_openai_client

    with pytest.raises(RuntimeError, match="boom"):
        client.generate([{"role": "user", "content": "test"}])


# ---------------------------------------------------------------------------
# _build_kwargs
# ---------------------------------------------------------------------------


def test_build_kwargs_minimal():
    client = ChatClient(endpoint="https://test.openai.azure.com", api_key="key")
    kwargs = client._build_kwargs([{"role": "user", "content": "hi"}])
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert kwargs["temperature"] == 1.0
    assert "response_format" not in kwargs


def test_build_kwargs_with_all_options():
    """Temperature is always forced to 1.0; other options pass through."""
    client = ChatClient(endpoint="https://test.openai.azure.com", api_key="key")
    kwargs = client._build_kwargs(
        [{"role": "user", "content": "hi"}],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    assert kwargs["temperature"] == 1.0
    assert kwargs["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


def test_close_clears_sync_client():
    client = ChatClient(endpoint="https://test.openai.azure.com", api_key="key")
    mock_client = MagicMock()
    client._client = mock_client

    client.close()

    assert client._client is None
    mock_client.close.assert_called_once()
