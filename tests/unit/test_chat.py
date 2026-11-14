"""Unit tests for azure.cosmos.agent_memory.chat.ChatClient (sync)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from azure.cosmos.agent_memory.chat import ChatClient
from azure.cosmos.agent_memory.exceptions import ConfigurationError, LLMError, OutputTruncatedError
from azure.cosmos.agent_memory.logging import describe_exception
from azure.cosmos.agent_memory.services._pipeline_helpers import is_retryable_llm_error

MESSAGES = [{"role": "user", "content": "hi"}]

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


def _api_status_error(status_code: int, headers: dict[str, str] | None = None):
    import httpx
    import openai

    request = httpx.Request("POST", "https://test.openai.azure.com/openai/deployments/test/chat/completions")
    response = httpx.Response(status_code, headers=headers or {}, request=request)
    return openai.APIStatusError(message=f"status {status_code}", response=response, body=None)


def test_generate_retryable_api_error_honors_retry_after(monkeypatch):
    client = ChatClient(endpoint="https://test.openai.azure.com", api_key="test-key")

    mock_choice = MagicMock()
    mock_choice.message.content = "recovered"
    mock_response = MagicMock(choices=[mock_choice], usage=None)
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.side_effect = [
        _api_status_error(503, {"retry-after": "30"}),
        mock_response,
    ]
    client._client = mock_openai_client
    sleeps: list[float] = []
    monkeypatch.setattr("azure.cosmos.agent_memory.chat.random.random", lambda: 0.0)
    monkeypatch.setattr("azure.cosmos.agent_memory.chat.time.sleep", sleeps.append)

    result = client.generate([{"role": "user", "content": "test"}], max_retries=2, base_delay=2.0)

    assert result == "recovered"
    assert sleeps == [30.0]


def test_generate_retryable_api_error_falls_back_without_retry_after(monkeypatch):
    client = ChatClient(endpoint="https://test.openai.azure.com", api_key="test-key")

    mock_choice = MagicMock()
    mock_choice.message.content = "recovered"
    mock_response = MagicMock(choices=[mock_choice], usage=None)
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.side_effect = [
        _api_status_error(503),
        mock_response,
    ]
    client._client = mock_openai_client
    sleeps: list[float] = []
    monkeypatch.setattr("azure.cosmos.agent_memory.chat.random.random", lambda: 0.0)
    monkeypatch.setattr("azure.cosmos.agent_memory.chat.time.sleep", sleeps.append)

    result = client.generate([{"role": "user", "content": "test"}], max_retries=2, base_delay=2.0)

    assert result == "recovered"
    assert sleeps == [1.6]


# ---------------------------------------------------------------------------
# generate() – empty and terminal replies
# ---------------------------------------------------------------------------


def _reply(content, *, finish_reason="stop", refusal=None, choices=True):
    choice = SimpleNamespace(message=SimpleNamespace(content=content, refusal=refusal), finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice] if choices else [], usage=None)


def _client_replying(*replies):
    client = ChatClient(endpoint="https://test.openai.azure.com", api_key="test-key")
    create = MagicMock(side_effect=list(replies))
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, create


def test_generate_retries_on_empty_content(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("azure.cosmos.agent_memory.chat.random.random", lambda: 0.0)
    monkeypatch.setattr("azure.cosmos.agent_memory.chat.time.sleep", sleeps.append)
    client, create = _client_replying(_reply(""), _reply('{"status": "recovered"}'))

    result = client.generate(MESSAGES, max_retries=2, base_delay=0.01)

    assert result == '{"status": "recovered"}'
    assert create.call_count == 2
    assert sleeps == [0.008]


@pytest.mark.parametrize("shape", ["no-choices", "null-content", "empty-string", "whitespace"])
def test_all_transient_empty_reply_shapes_share_the_retry_budget(shape):
    content = {"null-content": None, "empty-string": "", "whitespace": "  "}.get(shape)
    empty = _reply(content, choices=shape != "no-choices")
    client, create = _client_replying(empty, _reply("recovered"))

    assert client.generate(MESSAGES, max_retries=2, base_delay=0) == "recovered"
    assert create.call_count == 2

    create.reset_mock()
    create.side_effect = None
    create.return_value = empty
    with pytest.raises(LLMError, match="empty content after 2 attempts"):
        client.generate(MESSAGES, max_retries=2, base_delay=0)
    assert create.call_count == 2


@pytest.mark.parametrize(
    ("reason", "content", "refusal", "code"),
    [
        ("content_filter", None, None, "content_filter"),
        ("stop", None, "refused", "model_refusal"),
        ("tool_calls", None, None, "unexpected_tool_response"),
        ("tool_calls", "", None, "unexpected_tool_response"),
        ("tool_calls", "  \n", None, "unexpected_tool_response"),
        ("function_call", "", None, "unexpected_tool_response"),
        ("length", None, None, "output_truncated"),
        ("length", "", None, "output_truncated"),
        ("length", "  \n", None, "output_truncated"),
    ],
)
def test_terminal_empty_replies_are_not_retried(reason, content, refusal, code):
    client, create = _client_replying(_reply(content, finish_reason=reason, refusal=refusal))

    with pytest.raises(LLMError, match=code) as failure:
        client.generate(MESSAGES, max_retries=3, base_delay=0)

    assert create.call_count == 1
    assert not is_retryable_llm_error(failure.value)


@pytest.mark.parametrize("content", [None, "", "  \n"])
def test_a_blank_reply_cut_off_at_the_output_limit_raises_the_output_truncated_error(content):
    client, create = _client_replying(_reply(content, finish_reason="length"))

    with pytest.raises(OutputTruncatedError, match="output_truncated") as failure:
        client.generate(MESSAGES, max_retries=3, base_delay=0)

    assert create.call_count == 1
    assert failure.value.error_code == "output_truncated"
    assert "code=output_truncated" in describe_exception(failure.value), "a log shows the code, never the message"


def test_a_partial_reply_cut_off_at_the_output_limit_is_returned():
    client, create = _client_replying(_reply('{"facts": [', finish_reason="length"))

    assert client.generate(MESSAGES, max_retries=3, base_delay=0) == '{"facts": ['
    assert create.call_count == 1


def test_retry_warning_describes_the_error_without_its_message(monkeypatch, caplog):
    import openai

    monkeypatch.setattr("azure.cosmos.agent_memory.chat.time.sleep", lambda _: None)
    client, _ = _client_replying(
        openai.RateLimitError(message="quoted-model-text", response=MagicMock(status_code=429), body=None),
        _reply("recovered"),
    )

    with caplog.at_level("WARNING", logger="azure.cosmos.agent_memory.chat"):
        client.generate(MESSAGES, max_retries=2, base_delay=0)

    assert "RateLimitError" in caplog.text
    assert "quoted-model-text" not in caplog.text


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
