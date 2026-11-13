"""Unit tests for azure.cosmos.agent_memory.aio.chat.AsyncChatClient."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from azure.cosmos.agent_memory.aio.chat import (
    AsyncChatClient,
    _is_async_credential,
    _make_sync_token_provider_for_async,
)
from azure.cosmos.agent_memory.exceptions import LLMError
from azure.cosmos.agent_memory.services._pipeline_helpers import is_retryable_llm_error

MESSAGES = [{"role": "user", "content": "hi"}]

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_async_chat_client_init_defaults():
    client = AsyncChatClient(endpoint="https://test.openai.azure.com", api_key="test-key")
    assert client._model == "gpt-4o-mini"
    assert client._endpoint == "https://test.openai.azure.com"
    assert client._api_key == "test-key"
    assert client._client is None  # lazy init


def test_async_chat_client_custom_model():
    client = AsyncChatClient(
        endpoint="https://test.openai.azure.com",
        api_key="key",
        model="gpt-4o",
    )
    assert client._model == "gpt-4o"


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_clears_async_client():
    client = AsyncChatClient(endpoint="https://test.openai.azure.com", api_key="key")
    mock_async = MagicMock()
    mock_async.close = AsyncMock()
    client._client = mock_async

    await client.close()
    assert client._client is None
    mock_async.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Async-credential detection / sync-credential adapter
# ---------------------------------------------------------------------------


def test_is_async_credential_detects_sync():
    class SyncCred:
        def get_token(self, scope):  # not a coroutine function
            return MagicMock(token="t")

    assert _is_async_credential(SyncCred()) is False


def test_is_async_credential_detects_async():
    class AsyncCred:
        async def get_token(self, scope):  # coroutine function
            return MagicMock(token="t")

    assert _is_async_credential(AsyncCred()) is True


@pytest.mark.asyncio
async def test_sync_credential_token_provider_offloads_to_thread():
    class SyncCred:
        def __init__(self):
            self.calls = 0

        def get_token(self, scope):
            self.calls += 1
            return MagicMock(token=f"token-for-{scope}")

    cred = SyncCred()
    provider = _make_sync_token_provider_for_async(cred, "scope-x")
    token = await provider()
    assert token == "token-for-scope-x"
    assert cred.calls == 1


@pytest.mark.asyncio
async def test_ensure_client_accepts_sync_credential(monkeypatch):
    """Regression: passing a sync ``DefaultAzureCredential`` must not raise.

    The async client must adapt a sync TokenCredential to an async bearer
    token provider; otherwise ``AsyncAzureOpenAI`` blows up at runtime.
    """

    class SyncCred:
        def get_token(self, scope):
            return MagicMock(token="tok")

    captured = {}

    class FakeAsyncAzureOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import sys

    fake_openai = MagicMock()
    fake_openai.AsyncAzureOpenAI = FakeAsyncAzureOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    client = AsyncChatClient(
        endpoint="https://test.openai.azure.com",
        credential=SyncCred(),
    )
    result = client._ensure_client()

    assert result is client._client
    assert "azure_ad_token_provider" in captured
    # The provider must be an async callable that returns the token string.
    token = await captured["azure_ad_token_provider"]()
    assert token == "tok"


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_returns_content():
    client = AsyncChatClient(endpoint="https://test.openai.azure.com", api_key="key")
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="hello world"))],
            usage=None,
        )
    )
    client._client = fake

    result = await client.generate([{"role": "user", "content": "hi"}])
    assert result == "hello world"


def _api_status_error(status_code: int, headers: dict[str, str] | None = None):
    import httpx
    import openai

    request = httpx.Request("POST", "https://test.openai.azure.com/openai/deployments/test/chat/completions")
    response = httpx.Response(status_code, headers=headers or {}, request=request)
    return openai.APIStatusError(message=f"status {status_code}", response=response, body=None)


@pytest.mark.asyncio
async def test_generate_retryable_api_error_honors_retry_after(monkeypatch):
    client = AsyncChatClient(endpoint="https://test.openai.azure.com", api_key="key")
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(
        side_effect=[
            _api_status_error(503, {"retry-after": "30"}),
            MagicMock(choices=[MagicMock(message=MagicMock(content="recovered"))], usage=None),
        ]
    )
    client._client = fake
    sleeps: list[float] = []

    async def capture_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("azure.cosmos.agent_memory.chat.random.random", lambda: 0.0)
    monkeypatch.setattr("azure.cosmos.agent_memory.aio.chat.asyncio.sleep", capture_sleep)

    result = await client.generate([{"role": "user", "content": "hi"}], max_retries=2, base_delay=2.0)

    assert result == "recovered"
    assert sleeps == [30.0]


@pytest.mark.asyncio
async def test_generate_retryable_api_error_falls_back_without_retry_after(monkeypatch):
    client = AsyncChatClient(endpoint="https://test.openai.azure.com", api_key="key")
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(
        side_effect=[
            _api_status_error(503),
            MagicMock(choices=[MagicMock(message=MagicMock(content="recovered"))], usage=None),
        ]
    )
    client._client = fake
    sleeps: list[float] = []

    async def capture_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("azure.cosmos.agent_memory.chat.random.random", lambda: 0.0)
    monkeypatch.setattr("azure.cosmos.agent_memory.aio.chat.asyncio.sleep", capture_sleep)

    result = await client.generate([{"role": "user", "content": "hi"}], max_retries=2, base_delay=2.0)

    assert result == "recovered"
    assert sleeps == [1.6]


# ---------------------------------------------------------------------------
# generate() - empty and terminal replies
# ---------------------------------------------------------------------------


def _reply(content, *, finish_reason="stop", refusal=None, choices=True):
    choice = SimpleNamespace(message=SimpleNamespace(content=content, refusal=refusal), finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice] if choices else [], usage=None)


def _client_replying(*replies):
    client = AsyncChatClient(endpoint="https://test.openai.azure.com", api_key="key")
    create = AsyncMock(side_effect=list(replies))
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, create


@pytest.mark.asyncio
async def test_generate_retries_on_empty_content(monkeypatch):
    sleeps: list[float] = []

    async def capture_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("azure.cosmos.agent_memory.chat.random.random", lambda: 0.0)
    monkeypatch.setattr("azure.cosmos.agent_memory.aio.chat.asyncio.sleep", capture_sleep)
    client, create = _client_replying(_reply(""), _reply('{"status": "recovered"}'))

    result = await client.generate(MESSAGES, max_retries=2, base_delay=0.01)

    assert result == '{"status": "recovered"}'
    assert create.await_count == 2
    assert sleeps == [0.008]


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["no-choices", "null-content", "empty-string", "whitespace"])
async def test_all_transient_empty_reply_shapes_share_the_retry_budget(shape):
    content = {"null-content": None, "empty-string": "", "whitespace": "  "}.get(shape)
    empty = _reply(content, choices=shape != "no-choices")
    client, create = _client_replying(empty, _reply("recovered"))

    assert await client.generate(MESSAGES, max_retries=2, base_delay=0) == "recovered"
    assert create.await_count == 2

    create.reset_mock()
    create.side_effect = None
    create.return_value = empty
    with pytest.raises(LLMError, match="empty content after 2 attempts"):
        await client.generate(MESSAGES, max_retries=2, base_delay=0)
    assert create.await_count == 2


@pytest.mark.asyncio
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
async def test_terminal_empty_replies_are_not_retried(reason, content, refusal, code):
    client, create = _client_replying(_reply(content, finish_reason=reason, refusal=refusal))

    with pytest.raises(LLMError, match=code) as failure:
        await client.generate(MESSAGES, max_retries=3, base_delay=0)

    assert create.await_count == 1
    assert not is_retryable_llm_error(failure.value)


@pytest.mark.asyncio
async def test_a_partial_reply_cut_off_at_the_output_limit_is_returned():
    client, create = _client_replying(_reply('{"facts": [', finish_reason="length"))

    assert await client.generate(MESSAGES, max_retries=3, base_delay=0) == '{"facts": ['
    assert create.await_count == 1


@pytest.mark.asyncio
async def test_retry_warning_describes_the_error_without_its_message(monkeypatch, caplog):
    import openai

    monkeypatch.setattr("azure.cosmos.agent_memory.aio.chat.asyncio.sleep", AsyncMock())
    client, _ = _client_replying(
        openai.RateLimitError(message="quoted-model-text", response=MagicMock(status_code=429), body=None),
        _reply("recovered"),
    )

    with caplog.at_level("WARNING", logger="azure.cosmos.agent_memory.aio.chat"):
        await client.generate(MESSAGES, max_retries=2, base_delay=0)

    assert "RateLimitError" in caplog.text
    assert "quoted-model-text" not in caplog.text
