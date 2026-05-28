"""Unit tests for agent_memory_toolkit.aio.chat.AsyncChatClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory_toolkit.aio.chat import (
    AsyncChatClient,
    _is_async_credential,
    _make_sync_token_provider_for_async,
)

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
