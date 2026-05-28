"""Unit tests for AsyncEmbeddingsClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory_toolkit.aio.embeddings import AOAI_EMBEDDING_BATCH_SIZE, AsyncEmbeddingsClient
from agent_memory_toolkit.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedding_response(embeddings: list[list[float]]):
    """Build a mock response matching openai's embedding response shape."""
    data = []
    for i, emb in enumerate(embeddings):
        item = MagicMock()
        item.embedding = emb
        item.index = i
        data.append(item)
    resp = MagicMock()
    resp.data = data
    return resp


@pytest.fixture
def client():
    return AsyncEmbeddingsClient(
        endpoint="https://fake.openai.azure.com/",
        api_key="fake-key",
        model="text-embedding-3-large",
    )


# ===================================================================
# generate() — success
# ===================================================================


async def test_generate_success(client):
    expected = [0.1, 0.2, 0.3]
    mock_openai = MagicMock()
    mock_openai.embeddings.create = AsyncMock(return_value=_make_embedding_response([expected]))
    client._client = mock_openai

    result = await client.generate("hello")
    assert result == expected
    mock_openai.embeddings.create.assert_awaited_once()


# ===================================================================
# generate() — lazy init
# ===================================================================


async def test_generate_lazy_init():
    """Client is created on first call, not at construction."""
    client = AsyncEmbeddingsClient(
        endpoint="https://fake.openai.azure.com/",
        api_key="test-key",
    )
    assert client._client is None

    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_instance.embeddings.create = AsyncMock(return_value=_make_embedding_response([[1.0, 2.0]]))
    mock_cls.return_value = mock_instance

    with patch("openai.AsyncAzureOpenAI", mock_cls):
        result = await client.generate("test")

    assert result == [1.0, 2.0]
    assert client._client is mock_instance
    mock_cls.assert_called_once()


# ===================================================================
# generate() — api_key vs credential auth
# ===================================================================


async def test_generate_api_key_auth():
    client = AsyncEmbeddingsClient(
        endpoint="https://fake.openai.azure.com/",
        api_key="my-key",
    )
    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_instance.embeddings.create = AsyncMock(return_value=_make_embedding_response([[1.0]]))
    mock_cls.return_value = mock_instance

    with patch("openai.AsyncAzureOpenAI", mock_cls):
        await client.generate("x")

    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["api_key"] == "my-key"
    assert "azure_ad_token_provider" not in call_kwargs


async def test_generate_credential_auth():
    mock_cred = MagicMock()
    client = AsyncEmbeddingsClient(
        endpoint="https://fake.openai.azure.com/",
        credential=mock_cred,
    )

    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_instance.embeddings.create = AsyncMock(return_value=_make_embedding_response([[2.0]]))
    mock_cls.return_value = mock_instance

    mock_token_provider = MagicMock()
    with (
        patch("openai.AsyncAzureOpenAI", mock_cls),
        patch("azure.identity.aio.get_bearer_token_provider", return_value=mock_token_provider),
    ):
        await client.generate("x")

    call_kwargs = mock_cls.call_args.kwargs
    assert "azure_ad_token_provider" in call_kwargs
    assert "api_key" not in call_kwargs


# ===================================================================
# generate() — missing endpoint
# ===================================================================


async def test_generate_missing_endpoint():
    client = AsyncEmbeddingsClient(endpoint=None, api_key="key")
    with pytest.raises(ConfigurationError):
        await client.generate("hello")


async def test_generate_missing_credential_and_api_key():
    client = AsyncEmbeddingsClient(
        endpoint="https://fake.openai.azure.com/",
        credential=None,
        api_key=None,
    )
    with pytest.raises(ConfigurationError):
        await client.generate("hello")


# ===================================================================
# generate() — API failure
# ===================================================================


async def test_generate_api_failure(client):
    mock_openai = MagicMock()
    mock_openai.embeddings.create = AsyncMock(side_effect=Exception("API down"))
    client._client = mock_openai

    with pytest.raises(Exception, match="API down"):
        await client.generate("hello")


# ===================================================================
# generate_batch()
# ===================================================================


async def test_generate_batch_order_preservation(client):
    emb0 = [1.0, 2.0]
    emb1 = [3.0, 4.0]
    emb2 = [5.0, 6.0]
    # Return in scrambled order to test sorting by index
    data = []
    for idx, emb in [(2, emb2), (0, emb0), (1, emb1)]:
        item = MagicMock()
        item.embedding = emb
        item.index = idx
        data.append(item)
    resp = MagicMock()
    resp.data = data

    mock_openai = MagicMock()
    mock_openai.embeddings.create = AsyncMock(return_value=resp)
    client._client = mock_openai

    result = await client.generate_batch(["a", "b", "c"])
    assert result == [emb0, emb1, emb2]


async def test_generate_batch_empty(client):
    result = await client.generate_batch([])
    assert result == []


async def test_generate_batch_api_failure(client):
    mock_openai = MagicMock()
    mock_openai.embeddings.create = AsyncMock(side_effect=Exception("timeout"))
    client._client = mock_openai

    with pytest.raises(Exception, match="timeout"):
        await client.generate_batch(["hello"])


# ===================================================================
# generate_batch() — N=16 chunk guard
# ===================================================================


def test_default_batch_size_constant_is_16():
    assert AOAI_EMBEDDING_BATCH_SIZE == 16


def _chunked_create_side_effect():
    """Return a side_effect that builds a per-chunk response shaped to inputs.

    Encodes each input text ``"t{i}"`` as a single-float embedding ``[float(i)]``
    so callers can assert end-to-end ordering after concatenation.
    """

    async def _create(*, input, model, **_):  # noqa: A002 - mirror OpenAI kwarg
        items = []
        for chunk_idx, text in enumerate(input):
            global_i = int(text[1:])
            item = MagicMock()
            item.embedding = [float(global_i)]
            item.index = chunk_idx
            items.append(item)
        resp = MagicMock()
        resp.data = items
        return resp

    return _create


async def test_generate_batch_small_calls_api_once(client):
    mock_openai = MagicMock()
    mock_openai.embeddings.create = AsyncMock(side_effect=_chunked_create_side_effect())
    client._client = mock_openai

    texts = [f"t{i}" for i in range(10)]
    result = await client.generate_batch(texts)

    assert len(result) == 10
    assert result == [[float(i)] for i in range(10)]
    assert mock_openai.embeddings.create.await_count == 1


async def test_generate_batch_large_chunked_into_16_16_8(client):
    mock_openai = MagicMock()
    mock_openai.embeddings.create = AsyncMock(side_effect=_chunked_create_side_effect())
    client._client = mock_openai

    texts = [f"t{i}" for i in range(40)]
    result = await client.generate_batch(texts)

    assert mock_openai.embeddings.create.await_count == 3
    chunk_sizes = [len(call.kwargs["input"]) for call in mock_openai.embeddings.create.await_args_list]
    assert chunk_sizes == [16, 16, 8]

    assert len(result) == 40
    assert result == [[float(i)] for i in range(40)]


async def test_generate_batch_empty_no_api_call():
    client = AsyncEmbeddingsClient(
        endpoint="https://fake.openai.azure.com/",
        api_key="fake-key",
    )
    mock_openai = MagicMock()
    mock_openai.embeddings.create = AsyncMock()
    client._client = mock_openai

    result = await client.generate_batch([])

    assert result == []
    mock_openai.embeddings.create.assert_not_awaited()


async def test_generate_batch_custom_batch_size(client):
    mock_openai = MagicMock()
    mock_openai.embeddings.create = AsyncMock(side_effect=_chunked_create_side_effect())
    client._client = mock_openai

    texts = [f"t{i}" for i in range(12)]
    result = await client.generate_batch(texts, batch_size=5)

    assert mock_openai.embeddings.create.await_count == 3
    chunk_sizes = [len(call.kwargs["input"]) for call in mock_openai.embeddings.create.await_args_list]
    assert chunk_sizes == [5, 5, 2]
    assert result == [[float(i)] for i in range(12)]


# ===================================================================
# close()
# ===================================================================


async def test_close(client):
    mock_openai = AsyncMock()
    client._client = mock_openai
    await client.close()
    mock_openai.close.assert_awaited_once()
    assert client._client is None


async def test_close_noop_when_no_client():
    client = AsyncEmbeddingsClient(endpoint="https://x.openai.azure.com/")
    await client.close()  # should not raise


# ===================================================================
# async context manager
# ===================================================================


async def test_context_manager():
    mock_openai = AsyncMock()
    client = AsyncEmbeddingsClient(endpoint="https://fake.openai.azure.com/", api_key="k")
    client._client = mock_openai

    async with client as c:
        assert c is client
    mock_openai.close.assert_awaited_once()
