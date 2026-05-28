"""Unit tests for EmbeddingsClient (sync embedding client)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_memory_toolkit.embeddings import AOAI_EMBEDDING_BATCH_SIZE, EmbeddingsClient
from agent_memory_toolkit.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(**overrides) -> EmbeddingsClient:
    defaults = dict(
        endpoint="https://fake.openai.azure.com",
        api_key="sk-fake",
        model="text-embedding-3-large",
    )
    defaults.update(overrides)
    return EmbeddingsClient(**defaults)


def _mock_embedding_response(embedding: list[float]):
    """Build a mock response for a single embedding call."""
    item = MagicMock()
    item.embedding = embedding
    item.index = 0
    resp = MagicMock()
    resp.data = [item]
    return resp


def _mock_batch_response(embeddings: list[list[float]]):
    """Build a mock response for a batch embedding call with index fields."""
    items = []
    for i, emb in enumerate(embeddings):
        item = MagicMock()
        item.embedding = emb
        item.index = i
        items.append(item)
    resp = MagicMock()
    resp.data = items
    return resp


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


class TestGenerate:
    @patch("openai.AzureOpenAI")
    def test_success(self, MockAOAI, sample_embedding):
        mock_client = MagicMock()
        MockAOAI.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_embedding_response(sample_embedding)

        client = _make_client()
        result = client.generate("hello world")

        assert result == sample_embedding
        mock_client.embeddings.create.assert_called_once()

    @patch("openai.AzureOpenAI")
    def test_lazy_init_reuses_client(self, MockAOAI, sample_embedding):
        mock_client = MagicMock()
        MockAOAI.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_embedding_response(sample_embedding)

        client = _make_client()
        client.generate("first")
        client.generate("second")

        # AzureOpenAI constructor should be called only once
        MockAOAI.assert_called_once()
        assert mock_client.embeddings.create.call_count == 2

    @patch("openai.AzureOpenAI")
    def test_with_api_key(self, MockAOAI, sample_embedding):
        mock_client = MagicMock()
        MockAOAI.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_embedding_response(sample_embedding)

        client = _make_client(api_key="my-key", credential=None)
        client.generate("text")

        call_kwargs = MockAOAI.call_args.kwargs
        assert call_kwargs["api_key"] == "my-key"

    @patch("azure.identity.get_bearer_token_provider")
    @patch("openai.AzureOpenAI")
    def test_with_credential(self, MockAOAI, mock_get_token, sample_embedding, mock_credential):
        mock_client = MagicMock()
        MockAOAI.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_embedding_response(sample_embedding)
        mock_get_token.return_value = lambda: "token"

        client = _make_client(api_key=None, credential=mock_credential)
        client.generate("text")

        mock_get_token.assert_called_once()
        call_kwargs = MockAOAI.call_args.kwargs
        assert "azure_ad_token_provider" in call_kwargs

    def test_missing_endpoint(self):
        client = EmbeddingsClient(endpoint=None, api_key="key")
        with pytest.raises(ConfigurationError) as exc_info:
            client.generate("text")
        assert exc_info.value.parameter == "endpoint"

    def test_missing_key_and_credential(self):
        client = EmbeddingsClient(
            endpoint="https://fake.openai.azure.com",
            api_key=None,
            credential=None,
        )
        with pytest.raises(ConfigurationError) as exc_info:
            client.generate("text")
        assert exc_info.value.parameter == "credential"

    @patch("openai.AzureOpenAI")
    def test_api_failure(self, MockAOAI):
        mock_client = MagicMock()
        MockAOAI.return_value = mock_client
        mock_client.embeddings.create.side_effect = RuntimeError("API down")

        client = _make_client()
        with pytest.raises(RuntimeError) as exc_info:
            client.generate("text")
        assert "API down" in str(exc_info.value)

    @patch("openai.AzureOpenAI")
    def test_with_dimensions(self, MockAOAI, sample_embedding):
        mock_client = MagicMock()
        MockAOAI.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_embedding_response(sample_embedding)

        client = _make_client(dimensions=256)
        client.generate("text")

        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert call_kwargs["dimensions"] == 256


# ---------------------------------------------------------------------------
# generate_batch()
# ---------------------------------------------------------------------------


class TestGenerateBatch:
    @patch("openai.AzureOpenAI")
    def test_preserves_order(self, MockAOAI):
        mock_client = MagicMock()
        MockAOAI.return_value = mock_client

        emb_a = [0.1, 0.2]
        emb_b = [0.3, 0.4]

        # Return items out of order to verify sorting by index
        item_b = MagicMock()
        item_b.embedding = emb_b
        item_b.index = 1
        item_a = MagicMock()
        item_a.embedding = emb_a
        item_a.index = 0
        resp = MagicMock()
        resp.data = [item_b, item_a]  # intentionally reversed
        mock_client.embeddings.create.return_value = resp

        client = _make_client()
        result = client.generate_batch(["text_a", "text_b"])

        assert result == [emb_a, emb_b]

    def test_empty_list(self):
        client = _make_client()
        result = client.generate_batch([])
        assert result == []

    @patch("openai.AzureOpenAI")
    def test_batch_api_failure(self, MockAOAI):
        mock_client = MagicMock()
        MockAOAI.return_value = mock_client
        mock_client.embeddings.create.side_effect = RuntimeError("batch fail")

        client = _make_client()
        with pytest.raises(RuntimeError):
            client.generate_batch(["a", "b"])


# ---------------------------------------------------------------------------
# generate_batch() — N=16 chunk guard
# ---------------------------------------------------------------------------


class TestGenerateBatchChunking:
    def test_default_batch_size_constant_is_16(self):
        assert AOAI_EMBEDDING_BATCH_SIZE == 16

    @patch("openai.AzureOpenAI")
    def test_small_batch_calls_api_once(self, MockAOAI):
        mock_client = MagicMock()
        MockAOAI.return_value = mock_client
        # 10 items, within default batch_size=16 → single API call.
        mock_client.embeddings.create.return_value = _mock_batch_response([[float(i)] for i in range(10)])

        client = _make_client()
        result = client.generate_batch([f"t{i}" for i in range(10)])

        assert len(result) == 10
        assert result == [[float(i)] for i in range(10)]
        assert mock_client.embeddings.create.call_count == 1

    @patch("openai.AzureOpenAI")
    def test_large_batch_chunked_into_16_16_8(self, MockAOAI):
        mock_client = MagicMock()
        MockAOAI.return_value = mock_client

        # 40 items → ceil(40/16) = 3 calls with chunk sizes (16, 16, 8).
        # Build response per chunk so each call sees indices 0..chunk_size-1.
        def _create(*, input, model, **_):  # noqa: A002 - mirror OpenAI kwarg
            return _mock_batch_response([[float(t)] for t in [int(x[1:]) for x in input]])

        mock_client.embeddings.create.side_effect = _create

        client = _make_client()
        texts = [f"t{i}" for i in range(40)]
        result = client.generate_batch(texts)

        # Three calls with expected chunk sizes in order.
        assert mock_client.embeddings.create.call_count == 3
        chunk_sizes = [len(call.kwargs["input"]) for call in mock_client.embeddings.create.call_args_list]
        assert chunk_sizes == [16, 16, 8]

        # Results preserved in input order across all chunks.
        assert len(result) == 40
        assert result == [[float(i)] for i in range(40)]

    @patch("openai.AzureOpenAI")
    def test_empty_list_no_api_call(self, MockAOAI):
        mock_client = MagicMock()
        MockAOAI.return_value = mock_client

        client = _make_client()
        result = client.generate_batch([])

        assert result == []
        assert mock_client.embeddings.create.call_count == 0
        # Lazy-init must not even fire — empty short-circuit comes first.
        MockAOAI.assert_not_called()

    @patch("openai.AzureOpenAI")
    def test_custom_batch_size_chunks_accordingly(self, MockAOAI):
        mock_client = MagicMock()
        MockAOAI.return_value = mock_client

        def _create(*, input, model, **_):  # noqa: A002
            return _mock_batch_response([[float(t)] for t in [int(x[1:]) for x in input]])

        mock_client.embeddings.create.side_effect = _create

        client = _make_client()
        texts = [f"t{i}" for i in range(12)]
        result = client.generate_batch(texts, batch_size=5)

        # ceil(12/5) = 3 calls with chunk sizes (5, 5, 2).
        assert mock_client.embeddings.create.call_count == 3
        chunk_sizes = [len(call.kwargs["input"]) for call in mock_client.embeddings.create.call_args_list]
        assert chunk_sizes == [5, 5, 2]
        assert result == [[float(i)] for i in range(12)]
