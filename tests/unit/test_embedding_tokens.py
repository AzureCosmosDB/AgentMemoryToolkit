"""Unit tests for embedding token-budget truncation (option 2 safety net)."""

from __future__ import annotations

import tiktoken

from azure.cosmos.agent_memory._embedding_tokens import (
    EMBEDDING_MAX_INPUT_TOKENS,
    truncate_text_to_token_budget,
)
from azure.cosmos.agent_memory.aio.embeddings import AsyncEmbeddingsClient
from azure.cosmos.agent_memory.embeddings import EmbeddingsClient

_MODEL = "text-embedding-3-large"


def _token_count(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


# ---------------------------------------------------------------------------
# truncate_text_to_token_budget()
# ---------------------------------------------------------------------------


class TestTruncateHelper:
    def test_empty_string_unchanged(self) -> None:
        assert truncate_text_to_token_budget("", _MODEL) == ""

    def test_short_text_unchanged(self) -> None:
        text = "The user prefers dark mode."
        assert truncate_text_to_token_budget(text, _MODEL) is text

    def test_text_within_budget_unchanged(self) -> None:
        # ~1000 tokens, comfortably under the 8000 budget.
        text = "word " * 1000
        assert truncate_text_to_token_budget(text, _MODEL) is text

    def test_oversized_text_truncated_within_budget(self) -> None:
        # ``supercalifragilistic`` tokenizes to several tokens each, so this
        # is well over the 8000-token budget.
        text = "supercalifragilistic " * 3000
        assert _token_count(text) > EMBEDDING_MAX_INPUT_TOKENS

        result = truncate_text_to_token_budget(text, _MODEL)

        assert _token_count(result) <= EMBEDDING_MAX_INPUT_TOKENS
        assert len(result) < len(text)

    def test_truncation_preserves_head(self) -> None:
        head = "UNIQUE_HEAD_MARKER the user asked about billing. "
        text = head + ("filler token stream " * 5000)

        result = truncate_text_to_token_budget(text, _MODEL)

        assert result.startswith("UNIQUE_HEAD_MARKER")

    def test_custom_max_tokens(self) -> None:
        text = "word " * 1000
        result = truncate_text_to_token_budget(text, _MODEL, max_tokens=50)
        assert _token_count(result) <= 50

    def test_unknown_model_falls_back_to_cl100k(self) -> None:
        text = "supercalifragilistic " * 3000
        # An unknown deployment name must still truncate (via cl100k_base).
        result = truncate_text_to_token_budget(text, "some-custom-deployment")
        assert _token_count(result) <= EMBEDDING_MAX_INPUT_TOKENS

    def test_char_fallback_when_tiktoken_unavailable(self, monkeypatch) -> None:
        # Simulate tiktoken being unavailable so the char-based estimate runs.
        monkeypatch.setattr(
            "azure.cosmos.agent_memory._embedding_tokens._encoding_for_model",
            lambda model: None,
        )
        # 4 chars/token fallback → budget * 4 chars. Use a single long token-free
        # string longer than that to force truncation.
        text = "a" * (EMBEDDING_MAX_INPUT_TOKENS * 4 + 1000)
        result = truncate_text_to_token_budget(text, _MODEL)
        assert len(result) == EMBEDDING_MAX_INPUT_TOKENS * 4

    def test_char_fallback_leaves_short_text_untouched(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "azure.cosmos.agent_memory._embedding_tokens._encoding_for_model",
            lambda model: None,
        )
        # Longer than the token budget in chars, but within the char budget.
        text = "a" * (EMBEDDING_MAX_INPUT_TOKENS + 100)
        assert truncate_text_to_token_budget(text, _MODEL) is text


# ---------------------------------------------------------------------------
# Client wiring: _build_kwargs applies truncation on every embed path
# ---------------------------------------------------------------------------


def _oversized() -> str:
    return "supercalifragilistic " * 3000


class TestSyncClientWiring:
    def test_build_kwargs_truncates_single_input(self) -> None:
        client = EmbeddingsClient(endpoint="https://x", api_key="k", model=_MODEL)
        kwargs = client._build_kwargs(_oversized())
        assert _token_count(kwargs["input"][0]) <= EMBEDDING_MAX_INPUT_TOKENS

    def test_build_kwargs_truncates_each_item_in_batch(self) -> None:
        client = EmbeddingsClient(endpoint="https://x", api_key="k", model=_MODEL)
        kwargs = client._build_kwargs([_oversized(), "small text", _oversized()])
        assert all(_token_count(t) <= EMBEDDING_MAX_INPUT_TOKENS for t in kwargs["input"])
        assert kwargs["input"][1] == "small text"


class TestAsyncClientWiring:
    def test_build_kwargs_truncates_single_input(self) -> None:
        client = AsyncEmbeddingsClient(endpoint="https://x", api_key="k", model=_MODEL)
        kwargs = client._build_kwargs(_oversized())
        assert _token_count(kwargs["input"][0]) <= EMBEDDING_MAX_INPUT_TOKENS

    def test_build_kwargs_truncates_each_item_in_batch(self) -> None:
        client = AsyncEmbeddingsClient(endpoint="https://x", api_key="k", model=_MODEL)
        kwargs = client._build_kwargs([_oversized(), "small text"])
        assert all(_token_count(t) <= EMBEDDING_MAX_INPUT_TOKENS for t in kwargs["input"])
        assert kwargs["input"][1] == "small text"
