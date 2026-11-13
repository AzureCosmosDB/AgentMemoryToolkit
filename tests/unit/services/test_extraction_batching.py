"""Unit tests for Option-4 extraction helpers: failure classification + batching."""

from __future__ import annotations

import pytest

from azure.cosmos.agent_memory.exceptions import LLMError
from azure.cosmos.agent_memory.services._pipeline_helpers import (
    batch_turns_by_tokens,
    is_retryable_llm_error,
)


class TestIsRetryableLLMError:
    def test_content_filter_is_non_retryable(self) -> None:
        assert is_retryable_llm_error(Exception("Error code: 400 - content_filter triggered")) is False

    @pytest.mark.parametrize("cause", ["model_refusal", "unexpected_tool_response"])
    def test_terminal_empty_reply_causes_are_non_retryable(self, cause: str) -> None:
        assert is_retryable_llm_error(LLMError(f"LLM {cause} (model=gpt-4o-mini)")) is False

    def test_empty_reply_after_retries_stays_retryable(self) -> None:
        error = LLMError("LLM returned empty content after 6 attempts (model=gpt-4o-mini, finish_reason=stop)")

        assert is_retryable_llm_error(error) is True

    def test_content_management_policy_is_non_retryable(self) -> None:
        assert is_retryable_llm_error(Exception("blocked by Azure content management policy")) is False

    def test_context_length_is_non_retryable(self) -> None:
        assert is_retryable_llm_error(Exception("context_length_exceeded")) is False
        assert is_retryable_llm_error(Exception("This model's maximum context length is 8192")) is False

    def test_rate_limit_is_retryable(self) -> None:
        assert is_retryable_llm_error(Exception("Error code: 429 - rate limit exceeded")) is True

    def test_server_error_is_retryable(self) -> None:
        assert is_retryable_llm_error(Exception("Error code: 503 - service unavailable")) is True

    def test_unknown_error_defaults_retryable(self) -> None:
        # Conservative: never quarantine (drop) turns on an unclassified error.
        assert is_retryable_llm_error(Exception("something weird happened")) is True

    def test_programming_errors_are_non_retryable(self) -> None:
        # A bug in our own parse/extract code (e.g. .get on a non-dict) is
        # deterministic - retrying re-fails identically and would wedge the
        # segment/batch forever. Classify by type as non-retryable so it is
        # quarantined and surfaced instead of deferred indefinitely.
        assert is_retryable_llm_error(AttributeError("'list' object has no attribute 'get'")) is False
        assert is_retryable_llm_error(KeyError("episodes")) is False
        assert is_retryable_llm_error(TypeError("unhashable type")) is False
        assert is_retryable_llm_error(IndexError("list index out of range")) is False

    def test_programming_error_type_beats_retryable_looking_message(self) -> None:
        # Type wins over text: a programming error stays non-retryable even when
        # its message would otherwise look transient.
        assert is_retryable_llm_error(KeyError("429 rate limit")) is False

    def test_transient_provider_error_stays_retryable(self) -> None:
        # Genuine transient provider failures are their own SDK exception types
        # (not the programming-error allow-list) and must remain retryable - the
        # allow-list is deliberately narrow, NOT "any non-LLMError".
        assert is_retryable_llm_error(RuntimeError("Error code: 429 - rate limit")) is True


class TestBatchTurnsByTokens:
    def _turns(self, n, content="word " * 10):
        return [{"id": f"t{i}", "content": content} for i in range(n)]

    def test_empty_returns_empty(self) -> None:
        assert batch_turns_by_tokens([], 1000) == []

    def test_small_turns_single_batch_when_budget_large(self) -> None:
        turns = self._turns(5)
        batches = batch_turns_by_tokens(turns, 10_000)
        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_splits_when_budget_small(self) -> None:
        turns = self._turns(6, content="word " * 20)  # ~20 tokens each
        batches = batch_turns_by_tokens(turns, 25)  # ~1 turn per batch
        assert len(batches) == 6
        assert all(len(b) == 1 for b in batches)
        # order + coverage preserved
        assert [t["id"] for b in batches for t in b] == [f"t{i}" for i in range(6)]

    def test_oversized_single_turn_is_own_batch(self) -> None:
        turns = [{"id": "big", "content": "word " * 5000}, {"id": "small", "content": "hi"}]
        batches = batch_turns_by_tokens(turns, 100)
        assert [t["id"] for b in batches for t in b] == ["big", "small"]
        assert len(batches) == 2
