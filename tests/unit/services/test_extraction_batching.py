"""Unit tests for Option-4 extraction helpers: failure classification + batching."""

from __future__ import annotations

from azure.cosmos.agent_memory.services._pipeline_helpers import (
    batch_turns_by_tokens,
    is_retryable_llm_error,
)


class TestIsRetryableLLMError:
    def test_content_filter_is_non_retryable(self) -> None:
        assert is_retryable_llm_error(Exception("Error code: 400 - content_filter triggered")) is False

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
