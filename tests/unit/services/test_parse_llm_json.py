"""Unit tests for parse_llm_json — resilient LLM JSON decoding (Issue A)."""

from __future__ import annotations

import logging

import pytest

from azure.cosmos.agent_memory.exceptions import LLMError
from azure.cosmos.agent_memory.services._pipeline_helpers import parse_llm_json

_HELPER_LOGGER = "azure.cosmos.agent_memory.services._pipeline_helpers"

# The exact string gpt-5.4 produced on the Ruler dataset: a valid object
# followed by a concatenated duplicate of itself.
_DOUBLED = '{"facts":[],"episodic":[],"unclassified":[]} {"facts":[],"episodic":[],"unclassified":[]}'


class TestHappyPath:
    def test_plain_object(self) -> None:
        assert parse_llm_json('{"facts": [], "episodic": []}') == {"facts": [], "episodic": []}

    def test_markdown_fenced_object(self) -> None:
        assert parse_llm_json('```json\n{"facts": [1]}\n```') == {"facts": [1]}

    def test_object_with_surrounding_whitespace(self) -> None:
        assert parse_llm_json('   \n {"a": 1}\n  ') == {"a": 1}


class TestDoubledAndTrailing:
    def test_doubled_object_returns_first(self) -> None:
        # Previously raised "Extra data" and lost the whole extraction.
        assert parse_llm_json(_DOUBLED) == {"facts": [], "episodic": [], "unclassified": []}

    def test_object_then_garbage_is_salvaged(self) -> None:
        assert parse_llm_json('{"facts": [{"text": "x"}]} <end of turn>') == {"facts": [{"text": "x"}]}

    def test_fenced_object_with_trailing_duplicate(self) -> None:
        assert parse_llm_json('```json\n{"a": 1}\n``` {"a": 1}') == {"a": 1}

    def test_trailing_data_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=_HELPER_LOGGER):
            parse_llm_json(_DOUBLED)
        assert any("extra data after the first JSON object" in r.message for r in caplog.records)

    def test_clean_object_emits_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=_HELPER_LOGGER):
            parse_llm_json('{"facts": []}')
        assert not any("extra data" in r.message for r in caplog.records)


class TestTruncationDetectionPreserved:
    def test_unbalanced_braces_flagged_truncated(self) -> None:
        with pytest.raises(LLMError) as exc:
            parse_llm_json('{"facts": [{"text": "the user prefers')
        assert "TRUNCATED" in str(exc.value)

    def test_unterminated_string_flagged_truncated(self) -> None:
        with pytest.raises(LLMError) as exc:
            parse_llm_json('{"facts": [], "note": "unterminated')
        assert "TRUNCATED" in str(exc.value)


class TestGenuineErrors:
    def test_none_raises(self) -> None:
        with pytest.raises(LLMError) as exc:
            parse_llm_json(None)
        assert "no content" in str(exc.value)

    def test_empty_string_raises_invalid_not_truncated(self) -> None:
        with pytest.raises(LLMError) as exc:
            parse_llm_json("")
        assert "invalid JSON" in str(exc.value)
        assert "TRUNCATED" not in str(exc.value)

    def test_non_json_raises_invalid_not_truncated(self) -> None:
        with pytest.raises(LLMError) as exc:
            parse_llm_json("I could not find any memories.")
        assert "invalid JSON" in str(exc.value)
        assert "TRUNCATED" not in str(exc.value)
