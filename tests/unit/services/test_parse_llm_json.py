"""Unit tests for parse_llm_json - resilient LLM JSON decoding (Issue A)."""

from __future__ import annotations

import logging

import pytest

from azure.cosmos.agent_memory.exceptions import LLMError
from azure.cosmos.agent_memory.logging import describe_exception
from azure.cosmos.agent_memory.services._pipeline_helpers import is_retryable_llm_error, parse_llm_json

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

    def test_doubled_object_emits_merge_info(self, caplog: pytest.LogCaptureFixture) -> None:
        # Two clean concatenated objects are merged (so no items are dropped);
        # this is a recoverable case logged at INFO, not a warning.
        with caplog.at_level(logging.INFO, logger=_HELPER_LOGGER):
            parse_llm_json(_DOUBLED)
        assert any("concatenated JSON objects" in r.message for r in caplog.records)

    def test_doubled_object_with_nonempty_lists_concatenates(self) -> None:
        # Discriminating merge test: list fields from BOTH objects are combined,
        # so a broken (no-op) merge that kept only the first object would fail here.
        doubled = '{"facts":[{"t":"a"}],"episodic":[]} {"facts":[{"t":"b"}],"episodic":[{"e":"x"}]}'
        merged = parse_llm_json(doubled)
        assert merged["facts"] == [{"t": "a"}, {"t": "b"}]
        assert merged["episodic"] == [{"e": "x"}]

    def test_trailing_garbage_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        # A valid object followed by non-JSON garbage is salvaged but warned.
        with caplog.at_level(logging.WARNING, logger=_HELPER_LOGGER):
            parse_llm_json('{"facts": [{"text": "x"}]} <end of turn>')
        assert any("non-JSON trailing data" in r.message for r in caplog.records)

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

    @pytest.mark.parametrize(
        "reply",
        ['{"facts": [{"text": "the user prefers', '{"facts": [], "note": "unterminated'],
        ids=["unbalanced-braces", "unterminated-string"],
    )
    def test_a_truncated_reply_is_not_retryable_and_its_error_code_shows_in_a_log(self, reply: str) -> None:
        # Asking again sends the same input and gets a reply cut off at the same place, so the batch that drew it is
        # quarantined, not deferred to a run that fails the same way. describe_exception, which every log uses, leaves
        # the message and its 'maxOutputTokens' advice out: the error code is what tells an operator what happened.
        with pytest.raises(LLMError) as exc:
            parse_llm_json(reply)
        assert is_retryable_llm_error(exc.value) is False
        assert exc.value.error_code == "output_truncated"
        assert "code=output_truncated" in describe_exception(exc.value)
        assert "maxOutputTokens" in str(exc.value)


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


class TestNonObjectRoot:
    """A non-object JSON root must raise LLMError, not leak a non-dict.

    Callers immediately do ``parsed.get("facts"/"episodes")``; a bare list/scalar
    would raise AttributeError downstream, which the extraction error-handlers
    then misclassify as a transient (retryable) failure and defer forever.
    """

    def test_bare_array_root_raises(self) -> None:
        with pytest.raises(LLMError) as exc:
            parse_llm_json('[{"text": "x"}]')
        assert "non-object JSON root" in str(exc.value)

    def test_bare_array_root_with_trailing_data_raises(self) -> None:
        with pytest.raises(LLMError) as exc:
            parse_llm_json('[1, 2] {"a": 1}')
        assert "non-object JSON root" in str(exc.value)

    def test_scalar_number_root_raises(self) -> None:
        with pytest.raises(LLMError):
            parse_llm_json("42")

    def test_scalar_string_root_raises(self) -> None:
        with pytest.raises(LLMError):
            parse_llm_json('"just a sentence"')


_MODEL_TEXT = "zx-model-ana-prefers-window-seats-9902"


class TestErrorTextHoldsNoModelOutput:
    """The error names a position and size, never the response: it is logged and can reach Durable history."""

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            pytest.param(f'{{"facts": [{{"text": "{_MODEL_TEXT}"}}', "TRUNCATED", id="truncated"),
            pytest.param(f'{{"facts": {_MODEL_TEXT}}}', "invalid JSON", id="invalid"),
            pytest.param(f'["{_MODEL_TEXT}"]', "non-object JSON root", id="non-object-root"),
        ],
    )
    def test_unparseable_model_output_stays_out_of_the_error(self, response: str, expected: str) -> None:
        with pytest.raises(LLMError, match=expected) as exc:
            parse_llm_json(response)
        assert _MODEL_TEXT not in str(exc.value)
