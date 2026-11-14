"""The consolidation prompts (``dedup_verify`` and ``combine_memories``) against the code that feeds and reads them."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from azure.cosmos.agent_memory.services._combine import COMBINE_PROMPT, writer_inputs
from azure.cosmos.agent_memory.services._dedup import DEDUP_PROMPT, parse_verdicts, verify_inputs
from azure.cosmos.agent_memory.services._pipeline_helpers import PromptyLoader

PROMPTS = (DEDUP_PROMPT, COMBINE_PROMPT)
_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "azure" / "cosmos" / "agent_memory" / "prompts"


def _text(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _fact(fid: str, content: str) -> dict:
    return {"id": fid, "content": content, "created_at": "2024-03-03T10:00:00+00:00", "tags": [], "metadata": {}}


@pytest.mark.parametrize("name", PROMPTS)
def test_json_object_mode_needs_the_word_json_in_the_messages(name):
    messages, params = PromptyLoader().prepare(
        name, {"seed": "{}", "candidates": "[]", "memories": "[]", "feedback": ""}
    )

    assert params["response_format"] == {"type": "json_object"}
    assert any("JSON" in message["content"] for message in messages)


@pytest.mark.parametrize("name", PROMPTS)
def test_a_prompt_that_handles_memory_text_never_writes_a_literal_secret(name):
    assert "Never write a literal secret" in _text(name)


def test_every_input_the_dedup_check_declares_is_rendered_and_nothing_is_left_over():
    seed = _fact("s", "The user runs")
    candidate = _fact("c", "The user runs daily")

    messages, _ = PromptyLoader().prepare(DEDUP_PROMPT, verify_inputs(seed, [(candidate, 0.9)]))

    rendered = "\n".join(message["content"] for message in messages)
    assert "The user runs daily" in rendered and '"label": "C1"' in rendered
    assert "{{" not in rendered


def test_every_input_the_writer_declares_is_rendered_and_nothing_is_left_over():
    docs = [_fact("a", "The user plans a dinner"), _fact("b", "The dinner starts at 6:30 PM")]

    messages, _ = PromptyLoader().prepare(COMBINE_PROMPT, writer_inputs(docs, ["the dinner starts at 6:30 PM"]))

    rendered = "\n".join(message["content"] for message in messages)
    assert '"label": "M2"' in rendered and "- the dinner starts at 6:30 PM" in rendered
    assert "{{" not in rendered


def test_the_dedup_check_example_parses_into_the_verdict_it_describes():
    example = json.loads(re.search(r"```json\s*\n(.*?)```", _text(DEDUP_PROMPT), re.DOTALL).group(1))
    candidate = _fact("c", "The race is on 2024-10-05 and the user runs daily")

    [verdict] = parse_verdicts(example, [(candidate, 0.9)])

    assert verdict.relation == "candidate_contains_seed"
    assert (verdict.seed_missing, verdict.candidate_missing) == ((), ("the race is on 2024-10-05",))


def test_the_writers_answer_format_is_the_one_the_pipeline_reads():
    text = _text(COMBINE_PROMPT)

    assert '{"combine": ["M1", "M3"], "memory": "<the combined memory>"}' in text
    assert '{"combine": [], "memory": ""}' in text
