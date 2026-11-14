"""Every prompt that writes memory text must forbid literal secrets."""

from __future__ import annotations

from pathlib import Path

import pytest

from azure.cosmos.agent_memory.services._pipeline_helpers import PromptyLoader

# Prompts whose output is never stored as memory text: dedup and dedup_verify only classify memories that are
# already stored.
_NON_WRITING_PROMPTS = frozenset({"dedup.prompty", "dedup_verify.prompty"})


def test_memory_writing_prompts_forbid_literal_secrets() -> None:
    """The write guard only redacts fixed credential formats; every prompt that writes memory
    text must forbid the rest, so a new prompt fails here until it carries the rule."""
    prompts = sorted(Path(PromptyLoader().prompts_dir).rglob("*.prompty"))
    writing = [path for path in prompts if path.name not in _NON_WRITING_PROMPTS]
    assert len(writing) >= 8
    for path in writing:
        system = path.read_text(encoding="utf-8").split("\nsystem:\n", 1)[1].split("\nuser:\n", 1)[0]
        assert "Never write a literal secret" in system, path.name


@pytest.mark.parametrize(
    ("filename", "inputs"),
    [
        ("extract_episode.prompty", {"transcript": "[2026-01-01T00:00:00Z | user]: hi"}),
        ("extract_memories-v2.prompty", {"transcript": "[user]: hi"}),
        (
            "extract_memories-v3.prompty",
            {"transcript": "[user]: hi", "previous_turns": "", "thread_summary": "", "known_memories": ""},
        ),
        ("extract_memories.prompty", {"transcript": "[user]: hi"}),
        ("extract_procedure.prompty", {"behavioral_facts": "fact-1: Prefer concise replies.", "episodes": "(none)"}),
        ("combine_memories.prompty", {"memories": "[]", "feedback": ""}),
        ("summarize.prompty", {"transcript": "[user]: hi"}),
        ("summarize_update.prompty", {"prior_summary": "{}", "transcript": "[user]: hi"}),
        ("user_summary.prompty", {"transcript": "[user]: hi"}),
        ("user_summary_update.prompty", {"prior_summary": "{}", "transcript": "[user]: hi"}),
    ],
)
def test_prompts_with_the_secret_rule_still_render(filename: str, inputs: dict[str, str]) -> None:
    messages, params = PromptyLoader().prepare(filename, inputs=inputs)

    assert any("Never write a literal secret" in str(message["content"]) for message in messages)
    assert isinstance(params, dict)
