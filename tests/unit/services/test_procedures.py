"""Pure helpers of the procedure pass: evidence gates, support bookkeeping, sources, cursor and rule rendering."""

from __future__ import annotations

import pytest

from azure.cosmos.agent_memory.services._dedup import ACTIVE_FILTER, AFTER_POSITION
from azure.cosmos.agent_memory.services._procedures import (
    PROMPT_BATCH,
    STANDING_RULES_MAX,
    STANDING_RULES_MAX_BYTES,
    ProcedureAnswer,
    ProcedureScope,
    advanced_procedure_cursor,
    behavioral_facts_query,
    can_teach,
    in_force,
    is_user_correction,
    lesson_episodes_query,
    procedural_context,
    procedure_cursor_body,
    procedure_id,
    procedure_status,
    procedure_text,
    rank_related,
    rendered_rule,
    retirable_facts,
    revised_procedure,
    source_batches,
    sources_input,
    standing_rules,
    with_support,
)


def _source(source_id: str, *, kind: str = "fact", day: str = "2025-01-01", agent: bool = False, outcome: str = ""):
    doc = {
        "id": source_id,
        "type": kind,
        "created_at": f"{day}T00:00:00+00:00",
        "tags": ["sys:agent-fact"] if agent else [],
    }
    if outcome:
        doc["outcome"] = {"status": outcome}
    return doc


@pytest.mark.parametrize(
    ("kind", "evidence", "user_support", "dates", "practiced", "status"),
    [
        pytest.param("instruction", ["taught"], 1, ["d1"], [], "active", id="instruction-taught-by-the-user"),
        pytest.param("instruction", ["corrected"], 1, ["d1"], [], "active", id="instruction-corrected-by-the-user"),
        pytest.param("instruction", ["taught"], 0, ["d1"], [], "candidate", id="instruction-no-user-statement"),
        pytest.param("instruction", ["repeated"], 1, ["d1", "d2"], [], "candidate", id="instruction-never-taught"),
        pytest.param("habit", ["corrected"], 1, ["d1"], [], "active", id="habit-corrected"),
        pytest.param("habit", ["corrected"], 0, ["d1"], [], "candidate", id="habit-corrected-by-no-user-fact"),
        pytest.param("habit", ["repeated"], 1, ["d1"], [], "candidate", id="habit-one-date"),
        pytest.param("habit", ["repeated"], 0, ["d1", "d2"], [], "active", id="habit-two-dates"),
        pytest.param("skill", ["practiced"], 0, ["d1"], ["d1"], "candidate", id="skill-practiced-once"),
        pytest.param("skill", ["practiced"], 0, ["d1", "d2"], ["d1", "d2"], "active", id="skill-practiced-twice"),
        pytest.param("skill", ["endorsed"], 1, ["d1"], [], "active", id="skill-endorsed"),
        pytest.param("skill", ["endorsed"], 0, ["d1"], [], "candidate", id="skill-endorsed-by-no-user-fact"),
        pytest.param("workflow", ["taught"], 1, ["d1"], [], "candidate", id="unknown-kind"),
    ],
)
def test_procedure_status_follows_the_evidence_each_kind_needs(kind, evidence, user_support, dates, practiced, status):
    doc = {
        "procedure_kind": kind,
        "evidence": evidence,
        "user_support_count": user_support,
        "support_dates": dates,
        "practiced_dates": practiced,
    }

    assert procedure_status(doc) == status


def test_with_support_counts_only_new_sources_and_known_dates():
    doc = {"procedure_kind": "habit", "status": "candidate", "source_fact_ids": ["f1"], "support_dates": ["2025-01-01"]}
    sources = [
        _source("f1"),
        _source("f2", day="2025-02-01"),
        _source("f3", agent=True, day=""),
        _source("e1", kind="episodic", day="2025-03-01", outcome="partially_successful"),
    ]

    out = with_support(doc, sources, "repeated")

    assert out["source_fact_ids"] == ["f1", "f2", "f3"]
    assert out["source_episodic_ids"] == ["e1"]
    assert out["support_dates"] == ["2025-01-01", "2025-02-01", "2025-03-01"]
    assert (out["user_support_count"], out["practiced_dates"]) == (1, ["2025-03-01"])
    assert (out["evidence"], out["status"]) == (["repeated"], "active")
    assert with_support(out, sources, "repeated") is None


def test_a_habit_needs_the_users_own_evidence_on_two_dates():
    # What the assistant said and episodes that failed are not the user choosing the same way twice.
    out = with_support(
        {"procedure_kind": "habit", "status": "candidate"},
        [
            _source("a1", agent=True, day="2025-01-01"),
            _source("a2", agent=True, day="2025-01-02"),
            _source("e1", kind="episodic", day="2025-01-03", outcome="failed"),
        ],
        "repeated",
    )

    assert (out["support_dates"], out["status"]) == ([], "candidate")

    out = with_support(out, [_source("f1", day="2025-02-01"), _source("f2", day="2025-02-02")], "repeated")

    assert (out["support_dates"], out["status"]) == (["2025-02-01", "2025-02-02"], "active")


def test_with_support_never_demotes_an_active_procedure():
    active = {"procedure_kind": "instruction", "status": "active", "evidence": ["taught"], "user_support_count": 0}

    out = with_support(active, [_source("f9", agent=True)], "taught")

    assert (out["status"], out["source_kind"], out["source_authority"]) == (
        "active",
        "explicit_user_instruction",
        "high",
    )


@pytest.mark.parametrize(
    ("evidence", "kind", "authority"),
    [
        ("taught", "explicit_user_instruction", "high"),
        ("endorsed", "explicit_user_instruction", "high"),
        ("corrected", "observed_user_preference", "medium"),
        ("repeated", "observed_user_preference", "medium"),
        ("practiced", "episode_distillation", "medium"),
    ],
)
def test_an_active_procedures_provenance_follows_its_strongest_evidence(evidence, kind, authority):
    doc = {"procedure_kind": "habit", "status": "active", "evidence": []}

    out = with_support(doc, [_source("f1")], evidence)

    assert (out["source_kind"], out["source_authority"]) == (kind, authority)


def test_a_candidates_provenance_is_unverified():
    out = with_support({"procedure_kind": "habit", "status": "candidate"}, [_source("f1")], "repeated")

    assert (out["status"], out["source_kind"], out["source_authority"]) == ("candidate", "agent_inference", "low")


def test_standing_rules_stop_at_the_byte_cap():
    long_rule = "x" * (STANDING_RULES_MAX_BYTES // 2)
    docs = [
        {"type": "procedural", "status": "active", "applies": "always", "summary": f"{index} {long_rule}"}
        for index in range(3)
    ]

    kept, left_out = standing_rules(docs, "2025-01-01")

    assert (len(kept), left_out) == (1, 2)


def test_standing_rules_skip_a_rule_too_big_for_the_cap_and_keep_the_smaller_ones():
    docs = [
        {
            "type": "procedural",
            "status": "active",
            "applies": "always",
            "summary": "x" * (STANDING_RULES_MAX_BYTES + 1),
            "updated_at": "2025-03-01",
        },
        {
            "type": "procedural",
            "status": "active",
            "applies": "always",
            "summary": "Reply in English",
            "updated_at": "2025-02-01",
        },
    ]

    kept, left_out = standing_rules(docs, "2025-01-01")

    assert ([doc["summary"] for doc in kept], left_out) == (["Reply in English"], 1)


def test_standing_rules_stop_at_the_rule_cap():
    docs = [
        {"type": "procedural", "status": "active", "applies": "always", "summary": f"Rule {index}"}
        for index in range(STANDING_RULES_MAX + 3)
    ]

    kept, left_out = standing_rules(docs, "2025-01-01")

    assert (len(kept), left_out) == (STANDING_RULES_MAX, 3)


def test_standing_rules_rank_by_evidence_then_recency_and_skip_what_is_not_in_force():
    def rule(summary, evidence, updated, **extra):
        return {
            "type": "procedural",
            "status": "active",
            "applies": "always",
            "summary": summary,
            "evidence": evidence,
            "updated_at": updated,
            **extra,
        }

    docs = [
        rule("practiced", ["practiced"], "2025-05-01"),
        rule("repeated-old", ["repeated"], "2025-01-01"),
        rule("taught-old", ["taught"], "2025-01-01"),
        rule("corrected", ["corrected"], "2025-02-01"),
        rule("repeated-new", ["repeated", "practiced"], "2025-04-01"),
        rule("endorsed", ["endorsed"], "2025-03-01"),
        rule("taught-new", ["taught"], "2025-03-01"),
        rule("expired", ["taught"], "2025-06-01", valid_until="2024-12-31"),
        rule("candidate", ["taught"], "2025-06-01", status="candidate"),
        rule("situational", ["taught"], "2025-06-01", applies="when"),
        rule("superseded", ["taught"], "2025-06-01", superseded_by="proc_other"),
        rule("  ", ["taught"], "2025-06-01", name=" "),
    ]

    kept, left_out = standing_rules(docs, "2025-01-01")

    assert [doc["summary"] for doc in kept] == [
        "taught-new",
        "taught-old",
        "corrected",
        "repeated-new",
        "repeated-old",
        "endorsed",
        "practiced",
    ]
    assert left_out == 0


@pytest.mark.parametrize(
    ("until", "today", "expected"),
    [
        pytest.param(None, "2025-06-01", True, id="no-end"),
        pytest.param("", "2025-06-01", True, id="blank-end"),
        pytest.param("2025-06-01", "2025-06-01", True, id="last-day"),
        pytest.param("2025-05-31", "2025-06-01", False, id="past"),
        pytest.param("2025-07-01T09:00:00Z", "2025-06-01", True, id="timestamp-keeps-its-date"),
    ],
)
def test_a_procedure_is_in_force_through_its_valid_until_date(until, today, expected):
    assert in_force({"valid_until": until}, today) is expected


@pytest.mark.parametrize(
    ("doc", "rendered"),
    [
        pytest.param({"summary": "Reply in English.", "applies": "always"}, "Reply in English.", id="always"),
        pytest.param(
            {"summary": "Suggest a fix.", "applies": "when", "activation_conditions": ["reviewing code", " "]},
            "When reviewing code: Suggest a fix.",
            id="situational",
        ),
        pytest.param({"summary": "Suggest a fix.", "applies": "when"}, "Suggest a fix.", id="no-conditions"),
        pytest.param({"name": "Reply in English.", "applies": "always"}, "Reply in English.", id="name-only"),
    ],
)
def test_rendered_rule(doc, rendered):
    assert rendered_rule(doc) == rendered


def test_procedural_context_lists_standing_rules_then_the_situational_one():
    standing = [{"summary": "Reply in English.", "applies": "always"}]
    situational = [{"summary": "Suggest a fix.", "applies": "when", "activation_conditions": ["reviewing code"]}]

    assert procedural_context(standing, situational) == (
        "# Standing instructions from the user\n"
        "- Reply in English.\n"
        "\n"
        "## Apply only when the situation matches\n"
        "- When reviewing code: Suggest a fix."
    )
    assert procedural_context([], []) == ""
    assert procedural_context([], situational).startswith("# Standing instructions from the user\n\n## Apply only")


# ---------------------------------------------------------------------------
# Directives and corrections
# ---------------------------------------------------------------------------


def _asserted(source_id: str, assertion: str | None, **kwargs) -> dict:
    doc = _source(source_id, **kwargs)
    if assertion:
        doc["assertion"] = assertion
    return doc


@pytest.mark.parametrize(
    ("source", "teaches"),
    [
        pytest.param(_asserted("f", None), True, id="fact-from-before-assertions"),
        pytest.param(_asserted("f", "directive"), True, id="directive"),
        pytest.param(_asserted("f", "correction"), True, id="correction"),
        pytest.param(_asserted("f", "stated"), False, id="stated-preference"),
        pytest.param(_asserted("f", "decided"), False, id="decision"),
        pytest.param(_asserted("f", "directive", agent=True), False, id="assistant"),
        pytest.param(_source("e", kind="episodic", outcome="successful"), False, id="episode"),
    ],
)
def test_only_the_users_directives_and_corrections_can_teach(source, teaches):
    assert can_teach(source) is teaches


def test_a_preference_read_as_an_instruction_stays_a_candidate_until_the_user_directs_it():
    doc = {"procedure_kind": "instruction", "status": "candidate"}

    preference = with_support(doc, [_asserted("s", "stated")], "taught")
    directed = with_support(preference, [_asserted("d", "directive", day="2025-02-01")], "taught")

    assert (preference["evidence"], preference["user_support_count"], preference["status"]) == ([], 1, "candidate")
    assert (directed["evidence"], directed["status"]) == (["taught"], "active")


@pytest.mark.parametrize(("assertion", "corrects"), [("correction", True), ("directive", True), ("stated", False)])
def test_only_a_teaching_fact_may_change_a_rules_numbers(assertion, corrects):
    answer = ProcedureAnswer(
        action="update",
        kind="habit",
        evidence="corrected",
        name="Use the logger",
        sources=(_asserted("f", assertion),),
        target=None,
        item={},
    )

    assert is_user_correction(answer) is corrects


def test_fact_lines_show_how_a_fact_was_asserted_when_known():
    facts = [
        {**_asserted("d", "directive"), "content": "Use the logger.", "metadata": {"category": "preference"}},
        {**_source("l"), "content": "Likes tea.", "metadata": {"category": "preference"}},
        {**_asserted("x", "invented"), "content": "Odd.", "metadata": {"category": "other"}},
        {**_source("a", agent=True), "content": "Suggested a charger.", "metadata": {}},
    ]

    inputs, labels = sources_input(facts)

    assert inputs["behavioral_facts"].splitlines() == [
        "fact-1 [preference · user · directive · 2025-01-01]: Use the logger.",
        "fact-2 [preference · user · 2025-01-01]: Likes tea.",
        "fact-3 [other · user · 2025-01-01]: Odd.",
        "fact-4 [unknown · assistant · 2025-01-01]: Suggested a charger.",
    ]
    assert inputs["episodes"] == "(none)"
    assert labels["fact-1"]["id"] == "d"


def test_episode_lines_show_the_outcome_and_lessons():
    episode = {
        **_source("e", kind="episodic", day="2025-03-03", outcome="successful"),
        "title": "Fixed flaky tests",
        "content": "Reset the fixture.",
        "lessons": ["Reset fixtures.", " ", "Isolate state."],
    }
    episode["outcome"]["description"] = "The suite passed."

    inputs, labels = sources_input([episode])

    assert inputs["behavioral_facts"] == "(none)"
    assert inputs["episodes"] == (
        "ep-1 [2025-03-03] Fixed flaky tests (outcome: successful - The suite passed.). "
        "Summary: Reset the fixture. Lessons: Reset fixtures.; Isolate state."
    )
    assert labels["ep-1"]["id"] == "e"


def test_the_pass_reads_directives_and_corrections_whatever_their_category_or_salience():
    query, _ = behavioral_facts_query(ProcedureScope("u1"), None)

    assert "c.assertion IN ('directive', 'correction')" in query


def test_an_active_situational_procedure_takes_the_place_of_the_directives_it_cites():
    directive, correction = _asserted("d", "directive"), _asserted("c", "correction")
    others = [
        _asserted("s", "stated"),
        _asserted("l", None),
        {**_asserted("g", "directive"), "superseded_by": "proc_0"},
        _asserted("a", "directive", agent=True),
        _asserted("u", "directive"),
    ]
    proc = {"id": "proc_1", "status": "active", "applies": "when", "source_fact_ids": ["d", "c", "s", "l", "g", "a"]}

    assert [fact["id"] for fact in retirable_facts(proc, [directive, correction, *others])] == ["d", "c"]
    assert retirable_facts({**proc, "applies": "always"}, [directive]) == []
    assert retirable_facts({**proc, "status": "candidate"}, [directive]) == []
    assert retirable_facts({k: v for k, v in proc.items() if k != "id"}, [directive]) == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("before", "after", "revised"),
    [
        pytest.param(
            ("Keep replies short.", "Check the length."),
            ("Keep replies short.", "Verify the length."),
            True,
            id="reworded-step",
        ),
        pytest.param(
            ("Keep replies under 200 words.", "Check the length."),
            ("Keep replies under 200 words, plainly.", "Check the length."),
            True,
            id="added-detail",
        ),
        pytest.param(
            ("Keep replies under 200 words.", "Check the length."),
            ("Keep replies short.", "Check the length."),
            False,
            id="dropped-number",
        ),
        pytest.param(
            ("Never reply in bullets.", "Check the format."),
            ("Reply in bullets.", "Check the format."),
            False,
            id="dropped-negation",
        ),
        pytest.param(
            ("Reply in bullets.", "Check the format."),
            ("Don't reply in bullets.", "Check the format."),
            False,
            id="added-negation",
        ),
    ],
)
def test_a_revision_may_reword_a_step_but_never_drops_a_number_or_a_negation(before, after, revised):
    summary, step = before
    target = {
        "id": "proc_1",
        "name": "Reply briefly",
        "summary": summary,
        "procedure_kind": "habit",
        "steps": [{"sequence": 1, "instruction": step}],
        "version": 1,
        "embedding": [1.0],
    }
    item = {"summary": after[0], "steps": [{"sequence": 1, "instruction": after[1]}]}
    answer = ProcedureAnswer("update", "habit", "repeated", "Reply briefly", (), None, item)

    result = revised_procedure(target, answer, now="2025-01-01T00:00:00+00:00")

    assert (result is not None) is revised
    if result is not None:
        assert (result["version"], result["embedding"], result["id"]) == (2, None, "proc_1")


def test_procedure_text_joins_what_a_procedure_states():
    doc = {
        "name": "Suggest fixes",
        "summary": "Include a fix with every comment.",
        "activation_conditions": ["reviewing code", " "],
        "preconditions": ["has the diff"],
        "safety_constraints": ["no secrets"],
        "steps": [{"instruction": "Find the bug."}, {"instruction": ""}, {"instruction": "Propose the fix."}],
    }

    assert procedure_text(doc) == (
        "Suggest fixes: Include a fix with every comment. | When: reviewing code | Requires: has the diff"
        " | Constraints: no secrets | Steps: Find the bug.; Propose the fix."
    )
    assert procedure_text({"name": "Only a name"}) == "Only a name"


def test_procedure_ids_are_stable_per_user_kind_and_case_folded_name():
    first = procedure_id("u1", "instruction", "Run targeted tests")

    assert first == procedure_id("u1", "instruction", "  run TARGETED tests ")
    assert first.startswith("proc_") and len(first) == len("proc_") + 32
    assert len({first, procedure_id("u2", "instruction", "Run targeted tests")}) == 2
    assert first != procedure_id("u1", "habit", "Run targeted tests")


def test_related_procedures_rank_by_their_best_similarity_and_drop_repeats():
    a, b, c = ({"id": name} for name in "abc")

    ranked = rank_related([(a, 0.2), (b, 0.9), (a, 0.5), (c, 0.4), ({"name": "no id"}, 1.0)])

    assert [doc["id"] for doc in ranked] == ["b", "a", "c"]


def _read(doc_id: str, ts: int, *, kind: str = "fact", lessons: bool = False, content: str = "text") -> dict:
    doc = {"id": doc_id, "type": kind, "_ts": ts, "content": content}
    if kind == "episodic":
        doc["lessons"] = ["A lesson."] if lessons else []
    return doc


def test_source_batches_hold_up_to_a_prompt_of_sources_in_write_order():
    facts = [_read(f"f{n}", n * 2) for n in range(1, PROMPT_BATCH + 3)]
    episodes = [_read("e1", 3, kind="episodic", lessons=True)]

    batches = source_batches(facts, episodes)

    sources_per_batch = [[doc["id"] for doc in sources] for sources, _, _ in batches]
    assert [len(sources) for sources in sources_per_batch] == [PROMPT_BATCH, 3]
    assert sources_per_batch[0][:3] == ["f1", "e1", "f2"]
    first_facts, first_episodes = batches[0][1], batches[0][2]
    assert [doc["id"] for doc in first_episodes] == ["e1"]
    assert all(doc["type"] == "fact" for doc in first_facts)


def test_memories_with_nothing_to_learn_from_still_move_the_cursor_in_a_final_batch():
    facts = [_read("f1", 1), _read("f2", 2, content="  ")]
    episodes = [_read("e1", 3, kind="episodic", lessons=False)]

    batches = source_batches(facts, episodes)

    assert [[doc["id"] for doc in sources] for sources, _, _ in batches] == [["f1"]]
    assert [doc["id"] for doc in batches[0][1]] == ["f1", "f2"]
    assert [doc["id"] for doc in batches[0][2]] == ["e1"]


def test_the_cursor_advances_to_the_last_memory_read_of_each_kind():
    cursor = {"facts": {"last_ts": 5, "ids_at_last_ts": ["old"]}, "episodes": {"last_ts": 0, "ids_at_last_ts": []}}
    facts = [_read("f1", 7), _read("f2", 9), _read("f3", 9)]

    positions = advanced_procedure_cursor(cursor, facts, [])

    assert positions == ({"ts": 9, "ids": ["f2", "f3"]}, {"ts": 0, "ids": []})
    assert advanced_procedure_cursor(cursor, [], []) is None
    # Within one write second the ids already seen stay recorded.
    again = advanced_procedure_cursor(
        {"facts": {"last_ts": 9, "ids_at_last_ts": ["f1"]}}, [_read("f2", 9)], [_read("e1", 4, kind="episodic")]
    )
    assert again == ({"ts": 9, "ids": ["f1", "f2"]}, {"ts": 4, "ids": ["e1"]})


def test_the_cursor_body_keeps_what_the_stored_one_holds_and_drops_system_fields():
    existing = {"id": "procedure_cursor", "_etag": "e", "_ts": 3, "note": "kept"}

    body = procedure_cursor_body(existing, ProcedureScope("u1"), {"ts": 9, "ids": ["f2"]}, {"ts": 4, "ids": []})

    assert body["note"] == "kept"
    assert {key: body[key] for key in ("id", "type", "user_id", "thread_id")} == {
        "id": "procedure_cursor",
        "type": "procedure_cursor",
        "user_id": "u1",
        "thread_id": "__procedural__",
    }
    assert body["facts"] == {"last_ts": 9, "ids_at_last_ts": ["f2"]}
    assert body["episodes"] == {"last_ts": 4, "ids_at_last_ts": []}
    assert not {"_etag", "_ts"} & set(body)


def test_the_source_queries_read_after_the_cursor_within_the_users_memories():
    cursor = {"facts": {"last_ts": 5, "ids_at_last_ts": ["f1"]}, "episodes": {"last_ts": 8, "ids_at_last_ts": ["e1"]}}
    scope = ProcedureScope("u1")

    fact_query, fact_parameters = behavioral_facts_query(scope, cursor)
    episode_query, episode_parameters = lesson_episodes_query(scope, cursor)

    assert scope.partition_key == ["u1", "__procedural__"]
    for query in (fact_query, episode_query):
        assert "c.user_id = @uid" in query
        assert query.endswith(AFTER_POSITION), "the window is the one dedup reads its cursor with"
        assert ACTIVE_FILTER in query
    assert {p["name"]: p["value"] for p in fact_parameters}["@after_ts"] == 5
    assert {p["name"]: p["value"] for p in episode_parameters}["@after_ts"] == 8
    assert {p["name"]: p["value"] for p in episode_parameters}["@type"] == "episodic"
