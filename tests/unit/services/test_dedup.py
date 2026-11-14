"""Unit tests for the dedup pass's pure helpers (``azure.cosmos.agent_memory.services._dedup``)."""

from __future__ import annotations

import pytest

from azure.cosmos.agent_memory.exceptions import ValidationError
from azure.cosmos.agent_memory.models import MAX_TAGS_PER_RECORD
from azure.cosmos.agent_memory.services._dedup import (
    ACTIVE_FILTER,
    DEDUP_CURSOR_THREAD_ID,
    DEDUP_CURSOR_TYPE,
    MAX_FOLD_LOSERS,
    Verdict,
    checkpoint_position,
    close_candidates,
    dedup_cursor_body,
    dedup_cursor_id,
    dedup_result,
    is_agent_sourced,
    may_retire,
    neighbor_query,
    next_dedup_cursor,
    next_position,
    parse_verdicts,
    plan_folds,
    preferred_survivor,
    ranked_candidates,
    specifics_preserved,
    survivor_patch,
    unchecked_memories_query,
    validate_consolidation_args,
    verify_inputs,
    without_system_fields,
)
from azure.cosmos.agent_memory.thresholds import DEDUP_MAX_NEIGHBORS


def _doc(fid: str, content: str = "", **extra) -> dict:
    return {"id": fid, "content": content, "tags": ["sys:fact"], "metadata": {"source": "user"}, **extra}


class TestSpecificsPreserved:
    @pytest.mark.parametrize(
        ("retired", "survivor", "kept"),
        [
            ("User budgets 150 TRY for the gym", "User has a gym budget", False),
            ("User budgets 150 TRY for the gym", "User budgets 150 TRY a month for the gym", True),
            ("Rebecca corrected the user's deadlift form", "A trainer corrected the user's deadlift form", False),
            ("Rebecca's plan has three phases", "Rebecca suggested a plan with three phases", True),
            ("The user runs on Monday in March", "the user runs", True),
            ("User has never implemented retry logic", "User implemented retry logic", False),
            ("User doesn't drink coffee", "User does not drink coffee at all", True),
        ],
        ids=[
            "number-missing",
            "number-kept",
            "name-missing",
            "possessive-name-kept",
            "sentence-words-and-calendar-words-are-not-names",
            "negation-lost",
            "negation-kept",
        ],
    )
    def test_numbers_names_and_negation_must_survive(self, retired, survivor, kept):
        assert specifics_preserved(retired, survivor) is kept

    def test_names_can_be_left_out_of_the_check(self):
        # Procedures: a reworded imperative step reads as a changed name.
        before, after = "Steps: Check the length.", "Steps: Verify the length."

        assert specifics_preserved(before, after) is False
        assert specifics_preserved(before, after, names=False) is True
        assert specifics_preserved("Never keep 3 bullets", "Keep 3 bullets", names=False) is False
        assert specifics_preserved("Keep 3 bullets", "Keep bullets", names=False) is False


class TestCandidates:
    def test_close_candidates_are_scored_here_filtered_by_the_floor_and_capped(self):
        seed = _doc("seed", embedding=[1.0, 0.0])
        neighbours = [_doc("seed", embedding=[1.0, 0.0]), _doc("far", embedding=[0.8, 0.6])] + [
            _doc(f"n{i}", embedding=[1.0, 0.01 * i]) for i in range(DEDUP_MAX_NEIGHBORS + 2)
        ]

        close = close_candidates(seed, neighbours)

        assert [doc["id"] for doc, _ in close] == [f"n{i}" for i in range(DEDUP_MAX_NEIGHBORS)]
        assert all(score >= 0.85 for _, score in close)

    def test_a_candidate_must_also_pass_the_callers_own_test(self):
        seed = _doc("seed", embedding=[1.0, 0.0])
        neighbours = [_doc("low", embedding=[0.8, 0.6]), _doc("close", embedding=[1.0, 0.1]), _doc("seed")]

        close = ranked_candidates(seed, neighbours, lambda doc, score: score >= 0.5 and doc["id"] != "close")

        assert [doc["id"] for doc, _ in close] == ["low"]

    def test_a_neighbour_without_a_usable_vector_is_never_a_candidate(self):
        seed = _doc("seed", embedding=[1.0, 0.0])
        neighbours = [_doc("none"), _doc("empty", embedding=[]), _doc("short", embedding=[1.0])]

        assert close_candidates(seed, neighbours) == []

    def test_candidates_reach_the_model_labeled_without_ids(self):
        seed = _doc("fact-seed", 'She said "hi" | then left', created_at="2024-03-03T10:00:00+00:00")
        candidate = _doc("fact-cand", "Assistant noted she said hi", tags=["sys:agent-fact"])

        inputs = verify_inputs(seed, [(candidate, 0.9)])

        assert '"said": "2024-03-03"' in inputs["seed"] and '"by": "the user"' in inputs["seed"]
        assert '"label": "C1"' in inputs["candidates"] and '"by": "the assistant"' in inputs["candidates"]
        assert "fact-seed" not in str(inputs) and "fact-cand" not in str(inputs)


class TestAgentSourcing:
    @pytest.mark.parametrize(
        ("doc", "agent"),
        [
            ({"tags": ["sys:fact"], "metadata": {"source": "user"}}, False),
            ({"tags": ["sys:fact", "sys:agent-fact"], "metadata": {"source": "user"}}, True),
            ({"tags": ["sys:fact"], "metadata": {"source": "agent"}}, True),
            ({"tags": None, "metadata": None}, False),
            ({}, False),
        ],
        ids=["user", "tagged", "metadata-source", "null-fields", "bare"],
    )
    def test_a_fact_is_agent_sourced_by_its_tag_or_its_metadata_source(self, doc, agent):
        assert is_agent_sourced(doc) is agent


class TestVerdicts:
    def test_an_unusable_missing_details_list_blocks_the_fold_it_guards(self):
        seed, candidate = _doc("s", "User runs"), _doc("c", "User runs daily")
        [verdict] = parse_verdicts(
            {
                "candidates": [
                    {
                        "label": "C1",
                        "relation": "candidate_contains_seed",
                        "seed_details_missing_from_candidate": "none",
                        "candidate_details_missing_from_seed": [],
                    }
                ]
            },
            [(candidate, 0.95)],
        )

        assert verdict.seed_missing is None
        assert plan_folds([(seed, [verdict])]).retired == {}

    def test_a_relation_outside_the_schema_is_dropped(self):
        assert parse_verdicts({"candidates": [{"label": "C1", "relation": "duplicate"}]}, [(_doc("c"), 0.9)]) == []

    def test_unknown_and_repeated_labels_are_dropped(self):
        answer = {
            "candidates": [
                {"label": "C7", "relation": "same_claim"},
                {"label": "C1", "relation": "related"},
                {"label": "C1", "relation": "same_claim"},
            ]
        }

        verdicts = parse_verdicts(answer, [(_doc("c"), 0.9)])

        assert [(verdict.relation, verdict.candidate["id"]) for verdict in verdicts] == [("related", "c")]

    def test_updates_and_inconsistencies_are_never_folded(self):
        seed = _doc("s", "Endpoint is eastus")
        verdicts = [
            Verdict(_doc("a", "Endpoint is westus"), 0.9, "update", ("eastus",), ("westus",)),
            Verdict(_doc("b", "Endpoint is centralus"), 0.9, "inconsistent", ("eastus",), ("centralus",)),
        ]

        plan = plan_folds([(seed, verdicts)])

        assert (plan.retired, plan.vetoed) == ({}, 0)


class TestFoldOrder:
    def test_paraphrases_converge_on_the_statement_that_contains_them(self):
        lives = _doc("a", "The user lives in Seattle", created_at="2024-01-01")
        state = _doc("b", "The user lives in Seattle, WA", created_at="2024-01-02")
        resides = _doc("c", "The user resides in Seattle", created_at="2024-01-03")
        checked = [
            (
                lives,
                [
                    Verdict(resides, 0.97, "same_claim", (), ()),
                    Verdict(state, 0.93, "candidate_contains_seed", (), ("WA",)),
                ],
            ),
            (state, [Verdict(resides, 0.93, "seed_contains_candidate", ("WA",), ())]),
        ]

        plan = plan_folds(checked)

        assert {sid: [doc["id"] for doc in docs] for sid, docs in plan.retired.items()} == {"b": ["a", "c"]}

    def test_a_fact_is_retired_at_most_once_and_never_into_a_fact_already_retired(self):
        a, b, c = _doc("a", "User runs"), _doc("b", "User runs daily"), _doc("c", "User runs daily at dawn")
        checked = [
            (a, [Verdict(b, 0.95, "candidate_contains_seed", (), ("daily",))]),
            (b, [Verdict(c, 0.95, "candidate_contains_seed", (), ("at dawn",))]),
        ]

        plan = plan_folds(checked)

        # b already absorbed a in this pass, so it is not retired into c.
        assert {sid: [doc["id"] for doc in docs] for sid, docs in plan.retired.items()} == {"b": ["a"]}

    def test_a_survivor_takes_at_most_max_fold_losers(self):
        survivor = _doc("s", "User runs daily at dawn")
        losers = [_doc(f"l{i}", "User runs") for i in range(MAX_FOLD_LOSERS + 3)]
        checked = [(loser, [Verdict(survivor, 0.95, "candidate_contains_seed", (), ())]) for loser in losers]

        plan = plan_folds(checked)

        assert len(plan.retired["s"]) == MAX_FOLD_LOSERS


class TestAssertionsSurviveFolds:
    @staticmethod
    def _asserted(fid: str, assertion: str | None, **extra) -> dict:
        return _doc(
            fid,
            "The user wants the assistant to cite sources",
            **extra,
            **({"assertion": assertion} if assertion else {}),
        )

    @pytest.mark.parametrize(
        ("retired", "survivor", "allowed"),
        [
            pytest.param(None, None, True, id="neither-recorded"),
            pytest.param(None, "directive", True, id="older-fact-into-a-directive"),
            pytest.param("stated", None, True, id="stated-into-an-older-fact"),
            pytest.param("stated", "stated", True, id="same-assertion"),
            pytest.param("directive", "directive", True, id="directive-into-a-directive"),
            pytest.param("directive", None, False, id="directive-into-an-older-fact"),
            pytest.param("correction", "stated", False, id="correction-into-a-statement"),
            pytest.param("suggested", "decided", False, id="suggestion-into-its-decision"),
            pytest.param("decided", "suggested", False, id="decision-into-a-suggestion"),
        ],
    )
    def test_a_fact_is_retired_only_into_one_asserted_the_same_way(self, retired, survivor, allowed):
        assert may_retire(self._asserted("r", retired), self._asserted("s", survivor)) is allowed

    def test_a_directive_absorbs_a_newer_paraphrase_that_records_no_assertion(self):
        """Recency would keep the newer paraphrase; the directive survives instead, so it still teaches."""
        directive = self._asserted("d", "directive", created_at="2024-01-01")
        paraphrase = self._asserted("p", None, created_at="2024-02-01")

        plan = plan_folds([(directive, [Verdict(paraphrase, 0.97, "same_claim", (), ())])])

        assert {sid: [doc["id"] for doc in docs] for sid, docs in plan.retired.items()} == {"d": ["p"]}


class TestMayRetire:
    def test_a_fact_that_absorbed_others_is_never_retired(self):
        assert may_retire(_doc("a", merged_from=["x"]), _doc("b")) is False

    def test_a_users_statement_is_never_retired_into_an_assistant_fact(self):
        user, agent = _doc("u"), _doc("a", tags=["sys:fact", "sys:agent-fact"])

        assert may_retire(user, agent) is False
        assert may_retire(agent, user) is True
        assert may_retire(agent, _doc("a2", metadata={"source": "agent"})) is True


class TestPreferredSurvivor:
    def test_newer_then_more_confident_then_more_salient_then_first_id(self):
        old = _doc("b", created_at="2024-01-01", confidence=0.9)
        new = _doc("c", created_at="2024-02-01", confidence=0.1)
        assert preferred_survivor(old, new)["id"] == "c"

        sure = _doc("b", created_at="2024-01-01", confidence=0.9, salience=0.1)
        unsure = _doc("a", created_at="2024-01-01", confidence=0.5, salience=0.9)
        assert preferred_survivor(unsure, sure)["id"] == "b"

        tie_b = _doc("b", created_at="2024-01-01", confidence=0.5, salience=0.5)
        tie_a = _doc("a", created_at="2024-01-01", confidence=0.5, salience=0.5)
        assert preferred_survivor(tie_b, tie_a)["id"] == "a"


class TestCursorAndQueries:
    def test_the_position_after_a_prefix_of_a_window_never_moves_back(self):
        position = {"ts": 5, "ids": ["a"]}

        assert next_position(position, []) is None
        assert next_position(position, [{"id": "b", "_ts": 5}, {"id": "c", "_ts": 5}]) == {
            "ts": 5,
            "ids": ["a", "b", "c"],
        }
        assert next_position(position, [{"id": "b", "_ts": 6}, {"id": "d", "_ts": 7}]) == {"ts": 7, "ids": ["d"]}
        assert next_position(position, [{"id": "z", "_ts": 3}]) == position
        assert next_position(None, [{"id": "x", "_ts": 2}]) == {"ts": 2, "ids": ["x"]}

    def test_a_stored_cursor_reads_as_a_position_and_anything_else_as_the_start(self):
        assert checkpoint_position({"last_ts": "9", "ids_at_last_ts": ["b", "a", "", None]}) == {
            "ts": 9,
            "ids": ["a", "b"],
        }
        assert checkpoint_position(None) == checkpoint_position({"last_ts": None}) == {"ts": 0, "ids": []}

    def test_system_fields_are_left_out_of_a_body_to_write(self):
        stored = {"id": "x", "_etag": "e", "_ts": 1, "_rid": "r", "type": "fact"}

        assert without_system_fields(stored) == {"id": "x", "type": "fact"}
        assert without_system_fields(None) == {}

    def test_the_cursor_never_moves_back_and_collects_ids_written_in_the_same_second(self):
        existing = {"last_ts": 5, "ids_at_last_ts": ["a"]}

        assert next_dedup_cursor(existing, []) is None
        same_second = next_dedup_cursor(existing, [{"id": "b", "_ts": 5}, {"id": "c", "_ts": 5}])
        assert (same_second["ts"], sorted(same_second["ids"])) == (5, ["a", "b", "c"])
        later = next_dedup_cursor(existing, [{"id": "b", "_ts": 6}, {"id": "d", "_ts": 7}])
        assert (later["ts"], later["ids"]) == (7, ["d"])
        behind = next_dedup_cursor(existing, [{"id": "z", "_ts": 3}])
        assert (behind["ts"], behind["ids"]) == (5, ["a"])

    def test_a_cursor_document_is_user_wide_and_keeps_what_it_already_holds(self):
        existing = {"id": "dedup_cursor_u1_fact", "extra": 1, "_etag": "e1", "_ts": 9}

        body = dedup_cursor_body(existing, "u1", "fact", {"ts": 7, "ids": ["a"]})

        assert (body["id"], body["type"]) == (dedup_cursor_id("u1", "fact"), DEDUP_CURSOR_TYPE)
        assert (body["user_id"], body["thread_id"]) == ("u1", DEDUP_CURSOR_THREAD_ID)
        assert (body["last_ts"], body["ids_at_last_ts"], body["extra"]) == (7, ["a"], 1)
        assert not [key for key in body if key.startswith("_")]

    def test_unchecked_facts_are_the_users_active_ones_after_the_cursor_oldest_first(self):
        cursor = {"last_ts": 5, "ids_at_last_ts": ["f0"]}

        sql, params = unchecked_memories_query("u1", "fact", cursor, 7)

        assert "TOP 7" in sql and "ORDER BY c._ts ASC" in sql and "c.user_id = @user_id" in sql
        assert ACTIVE_FILTER in sql and "c.superseded_by = null" not in sql
        assert {p["name"]: p["value"] for p in params} == {
            "@user_id": "u1",
            "@memory_type": "fact",
            "@after_ts": 5,
            "@after_ids": ["f0"],
        }

    def test_neighbours_are_the_nearest_active_facts_of_the_same_user(self):
        sql, params = neighbor_query("u1", "fact", {"id": "f1", "embedding": [1.0, 0.0]})

        assert f"TOP {DEDUP_MAX_NEIGHBORS}" in sql and "ORDER BY VectorDistance(c.embedding, @embedding)" in sql
        assert ACTIVE_FILTER in sql and "c.user_id = @user_id" in sql and "c.id != @seed_id" in sql
        assert {p["name"]: p["value"] for p in params} == {
            "@user_id": "u1",
            "@memory_type": "fact",
            "@seed_id": "f1",
            "@embedding": [1.0, 0.0],
        }


class TestSurvivorProvenance:
    def test_a_survivor_keeps_its_content_and_gains_its_duplicates_tags_and_best_scores(self):
        survivor = _doc("b", "The user runs daily", salience=0.4, confidence=0.9, tags=["sys:fact", "topic:a"])
        survivor["_etag"] = "etag-b"
        loser = _doc("a", "User runs", salience=0.8, confidence=0.5, tags=["sys:fact", "topic:b"])

        patch = survivor_patch(survivor, [loser])

        assert patch["content"] == "The user runs daily"
        assert patch["tags"] == ["sys:fact", "topic:a", "topic:b"]
        assert (patch["salience"], patch["confidence"]) == (0.8, 0.9)
        assert patch["merged_from"] == ["a"]
        assert not [key for key in patch if key.startswith("_")]

    def test_what_a_survivor_already_absorbed_is_kept_and_extended(self):
        survivor = _doc("s", merged_from=["old"])

        patch = survivor_patch(survivor, [_doc("a"), _doc("b")])

        assert patch["merged_from"] == ["a", "b", "old"]

    def test_a_user_fact_that_absorbs_an_agent_paraphrase_stays_open_to_contradiction(self):
        survivor = _doc("s", "User runs daily")
        loser = _doc("a", "Assistant noted the user runs", tags=["sys:fact", "sys:agent-fact"])

        assert "sys:agent-fact" not in survivor_patch(survivor, [loser])["tags"]
        agent_survivor = _doc("s2", tags=["sys:fact", "sys:agent-fact"])
        assert "sys:agent-fact" in survivor_patch(agent_survivor, [_doc("b")])["tags"]

    def test_source_turns_are_unioned_only_when_facts_record_them(self):
        assert "source_turn_ids" not in survivor_patch(_doc("s"), [_doc("a")])

        patch = survivor_patch(_doc("s", source_turn_ids=["t2"]), [_doc("a", source_turn_ids=["t1"]), _doc("b")])

        assert patch["source_turn_ids"] == ["t1", "t2"]

    def test_a_survivor_never_ends_up_with_more_tags_than_a_record_may_hold(self):
        survivor = _doc("s", tags=["sys:fact", "topic:own"])
        losers = [_doc(f"l{i}", tags=[f"topic:t{i:02d}" for i in range(30)]) for i in range(3)]
        losers.append(_doc("last", tags=[f"topic:u{i:02d}" for i in range(40)]))

        patch = survivor_patch(survivor, losers)

        assert len(patch["tags"]) == MAX_TAGS_PER_RECORD
        assert {"sys:fact", "topic:own"} <= set(patch["tags"])

    def test_no_freshness_fields_are_written(self):
        patch = survivor_patch(_doc("s", created_at="2024-01-01"), [_doc("a", created_at="2024-03-01")])

        assert not {"confirmed_at", "valid_until", "durability"} & set(patch)


class TestResultAndArguments:
    def test_a_passes_counts_report_what_it_kept_and_retired(self):
        result = dedup_result(checked=4, retired_seeds=1, merged=2, compared=3, vetoed=1)

        assert result == {
            "kept": 3,
            "merged": 2,
            "contradicted": 0,
            "checked": 4,
            "compared": 3,
            "vetoed": 1,
            "combined": 0,
            "synthesized": 0,
            "combine_groups": 0,
            "combine_failed": 0,
        }

    @pytest.mark.parametrize(
        ("user_id", "n", "memory_type", "message"),
        [
            ("", 5, "fact", "user_id is required"),
            ("u1", 0, "fact", "positive integer"),
            ("u1", True, "fact", "positive integer"),
            ("u1", 501, "fact", "<= 500"),
            ("u1", 5, "turn", "memory_type must be one of"),
        ],
    )
    def test_invalid_arguments_are_refused(self, user_id, n, memory_type, message):
        with pytest.raises(ValidationError, match=message):
            validate_consolidation_args(user_id, n, memory_type)

    def test_valid_arguments_pass(self):
        validate_consolidation_args("u1", 500, "episodic")
