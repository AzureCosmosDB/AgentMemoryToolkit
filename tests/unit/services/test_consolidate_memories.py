"""Tests for ``PipelineService.consolidate_memories``: the dedup pass.

Covers:
* input validation and the memory types the pass leaves alone
* which facts a pass checks (the cursor) and compares (nearest neighbours above the floor)
* which verdicts fold, which keep both facts, and the rules no verdict overrides
  (facts that absorbed others, users' statements, the deterministic veto)
* model failures, lost commit races, the cursor write and the non-transactional fold commit
* the ``deadline`` that ends a pass early

The pipeline is constructed via ``PipelineService.__new__`` and patched in place to avoid
requiring a real Cosmos / LLM / embeddings stack. Combining is off here; ``test_combine.py``
covers it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from azure.core import MatchConditions
from azure.cosmos.exceptions import (
    CosmosAccessConditionFailedError,
    CosmosResourceExistsError,
    CosmosResourceNotFoundError,
)

from azure.cosmos.agent_memory.exceptions import LLMError, ValidationError
from azure.cosmos.agent_memory.services import pipeline as pipeline_module
from azure.cosmos.agent_memory.services._dedup import dedup_result, survivor_patch
from azure.cosmos.agent_memory.services.pipeline import PipelineService

NEAR = [1.0, 0.0, 0.0]
NEAR_ENOUGH = [0.9, 0.3, 0.0]  # cosine 0.95 with NEAR
TOO_FAR = [0.8, 0.6, 0.0]  # cosine 0.80 with NEAR: below the 0.85 floor


@pytest.fixture(autouse=True)
def _dedup_only(monkeypatch):
    monkeypatch.setattr("azure.cosmos.agent_memory.thresholds.COMBINE_ENABLED", False)


def _make_pipeline() -> PipelineService:
    p = PipelineService.__new__(PipelineService)
    p._embeddings = MagicMock()
    p._commit_fold = MagicMock(side_effect=lambda survivor, losers, **kwargs: [doc["id"] for doc in losers])
    p._container = MagicMock()
    p._memories_container = p._container
    p._turns_container = p._container
    p._summaries_container = p._container
    p._chat = MagicMock()
    p._container.read_item.side_effect = CosmosResourceNotFoundError(message="no dedup cursor yet")
    return p


def _fact(
    fid: str,
    content: str,
    *,
    vector: list[float] | None = NEAR,
    ts: int = 1,
    created_at: str = "2024-01-01T00:00:00+00:00",
    agent: bool = False,
    merged_from: list[str] | None = None,
) -> dict:
    doc = {
        "id": fid,
        "user_id": "u1",
        "thread_id": "t1",
        "type": "fact",
        "content": content,
        "confidence": 0.8,
        "salience": 0.5,
        "tags": ["sys:fact", "sys:agent-fact"] if agent else ["sys:fact"],
        "metadata": {"source": "agent" if agent else "user"},
        "created_at": created_at,
        "_ts": ts,
        "_etag": f"etag-{fid}",
    }
    if vector is not None:
        doc["embedding"] = vector
    if merged_from:
        doc["merged_from"] = merged_from
    return doc


def _verdict(relation: str = "same_claim", label: str = "C1", *, seed_missing=(), candidate_missing=()) -> dict:
    return {
        "label": label,
        "relation": relation,
        "seed_details_missing_from_candidate": list(seed_missing),
        "candidate_details_missing_from_seed": list(candidate_missing),
    }


def _run(p, seeds, neighbors=None, replies=(), *, cursor=None, n=50, deadline=None):
    """Drive one pass. ``seeds`` answer the cursor query, ``neighbors[seed id]`` the neighbour
    query, and each model call returns the next reply: a list of verdicts, a raw string, or an
    exception to raise. Returns the result and every query the pass ran."""
    neighbors = neighbors or {}
    queries = []

    def query_items(query, parameters, **kwargs):
        params = {item["name"]: item["value"] for item in parameters}
        queries.append((query, params, kwargs))
        if "VectorDistance" in query:
            return iter(neighbors.get(params["@seed_id"], []))
        return iter(seeds)

    p._container.query_items.side_effect = query_items
    if cursor is not None:
        p._container.read_item.side_effect = None
        p._container.read_item.return_value = cursor
    p._run_prompty = MagicMock(
        side_effect=[
            reply if isinstance(reply, (BaseException, str)) else json.dumps({"candidates": reply}) for reply in replies
        ]
    )
    return p.consolidate_memories("u1", n=n, deadline=deadline), queries


def _folds(p) -> list[tuple[str, list[str]]]:
    """Each committed fold as ``(survivor id, retired ids)``."""
    return [(c.args[0]["id"], [doc["id"] for doc in c.args[1]]) for c in p._commit_fold.call_args_list]


class TestConsolidateValidation:
    def test_validates_user_id(self):
        p = _make_pipeline()
        with pytest.raises(ValidationError):
            p.consolidate_memories("")

    def test_validates_n(self):
        p = _make_pipeline()
        with pytest.raises(ValidationError):
            p.consolidate_memories("u1", n=0)
        with pytest.raises(ValidationError):
            p.consolidate_memories("u1", n=-3)
        with pytest.raises(ValidationError, match="<= 500"):
            p.consolidate_memories("u1", n=501)

    @pytest.mark.parametrize("memory_type", ["episodic", "procedural"])
    def test_other_memory_types_are_left_alone(self, memory_type):
        p = _make_pipeline()

        result = p.consolidate_memories("u1", memory_type=memory_type)

        assert result == dedup_result(checked=0, retired_seeds=0, merged=0, compared=0, vetoed=0)
        p._container.query_items.assert_not_called()


class TestWhatAPassChecks:
    def test_without_new_memories_nothing_is_compared_or_written(self):
        p = _make_pipeline()

        result, _ = _run(p, [])

        assert result == dedup_result(checked=0, retired_seeds=0, merged=0, compared=0, vetoed=0)
        p._run_prompty.assert_not_called()
        p._container.create_item.assert_not_called()
        p._container.replace_item.assert_not_called()

    def test_a_fact_without_a_close_neighbour_is_checked_without_asking_the_model(self):
        p = _make_pipeline()
        seed = _fact("f1", "User likes coffee", ts=3)

        result, _ = _run(p, [seed], {"f1": [_fact("f2", "User likes tea", vector=TOO_FAR)]})

        assert (result["checked"], result["compared"], result["kept"]) == (1, 0, 1)
        p._run_prompty.assert_not_called()
        body = p._container.create_item.call_args.kwargs["body"]
        assert (body["id"], body["type"]) == ("dedup_cursor_u1_fact", "dedup_cursor")
        assert (body["user_id"], body["thread_id"]) == ("u1", "__dedup__")
        assert (body["last_ts"], body["ids_at_last_ts"]) == (3, ["f1"])

    def test_a_fact_without_an_embedding_is_checked_without_a_neighbour_query(self):
        p = _make_pipeline()

        result, queries = _run(p, [_fact("f1", "User likes coffee", vector=None)])

        assert (result["checked"], result["compared"]) == (1, 0)
        assert not [q for q, _, _ in queries if "VectorDistance" in q]
        p._container.create_item.assert_called_once()

    def test_a_pass_reads_active_facts_after_its_cursor_oldest_first(self):
        p = _make_pipeline()
        cursor = {
            "id": "dedup_cursor_u1_fact",
            "type": "dedup_cursor",
            "last_ts": 5,
            "ids_at_last_ts": ["f0"],
            "_etag": "c1",
        }

        _, queries = _run(p, [_fact("f1", "User likes coffee", ts=6)], cursor=cursor, n=7)

        sql, params, kwargs = queries[0]
        assert "TOP 7" in sql and "ORDER BY c._ts ASC" in sql
        assert "IS_NULL(c.superseded_by)" in sql and "c.superseded_by = null" not in sql
        assert (params["@user_id"], params["@after_ts"], params["@after_ids"]) == ("u1", 5, ["f0"])
        assert (params["@memory_type"], kwargs["enable_cross_partition_query"]) == ("fact", True)
        replaced = p._container.replace_item.call_args.kwargs
        assert (replaced["item"], replaced["etag"]) == ("dedup_cursor_u1_fact", "c1")
        assert replaced["match_condition"] == MatchConditions.IfNotModified
        assert (replaced["body"]["last_ts"], replaced["body"]["ids_at_last_ts"]) == (6, ["f1"])

    def test_the_cursor_document_is_read_from_the_users_dedup_partition(self):
        p = _make_pipeline()

        _run(p, [])

        assert p._container.read_item.call_args.kwargs == {
            "item": "dedup_cursor_u1_fact",
            "partition_key": ["u1", "__dedup__"],
        }

    def test_neighbours_are_the_nearest_active_facts_of_the_same_user(self):
        p = _make_pipeline()

        _, queries = _run(p, [_fact("f1", "User likes coffee")])

        sql, params, kwargs = queries[1]
        assert "ORDER BY VectorDistance(c.embedding, @embedding)" in sql
        assert "IS_NULL(c.superseded_by)" in sql
        assert (params["@seed_id"], params["@memory_type"], params["@embedding"]) == ("f1", "fact", NEAR)
        assert (params["@user_id"], kwargs["enable_cross_partition_query"]) == ("u1", True)

    def test_only_neighbours_at_or_above_the_similarity_floor_reach_the_model(self):
        p = _make_pipeline()
        seed = _fact("f1", "User likes coffee")
        close = _fact("f2", "The user likes coffee", vector=NEAR_ENOUGH)
        far = _fact("f3", "User likes tea", vector=TOO_FAR)

        result, _ = _run(p, [seed], {"f1": [far, close]}, [[_verdict("related")]])

        candidates = json.loads(p._run_prompty.call_args.kwargs["inputs"]["candidates"])
        assert [c["text"] for c in candidates] == ["The user likes coffee"]
        assert result["compared"] == 1

    def test_a_pair_two_new_facts_share_is_judged_once(self):
        p = _make_pipeline()
        first, second = _fact("f1", "User likes coffee", ts=1), _fact("f2", "User loves coffee", ts=2)

        _run(p, [first, second], {"f1": [second], "f2": [first]}, [[_verdict("related")]])

        assert p._run_prompty.call_count == 1

    def test_the_model_sees_labels_dates_sources_and_text_but_no_ids(self):
        p = _make_pipeline()
        seed = _fact("fact-seed", 'She said "hi" | then left', created_at="2024-03-03T10:00:00+00:00")
        candidate = _fact("fact-cand", "Assistant noted she said hi", agent=True)

        _run(p, [seed], {"fact-seed": [candidate]}, [[_verdict("related")]])

        assert p._run_prompty.call_args.args[0] == "dedup_verify.prompty"
        inputs = p._run_prompty.call_args.kwargs["inputs"]
        assert json.loads(inputs["seed"]) == {
            "said": "2024-03-03",
            "by": "the user",
            "text": 'She said "hi" | then left',
        }
        assert json.loads(inputs["candidates"]) == [
            {"label": "C1", "said": "2024-01-01", "by": "the assistant", "text": "Assistant noted she said hi"}
        ]
        assert "fact-seed" not in json.dumps(inputs) and "fact-cand" not in json.dumps(inputs)


class TestWhichVerdictsFold:
    def test_the_same_claim_retires_the_older_statement_into_the_newer(self):
        p = _make_pipeline()
        older = _fact("f1", "User likes coffee", created_at="2024-01-01T00:00:00+00:00")
        newer = _fact("f2", "The user likes coffee", created_at="2024-02-01T00:00:00+00:00")

        result, _ = _run(p, [older], {"f1": [newer]}, [[_verdict("same_claim")]])

        assert _folds(p) == [("f2", ["f1"])]
        survivor = p._commit_fold.call_args.args[0]
        assert survivor["content"] == "The user likes coffee"
        assert survivor["merged_from"] == ["f1"]
        assert p._commit_fold.call_args.kwargs["survivor_etag"] == "etag-f2"
        assert (result["merged"], result["kept"]) == (1, 0)

    @pytest.mark.parametrize(
        ("relation", "expected"),
        [("seed_contains_candidate", ("f1", ["f2"])), ("candidate_contains_seed", ("f2", ["f1"]))],
    )
    def test_a_contained_fact_is_retired_into_the_one_that_contains_it(self, relation, expected):
        p = _make_pipeline()
        seed, candidate = _fact("f1", "User runs"), _fact("f2", "User runs daily")

        _run(p, [seed], {"f1": [candidate]}, [[_verdict(relation)]])

        assert _folds(p) == [expected]

    def test_a_fold_that_would_lose_a_listed_detail_keeps_both(self):
        p = _make_pipeline()
        seed = _fact("f1", "User keeps karaoke at a safe volume")
        candidate = _fact("f2", "User keeps karaoke volume at 70 decibels")

        result, _ = _run(
            p, [seed], {"f1": [candidate]}, [[_verdict("candidate_contains_seed", seed_missing=["for safety"])]]
        )

        p._commit_fold.assert_not_called()
        assert result["kept"] == 1

    @pytest.mark.parametrize("relation", ["partial_overlap", "related", "unrelated", "update", "inconsistent"])
    def test_facts_that_are_not_duplicates_are_kept(self, relation):
        p = _make_pipeline()

        result, _ = _run(
            p,
            [_fact("f1", "User lives in Portland")],
            {"f1": [_fact("f2", "User moved to Seattle")]},
            [[_verdict(relation)]],
        )

        p._commit_fold.assert_not_called()
        assert (result["merged"], result["contradicted"]) == (0, 0)

    def test_unknown_and_repeated_labels_are_ignored(self):
        p = _make_pipeline()

        _run(
            p,
            [_fact("f1", "User likes coffee")],
            {"f1": [_fact("f2", "The user likes coffee")]},
            [[_verdict("same_claim", label="C7"), _verdict("related"), _verdict("same_claim")]],
        )

        p._commit_fold.assert_not_called()


class TestRulesNoVerdictOverrides:
    def test_the_veto_keeps_a_fact_whose_numbers_the_survivor_lacks(self):
        p = _make_pipeline()
        seed = _fact("f1", "User budgets 150 TRY a month for the gym")
        candidate = _fact("f2", "User has a monthly gym budget and a trainer")

        result, _ = _run(p, [seed], {"f1": [candidate]}, [[_verdict("candidate_contains_seed")]])

        p._commit_fold.assert_not_called()
        assert result["vetoed"] == 1

    def test_a_fact_that_absorbed_others_is_never_retired(self):
        p = _make_pipeline()
        absorber = _fact("f1", "User runs", merged_from=["f0"])
        candidate = _fact("f2", "User runs daily")

        _run(p, [absorber], {"f1": [candidate]}, [[_verdict("candidate_contains_seed")]])

        p._commit_fold.assert_not_called()

    def test_a_fact_that_absorbed_others_can_still_absorb_more(self):
        p = _make_pipeline()
        absorber = _fact("f1", "User runs daily", merged_from=["f0"])

        _run(p, [absorber], {"f1": [_fact("f2", "User runs")]}, [[_verdict("seed_contains_candidate")]])

        assert _folds(p) == [("f1", ["f2"])]
        assert p._commit_fold.call_args.args[0]["merged_from"] == ["f0", "f2"]

    def test_a_users_statement_is_never_retired_into_an_assistant_fact(self):
        p = _make_pipeline()
        said_by_user = _fact("f1", "User runs")
        said_by_assistant = _fact("f2", "User runs daily", agent=True)

        _run(p, [said_by_user], {"f1": [said_by_assistant]}, [[_verdict("candidate_contains_seed")]])

        p._commit_fold.assert_not_called()

    def test_an_assistant_fact_can_be_retired_into_a_users_statement(self):
        p = _make_pipeline()
        said_by_assistant = _fact("f1", "User runs", agent=True)

        _run(p, [said_by_assistant], {"f1": [_fact("f2", "User runs daily")]}, [[_verdict("candidate_contains_seed")]])

        assert _folds(p) == [("f2", ["f1"])]

    def test_a_fact_is_never_retired_into_one_asserted_differently(self):
        p = _make_pipeline()
        suggestion = {**_fact("f1", "User could try a standing desk"), "assertion": "suggested"}
        decision = {**_fact("f2", "User could try a standing desk"), "assertion": "decided"}

        _run(p, [suggestion], {"f1": [decision]}, [[_verdict("same_claim")]])

        p._commit_fold.assert_not_called()

    def test_a_survivor_of_this_pass_is_not_retired_later_in_it(self):
        p = _make_pipeline()
        f1 = _fact("f1", "User likes coffee", ts=1, created_at="2024-01-01T00:00:00+00:00")
        f2 = _fact("f2", "The user likes coffee", ts=2, created_at="2024-02-01T00:00:00+00:00")
        f3 = _fact("f3", "The user likes coffee, black")

        _run(
            p,
            [f1, f2],
            {"f1": [f2], "f2": [f1, f3]},
            [[_verdict("same_claim")], [_verdict("candidate_contains_seed")]],
        )

        assert _folds(p) == [("f2", ["f1"])]

    def test_facts_retired_into_one_survivor_commit_together(self):
        p = _make_pipeline()
        f1, f2 = _fact("f1", "User runs", ts=1), _fact("f2", "User jogs", ts=2)
        f3 = _fact("f3", "User runs and jogs daily")

        result, _ = _run(
            p,
            [f1, f2],
            {"f1": [f3], "f2": [f3]},
            [[_verdict("candidate_contains_seed")], [_verdict("candidate_contains_seed")]],
        )

        assert _folds(p) == [("f3", ["f1", "f2"])]
        assert (result["merged"], result["kept"]) == (2, 0)


class TestModelFailures:
    def test_a_retryable_failure_ends_the_pass_but_keeps_its_progress(self):
        p = _make_pipeline()
        f1, f2, f3 = (_fact(f"f{i}", f"memory {i}", ts=i) for i in (1, 2, 3))
        neighbours = {"f1": [_fact("n1", "memory 1")], "f2": [_fact("n2", "memory 2")], "f3": [_fact("n3", "memory 3")]}

        result, _ = _run(p, [f1, f2, f3], neighbours, [[_verdict("same_claim")], RuntimeError("429 rate limited")])

        assert result["checked"] == 1
        assert len(_folds(p)) == 1
        body = p._container.create_item.call_args.kwargs["body"]
        assert (body["last_ts"], body["ids_at_last_ts"]) == (1, ["f1"])

    @pytest.mark.parametrize(
        "reply",
        [LLMError("content_filter: the response was filtered"), "this is not JSON"],
        ids=["non-retryable-error", "unparseable-answer"],
    )
    def test_an_unusable_check_keeps_the_facts_and_the_pass_moves_on(self, reply):
        p = _make_pipeline()
        f1, f2 = _fact("f1", "memory one", ts=1), _fact("f2", "memory two", ts=2)

        result, _ = _run(
            p,
            [f1, f2],
            {"f1": [_fact("n1", "memory one")], "f2": [_fact("n2", "memory two")]},
            [reply, [_verdict("same_claim")]],
        )

        assert result["checked"] == 2
        # Equal dates and scores: the lexicographically first ID survives.
        assert _folds(p) == [("f2", ["n2"])]


class TestCommitAndCursor:
    def test_a_fold_that_retired_nothing_is_not_counted(self):
        p = _make_pipeline()
        p._commit_fold = MagicMock(return_value=[])

        result, _ = _run(
            p,
            [_fact("f1", "User likes coffee")],
            {"f1": [_fact("f2", "User likes coffee")]},
            [[_verdict("same_claim")]],
        )

        assert (result["merged"], result["kept"]) == (0, 1)
        p._container.create_item.assert_called_once()

    def test_only_the_facts_a_fold_retired_count(self):
        p = _make_pipeline()
        p._commit_fold = MagicMock(return_value=["f1"])
        f1, f2 = _fact("f1", "User runs", ts=1), _fact("f2", "User jogs", ts=2)
        f3 = _fact("f3", "User runs and jogs daily")

        result, _ = _run(
            p,
            [f1, f2],
            {"f1": [f3], "f2": [f3]},
            [[_verdict("candidate_contains_seed")], [_verdict("candidate_contains_seed")]],
        )

        assert (result["merged"], result["kept"]) == (1, 1)

    @pytest.mark.parametrize("error", [CosmosResourceExistsError, CosmosAccessConditionFailedError])
    def test_a_cursor_moved_by_a_concurrent_pass_is_left_alone(self, error):
        p = _make_pipeline()
        cursor = {
            "id": "dedup_cursor_u1_fact",
            "type": "dedup_cursor",
            "last_ts": 1,
            "ids_at_last_ts": [],
            "_etag": "c1",
        }
        p._container.replace_item.side_effect = error(status_code=412, message="moved")

        result, _ = _run(p, [_fact("f1", "User likes coffee", ts=2)], cursor=cursor)

        assert result["checked"] == 1

    def test_the_cursor_is_created_when_none_exists_and_a_concurrent_create_is_left_alone(self):
        p = _make_pipeline()
        p._container.create_item.side_effect = CosmosResourceExistsError(status_code=409, message="exists")

        result, _ = _run(p, [_fact("f1", "User likes coffee", ts=2)])

        assert result["checked"] == 1
        p._container.replace_item.assert_not_called()

    def test_an_unexpected_cursor_error_is_not_swallowed(self):
        p = _make_pipeline()
        p._container.create_item.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            _run(p, [_fact("f1", "User likes coffee", ts=2)])

    def test_a_document_that_is_not_a_cursor_is_ignored(self):
        p = _make_pipeline()

        _, queries = _run(p, [], cursor={"id": "dedup_cursor_u1_fact", "type": "fact", "last_ts": 99})

        assert queries[0][1]["@after_ts"] == 0


class TestCommitFold:
    """The survivor and its duplicates are written one after another: the survivor first."""

    @staticmethod
    def _build(superseded: bool | list[bool] = True) -> PipelineService:
        p = PipelineService.__new__(PipelineService)
        p._container = MagicMock()
        p._memories_container = p._container
        p._mark_superseded = MagicMock(side_effect=superseded if isinstance(superseded, list) else None)
        if not isinstance(superseded, list):
            p._mark_superseded.return_value = superseded
        return p

    @staticmethod
    def _fold(p, *, survivor_etag="etag-s"):
        survivor = _fact("s", "User runs daily")
        losers = [_fact("a", "User runs"), _fact("b", "User jogs")]
        return p._commit_fold(survivor_patch(survivor, losers), losers, survivor_etag=survivor_etag)

    def test_the_survivor_is_replaced_if_unchanged_before_its_duplicates_are_retired(self):
        p = self._build()
        order = []
        p._container.replace_item.side_effect = lambda **kwargs: order.append(("replace", kwargs["item"]))
        p._mark_superseded.side_effect = lambda doc, superseder, **kwargs: order.append(("retire", doc["id"])) or True

        assert self._fold(p) == ["a", "b"]

        assert order == [("replace", "s"), ("retire", "a"), ("retire", "b")]
        replaced = p._container.replace_item.call_args.kwargs
        assert (replaced["etag"], replaced["match_condition"]) == ("etag-s", MatchConditions.IfNotModified)
        assert replaced["body"]["merged_from"] == ["a", "b"]
        assert [(c.args[1], c.kwargs["reason"]) for c in p._mark_superseded.call_args_list] == [
            ("s", "duplicate"),
            ("s", "duplicate"),
        ]

    @pytest.mark.parametrize(
        "error",
        [CosmosAccessConditionFailedError(status_code=412, message="x"), CosmosResourceNotFoundError(message="x")],
        ids=["changed", "deleted"],
    )
    def test_a_survivor_that_changed_or_disappeared_leaves_everything_as_it_was(self, error):
        p = self._build()
        p._container.replace_item.side_effect = error

        assert self._fold(p) == []

        p._mark_superseded.assert_not_called()

    def test_a_duplicate_that_changed_since_the_pass_read_it_stays_active(self):
        p = self._build([True, False])

        assert self._fold(p) == ["a"]

    def test_a_survivor_without_an_etag_is_upserted(self):
        p = self._build()

        assert self._fold(p, survivor_etag=None) == ["a", "b"]

        p._container.replace_item.assert_not_called()
        assert p._container.upsert_item.call_args.kwargs["body"]["id"] == "s"

    def test_a_server_error_while_writing_the_survivor_propagates(self):
        p = self._build()
        p._container.replace_item.side_effect = RuntimeError("503")

        with pytest.raises(RuntimeError):
            self._fold(p)

        p._mark_superseded.assert_not_called()

    def test_folding_again_after_a_crash_between_the_writes_is_idempotent(self):
        p = self._build()
        survivor = _fact("s", "User runs daily")
        losers = [_fact("a", "User runs")]

        first = survivor_patch(survivor, losers)
        again = survivor_patch({**first, "_etag": "etag-s2"}, losers)

        assert again["merged_from"] == first["merged_from"] == ["a"]
        assert p._commit_fold(again, losers, survivor_etag="etag-s2") == ["a"]


class TestConsolidateDeadline:
    def test_a_pass_past_its_deadline_checks_nothing_and_keeps_its_place(self):
        p = _make_pipeline()

        result, _ = _run(p, [_fact("f1", "Lives in Seattle")], deadline=-1.0)

        assert result["checked"] == 0
        p._run_prompty.assert_not_called()
        p._container.create_item.assert_not_called()

    def test_a_pass_stops_at_its_deadline_and_keeps_what_it_checked(self, monkeypatch):
        p = _make_pipeline()
        clock = {"now": 0.0}
        monkeypatch.setattr(pipeline_module, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
        seeds = [_fact("f1", "Lives in Seattle", ts=1), _fact("f2", "Works at Contoso", ts=2)]
        neighbors = {"f1": [_fact("n1", "Lives in Seattle, WA", vector=NEAR_ENOUGH, ts=0)]}

        def slow_check(filename, inputs):
            del filename, inputs
            clock["now"] = 100.0
            return json.dumps({"candidates": [_verdict("candidate_contains_seed")]})

        p._container.query_items.side_effect = lambda query, parameters, **kwargs: iter(
            neighbors.get({i["name"]: i["value"] for i in parameters}.get("@seed_id"), [])
            if "VectorDistance" in query
            else seeds
        )
        p._run_prompty = MagicMock(side_effect=slow_check)

        result = p.consolidate_memories("u1", deadline=50.0)

        assert (result["checked"], result["merged"]) == (1, 1)
        assert _folds(p) == [("n1", ["f1"])]
        assert p._container.create_item.call_args.kwargs["body"]["last_ts"] == 1
