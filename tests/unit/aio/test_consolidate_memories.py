"""Async mirror of the ``consolidate_memories`` tests (the dedup pass).

``AsyncPipelineService.consolidate_memories`` makes the same decisions as the sync pipeline
(see ``tests/unit/services/test_consolidate_memories.py``); these tests drive the async I/O
paths: awaited queries and model calls, the cursor write and the non-transactional fold commit.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from azure.core import MatchConditions
from azure.cosmos.exceptions import (
    CosmosAccessConditionFailedError,
    CosmosResourceExistsError,
    CosmosResourceNotFoundError,
)

from azure.cosmos.agent_memory.aio.services import pipeline as pipeline_module
from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService
from azure.cosmos.agent_memory.exceptions import LLMError, ValidationError
from azure.cosmos.agent_memory.services._dedup import dedup_result, survivor_patch

NEAR = [1.0, 0.0, 0.0]
NEAR_ENOUGH = [0.9, 0.3, 0.0]  # cosine 0.95 with NEAR
TOO_FAR = [0.8, 0.6, 0.0]  # cosine 0.80 with NEAR: below the 0.85 floor


@pytest.fixture(autouse=True)
def _dedup_only(monkeypatch):
    monkeypatch.setattr("azure.cosmos.agent_memory.thresholds.COMBINE_ENABLED", False)


def _make_pipeline() -> AsyncPipelineService:
    p = AsyncPipelineService.__new__(AsyncPipelineService)
    p._embeddings = MagicMock()
    p._commit_fold = AsyncMock(side_effect=lambda survivor, losers, **kwargs: [doc["id"] for doc in losers])
    p._container = MagicMock()
    p._container.query_items = AsyncMock(return_value=[])
    p._container.read_item = AsyncMock(side_effect=CosmosResourceNotFoundError(message="no dedup cursor yet"))
    p._container.create_item = AsyncMock()
    p._container.replace_item = AsyncMock()
    p._memories_container = p._container
    p._turns_container = p._container
    p._summaries_container = p._container
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


async def _run(p, seeds, neighbors=None, replies=(), *, cursor=None, n=50, deadline=None):
    """Drive one pass; see the sync tests' ``_run``. Returns the result and every query the pass ran."""
    neighbors = neighbors or {}
    queries = []

    def query_items(query, parameters, **kwargs):
        params = {item["name"]: item["value"] for item in parameters}
        queries.append((query, params, kwargs))
        if "VectorDistance" in query:
            return list(neighbors.get(params["@seed_id"], []))
        return list(seeds)

    p._container.query_items = AsyncMock(side_effect=query_items)
    if cursor is not None:
        p._container.read_item = AsyncMock(return_value=cursor)
    p._run_prompty = AsyncMock(
        side_effect=[
            reply if isinstance(reply, (BaseException, str)) else json.dumps({"candidates": reply}) for reply in replies
        ]
    )
    return await p.consolidate_memories("u1", n=n, deadline=deadline), queries


def _folds(p) -> list[tuple[str, list[str]]]:
    return [(c.args[0]["id"], [doc["id"] for doc in c.args[1]]) for c in p._commit_fold.call_args_list]


class TestConsolidateValidation:
    async def test_validates_arguments(self):
        p = _make_pipeline()
        with pytest.raises(ValidationError):
            await p.consolidate_memories("")
        with pytest.raises(ValidationError):
            await p.consolidate_memories("u1", n=0)
        with pytest.raises(ValidationError, match="<= 500"):
            await p.consolidate_memories("u1", n=501)

    @pytest.mark.parametrize("memory_type", ["episodic", "procedural"])
    async def test_other_memory_types_are_left_alone(self, memory_type):
        p = _make_pipeline()

        result = await p.consolidate_memories("u1", memory_type=memory_type)

        assert result == dedup_result(checked=0, retired_seeds=0, merged=0, compared=0, vetoed=0)
        p._container.query_items.assert_not_called()


class TestWhatAPassChecks:
    async def test_without_new_memories_nothing_is_compared_or_written(self):
        p = _make_pipeline()

        result, _ = await _run(p, [])

        assert result == dedup_result(checked=0, retired_seeds=0, merged=0, compared=0, vetoed=0)
        p._run_prompty.assert_not_called()
        p._container.create_item.assert_not_called()

    async def test_a_fact_without_a_close_neighbour_is_checked_without_asking_the_model(self):
        p = _make_pipeline()
        seed = _fact("f1", "User likes coffee", ts=3)

        result, _ = await _run(p, [seed], {"f1": [_fact("f2", "User likes tea", vector=TOO_FAR)]})

        assert (result["checked"], result["compared"], result["kept"]) == (1, 0, 1)
        p._run_prompty.assert_not_called()
        body = p._container.create_item.call_args.kwargs["body"]
        assert (body["id"], body["type"]) == ("dedup_cursor_u1_fact", "dedup_cursor")
        assert (body["user_id"], body["thread_id"]) == ("u1", "__dedup__")
        assert (body["last_ts"], body["ids_at_last_ts"]) == (3, ["f1"])

    async def test_a_pass_reads_active_facts_after_its_cursor_and_replaces_it_if_unchanged(self):
        p = _make_pipeline()
        cursor = {
            "id": "dedup_cursor_u1_fact",
            "type": "dedup_cursor",
            "last_ts": 5,
            "ids_at_last_ts": ["f0"],
            "_etag": "c1",
        }

        _, queries = await _run(p, [_fact("f1", "User likes coffee", ts=6)], cursor=cursor, n=7)

        sql, params, _ = queries[0]
        assert "TOP 7" in sql and "ORDER BY c._ts ASC" in sql and "IS_NULL(c.superseded_by)" in sql
        assert (params["@user_id"], params["@after_ts"], params["@after_ids"]) == ("u1", 5, ["f0"])
        assert p._container.read_item.call_args.kwargs["partition_key"] == ["u1", "__dedup__"]
        replaced = p._container.replace_item.call_args.kwargs
        assert (replaced["item"], replaced["etag"]) == ("dedup_cursor_u1_fact", "c1")
        assert replaced["match_condition"] == MatchConditions.IfNotModified
        assert (replaced["body"]["last_ts"], replaced["body"]["ids_at_last_ts"]) == (6, ["f1"])

    async def test_only_neighbours_at_or_above_the_similarity_floor_reach_the_model(self):
        p = _make_pipeline()
        seed = _fact("f1", "User likes coffee")
        close = _fact("f2", "The user likes coffee", vector=NEAR_ENOUGH)
        far = _fact("f3", "User likes tea", vector=TOO_FAR)

        result, queries = await _run(p, [seed], {"f1": [far, close]}, [[_verdict("related")]])

        assert "ORDER BY VectorDistance(c.embedding, @embedding)" in queries[1][0]
        assert p._run_prompty.call_args.args[0] == "dedup_verify.prompty"
        candidates = json.loads(p._run_prompty.call_args.kwargs["inputs"]["candidates"])
        assert [c["text"] for c in candidates] == ["The user likes coffee"]
        assert result["compared"] == 1

    async def test_a_pair_two_new_facts_share_is_judged_once(self):
        p = _make_pipeline()
        first, second = _fact("f1", "User likes coffee", ts=1), _fact("f2", "User loves coffee", ts=2)

        await _run(p, [first, second], {"f1": [second], "f2": [first]}, [[_verdict("related")]])

        assert p._run_prompty.call_count == 1


class TestWhichVerdictsFold:
    async def test_the_same_claim_retires_the_older_statement_into_the_newer(self):
        p = _make_pipeline()
        older = _fact("f1", "User likes coffee", created_at="2024-01-01T00:00:00+00:00")
        newer = _fact("f2", "The user likes coffee", created_at="2024-02-01T00:00:00+00:00")

        result, _ = await _run(p, [older], {"f1": [newer]}, [[_verdict("same_claim")]])

        assert _folds(p) == [("f2", ["f1"])]
        assert p._commit_fold.call_args.args[0]["merged_from"] == ["f1"]
        assert p._commit_fold.call_args.kwargs["survivor_etag"] == "etag-f2"
        assert (result["merged"], result["kept"]) == (1, 0)

    @pytest.mark.parametrize(
        ("relation", "expected"),
        [("seed_contains_candidate", ("f1", ["f2"])), ("candidate_contains_seed", ("f2", ["f1"]))],
    )
    async def test_a_contained_fact_is_retired_into_the_one_that_contains_it(self, relation, expected):
        p = _make_pipeline()

        await _run(p, [_fact("f1", "User runs")], {"f1": [_fact("f2", "User runs daily")]}, [[_verdict(relation)]])

        assert _folds(p) == [expected]

    async def test_a_fold_that_would_lose_a_listed_detail_keeps_both(self):
        p = _make_pipeline()
        seed = _fact("f1", "User keeps karaoke at a safe volume")
        candidate = _fact("f2", "User keeps karaoke volume at 70 decibels")

        result, _ = await _run(
            p, [seed], {"f1": [candidate]}, [[_verdict("candidate_contains_seed", seed_missing=["for safety"])]]
        )

        p._commit_fold.assert_not_called()
        assert result["kept"] == 1

    @pytest.mark.parametrize("relation", ["partial_overlap", "related", "unrelated", "update", "inconsistent"])
    async def test_facts_that_are_not_duplicates_are_kept(self, relation):
        p = _make_pipeline()

        result, _ = await _run(
            p,
            [_fact("f1", "User lives in Portland")],
            {"f1": [_fact("f2", "User moved to Seattle")]},
            [[_verdict(relation)]],
        )

        p._commit_fold.assert_not_called()
        assert (result["merged"], result["contradicted"]) == (0, 0)


class TestRulesNoVerdictOverrides:
    async def test_the_veto_keeps_a_fact_whose_numbers_the_survivor_lacks(self):
        p = _make_pipeline()
        seed = _fact("f1", "User budgets 150 TRY a month for the gym")
        candidate = _fact("f2", "User has a monthly gym budget and a trainer")

        result, _ = await _run(p, [seed], {"f1": [candidate]}, [[_verdict("candidate_contains_seed")]])

        p._commit_fold.assert_not_called()
        assert result["vetoed"] == 1

    async def test_a_fact_that_absorbed_others_is_never_retired_but_can_absorb_more(self):
        p = _make_pipeline()
        absorber = _fact("f1", "User runs", merged_from=["f0"])
        await _run(p, [absorber], {"f1": [_fact("f2", "User runs daily")]}, [[_verdict("candidate_contains_seed")]])
        p._commit_fold.assert_not_called()

        grows = _fact("f1", "User runs daily", merged_from=["f0"])
        await _run(p, [grows], {"f1": [_fact("f2", "User runs")]}, [[_verdict("seed_contains_candidate")]])

        assert _folds(p) == [("f1", ["f2"])]

    async def test_a_users_statement_is_never_retired_into_an_assistant_fact(self):
        p = _make_pipeline()

        await _run(
            p,
            [_fact("f1", "User runs")],
            {"f1": [_fact("f2", "User runs daily", agent=True)]},
            [[_verdict("candidate_contains_seed")]],
        )

        p._commit_fold.assert_not_called()

    async def test_an_assistant_fact_can_be_retired_into_a_users_statement(self):
        p = _make_pipeline()

        await _run(
            p,
            [_fact("f1", "User runs", agent=True)],
            {"f1": [_fact("f2", "User runs daily")]},
            [[_verdict("candidate_contains_seed")]],
        )

        assert _folds(p) == [("f2", ["f1"])]

    async def test_facts_retired_into_one_survivor_commit_together(self):
        p = _make_pipeline()
        f1, f2 = _fact("f1", "User runs", ts=1), _fact("f2", "User jogs", ts=2)
        f3 = _fact("f3", "User runs and jogs daily")

        result, _ = await _run(
            p,
            [f1, f2],
            {"f1": [f3], "f2": [f3]},
            [[_verdict("candidate_contains_seed")], [_verdict("candidate_contains_seed")]],
        )

        assert _folds(p) == [("f3", ["f1", "f2"])]
        assert (result["merged"], result["kept"]) == (2, 0)


class TestModelFailures:
    async def test_a_retryable_failure_ends_the_pass_but_keeps_its_progress(self):
        p = _make_pipeline()
        f1, f2, f3 = (_fact(f"f{i}", f"memory {i}", ts=i) for i in (1, 2, 3))
        neighbours = {"f1": [_fact("n1", "memory 1")], "f2": [_fact("n2", "memory 2")], "f3": [_fact("n3", "memory 3")]}

        result, _ = await _run(
            p, [f1, f2, f3], neighbours, [[_verdict("same_claim")], RuntimeError("429 rate limited")]
        )

        assert result["checked"] == 1
        assert len(_folds(p)) == 1
        body = p._container.create_item.call_args.kwargs["body"]
        assert (body["last_ts"], body["ids_at_last_ts"]) == (1, ["f1"])

    @pytest.mark.parametrize(
        "reply",
        [LLMError("content_filter: the response was filtered"), "this is not JSON"],
        ids=["non-retryable-error", "unparseable-answer"],
    )
    async def test_an_unusable_check_keeps_the_facts_and_the_pass_moves_on(self, reply):
        p = _make_pipeline()
        f1, f2 = _fact("f1", "memory one", ts=1), _fact("f2", "memory two", ts=2)

        result, _ = await _run(
            p,
            [f1, f2],
            {"f1": [_fact("n1", "memory one")], "f2": [_fact("n2", "memory two")]},
            [reply, [_verdict("same_claim")]],
        )

        assert result["checked"] == 2
        assert _folds(p) == [("f2", ["n2"])]


class TestCommitAndCursor:
    async def test_a_fold_that_retired_nothing_is_not_counted(self):
        p = _make_pipeline()
        p._commit_fold = AsyncMock(return_value=[])

        result, _ = await _run(
            p,
            [_fact("f1", "User likes coffee")],
            {"f1": [_fact("f2", "User likes coffee")]},
            [[_verdict("same_claim")]],
        )

        assert (result["merged"], result["kept"]) == (0, 1)

    @pytest.mark.parametrize("error", [CosmosResourceExistsError, CosmosAccessConditionFailedError])
    async def test_a_cursor_moved_by_a_concurrent_pass_is_left_alone(self, error):
        p = _make_pipeline()
        cursor = {
            "id": "dedup_cursor_u1_fact",
            "type": "dedup_cursor",
            "last_ts": 1,
            "ids_at_last_ts": [],
            "_etag": "c1",
        }
        p._container.replace_item = AsyncMock(side_effect=error(status_code=412, message="moved"))

        result, _ = await _run(p, [_fact("f1", "User likes coffee", ts=2)], cursor=cursor)

        assert result["checked"] == 1

    async def test_an_unexpected_cursor_error_is_not_swallowed(self):
        p = _make_pipeline()
        p._container.create_item = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError):
            await _run(p, [_fact("f1", "User likes coffee", ts=2)])


class TestCommitFold:
    """The survivor and its duplicates are written one after another: the survivor first."""

    @staticmethod
    def _build(superseded=True) -> AsyncPipelineService:
        p = AsyncPipelineService.__new__(AsyncPipelineService)
        p._container = MagicMock()
        p._container.replace_item = AsyncMock()
        p._container.upsert_item = AsyncMock()
        p._memories_container = p._container
        p._mark_superseded = AsyncMock(side_effect=superseded if isinstance(superseded, list) else None)
        if not isinstance(superseded, list):
            p._mark_superseded.return_value = superseded
        return p

    @staticmethod
    async def _fold(p, *, survivor_etag="etag-s"):
        survivor = _fact("s", "User runs daily")
        losers = [_fact("a", "User runs"), _fact("b", "User jogs")]
        return await p._commit_fold(survivor_patch(survivor, losers), losers, survivor_etag=survivor_etag)

    async def test_the_survivor_is_replaced_if_unchanged_before_its_duplicates_are_retired(self):
        p = self._build()
        order = []

        async def replace(**kwargs):
            order.append(("replace", kwargs["item"]))

        async def retire(doc, superseder, **kwargs):
            order.append(("retire", doc["id"]))
            return True

        p._container.replace_item = AsyncMock(side_effect=replace)
        p._mark_superseded = AsyncMock(side_effect=retire)

        assert await self._fold(p) == ["a", "b"]

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
    async def test_a_survivor_that_changed_or_disappeared_leaves_everything_as_it_was(self, error):
        p = self._build()
        p._container.replace_item = AsyncMock(side_effect=error)

        assert await self._fold(p) == []

        p._mark_superseded.assert_not_called()

    async def test_a_duplicate_that_changed_since_the_pass_read_it_stays_active(self):
        p = self._build([True, False])

        assert await self._fold(p) == ["a"]

    async def test_a_survivor_without_an_etag_is_upserted(self):
        p = self._build()

        assert await self._fold(p, survivor_etag=None) == ["a", "b"]

        p._container.replace_item.assert_not_called()
        assert p._container.upsert_item.call_args.kwargs["body"]["id"] == "s"


class TestConsolidateDeadline:
    async def test_a_pass_past_its_deadline_checks_nothing_and_keeps_its_place(self):
        p = _make_pipeline()

        result, _ = await _run(p, [_fact("f1", "Lives in Seattle")], deadline=-1.0)

        assert result["checked"] == 0
        p._run_prompty.assert_not_called()
        p._container.create_item.assert_not_called()

    async def test_a_pass_stops_at_its_deadline_and_keeps_what_it_checked(self, monkeypatch):
        p = _make_pipeline()
        clock = {"now": 0.0}
        monkeypatch.setattr(pipeline_module, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
        seeds = [_fact("f1", "Lives in Seattle", ts=1), _fact("f2", "Works at Contoso", ts=2)]
        neighbors = {"f1": [_fact("n1", "Lives in Seattle, WA", vector=NEAR_ENOUGH, ts=0)]}

        async def slow_check(filename, inputs):
            del filename, inputs
            clock["now"] = 100.0
            return json.dumps({"candidates": [_verdict("candidate_contains_seed")]})

        p._container.query_items = AsyncMock(
            side_effect=lambda query, parameters, **kwargs: (
                list(neighbors.get({i["name"]: i["value"] for i in parameters}.get("@seed_id"), []))
                if "VectorDistance" in query
                else list(seeds)
            )
        )
        p._run_prompty = AsyncMock(side_effect=slow_check)

        result = await p.consolidate_memories("u1", deadline=50.0)

        assert (result["checked"], result["merged"]) == (1, 1)
        assert _folds(p) == [("n1", ["f1"])]
        assert p._container.create_item.call_args.kwargs["body"]["last_ts"] == 1
