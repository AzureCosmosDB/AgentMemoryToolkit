"""Async mirror of the combining tests (``tests/unit/services/test_combine.py``).

``AsyncPipelineService.consolidate_memories`` combines facts the same way as the sync pipeline; these
tests drive the async I/O paths: awaited model calls on concurrent tasks, the embedding, and the ordered
writes (combined fact first, then each member retired into it).
"""

from __future__ import annotations

import asyncio
import json
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from azure.cosmos.exceptions import CosmosResourceExistsError, CosmosResourceNotFoundError

from azure.cosmos.agent_memory._utils import compute_content_hash
from azure.cosmos.agent_memory.aio.services import pipeline as pipeline_module
from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService
from azure.cosmos.agent_memory.exceptions import LLMError
from azure.cosmos.agent_memory.services._combine import COMBINE_PROMPT, CombineWrite, combined_record

NEAR = [1.0, 0.0, 0.0]
BAND = [0.75, math.sqrt(1 - 0.75**2), 0.0]  # cosine 0.75: below the dedup floor, above the combine floor

PLANS = "The user is planning a Christmas Eve dinner for 15 guests"
START = "The user wants the Christmas Eve dinner to start at 6:30 PM"
BOTH = "For the user's Christmas Eve dinner, the user is planning for 15 guests and wants it to start at 6:30 PM."


def _fact(fid: str, content: str, *, vector=NEAR, ts: int = 1, created_at: str = "2024-01-01T00:00:00+00:00", **extra):
    doc = {
        "id": fid,
        "user_id": "u1",
        "thread_id": "t1",
        "role": "system",
        "type": "fact",
        "content": content,
        "confidence": 0.8,
        "salience": 0.5,
        "tags": ["sys:fact"],
        "metadata": {"source": "user", "category": "other"},
        "created_at": created_at,
        "embedding": vector,
        "_ts": ts,
        "_etag": f"etag-{fid}",
    }
    doc.update(extra)
    return doc


def _verdict(relation: str, label: str = "C1", *, candidate_missing=()) -> dict:
    return {
        "label": label,
        "relation": relation,
        "seed_details_missing_from_candidate": [],
        "candidate_details_missing_from_seed": list(candidate_missing),
    }


def _write(labels: list[str], text: str) -> dict:
    return {"combine": labels, "memory": text}


def _holds(count: int) -> dict:
    return {"candidates": [_verdict("seed_contains_candidate", f"C{i}") for i in range(1, count + 1)]}


class _Model:
    """Answers a pass's model calls (see the sync tests' ``_Model``), awaited, and counts concurrent writers."""

    def __init__(self, verdicts=None, writes=(), checks=()):
        self.verdicts = dict(verdicts or {})
        self.writes = list(writes)
        self.checks = list(checks)
        self.calls: list[tuple[str, dict]] = []
        self.active = 0
        self.max_active = 0

    async def __call__(self, filename: str, inputs: dict):
        self.calls.append((filename, inputs))
        if filename == COMBINE_PROMPT:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            reply = self.writes.pop(0)
        else:
            seed = json.loads(inputs["seed"])["text"]
            reply = {"candidates": self.verdicts[seed]} if seed in self.verdicts else self.checks.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)

    def writer_inputs(self) -> list[dict]:
        return [inputs for filename, inputs in self.calls if filename == COMBINE_PROMPT]


def _make_pipeline() -> AsyncPipelineService:
    p = AsyncPipelineService.__new__(AsyncPipelineService)
    p._embeddings = MagicMock()
    p._embed_one = AsyncMock(return_value=[0.1, 0.2, 0.3])
    p._commit_fold = AsyncMock(side_effect=lambda survivor, losers, **kwargs: [doc["id"] for doc in losers])
    p._mark_superseded = AsyncMock(return_value=True)
    p._container = MagicMock()
    p._docs = {}  # the facts a re-read returns, by id (filled by `_run`); anything else is not found

    def read_item(item, partition_key):
        if item not in p._docs:
            raise CosmosResourceNotFoundError(message="not found")
        return dict(p._docs[item])

    p._container.read_item = AsyncMock(side_effect=read_item)
    p._container.create_item = AsyncMock(side_effect=lambda body: body)
    p._container.replace_item = AsyncMock()
    p._memories_container = p._container
    p._turns_container = p._container
    p._summaries_container = p._container
    return p


async def _run(p, seeds, neighbors, model, *, deadline=None, changed=None):
    # The facts the pass read are still there, unchanged, when it re-reads them before writing; `changed`
    # maps a fact's ID to what it became meanwhile, or to None when it was deleted.
    p._docs.update({doc["id"]: doc for doc in [*seeds, *(doc for docs in neighbors.values() for doc in docs)]})
    for fact_id, replacement in (changed or {}).items():
        if replacement is None:
            p._docs.pop(fact_id, None)
        else:
            p._docs[fact_id] = replacement

    def query_items(query, parameters, **kwargs):
        params = {item["name"]: item["value"] for item in parameters}
        if "VectorDistance" in query:
            return list(neighbors.get(params["@seed_id"], []))
        return list(seeds)

    async def run_prompty(filename, inputs):
        return await model(filename, inputs)

    p._container.query_items = AsyncMock(side_effect=query_items)
    p._run_prompty = run_prompty
    return await p.consolidate_memories("u1", deadline=deadline)


def _created(p) -> list[dict]:
    return [c.kwargs["body"] for c in p._container.create_item.call_args_list if c.kwargs["body"]["type"] == "fact"]


def _retired(p) -> list[tuple[str, str, str]]:
    return [(c.args[0]["id"], c.args[1], c.kwargs["reason"]) for c in p._mark_superseded.call_args_list]


def _cursor(p) -> tuple[int, list[str]]:
    [body] = [c.kwargs["body"] for c in p._container.create_item.call_args_list if c.kwargs["body"]["type"] != "fact"]
    return body["last_ts"], body["ids_at_last_ts"]


def _pair():
    seed = _fact("f2", START, ts=2, created_at="2023-11-20T00:00:00+00:00")
    partner = _fact("f1", PLANS, vector=BAND, created_at="2023-11-01T00:00:00+00:00")
    return seed, partner, {seed["content"]: [_verdict("partial_overlap")]}


class TestAsyncCombine:
    async def test_overlapping_facts_become_one_fact_and_retire_into_it(self):
        p = _make_pipeline()
        seed, partner, verdicts = _pair()
        model = _Model(verdicts, [_write(["M1", "M2"], BOTH)], [_holds(2)])

        result = await _run(p, [seed], {"f2": [partner]}, model)

        [record] = _created(p)
        assert (record["content"], record["merged_from"]) == (BOTH, ["f1", "f2"])
        assert "sys:combined" in record["tags"] and record["embedding"] == [0.1, 0.2, 0.3]
        assert (record["prompt_id"], record["prompt_version"]) == ("combine_memories.prompty", "v1")
        assert _retired(p) == [("f1", record["id"], "combined"), ("f2", record["id"], "combined")]
        assert (result["combined"], result["synthesized"], result["kept"], result["combine_groups"]) == (2, 1, 0, 1)
        assert _cursor(p) == (2, ["f2"])

    async def test_the_combined_fact_is_created_before_any_member_is_retired(self):
        p = _make_pipeline()
        order = []

        async def create(body):
            order.append(("create", body["type"]))
            return body

        async def retire(doc, superseder, **kwargs):
            order.append(("retire", doc["id"]))
            return True

        p._container.create_item = AsyncMock(side_effect=create)
        p._mark_superseded = AsyncMock(side_effect=retire)
        seed, partner, verdicts = _pair()

        await _run(p, [seed], {"f2": [partner]}, _Model(verdicts, [_write(["M1", "M2"], BOTH)], [_holds(2)]))

        assert order[:3] == [("create", "fact"), ("retire", "f1"), ("retire", "f2")]

    async def test_with_combining_off_nothing_below_the_dedup_floor_is_compared(self, monkeypatch):
        monkeypatch.setattr("azure.cosmos.agent_memory.thresholds.COMBINE_ENABLED", False)
        p = _make_pipeline()
        seed, partner, verdicts = _pair()
        model = _Model(verdicts)

        result = await _run(p, [seed], {"f2": [partner]}, model)

        assert model.calls == [] and result["compared"] == 0 and _created(p) == []

    async def test_a_group_the_writer_declines_is_left_alone(self):
        p = _make_pipeline()
        seed, partner, verdicts = _pair()

        result = await _run(p, [seed], {"f2": [partner]}, _Model(verdicts, [_write(["M1"], PLANS)]))

        assert _created(p) == [] and _retired(p) == []
        assert (result["combine_groups"], result["combine_failed"], result["combined"]) == (1, 1, 0)

    async def test_a_missing_detail_is_written_again_once_with_the_detail_listed(self):
        p = _make_pipeline()
        seed, partner, verdicts = _pair()
        short = "For the user's Christmas Eve dinner, the user is planning for 15 guests at 6:30 PM."
        lacking = {
            "candidates": [
                _verdict("seed_contains_candidate", "C1"),
                _verdict("partial_overlap", "C2", candidate_missing=["the dinner starts at 6:30 PM"]),
            ]
        }
        model = _Model(verdicts, [_write(["M1", "M2"], short), _write(["M1", "M2"], BOTH)], [lacking, _holds(2)])

        result = await _run(p, [seed], {"f2": [partner]}, model)

        assert "- the dinner starts at 6:30 PM" in model.writer_inputs()[1]["feedback"]
        assert result["synthesized"] == 1 and _created(p)[0]["content"] == BOTH

    async def test_a_retryable_error_while_combining_keeps_the_cursor_before_the_group(self):
        p = _make_pipeline()
        alone = _fact("f0", "The user likes tea", ts=1)
        seed, partner, verdicts = _pair()

        result = await _run(p, [alone, seed], {"f2": [partner]}, _Model(verdicts, [RuntimeError("429 rate limited")]))

        assert _created(p) == []
        assert (result["checked"], result["combine_groups"]) == (2, 0)
        assert _cursor(p) == (1, ["f0"])

    async def test_any_other_model_error_leaves_the_group_and_moves_on(self):
        p = _make_pipeline()
        seed, partner, verdicts = _pair()

        result = await _run(
            p, [seed], {"f2": [partner]}, _Model(verdicts, [LLMError("content_filter: the response was filtered")])
        )

        assert (result["combine_groups"], result["combine_failed"]) == (1, 1)
        assert _cursor(p) == (2, ["f2"])

    async def test_a_retryable_embedding_error_keeps_the_cursor_before_the_group(self):
        p = _make_pipeline()
        p._embed_one = AsyncMock(side_effect=RuntimeError("429 rate limited"))
        alone = _fact("f0", "The user likes tea", ts=1)
        seed, partner, verdicts = _pair()

        result = await _run(
            p, [alone, seed], {"f2": [partner]}, _Model(verdicts, [_write(["M1", "M2"], BOTH)], [_holds(2)])
        )

        assert _created(p) == [] and result["combine_groups"] == 0
        assert _cursor(p) == (1, ["f0"])

    async def test_a_combined_fact_that_fails_validation_is_dropped(self):
        p = _make_pipeline()
        seed = _fact("f2", START, ts=2, created_at="2023-11-20T00:00:00+00:00", metadata={"source": "user"})
        partner = _fact("f1", PLANS, vector=BAND, created_at="2023-11-01T00:00:00+00:00")
        verdicts = {START: [_verdict("partial_overlap")]}

        result = await _run(p, [seed], {"f2": [partner]}, _Model(verdicts, [_write(["M1", "M2"], BOTH)], [_holds(2)]))

        assert _created(p) == []
        p._embed_one.assert_not_called()
        assert (result["combine_groups"], result["combine_failed"]) == (1, 1)

    async def test_members_that_changed_since_the_pass_read_them_stay_active_and_are_not_counted(self):
        p = _make_pipeline()
        p._mark_superseded = AsyncMock(return_value=False)
        seed, partner, verdicts = _pair()

        result = await _run(p, [seed], {"f2": [partner]}, _Model(verdicts, [_write(["M1", "M2"], BOTH)], [_holds(2)]))

        assert (result["combined"], result["synthesized"], result["kept"]) == (0, 1, 1)

    async def test_checking_stops_halfway_to_the_deadline_so_groups_can_be_written(self, monkeypatch):
        p = _make_pipeline()
        clock = {"now": 0.0}
        # Only the pipeline's clock is replaced: patching ``time.monotonic`` itself would freeze the event loop.
        monkeypatch.setattr(pipeline_module, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
        seed, partner, verdicts = _pair()
        later = _fact("f3", "The user is hosting a New Year party", ts=3)
        model = _Model(verdicts, [_write(["M1", "M2"], BOTH)], [_holds(2)])
        check = model.__call__

        async def slow(filename, inputs):
            clock["now"] = 60.0
            return await check(filename, inputs)

        result = await _run(p, [seed, later], {"f2": [partner], "f3": [partner]}, slow, deadline=100.0)

        assert (result["checked"], result["combined"]) == (1, 2)
        assert _cursor(p) == (2, ["f2"])

    @pytest.mark.parametrize("workers", [1, 2])
    async def test_the_worker_limit_bounds_how_many_groups_are_written_at_once(self, monkeypatch, workers):
        monkeypatch.setattr("azure.cosmos.agent_memory.thresholds.COMBINE_WORKERS", workers)
        p = _make_pipeline()
        seeds, neighbours, verdicts, writes, checks = [], {}, {}, [], []
        for i in range(3):
            seed = _fact(f"s{i}", f"Seed {i} about trip {i}", ts=i + 1, created_at=f"2023-11-0{i + 1}T00:00:00+00:00")
            partner = _fact(f"p{i}", f"Partner {i} about trip {i}", vector=BAND)
            seeds.append(seed)
            neighbours[seed["id"]] = [partner]
            verdicts[seed["content"]] = [_verdict("partial_overlap")]
            writes.append(_write(["M1", "M2"], f"Trip {i}: seed {i} and partner {i}."))
            checks.append(_holds(2))
        model = _Model(verdicts, writes, checks)

        result = await _run(p, seeds, neighbours, model)

        assert result["synthesized"] == 3 and model.max_active == workers


class TestAsyncAMemberChangedSinceThePassReadIt:
    @pytest.mark.parametrize(
        "change",
        [
            pytest.param(None, id="deleted"),
            pytest.param({"_etag": "etag-newer"}, id="edited"),
            pytest.param({"superseded_by": "other"}, id="retired"),
        ],
    )
    async def test_the_combined_fact_is_not_written(self, change):
        p = _make_pipeline()
        seed, partner, verdicts = _pair()
        replacement = None if change is None else {**partner, **change}
        model = _Model(verdicts, [_write(["M1", "M2"], BOTH)], [_holds(2)])

        result = await _run(p, [seed], {"f2": [partner]}, model, changed={"f1": replacement})

        assert _created(p) == [] and _retired(p) == []
        assert (result["synthesized"], result["combined"], result["combine_failed"]) == (0, 0, 1)


class TestAsyncACombinedFactIsRedactedBeforeItIsEmbedded:
    async def test_the_vector_survives_the_write_guard_and_matches_the_stored_text(self):
        token = "ghp_" + "A1b2" * 9  # built here so the source holds no token literal
        p = _make_pipeline()
        seed, partner, verdicts = _pair()
        model = _Model(verdicts, [_write(["M1", "M2"], f"{BOTH} Its token is {token}.")], [_holds(2)])

        await _run(p, [seed], {"f2": [partner]}, model)

        [body] = _created(p)
        assert token not in body["content"] and "[REDACTED-SECRET]" in body["content"]
        assert body["embedding"] == [0.1, 0.2, 0.3]
        assert token not in p._embed_one.call_args.args[0]

    async def test_the_hash_of_what_combine_wrote_is_that_of_the_text_that_is_stored(self):
        token = "ghp_" + "A1b2" * 9
        p = _make_pipeline()
        seed, partner, verdicts = _pair()
        model = _Model(verdicts, [_write(["M1", "M2"], f"{BOTH} Its token is {token}.")], [_holds(2)])

        await _run(p, [seed], {"f2": [partner]}, model)

        [body] = _created(p)
        assert token not in body["content"]
        assert body["combined_hash"] == compute_content_hash(body["content"]), "redacting is not an edit"


class TestAsyncTheContradictionPoolDatesACombinedFactByItsOldestMember:
    async def test_a_combined_fact_is_dated_by_combined_since_and_any_other_fact_by_created_at(self):
        p = AsyncPipelineService.__new__(AsyncPipelineService)
        shown: list[str] = []

        async def run_prompty(filename: str, inputs: dict) -> str:
            shown.extend(line.rsplit("Created: ", 1)[1] for line in inputs["facts_text"].splitlines())
            return json.dumps({"contradicted_pairs": [], "kept_ids": []})

        p._run_prompty = run_prompty
        tags = ["sys:fact", "sys:combined"]
        combined = _fact(
            "c1", BOTH, created_at="2023-11-20T00:00:00+00:00", combined_since="2023-11-01T00:00:00+00:00", tags=tags
        )
        written_before_the_field_existed = _fact("c2", BOTH, created_at="2023-11-21T00:00:00+00:00", tags=tags)
        ordinary = _fact("f1", PLANS, created_at="2023-11-22T00:00:00+00:00")
        undated = _fact("f2", START, created_at=None)

        await p._reconcile_contradictions("u1", "fact", [combined, written_before_the_field_existed, ordinary, undated])

        assert shown == ["2023-11-01T00:00:00+00:00", "2023-11-21T00:00:00+00:00", "2023-11-22T00:00:00+00:00", "N/A"]


class TestAsyncReconcilePoolKeepsCombinedFacts:
    async def test_the_pool_query_leaves_out_only_agent_facts(self):
        p = AsyncPipelineService.__new__(AsyncPipelineService)
        p._memories_container = MagicMock()
        p._query_items = AsyncMock(return_value=[])

        await p._active_memories_for_reconcile("u1", "fact", 50)

        query = p._query_items.call_args.kwargs["query"]
        assert "sys:combined" not in query
        assert query.count("ARRAY_CONTAINS") == 1 and "NOT ARRAY_CONTAINS(c.tags, 'sys:agent-fact')" in query


class TestAsyncCommitCombine:
    @staticmethod
    def _build() -> AsyncPipelineService:
        p = AsyncPipelineService.__new__(AsyncPipelineService)
        p._container = MagicMock()
        p._container.create_item = AsyncMock()
        p._memories_container = p._container
        p._mark_superseded = AsyncMock(return_value=True)
        return p

    @staticmethod
    def _inputs():
        members = [_fact("f1", PLANS), _fact("f2", START)]
        return combined_record(members, BOTH, lineage={"prompt_id": COMBINE_PROMPT}), members

    async def test_every_member_is_retired_into_the_created_fact(self):
        p = self._build()
        record, members = self._inputs()

        assert await p._commit_combine(record, members) == ["f1", "f2"]

        p._container.create_item.assert_awaited_once_with(body=record)
        assert [(c.args[1], c.kwargs["reason"]) for c in p._mark_superseded.call_args_list] == [
            (record["id"], "combined"),
            (record["id"], "combined"),
        ]

    async def test_a_fact_that_already_exists_is_not_created_again_and_its_members_are_still_retired(self):
        p = self._build()
        p._container.create_item = AsyncMock(side_effect=CosmosResourceExistsError(status_code=409, message="x"))
        p._container.read_item = AsyncMock(return_value={"id": "x", "type": "fact"})
        record, members = self._inputs()

        assert await p._commit_combine(record, members) == ["f1", "f2"]

    @pytest.mark.parametrize(
        "existing",
        [
            {"id": "x", "superseded_by": "other", "supersede_reason": "contradict"},
            CosmosResourceNotFoundError(message="x"),
        ],
        ids=["retired-since", "deleted-since"],
    )
    async def test_a_fact_that_is_there_but_no_longer_active_keeps_the_members_active(self, existing):
        p = self._build()
        p._container.create_item = AsyncMock(side_effect=CosmosResourceExistsError(status_code=409, message="x"))
        if isinstance(existing, BaseException):
            p._container.read_item = AsyncMock(side_effect=existing)
        else:
            p._container.read_item = AsyncMock(return_value=existing)
        record, members = self._inputs()

        assert await p._commit_combine(record, members) is None

        p._mark_superseded.assert_not_called()

    async def test_a_server_error_creating_the_fact_retires_nothing(self):
        p = self._build()
        p._container.create_item = AsyncMock(side_effect=RuntimeError("503"))
        record, members = self._inputs()

        with pytest.raises(RuntimeError):
            await p._commit_combine(record, members)

        p._mark_superseded.assert_not_called()

    async def test_nothing_to_write_reports_lost(self):
        p = AsyncPipelineService.__new__(AsyncPipelineService)

        assert await p._write_combined(None) == CombineWrite("lost")
