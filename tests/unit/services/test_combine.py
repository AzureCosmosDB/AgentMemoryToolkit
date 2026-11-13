"""Tests for combining in the dedup pass (``azure.cosmos.agent_memory.services._combine``).

Covers:
* what a combining pass compares: neighbours it could combine with down to the combine floor, others at the
  dedup floor, and folds still at the dedup floor
* which facts may be combined and how groups form
* writing a combined fact: the writer's inputs, the check, the retry, the veto, the length cap, and the record
  and the ordered writes it commits
* the model failing, a lost race, the deadline, and where the cursor stops

The pipelines are built with ``__new__`` and patched in place, as in ``test_consolidate_memories.py``.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from azure.cosmos.exceptions import CosmosResourceExistsError, CosmosResourceNotFoundError

from azure.cosmos.agent_memory.exceptions import LLMError
from azure.cosmos.agent_memory.services import pipeline as pipeline_module
from azure.cosmos.agent_memory.services._combine import (
    COMBINE_PROMPT,
    CombineWrite,
    combinable_pair,
    combined_record,
    in_seed_order,
    kept_specifics,
    may_combine,
    plan_combine_groups,
    settled_seeds,
)
from azure.cosmos.agent_memory.services._dedup import Verdict
from azure.cosmos.agent_memory.services.pipeline import PipelineService

NEAR = [1.0, 0.0, 0.0]
NEAR_ENOUGH = [0.9, 0.3, 0.0]  # cosine 0.95 with NEAR: at or above the dedup floor
BAND = [0.75, math.sqrt(1 - 0.75**2), 0.0]  # cosine 0.75: below the dedup floor, above the combine floor
FAR = [0.6, 0.8, 0.0]  # cosine 0.60: below both floors

PLANS = "The user is planning a Christmas Eve dinner for 15 guests"
START = "The user wants the Christmas Eve dinner to start at 6:30 PM"
BOTH = "For the user's Christmas Eve dinner, the user is planning for 15 guests and wants it to start at 6:30 PM."


def _fact(
    fid: str,
    content: str,
    *,
    vector: list[float] = NEAR,
    ts: int = 1,
    created_at: str = "2024-01-01T00:00:00+00:00",
    agent: bool = False,
    **extra,
) -> dict:
    doc = {
        "id": fid,
        "user_id": "u1",
        "thread_id": "t1",
        "role": "system",
        "type": "fact",
        "content": content,
        "confidence": 0.8,
        "salience": 0.5,
        "tags": ["sys:fact", "sys:agent-fact"] if agent else ["sys:fact"],
        "metadata": {"source": "agent" if agent else "user", "category": "other"},
        "created_at": created_at,
        "embedding": vector,
        "_ts": ts,
        "_etag": f"etag-{fid}",
    }
    doc.update(extra)
    return doc


def _verdict(relation: str, label: str = "C1", *, seed_missing=(), candidate_missing=()) -> dict:
    return {
        "label": label,
        "relation": relation,
        "seed_details_missing_from_candidate": list(seed_missing),
        "candidate_details_missing_from_seed": list(candidate_missing),
    }


def _write(labels: list[str], text: str) -> dict:
    return {"combine": labels, "memory": text}


def _holds(count: int, *, missing: dict[str, list[str]] | None = None) -> dict:
    """A check finding the combined fact holds each of ``count`` members, except those in ``missing``."""
    missing = missing or {}
    return {
        "candidates": [
            _verdict(
                "partial_overlap" if f"C{i}" in missing else "seed_contains_candidate",
                f"C{i}",
                candidate_missing=missing.get(f"C{i}", ()),
            )
            for i in range(1, count + 1)
        ]
    }


class _Model:
    """Answers a pass's model calls: each seed's verdicts by the seed's text, then the writer's and the check's
    answers in order. A reply that is an exception is raised."""

    def __init__(self, verdicts=None, writes=(), checks=()):
        self.verdicts = dict(verdicts or {})
        self.writes = list(writes)
        self.checks = list(checks)
        self.calls: list[tuple[str, dict]] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def answer(self, filename: str, inputs: dict):
        with self._lock:
            self.calls.append((filename, inputs))
            if filename == COMBINE_PROMPT:
                reply = self.writes.pop(0)
            else:
                seed = json.loads(inputs["seed"])["text"]
                reply = {"candidates": self.verdicts[seed]} if seed in self.verdicts else self.checks.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)

    def __call__(self, filename: str, inputs: dict):
        return self.answer(filename, inputs)

    def writer_inputs(self) -> list[dict]:
        return [inputs for filename, inputs in self.calls if filename == COMBINE_PROMPT]

    def checked_texts(self) -> list[list[str]]:
        """The candidates' texts of each pass check (not the combined-fact checks)."""
        return [
            [c["text"] for c in json.loads(inputs["candidates"])]
            for filename, inputs in self.calls
            if filename != COMBINE_PROMPT and json.loads(inputs["seed"])["text"] in self.verdicts
        ]


def _make_pipeline() -> PipelineService:
    p = PipelineService.__new__(PipelineService)
    p._embeddings = MagicMock()
    p._embed_one = MagicMock(return_value=[0.1, 0.2, 0.3])
    p._commit_fold = MagicMock(side_effect=lambda survivor, losers, **kwargs: [doc["id"] for doc in losers])
    p._mark_superseded = MagicMock(return_value=True)
    p._container = MagicMock()
    p._memories_container = p._container
    p._turns_container = p._container
    p._summaries_container = p._container
    p._chat = MagicMock()
    p._docs = {}  # the facts a re-read returns, by id (filled by `_run`); anything else is not found

    def read_item(item, partition_key):
        if item not in p._docs:
            raise CosmosResourceNotFoundError(message="not found")
        return dict(p._docs[item])

    p._container.read_item.side_effect = read_item
    return p


def _run(p, seeds, neighbors=None, model=None, *, deadline=None, changed=None):
    neighbors = neighbors or {}
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
            return iter(neighbors.get(params["@seed_id"], []))
        return iter(seeds)

    p._container.query_items.side_effect = query_items
    model = model or _Model()
    p._run_prompty = model
    return p.consolidate_memories("u1", deadline=deadline), model


def _created(p) -> list[dict]:
    """The combined facts created (the cursor document is created through the same container)."""
    return [c.kwargs["body"] for c in p._container.create_item.call_args_list if c.kwargs["body"]["type"] == "fact"]


def _retired(p) -> list[tuple[str, str, str]]:
    """Each member retired as ``(id, superseded_by, reason)``."""
    return [(c.args[0]["id"], c.args[1], c.kwargs["reason"]) for c in p._mark_superseded.call_args_list]


def _cursor(p) -> tuple[int, list[str]]:
    [body] = [c.kwargs["body"] for c in p._container.create_item.call_args_list if c.kwargs["body"]["type"] != "fact"]
    return body["last_ts"], body["ids_at_last_ts"]


def _pair(seed=None, partner=None, *, relation="partial_overlap"):
    """A new fact (the seed) and an older one it overlaps with, below the dedup floor."""
    seed = seed or _fact("f2", START, ts=2, created_at="2023-11-20T00:00:00+00:00")
    partner = partner or _fact("f1", PLANS, vector=BAND, created_at="2023-11-01T00:00:00+00:00")
    return seed, partner, {seed["content"]: [_verdict(relation)]}


class TestWhatCombiningCompares:
    def test_a_neighbour_it_could_combine_with_is_compared_down_to_the_combine_floor(self):
        p = _make_pipeline()
        seed, partner, verdicts = _pair()
        far = _fact("f3", "The user likes tea", vector=FAR)

        result, model = _run(p, [seed], {"f2": [far, partner]}, _Model(verdicts, [_write([], "")]))

        assert model.checked_texts() == [[PLANS]]
        assert result["compared"] == 1

    @pytest.mark.parametrize(
        "other",
        [
            {"agent": True},
            {"assertion": "directive"},
            {"assertion": "decided"},
            {"tags": ["sys:fact", "sys:combined"]},
            {"type": "procedural"},
        ],
        ids=["another-speaker", "a-directive", "another-assertion", "a-combined-fact", "not-a-fact"],
    )
    def test_a_neighbour_it_could_not_combine_with_needs_the_dedup_floor(self, other):
        p = _make_pipeline()
        seed = _fact("f2", START, ts=2, assertion="stated")
        neighbour = _fact("f1", PLANS, vector=BAND, **{"assertion": "stated", **other})

        result, model = _run(p, [seed], {"f2": [neighbour]})

        assert model.calls == [] and result["compared"] == 0

    def test_with_combining_off_neighbours_need_the_dedup_floor(self, monkeypatch):
        monkeypatch.setattr("azure.cosmos.agent_memory.thresholds.COMBINE_ENABLED", False)
        p = _make_pipeline()
        seed, partner, _ = _pair()

        result, model = _run(p, [seed], {"f2": [partner]})

        assert model.calls == [] and result["compared"] == 0
        assert _created(p) == []

    def test_a_same_claim_below_the_dedup_floor_is_combined_not_folded(self):
        p = _make_pipeline()
        seed = _fact("f2", "Every morning the user drinks coffee", ts=2, created_at="2023-11-20T00:00:00+00:00")
        partner = _fact("f1", "The user drinks coffee every morning", vector=BAND)
        verdicts = {seed["content"]: [_verdict("same_claim")]}
        model = _Model(verdicts, [_write(["M1", "M2"], "The user drinks coffee every morning.")], [_holds(2)])

        result, _ = _run(p, [seed], {"f2": [partner]}, model)

        p._commit_fold.assert_not_called()
        assert (result["merged"], result["combined"]) == (0, 2)

    def test_folds_at_the_dedup_floor_are_unchanged_and_their_facts_wait(self):
        p = _make_pipeline()
        seed = _fact("f2", "Lives in Seattle", ts=2)
        duplicate = _fact("n1", "Lives in Seattle, WA", vector=NEAR_ENOUGH)
        overlap = _fact("n2", "Lives in Seattle near the lake", vector=BAND)
        verdicts = {"Lives in Seattle": [_verdict("candidate_contains_seed", "C1"), _verdict("partial_overlap", "C2")]}

        result, model = _run(p, [seed], {"f2": [overlap, duplicate]}, _Model(verdicts))

        assert [(c.args[0]["id"], [d["id"] for d in c.args[1]]) for c in p._commit_fold.call_args_list] == [
            ("n1", ["f2"])
        ]
        assert model.writer_inputs() == [] and result["combine_groups"] == 0

    def test_a_fold_below_the_dedup_floor_is_never_made(self):
        p = _make_pipeline()
        seed, partner, verdicts = _pair(relation="candidate_contains_seed")

        _run(p, [seed], {"f2": [partner]}, _Model(verdicts, [_write([], "")]))

        p._commit_fold.assert_not_called()


class TestWritingACombinedFact:
    def test_overlapping_facts_become_one_fact_and_retire_into_it(self):
        p = _make_pipeline()
        seed, partner, verdicts = _pair()
        model = _Model(verdicts, [_write(["M1", "M2"], BOTH)], [_holds(2)])

        result, _ = _run(p, [seed], {"f2": [partner]}, model)

        [record] = _created(p)
        assert record["content"] == BOTH and record["id"].startswith("fact_")
        assert record["merged_from"] == ["f1", "f2"] and "sys:combined" in record["tags"]
        assert (record["user_id"], record["thread_id"], record["type"]) == ("u1", "t1", "fact")
        assert record["created_at"] == "2023-11-20T00:00:00+00:00"
        assert (record["prompt_id"], record["prompt_version"]) == ("combine_memories.prompty", "v1")
        assert record["embedding"] == [0.1, 0.2, 0.3]
        assert _retired(p) == [("f1", record["id"], "combined"), ("f2", record["id"], "combined")]
        assert [c.args[0]["_etag"] for c in p._mark_superseded.call_args_list] == ["etag-f1", "etag-f2"]
        assert {k: result[k] for k in ("checked", "kept", "combined", "synthesized", "combine_groups")} == {
            "checked": 1,
            "kept": 0,
            "combined": 2,
            "synthesized": 1,
            "combine_groups": 1,
        }
        assert _cursor(p) == (2, ["f2"])

    def test_the_combined_fact_is_created_before_any_member_is_retired(self):
        p = _make_pipeline()
        order = []
        p._container.create_item.side_effect = lambda body: order.append(("create", body["type"])) or body
        p._mark_superseded.side_effect = lambda doc, superseder, **kwargs: order.append(("retire", doc["id"])) or True
        seed, partner, verdicts = _pair()

        _run(p, [seed], {"f2": [partner]}, _Model(verdicts, [_write(["M1", "M2"], BOTH)], [_holds(2)]))

        assert order[:3] == [("create", "fact"), ("retire", "f1"), ("retire", "f2")]

    def test_the_writer_sees_labels_dates_and_speakers_but_no_ids(self):
        p = _make_pipeline()
        seed, partner, verdicts = _pair()

        _, model = _run(p, [seed], {"f2": [partner]}, _Model(verdicts, [_write([], "")]))

        [inputs] = model.writer_inputs()
        assert json.loads(inputs["memories"]) == [
            {"label": "M1", "said": "2023-11-01", "by": "the user", "text": PLANS},
            {"label": "M2", "said": "2023-11-20", "by": "the user", "text": START},
        ]
        assert inputs["feedback"] == "" and "f1" not in inputs["memories"]

    def test_a_group_the_writer_declines_is_left_alone(self):
        p = _make_pipeline()
        seed, partner, verdicts = _pair()

        result, model = _run(p, [seed], {"f2": [partner]}, _Model(verdicts, [_write(["M1"], PLANS)]))

        assert _created(p) == [] and _retired(p) == []
        assert len(model.calls) == 2  # the pass's check and the writer; no combined-fact check
        assert (result["combine_groups"], result["combine_failed"], result["combined"]) == (1, 1, 0)

    def test_a_missing_detail_is_written_again_once_with_the_detail_listed(self):
        p = _make_pipeline()
        seed, partner, verdicts = _pair()
        short = "For the user's Christmas Eve dinner, the user is planning for 15 guests at 6:30 PM."
        model = _Model(
            verdicts,
            [_write(["M1", "M2"], short), _write(["M1", "M2"], BOTH)],
            [_holds(2, missing={"C2": ["the user wants the dinner to start at 6:30 PM"]}), _holds(2)],
        )

        result, _ = _run(p, [seed], {"f2": [partner]}, model)

        retry = model.writer_inputs()[1]["feedback"]
        assert (
            "left out or changed these details" in retry and "- the user wants the dinner to start at 6:30 PM" in retry
        )
        assert result["synthesized"] == 1 and _created(p)[0]["content"] == BOTH

    def test_a_lost_number_fails_even_when_the_check_passes(self):
        p = _make_pipeline()
        seed, partner, verdicts = _pair()
        lossy = "For the user's Christmas Eve dinner, the user is planning for guests and wants it to start at 6:30 PM."
        model = _Model(verdicts, [_write(["M1", "M2"], lossy), _write(["M1", "M2"], lossy)], [_holds(2), _holds(2)])

        result, _ = _run(p, [seed], {"f2": [partner]}, model)

        assert f"every number, name and negation in: {PLANS}" in model.writer_inputs()[1]["feedback"]
        assert _created(p) == []
        assert (result["combine_failed"], result["combined"]) == (1, 0)

    def test_a_fact_over_the_length_cap_is_not_written(self):
        p = _make_pipeline()
        seed, partner, verdicts = _pair()
        long = BOTH + " " + "The user mentioned it again." * 40
        model = _Model(verdicts, [_write(["M1", "M2"], long)] * 2, [_holds(2)] * 2)

        result, _ = _run(p, [seed], {"f2": [partner]}, model)

        assert "under 800 characters" in model.writer_inputs()[1]["feedback"]
        assert result["combined"] == 0

    def test_the_retry_offers_only_the_members_the_writer_chose(self):
        p = _make_pipeline()
        seed = _fact("f3", START, ts=3, created_at="2023-11-20T00:00:00+00:00")
        plans = _fact("f1", PLANS, vector=BAND, created_at="2023-11-01T00:00:00+00:00")
        other = _fact("f2", "The user is hosting a New Year party", vector=BAND)
        verdicts = {START: [_verdict("partial_overlap", "C1"), _verdict("partial_overlap", "C2")]}
        model = _Model(
            verdicts,
            [_write(["M2", "M1"], BOTH), _write(["M1", "M2"], BOTH)],
            [_holds(2, missing={"C1": ["15 guests"]}), _holds(2)],
        )

        _run(p, [seed], {"f3": [plans, other]}, model)

        offered = [m["text"] for m in json.loads(model.writer_inputs()[1]["memories"])]
        assert sorted(offered) == sorted([PLANS, START])

    def test_the_prompts_the_group_uses_are_the_writer_and_the_dedup_check(self):
        p = _make_pipeline()
        seed, partner, verdicts = _pair()

        _, model = _run(p, [seed], {"f2": [partner]}, _Model(verdicts, [_write(["M1", "M2"], BOTH)], [_holds(2)]))

        assert [filename for filename, _ in model.calls] == [
            "dedup_verify.prompty",
            "combine_memories.prompty",
            "dedup_verify.prompty",
        ]
        check = json.loads(model.calls[2][1]["seed"])
        assert check == {"said": "2023-11-20", "by": "the user", "text": BOTH}


class TestCombinedRecord:
    LINEAGE = {"prompt_id": COMBINE_PROMPT, "prompt_version": "v1"}

    def test_it_takes_the_newest_members_fields_and_every_members_tags_and_sources(self):
        older = _fact(
            "f1", PLANS, created_at="2023-11-01T00:00:00+00:00", thread_id="t0", tags=["sys:fact", "topic:dinner"]
        )
        older["source_turn_ids"] = ["turn-1"]
        newer = _fact("f2", START, created_at="2023-11-20T00:00:00+00:00", salience=0.9, confidence=0.4)
        newer["source_turn_ids"] = ["turn-2"]

        record = combined_record([older, newer], BOTH, lineage=self.LINEAGE)

        assert (record["thread_id"], record["created_at"]) == ("t1", "2023-11-20T00:00:00+00:00")
        assert record["tags"] == ["sys:combined", "sys:fact", "topic:dinner"]
        assert record["source_turn_ids"] == ["turn-1", "turn-2"]
        assert (record["salience"], record["confidence"]) == (0.9, 0.8)
        assert (record["content"], record["prompt_id"], record["prompt_version"]) == (BOTH, COMBINE_PROMPT, "v1")
        assert not {"embedding", "superseded_by", "supersede_reason", "superseded_at"} & set(record)
        assert not [key for key in record if key.startswith("_")]

    def test_it_names_its_members_and_what_they_had_absorbed(self):
        older = _fact("f1", PLANS, created_at="2023-11-01T00:00:00+00:00")
        newer = _fact("f2", START, created_at="2023-11-20T00:00:00+00:00", merged_from=["f0"])

        record = combined_record([older, newer], BOTH, lineage=self.LINEAGE)

        assert record["merged_from"] == ["f0", "f1", "f2"]

    def test_no_freshness_fields_are_merged_from_older_members(self):
        older = _fact(
            "f1",
            PLANS,
            created_at="2023-11-01T00:00:00+00:00",
            confirmed_at="2023-12-01T00:00:00+00:00",
            durability="plan",
            valid_until="2024-01-01",
        )
        newer = _fact("f2", START, created_at="2023-11-20T00:00:00+00:00")

        record = combined_record([older, newer], BOTH, lineage=self.LINEAGE)

        assert not {"confirmed_at", "durability", "valid_until"} & set(record)

    def test_a_user_fact_that_absorbs_an_agent_member_stays_a_user_fact(self):
        user = _fact("f2", START, created_at="2023-11-20T00:00:00+00:00")
        agent = _fact("f1", PLANS, created_at="2023-11-01T00:00:00+00:00", agent=True)

        assert "sys:agent-fact" not in combined_record([agent, user], BOTH, lineage=self.LINEAGE)["tags"]

    def test_the_record_id_follows_its_user_and_members_but_not_its_text(self):
        a, b, c = _fact("f1", PLANS), _fact("f2", START), _fact("f3", START)

        first = combined_record([a, b], BOTH, lineage=self.LINEAGE)["id"]

        assert first == combined_record([b, a], BOTH, lineage=self.LINEAGE)["id"]
        assert first == combined_record([a, b], BOTH + " Again.", lineage=self.LINEAGE)["id"]
        assert first != combined_record([a, c], BOTH, lineage=self.LINEAGE)["id"]
        other_user = [{**a, "user_id": "u2"}, {**b, "user_id": "u2"}]
        assert first != combined_record(other_user, BOTH, lineage=self.LINEAGE)["id"]


class TestCommitCombine:
    """The combined fact is created first; its members are retired one after another."""

    @staticmethod
    def _build() -> PipelineService:
        p = PipelineService.__new__(PipelineService)
        p._container = MagicMock()
        p._memories_container = p._container
        p._mark_superseded = MagicMock(return_value=True)
        return p

    @staticmethod
    def _inputs():
        members = [_fact("f1", PLANS), _fact("f2", START)]
        return combined_record(members, BOTH, lineage={"prompt_id": COMBINE_PROMPT}), members

    def test_every_member_is_retired_into_the_created_fact(self):
        p = self._build()
        record, members = self._inputs()

        assert p._commit_combine(record, members) == ["f1", "f2"]

        p._container.create_item.assert_called_once_with(body=record)
        assert [(c.args[1], c.kwargs["reason"]) for c in p._mark_superseded.call_args_list] == [
            (record["id"], "combined"),
            (record["id"], "combined"),
        ]

    def test_a_fact_that_already_exists_is_not_created_again_and_its_members_are_still_retired(self):
        p = self._build()
        p._container.create_item.side_effect = CosmosResourceExistsError(status_code=409, message="exists")
        p._container.read_item.return_value = {"id": "x", "type": "fact"}
        record, members = self._inputs()

        assert p._commit_combine(record, members) == ["f1", "f2"]
        assert p._container.read_item.call_args.kwargs == {"item": record["id"], "partition_key": ["u1", "t1"]}

    @pytest.mark.parametrize(
        "existing",
        [
            {"id": "x", "superseded_by": "other", "supersede_reason": "contradict"},
            CosmosResourceNotFoundError(message="x"),
        ],
        ids=["retired-since", "deleted-since"],
    )
    def test_a_fact_that_is_there_but_no_longer_active_keeps_the_members_active(self, existing):
        p = self._build()
        p._container.create_item.side_effect = CosmosResourceExistsError(status_code=409, message="exists")
        if isinstance(existing, BaseException):
            p._container.read_item.side_effect = existing
        else:
            p._container.read_item.return_value = existing
        record, members = self._inputs()

        assert p._commit_combine(record, members) is None

        p._mark_superseded.assert_not_called()

    def test_a_write_that_finds_a_retired_fact_is_reported_lost_and_not_counted(self):
        p = _make_pipeline()
        p._container.create_item.side_effect = lambda body: (
            (_ for _ in ()).throw(CosmosResourceExistsError(status_code=409, message="exists"))
            if body["type"] == "fact"
            else body
        )
        registry_read = p._container.read_item.side_effect

        def read_item(item, partition_key):
            if item in p._docs:
                return registry_read(item, partition_key)
            if item.startswith("dedup_cursor"):
                raise CosmosResourceNotFoundError(message="no dedup cursor yet")
            return {"id": item, "superseded_by": "other"}  # the combined fact, retired since

        p._container.read_item.side_effect = read_item
        seed, partner, verdicts = _pair()

        result, _ = _run(p, [seed], {"f2": [partner]}, _Model(verdicts, [_write(["M1", "M2"], BOTH)], [_holds(2)]))

        assert _retired(p) == []
        assert (result["synthesized"], result["combined"], result["combine_failed"]) == (0, 0, 1)

    def test_a_member_that_changed_since_the_pass_read_it_stays_active(self):
        p = self._build()
        p._mark_superseded.side_effect = [False, True]
        record, members = self._inputs()

        assert p._commit_combine(record, members) == ["f2"]

    def test_a_server_error_creating_the_fact_retires_nothing(self):
        p = self._build()
        p._container.create_item.side_effect = RuntimeError("503")
        record, members = self._inputs()

        with pytest.raises(RuntimeError):
            p._commit_combine(record, members)

        p._mark_superseded.assert_not_called()


class TestAMemberChangedSinceThePassReadIt:
    @pytest.mark.parametrize(
        "change",
        [
            pytest.param(None, id="deleted"),
            pytest.param({"_etag": "etag-newer"}, id="edited"),
            pytest.param({"superseded_by": "other"}, id="retired"),
        ],
    )
    def test_the_combined_fact_is_not_written(self, change):
        p = _make_pipeline()
        seed, partner, verdicts = _pair()
        replacement = None if change is None else {**partner, **change}
        model = _Model(verdicts, [_write(["M1", "M2"], BOTH)], [_holds(2)])

        result, _ = _run(p, [seed], {"f2": [partner]}, model, changed={"f1": replacement})

        assert _created(p) == [] and _retired(p) == []
        assert (result["synthesized"], result["combined"], result["combine_failed"]) == (0, 0, 1)


class TestACombinedFactIsRedactedBeforeItIsEmbedded:
    def test_the_vector_survives_the_write_guard_and_matches_the_stored_text(self):
        token = "ghp_" + "A1b2" * 9  # built here so the source holds no token literal
        p = _make_pipeline()
        seed, partner, verdicts = _pair()
        model = _Model(verdicts, [_write(["M1", "M2"], f"{BOTH} Its token is {token}.")], [_holds(2)])

        _run(p, [seed], {"f2": [partner]}, model)

        [body] = _created(p)
        assert token not in body["content"] and "[REDACTED-SECRET]" in body["content"]
        assert body["embedding"] == [0.1, 0.2, 0.3]
        assert token not in p._embed_one.call_args.args[0]


class TestWhichFactsCombine:
    @pytest.mark.parametrize(
        "change",
        [
            {"assertion": "directive"},
            {"assertion": "correction"},
            {"type": "procedural"},
            {"tags": ["sys:fact", "sys:combined"]},
        ],
    )
    def test_some_facts_are_never_combined(self, change):
        assert may_combine(_fact("f1", PLANS)) and not may_combine(_fact("f1", PLANS, **change))

    def test_a_fact_that_absorbed_duplicates_can_still_be_combined(self):
        assert may_combine(_fact("f1", PLANS, merged_from=["f0"]))

    @pytest.mark.parametrize(
        ("change", "allowed"),
        [
            ({}, True),
            ({"assertion": "decided"}, False),
            ({"agent": True}, False),
            ({"thread_id": "another-thread"}, True),
        ],
    )
    def test_a_pair_needs_one_speaker_and_assertion(self, change, allowed):
        assert combinable_pair(_fact("f1", PLANS), _fact("f2", START, **change)) is allowed

    def test_a_fact_without_an_assertion_pairs_with_one_that_has_none(self):
        assert combinable_pair(_fact("f1", PLANS), _fact("f2", START)) is True
        assert combinable_pair(_fact("f1", PLANS, assertion="stated"), _fact("f2", START)) is False

    def test_the_fact_with_most_partners_leads_a_group_of_at_most_four(self):
        hub = _fact("hub", "hub", created_at="2024-02-01T00:00:00+00:00")
        spokes = [_fact(f"s{i}", f"spoke {i}") for i in range(1, 6)]
        checked = [(s, [Verdict(hub, 0.70 + i / 100, "partial_overlap", (), ())]) for i, s in enumerate(spokes, 1)]

        groups = plan_combine_groups(checked, set())

        assert [[d["id"] for d in g] for g in groups] == [["hub", "s5", "s4", "s3"]]

    def test_each_fact_joins_one_group_and_skipped_ones_none(self):
        a, b, c, d = (_fact(i, i) for i in ("a", "b", "c", "d"))
        checked = [
            (a, [Verdict(b, 0.9, "partial_overlap", (), ())]),
            (c, [Verdict(b, 0.8, "partial_overlap", (), ()), Verdict(d, 0.7, "related", (), ())]),
        ]

        assert [[x["id"] for x in g] for g in plan_combine_groups(checked, set())] == [["b", "a", "c"]]
        assert plan_combine_groups(checked, {"b"}) == []

    def test_groups_are_tried_in_seed_order_and_the_cursor_stops_before_an_untried_one(self):
        s1, s2, s3, old = (_fact(i, i) for i in ("s1", "s2", "s3", "old"))
        groups = [[old, s3], [s2, s1]]

        assert [[d["id"] for d in g] for g in in_seed_order(groups, [s1, s2, s3])] == [["s2", "s1"], ["old", "s3"]]
        assert [d["id"] for d in settled_seeds([s1, s2, s3], [[old, s3]])] == ["s1", "s2"]
        assert settled_seeds([s1, s2, s3], []) == [s1, s2, s3]

    def test_a_negation_the_member_states_must_survive(self):
        assert kept_specifics("The user does not eat meat", "The user does not eat meat and likes Thai food")
        assert not kept_specifics("The user does not eat meat", "The user eats meat")
        assert kept_specifics("The user eats fish", "The user eats fish but never meat")
        assert not kept_specifics("The user met Patricia on 2024-03-02", "The user met someone on 2024-03-02")


class TestFailuresAndTheCursor:
    def test_a_retryable_error_while_combining_keeps_the_cursor_before_the_group(self):
        p = _make_pipeline()
        alone = _fact("f0", "The user likes tea", ts=1)
        seed, partner, verdicts = _pair()
        model = _Model(verdicts, [RuntimeError("429 rate limited")])

        result, _ = _run(p, [alone, seed], {"f2": [partner]}, model)

        assert _created(p) == []
        assert (result["checked"], result["combine_groups"]) == (2, 0)
        assert _cursor(p) == (1, ["f0"])

    def test_any_other_model_error_leaves_the_group_and_moves_on(self):
        p = _make_pipeline()
        seed, partner, verdicts = _pair()

        result, _ = _run(
            p, [seed], {"f2": [partner]}, _Model(verdicts, [LLMError("content_filter: the response was filtered")])
        )

        assert (result["combine_groups"], result["combine_failed"]) == (1, 1)
        assert _cursor(p) == (2, ["f2"])

    def test_a_combined_fact_that_fails_validation_is_dropped_and_the_cursor_moves_on(self):
        p = _make_pipeline()
        seed = _fact("f2", START, ts=2, created_at="2023-11-20T00:00:00+00:00", metadata={"source": "user"})
        seed, partner, verdicts = _pair(seed=seed)

        result, _ = _run(p, [seed], {"f2": [partner]}, _Model(verdicts, [_write(["M1", "M2"], BOTH)], [_holds(2)]))

        assert _created(p) == []
        p._embed_one.assert_not_called()
        assert (result["combine_groups"], result["combine_failed"]) == (1, 1)
        assert _cursor(p) == (2, ["f2"])

    def test_a_retryable_embedding_error_keeps_the_cursor_before_the_group(self):
        p = _make_pipeline()
        p._embed_one = MagicMock(side_effect=RuntimeError("429 rate limited"))
        alone = _fact("f0", "The user likes tea", ts=1)
        seed, partner, verdicts = _pair()

        result, _ = _run(
            p, [alone, seed], {"f2": [partner]}, _Model(verdicts, [_write(["M1", "M2"], BOTH)], [_holds(2)])
        )

        assert _created(p) == []
        assert result["combine_groups"] == 0
        assert _cursor(p) == (1, ["f0"])

    def test_a_non_retryable_embedding_error_drops_the_group(self):
        p = _make_pipeline()
        p._embed_one = MagicMock(side_effect=LLMError("context_length_exceeded"))
        seed, partner, verdicts = _pair()

        result, _ = _run(p, [seed], {"f2": [partner]}, _Model(verdicts, [_write(["M1", "M2"], BOTH)], [_holds(2)]))

        assert _created(p) == [] and (result["combine_groups"], result["combine_failed"]) == (1, 1)
        assert _cursor(p) == (2, ["f2"])

    def test_members_that_changed_since_the_pass_read_them_stay_active_and_are_not_counted(self):
        p = _make_pipeline()
        p._mark_superseded.return_value = False
        seed, partner, verdicts = _pair()

        result, _ = _run(p, [seed], {"f2": [partner]}, _Model(verdicts, [_write(["M1", "M2"], BOTH)], [_holds(2)]))

        assert (result["combined"], result["synthesized"], result["combine_failed"]) == (0, 1, 0)
        assert result["kept"] == 1
        assert _cursor(p) == (2, ["f2"])

    def test_checking_stops_halfway_to_the_deadline_so_groups_can_be_written(self, monkeypatch):
        p = _make_pipeline()
        clock = {"now": 0.0}
        monkeypatch.setattr(pipeline_module, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
        seed, partner, verdicts = _pair()
        later = _fact("f3", "The user is hosting a New Year party", ts=3)
        model = _Model(verdicts, [_write(["M1", "M2"], BOTH)], [_holds(2)])
        check = model.answer

        def slow(filename, inputs):
            clock["now"] = 60.0
            return check(filename, inputs)

        model.answer = slow
        result, _ = _run(p, [seed, later], {"f2": [partner], "f3": [partner]}, model, deadline=100.0)

        assert (result["checked"], result["combined"]) == (1, 2)
        assert _cursor(p) == (2, ["f2"])

    def test_groups_not_started_by_the_deadline_are_checked_again_later(self, monkeypatch):
        p = _make_pipeline()
        clock = {"now": 0.0}
        monkeypatch.setattr(pipeline_module, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
        alone = _fact("f0", "The user likes tea", ts=1)
        seed, partner, verdicts = _pair()
        model = _Model(verdicts)
        check = model.answer

        def slow(filename, inputs):
            clock["now"] = 120.0
            return check(filename, inputs)

        model.answer = slow
        result, _ = _run(p, [alone, seed], {"f2": [partner]}, model, deadline=100.0)

        assert model.writer_inputs() == [] and result["checked"] == 2
        assert _cursor(p) == (1, ["f0"])

    def test_a_pass_without_a_deadline_never_stops_checking_early(self):
        p = _make_pipeline()
        seed, partner, verdicts = _pair()

        result, _ = _run(p, [seed], {"f2": [partner]}, _Model(verdicts, [_write(["M1", "M2"], BOTH)], [_holds(2)]))

        assert result["combined"] == 2

    def test_the_worker_limit_bounds_how_many_groups_are_written_at_once(self, monkeypatch):
        monkeypatch.setattr("azure.cosmos.agent_memory.thresholds.COMBINE_WORKERS", 1)
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
        answer = model.answer

        def counted(filename, inputs):
            if filename == COMBINE_PROMPT:
                with model._lock:
                    model.active += 1
                    model.max_active = max(model.max_active, model.active)
                time.sleep(0.01)
                try:
                    return answer(filename, inputs)
                finally:
                    with model._lock:
                        model.active -= 1
            return answer(filename, inputs)

        model.answer = counted

        result, _ = _run(p, seeds, neighbours, model)

        assert result["synthesized"] == 3 and model.max_active == 1


class TestWriteResult:
    def test_a_combined_fact_the_pipeline_could_not_build_reports_lost(self):
        p = PipelineService.__new__(PipelineService)

        assert p._write_combined(None) == CombineWrite("lost")

    def test_reconcile_reports_the_combine_counts(self, caplog):
        p = _make_pipeline()
        p._reconcile_contradictions = MagicMock(return_value={"kept": 0, "merged": 0, "contradicted": 0})
        p._active_memories_for_reconcile = MagicMock(return_value=[])
        seed, partner, verdicts = _pair()
        p._docs.update({seed["id"]: seed, partner["id"]: partner})
        p._container.query_items.side_effect = lambda query, parameters, **kwargs: iter(
            [partner] if "VectorDistance" in query else [seed]
        )
        p._run_prompty = _Model(verdicts, [_write(["M1", "M2"], BOTH)], [_holds(2)])

        with caplog.at_level(logging.INFO, logger="azure.cosmos.agent_memory.pipeline"):
            result = p.reconcile_memories("u1")

        [record] = [r for r in caplog.records if r.getMessage() == "reconcile.outcome"]
        assert (result["combined"], result["synthesized"], result["combine_groups"]) == (2, 1, 1)
        assert (record.combined, record.synthesized, record.combine_groups, record.combine_failed) == (2, 1, 1, 0)
