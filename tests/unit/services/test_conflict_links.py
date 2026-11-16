"""``reconcile_memories`` links facts that conflict instead of retiring one of them.

The duplicate check labels how each new fact relates to its nearest neighbours; ``update`` (the same attribute with a
different value at a later time) and ``inconsistent`` (the two cannot both be true) name a conflict. A pass that finds
one keeps both facts active and writes one small link document for the pair, in the user's ``__conflicts__``
partition, from the verdicts it already has: no model call is added.

``reconcile_memories`` and ``consolidate_memories`` run for real here, with the real memory store and prompt
templates, against a fake memories container that answers the passes' queries the way Cosmos DB does (it applies every
filter, the ``ORDER BY`` and the ``TOP``, keeps one ETag per record, rejects a conditional write whose ETag is stale,
and finds a record only under its own partition) and a scripted chat model that reads each rendered prompt. A
predicate the fake does not understand, or a prompt the test did not plan for, fails the test.

Covers:
* a conflict (``update`` or ``inconsistent``): both facts stay active, nothing is written to retire either, one link
  document names the pair, ``conflicts`` counts it and the pass asks the model nothing more; judging the pair again
  rewrites the same document
* a conflicting fact that is combined in the same pass is linked as the combined fact, and one that is folded into
  another as a duplicate as the fact that took its place
* what is not linked: other relations, and any pair with a fact the assistant wrote
* a link that cannot be written neither fails the pass nor holds the cursor back
* the no-op types report ``conflicts: 0``
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
import math
import re
import threading
from typing import Any, Callable, Iterable, Iterator
from unittest.mock import MagicMock

import pytest
from azure.core import MatchConditions
from azure.cosmos.exceptions import (
    CosmosAccessConditionFailedError,
    CosmosHttpResponseError,
    CosmosResourceExistsError,
    CosmosResourceNotFoundError,
)

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory._utils import cosine_similarity
from azure.cosmos.agent_memory.services._combine import COMBINED_TAG
from azure.cosmos.agent_memory.services.pipeline import PipelineService
from azure.cosmos.agent_memory.store import MemoryStore

USER = "u1"
CURSOR_ID = "dedup_cursor_u1_fact"

RESULT_KEYS = {
    "kept",
    "merged",
    "conflicts",
    "checked",
    "compared",
    "vetoed",
    "combined",
    "synthesized",
    "combine_groups",
    "combine_failed",
}
RETIREMENT_FIELDS = ("superseded_by", "supersede_reason", "superseded_at")

# One attribute with two values, the second said later, and two statements that cannot both be true.
PORTLAND = "The user lives in Portland."
SEATTLE = "The user lives in Seattle."
VEGETARIAN = "The user is vegetarian."
STEAK = "The user eats a steak every week."
CONFLICTS = {"update": (PORTLAND, SEATTLE), "inconsistent": (VEGETARIAN, STEAK)}

# A third place, and a fact that holds everything the Seattle one states.
DENVER = "The user lives in Denver."
LAKE = "The user lives in Seattle near Green Lake."

# The flight: an old departure, and two newer facts that say more about the new date.
OLD_FLIGHT = "The user's flight to Paris departs June 1."
GATE = "The user's flight to Paris now departs June 3 from gate B12."
LEAVES = "The user's June 3 flight to Paris leaves at 9am."
FLIGHT = "The user's flight to Paris departs June 3 at 9am from gate B12."

V_A = [1.0, 0.0, 0.0]
V_B = [0.9, 0.3, 0.0]  # 0.95 from V_A: above the dedup floor, so the two are compared
V_C = [0.9, 0.0, 0.3]  # 0.95 from V_A and 0.90 from V_B
V_PORTLAND, V_LAKE = [0.87, -0.4931, 0.0], [0.87, 0.4931, 0.0]  # 0.87 from V_A each, 0.51 from one another
V_OLD = [0.8, 0.0, 0.6]  # 0.80 from V_A: below the dedup floor, above the combine floor
V_LEAVES = [0.75, math.sqrt(1 - 0.75**2), 0.0]  # 0.75 from V_A, 0.60 from V_OLD
V_FLIGHT = [0.9, 0.3, math.sqrt(0.1)]  # the vector the combined fact is embedded as


# ---------------------------------------------------------------------------
# The fake container
# ---------------------------------------------------------------------------
_SELECT = re.compile(
    r"SELECT (?:TOP (?P<top>\d+) )?(?P<cols>.+?) FROM c WHERE (?P<where>.+?)(?: ORDER BY (?P<order>.+))?"
)
_ACTIVE = "(NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))"


def _conjuncts(where: str) -> list[str]:
    """``where`` split at its top-level ``AND``s (one inside parentheses belongs to its predicate)."""
    parts, depth, start, index = [], 0, 0, 0
    while index < len(where):
        if where[index] == "(":
            depth += 1
        elif where[index] == ")":
            depth -= 1
        elif depth == 0 and where.startswith(" AND ", index):
            parts.append(where[start:index])
            start, index = index + 5, index + 4
        index += 1
    return [*parts, where[start:]]


def _holds(doc: dict[str, Any], predicate: str, params: dict[str, Any]) -> bool:
    if predicate == _ACTIVE:
        return doc.get("superseded_by") is None
    if m := re.fullmatch(r"c\.(\w+) = (@\w+)", predicate):
        return doc.get(m[1]) == params[m[2]]
    if m := re.fullmatch(r"c\.(\w+) != (@\w+)", predicate):
        return doc.get(m[1]) != params[m[2]]
    if m := re.fullmatch(r"IS_DEFINED\(c\.(\w+)\)", predicate):
        return m[1] in doc
    if m := re.fullmatch(r"c\._ts >= (@\w+)", predicate):
        return doc["_ts"] >= params[m[1]]
    if m := re.fullmatch(r"NOT \(c\._ts = (@\w+) AND ARRAY_CONTAINS\((@\w+), c\.id\)\)", predicate):
        return not (doc["_ts"] == params[m[1]] and doc["id"] in params[m[2]])
    raise AssertionError(f"the fake container does not understand {predicate!r}")


def _ordered(rows: list[dict[str, Any]], order: str | None, params: dict[str, Any]) -> list[dict[str, Any]]:
    if order is None:
        return rows
    if order == "c._ts ASC":
        return sorted(rows, key=lambda doc: doc["_ts"])
    if m := re.fullmatch(r"VectorDistance\(c\.embedding, (@\w+)\)", order):
        return sorted(rows, key=lambda doc: -cosine_similarity(doc["embedding"], params[m[1]]))
    raise AssertionError(f"the fake container does not understand ORDER BY {order!r}")


class Account:
    """The memories container as Cosmos DB serves the passes: one partition per ``[user_id, thread_id]``."""

    def __init__(self, docs: Iterable[dict[str, Any]] = ()) -> None:
        self._docs: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._writes = itertools.count(1000)
        self.queries: list[str] = []
        # The errors the next writes of link documents raise, one each.
        self.link_write_errors: list[Exception] = []
        for doc in docs:
            self.put(doc)

    # What another client, or the test, does.
    def put(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Write ``doc`` as it is, stamping a new ETag and write time (a second client's write, or a seed)."""
        stored = {key: value for key, value in doc.items() if not key.startswith("_")}
        number = next(self._writes)
        stored["_etag"], stored["_ts"] = f'"{number}"', number
        self._docs[(stored["user_id"], stored["thread_id"], stored["id"])] = stored
        return copy.deepcopy(stored)

    def get(self, record_id: str) -> dict[str, Any] | None:
        return next((copy.deepcopy(doc) for doc in self._docs.values() if doc["id"] == record_id), None)

    def delete(self, record_id: str) -> None:
        self._docs = {key: doc for key, doc in self._docs.items() if doc["id"] != record_id}

    def active_ids(self) -> set[str]:
        return {doc["id"] for doc in self._docs.values() if doc["type"] == "fact" and doc.get("superseded_by") is None}

    def retirements(self) -> dict[str, tuple[str, str]]:
        """Each retired fact's ID with who it was retired into and why."""
        return {
            doc["id"]: (doc["superseded_by"], doc["supersede_reason"])
            for doc in self._docs.values()
            if doc["type"] == "fact" and doc.get("superseded_by") is not None
        }

    def combined(self) -> dict[str, Any]:
        [doc] = [copy.deepcopy(d) for d in self._docs.values() if COMBINED_TAG in (d.get("tags") or [])]
        return doc

    def links(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(doc) for doc in self._docs.values() if doc["type"] == "conflict_link"]

    # What the pipeline calls.
    def query_items(
        self,
        *,
        query: str,
        parameters: list[dict[str, Any]] | None = None,
        partition_key: Any = None,
        enable_cross_partition_query: bool = False,
    ) -> Iterator[dict[str, Any]]:
        if partition_key is None and not enable_cross_partition_query:
            raise CosmosHttpResponseError(status_code=400, message="Cross partition query is required but disabled.")
        self.queries.append(query)
        match = _SELECT.fullmatch(query)
        assert match, f"the fake container cannot parse {query!r}"
        params = {parameter["name"]: parameter["value"] for parameter in parameters or []}
        predicates = _conjuncts(match["where"])
        rows = [doc for doc in self._docs.values() if all(_holds(doc, predicate, params) for predicate in predicates)]
        if partition_key is not None:
            rows = [doc for doc in rows if [doc["user_id"], doc["thread_id"]] == list(partition_key)]
        rows = _ordered(rows, match["order"], params)[: int(match["top"] or len(rows))]
        if match["cols"] == "*":
            return iter(copy.deepcopy(rows))
        columns = [column.strip().removeprefix("c.") for column in match["cols"].split(",")]
        return iter([{column: doc.get(column) for column in columns} for doc in rows])

    def read_item(self, item: str, partition_key: Any) -> dict[str, Any]:
        doc = self._docs.get((partition_key[0], partition_key[1], item))
        if doc is None:
            raise CosmosResourceNotFoundError(status_code=404, message="Entity with the specified id does not exist.")
        return copy.deepcopy(doc)

    def create_item(self, body: dict[str, Any]) -> dict[str, Any]:
        if (body["user_id"], body["thread_id"], body["id"]) in self._docs:
            raise CosmosResourceExistsError(status_code=409, message="Entity with the specified id already exists.")
        return self.put(body)

    def upsert_item(self, body: dict[str, Any]) -> dict[str, Any]:
        if body.get("type") == "conflict_link" and self.link_write_errors:
            raise self.link_write_errors.pop(0)
        return self.put(body)

    def replace_item(
        self, item: str, body: dict[str, Any], *, etag: str | None = None, match_condition: Any = None, **_: Any
    ) -> dict[str, Any]:
        assert (etag is None) == (match_condition is None), "a conditional write names the ETag and how to match it"
        current = self._docs.get((body["user_id"], body["thread_id"], item))
        if current is None:
            raise CosmosResourceNotFoundError(status_code=404, message="Entity with the specified id does not exist.")
        if match_condition == MatchConditions.IfNotModified and etag != current["_etag"]:
            raise CosmosAccessConditionFailedError(status_code=412, message="the ETag does not match")
        return self.put(body)


# ---------------------------------------------------------------------------
# The scripted chat model
# ---------------------------------------------------------------------------
def _tagged(prompt: str, tag: str) -> Any:
    """The JSON a rendered prompt holds between ``<tag>`` and ``</tag>``, or ``None`` when it has no such block."""
    found = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", prompt, re.DOTALL)
    return json.loads(found[1]) if found else None


class Chat:
    """The chat client as a script. It reads each rendered prompt and answers by it: ``verify(seed, candidate)`` says
    how two facts relate (each given as its text), and ``write(texts)`` picks the facts to combine from those offered
    and writes the combined one. ``calls`` keeps the name of each prompt it was asked, in order; a prompt it does not
    know is an error, so a model call the test did not plan for fails it."""

    def __init__(
        self,
        verify: Callable[[str, str], str],
        write: Callable[[list[str]], tuple[list[str], str]] | None = None,
    ) -> None:
        self.verify, self.write = verify, write
        self.calls: list[str] = []
        self._lock = threading.Lock()  # the combine writer runs on a worker thread

    def generate(self, messages: list[dict[str, str]], **options: Any) -> str:
        del options
        with self._lock:
            return self._answer(messages[-1]["content"])

    def _answer(self, prompt: str) -> str:
        if (seed := _tagged(prompt, "untrusted_seed_json")) is not None:
            self.calls.append("dedup_verify")
            return json.dumps(
                {
                    "candidates": [
                        {
                            "label": candidate["label"],
                            "relation": self.verify(seed["text"], candidate["text"]),
                            "seed_details_missing_from_candidate": [],
                            "candidate_details_missing_from_seed": [],
                        }
                        for candidate in _tagged(prompt, "untrusted_candidates_json")
                    ]
                }
            )
        if (memories := _tagged(prompt, "untrusted_memories_json")) is not None:
            assert self.write is not None, "the model was asked to combine facts the test did not plan to combine"
            self.calls.append("combine_memories")
            picked, text = self.write([memory["text"] for memory in memories])
            labels = [memory["label"] for memory in memories if memory["text"] in picked]
            return json.dumps({"combine": labels, "memory": text})
        raise AssertionError(f"the model was asked something the test did not plan for: {prompt[:80]!r}")


def between(pairs: dict[tuple[str, str], str]) -> Callable[[str, str], str]:
    """A ``verify`` that says how each listed pair of facts relates, whichever of the two is the seed."""

    def verify(seed: str, candidate: str) -> str:
        return pairs.get((seed, candidate)) or pairs.get((candidate, seed)) or "unrelated"

    return verify


def _flight_verify(seed: str, candidate: str) -> str:
    texts = {seed, candidate}
    if OLD_FLIGHT in texts:
        return "update"
    if FLIGHT in texts:  # the check of the combined fact against the members it was written from
        return "seed_contains_candidate" if seed == FLIGHT else "candidate_contains_seed"
    return "partial_overlap" if texts == {GATE, LEAVES} else "unrelated"


def lake_verify(seed: str, candidate: str) -> str:
    texts = {seed, candidate}
    if texts == {PORTLAND, SEATTLE}:
        return "update"
    if texts == {SEATTLE, LAKE}:
        return "seed_contains_candidate" if seed == LAKE else "candidate_contains_seed"
    return "unrelated"


def flight_chat() -> Chat:
    return Chat(_flight_verify, lambda texts: ([t for t in texts if t in (GATE, LEAVES)], FLIGHT))


class Embeddings:
    """Embeds the combined facts; any other text would be a call the test did not plan for."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def generate(self, text: str) -> list[float]:
        return list(self.vectors[text])


VECTORS = {FLIGHT: V_FLIGHT}


def fact(
    fid: str,
    content: str,
    vector: list[float],
    created_at: str,
    *,
    thread_id: str = "t1",
    agent: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": fid,
        "user_id": USER,
        "thread_id": thread_id,
        "type": "fact",
        "role": "system",
        "content": content,
        "confidence": 0.8,
        "salience": 0.6,
        "tags": ["sys:fact", "sys:agent-fact"] if agent else ["sys:fact"],
        "metadata": {"source": "agent" if agent else "user", "category": "other"},
        "assertion": "stated",
        "created_at": created_at,
        "embedding": vector,
        **extra,
    }


def service_over(account: Any, chat: Any) -> PipelineService:
    """The real pipeline and memory store over ``account``, with ``chat`` as the model."""
    embeddings = Embeddings(VECTORS)
    containers = {ContainerKey.MEMORIES: account, ContainerKey.TURNS: MagicMock(), ContainerKey.SUMMARIES: MagicMock()}
    store = MemoryStore(containers=containers, embeddings_client=embeddings)
    return PipelineService(store, chat, embeddings, containers=containers)


# The two facts of a conflict: in different threads, so in different partitions.
def conflicting_facts(older: str, newer: str) -> list[dict[str, Any]]:
    return [
        fact("f1", older, V_A, "2026-03-01T10:00:00Z", thread_id="t1"),
        fact("f2", newer, V_B, "2026-04-01T10:00:00Z", thread_id="t2"),
    ]


def a_conflict(kind: str) -> tuple[Account, Chat, PipelineService]:
    older, newer = CONFLICTS[kind]
    account, chat = Account(conflicting_facts(older, newer)), Chat(between({(older, newer): kind}))
    return account, chat, service_over(account, chat)


def link_fields(link: dict[str, Any]) -> dict[str, Any]:
    """What a link document holds, without its write time and the Cosmos DB system fields."""
    return {key: value for key, value in link.items() if key != "created_at" and not key.startswith("_")}


def assert_no_retirement(account: Account, *fact_ids: str) -> None:
    assert account.retirements() == {}
    for fact_id in fact_ids:
        assert not set(RETIREMENT_FIELDS) & set(account.get(fact_id)), "nothing of a retirement is on the record"


def cursor_of(account: Account) -> tuple[int, list[str]]:
    cursor = account.get(CURSOR_ID)
    return cursor["last_ts"], cursor["ids_at_last_ts"]


# ---------------------------------------------------------------------------
# A conflict keeps both facts active and links them
# ---------------------------------------------------------------------------
class TestAConflictIsLinkedNotRetired:
    @pytest.mark.parametrize("entry", ["reconcile_memories", "consolidate_memories"])
    @pytest.mark.parametrize("kind", ["update", "inconsistent"])
    def test_both_facts_stay_active_and_one_link_names_the_pair(self, kind, entry):
        account, chat, service = a_conflict(kind)

        result = getattr(service, entry)(USER)

        assert_no_retirement(account, "f1", "f2")
        assert account.active_ids() == {"f1", "f2"}
        [link] = account.links()
        assert link_fields(link) == {
            "id": link["id"],
            "type": "conflict_link",
            "user_id": USER,
            "thread_id": "__conflicts__",
            "first": "f1",
            "later": "f2",
            "kind": kind,
        }
        assert link["id"].startswith("conflict_") and link["created_at"]
        assert account.read_item(link["id"], [USER, "__conflicts__"])["id"] == link["id"], (
            "it is in the user's partition"
        )
        assert not {"content", "text"} & set(link), "a link names the facts and holds none of their text"
        assert (result["conflicts"], result["checked"], result["compared"], result["kept"]) == (1, 2, 1, 2)
        assert (result["merged"], result["combined"], result["synthesized"]) == (0, 0, 0)

    def test_the_result_has_no_contradicted_count_and_the_model_is_asked_only_the_duplicate_check(self):
        account, chat, service = a_conflict("update")

        result = service.reconcile_memories(USER)

        assert set(result) == RESULT_KEYS and "contradicted" not in result
        assert chat.calls == ["dedup_verify"], "one call judged the pair; nothing judges the facts again"
        assert [call for call in account.queries if "ORDER BY c.created_at DESC" in call] == []

    def test_the_link_is_written_before_the_cursor_moves_past_the_facts(self):
        account, _, service = a_conflict("update")
        last = account.get("f2")["_ts"]

        service.reconcile_memories(USER)

        assert cursor_of(account) == (last, ["f2"])
        [link] = account.links()
        assert link["_ts"] < account.get(CURSOR_ID)["_ts"]

    def test_judging_the_pair_again_in_a_later_pass_rewrites_the_same_document(self):
        account, chat, service = a_conflict("update")
        service.reconcile_memories(USER)
        [first] = account.links()
        account.delete(CURSOR_ID)  # a pass that checks the facts again, as one whose cursor write lost a race does

        second = service.reconcile_memories(USER)

        assert second["conflicts"] == 1, "a pair whose link was written again counts"
        [again] = account.links()
        assert again["id"] == first["id"] and link_fields(again) == link_fields(first)
        assert again["_etag"] != first["_etag"], "it was written again, not left as it was"
        assert chat.calls == ["dedup_verify", "dedup_verify"]

    def test_a_later_verdict_on_the_pair_replaces_the_kind(self):
        older, newer = CONFLICTS["update"]
        account, relation = Account(conflicting_facts(older, newer)), {"now": "update"}
        chat = Chat(lambda seed, candidate: relation["now"])
        service = service_over(account, chat)
        service.reconcile_memories(USER)
        relation["now"] = "inconsistent"
        account.delete(CURSOR_ID)

        service.reconcile_memories(USER)

        [link] = account.links()
        assert link["kind"] == "inconsistent" and (link["first"], link["later"]) == ("f1", "f2")
        assert account.active_ids() == {"f1", "f2"}

    def test_the_outcome_log_reports_the_conflicts(self, caplog):
        _, _, service = a_conflict("inconsistent")

        with caplog.at_level(logging.INFO, logger="azure.cosmos.agent_memory.pipeline"):
            result = service.reconcile_memories(USER)

        [record] = [r for r in caplog.records if r.getMessage() == "reconcile.outcome"]
        assert record.conflicts == result["conflicts"] == 1
        assert not hasattr(record, "contradicted")
        assert (record.facts_checked, record.candidates_considered) == (2, 2)
        assert (record.prompt_id, record.prompt_version) == ("dedup_verify.prompty", "v1")

    @pytest.mark.parametrize("relation", ["related", "unrelated"])
    def test_a_pair_that_is_not_a_conflict_is_not_linked(self, relation):
        older, newer = CONFLICTS["update"]
        account = Account(conflicting_facts(older, newer))
        service = service_over(account, Chat(between({(older, newer): relation})))

        result = service.reconcile_memories(USER)

        assert account.links() == [] and result["conflicts"] == 0
        assert account.active_ids() == {"f1", "f2"}


# ---------------------------------------------------------------------------
# A conflicting fact that this pass retires into another
# ---------------------------------------------------------------------------
class TestALinkFollowsAFactThatTookAnotherPlace:
    def test_a_conflicting_fact_that_is_combined_is_linked_as_the_combined_fact(self):
        account, chat = Account([fact("old", OLD_FLIGHT, V_OLD, "2026-03-01T10:00:00Z", thread_id="t0")]), flight_chat()
        service = service_over(account, chat)
        service.reconcile_memories(USER)
        account.put(fact("g", GATE, V_A, "2026-05-01T10:00:00Z", thread_id="t1"))
        account.put(fact("l", LEAVES, V_LEAVES, "2026-05-01T10:05:00Z", thread_id="t2"))

        result = service.reconcile_memories(USER)

        combined = account.combined()
        assert combined["content"] == FLIGHT
        assert account.retirements() == {"g": (combined["id"], "combined"), "l": (combined["id"], "combined")}
        assert account.active_ids() == {"old", combined["id"]}, "the old departure is not retired for the conflict"
        assert not set(RETIREMENT_FIELDS) & set(account.get("old"))
        [link] = account.links()
        assert (link["first"], link["later"], link["kind"]) == ("old", combined["id"], "update")
        assert (result["conflicts"], result["combined"], result["synthesized"]) == (1, 2, 1)
        assert chat.calls == ["dedup_verify", "combine_memories", "dedup_verify"], "no model call is added"

    def test_a_conflicting_fact_that_is_folded_as_a_duplicate_is_linked_as_the_fact_it_was_folded_into(self):
        account = Account(
            [
                fact("pdx", PORTLAND, V_PORTLAND, "2026-03-01T10:00:00Z", thread_id="t1"),
                fact("lake", LAKE, V_LAKE, "2026-04-01T10:00:00Z", thread_id="t2"),
                fact("sea", SEATTLE, V_A, "2026-04-10T10:00:00Z", thread_id="t3"),
            ]
        )
        chat = Chat(lake_verify)
        service = service_over(account, chat)

        result = service.reconcile_memories(USER)

        assert account.retirements() == {"sea": ("lake", "duplicate")}, "Seattle is folded into the fuller fact"
        assert account.active_ids() == {"pdx", "lake"}, "Portland is not retired for conflicting with it"
        [link] = account.links()
        assert (link["first"], link["later"], link["kind"]) == ("pdx", "lake", "update")
        assert (result["conflicts"], result["merged"], result["kept"]) == (1, 1, 2)
        assert chat.calls == ["dedup_verify", "dedup_verify"]


# ---------------------------------------------------------------------------
# What is not linked
# ---------------------------------------------------------------------------
class TestAPairWithAnAssistantFactIsNotLinked:
    @pytest.mark.parametrize("kind", ["update", "inconsistent"])
    @pytest.mark.parametrize("assistant_is_newer", [True, False], ids=["newer", "older"])
    def test_the_pair_is_judged_and_both_facts_stay_unlinked(self, kind, assistant_is_newer):
        said = "The assistant noted that the user lives in Seattle."
        user_fact = fact("f1", PORTLAND, V_A, "2026-03-01T10:00:00Z", thread_id="t1")
        agent_fact = fact("f2", said, V_B, "2026-04-01T10:00:00Z", thread_id="t2", agent=True)
        if not assistant_is_newer:
            user_fact["created_at"], agent_fact["created_at"] = agent_fact["created_at"], user_fact["created_at"]
        account = Account([user_fact, agent_fact])
        chat = Chat(between({(PORTLAND, said): kind}))
        service = service_over(account, chat)

        result = service.reconcile_memories(USER)

        assert chat.calls == ["dedup_verify"] and result["compared"] == 1, "the pair was judged"
        assert account.links() == [] and result["conflicts"] == 0
        assert_no_retirement(account, "f1", "f2")
        assert account.active_ids() == {"f1", "f2"}


# ---------------------------------------------------------------------------
# A link that cannot be written
# ---------------------------------------------------------------------------
class TestALinkThatCannotBeWrittenNeitherFailsThePassNorHoldsTheCursorBack:
    def test_the_pass_goes_on_and_the_cursor_moves_past_the_facts(self, caplog):
        account, chat, service = a_conflict("update")
        last = account.get("f2")["_ts"]
        account.link_write_errors.append(CosmosHttpResponseError(status_code=503, message=f"unavailable: {PORTLAND}"))

        with caplog.at_level(logging.INFO, logger="azure.cosmos.agent_memory"):
            result = service.reconcile_memories(USER)

        assert (result["conflicts"], result["checked"], result["kept"]) == (0, 2, 2)
        assert account.links() == [] and account.active_ids() == {"f1", "f2"}
        assert cursor_of(account) == (last, ["f2"]), "the cursor moved past both facts"
        warning = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
        assert "CosmosHttpResponseError" in warning and "status=503" in warning, "what failed is described"
        assert not [text for text in (PORTLAND, SEATTLE, "unavailable") if text in caplog.text], "no text is logged"

    def test_the_other_pairs_are_still_written(self):
        account = Account(
            [
                fact("f1", PORTLAND, V_A, "2026-03-01T10:00:00Z", thread_id="t1"),
                fact("f2", DENVER, V_C, "2026-03-15T10:00:00Z", thread_id="t2"),
                fact("f3", SEATTLE, V_B, "2026-04-01T10:00:00Z", thread_id="t3"),
            ]
        )
        service = service_over(account, Chat(lambda seed, candidate: "inconsistent"))
        account.link_write_errors.append(CosmosHttpResponseError(status_code=429, message="throttled"))

        result = service.reconcile_memories(USER)

        assert result["conflicts"] == 2 and len(account.links()) == 2, "three pairs were judged; one write failed"
        assert account.active_ids() == {"f1", "f2", "f3"}
        assert cursor_of(account)[1] == ["f3"]

    def test_a_failed_link_is_written_when_the_pair_is_judged_again(self):
        account, _, service = a_conflict("inconsistent")
        account.link_write_errors.append(RuntimeError("the host was recycled"))
        service.reconcile_memories(USER)
        assert account.links() == []
        account.delete(CURSOR_ID)

        result = service.reconcile_memories(USER)

        assert result["conflicts"] == 1 and len(account.links()) == 1


# ---------------------------------------------------------------------------
# The types a pass leaves alone
# ---------------------------------------------------------------------------
class TestTheNoOpTypesReportNoConflicts:
    @pytest.mark.parametrize("entry", ["reconcile_memories", "consolidate_memories"])
    @pytest.mark.parametrize("memory_type", ["episodic", "procedural"])
    def test_the_result_has_the_full_shape_with_no_conflicts_and_nothing_is_read_or_written(self, memory_type, entry):
        account, chat, service = a_conflict("update")
        before = account.get("f1")

        result = getattr(service, entry)(USER, memory_type=memory_type)

        assert set(result) == RESULT_KEYS and "contradicted" not in result
        assert result["conflicts"] == 0 and sum(result.values()) == 0
        assert account.queries == [] and account.links() == [] and chat.calls == []
        assert account.get("f1") == before
