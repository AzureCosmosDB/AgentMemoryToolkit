"""The procedure pass (instructions, habits and skills, each gated by its evidence), the procedural context built
from what it learns, and the retirement of taught directives into the procedures that hold them."""

from __future__ import annotations

import itertools
import json
import re
from typing import Any
from unittest.mock import MagicMock

import httpx
import openai
import pytest
from azure.cosmos.exceptions import (
    CosmosAccessConditionFailedError,
    CosmosHttpResponseError,
    CosmosResourceExistsError,
    CosmosResourceNotFoundError,
)

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory._utils import cosine_similarity
from azure.cosmos.agent_memory.services import pipeline as pipeline_module
from azure.cosmos.agent_memory.services._procedures import (
    SITUATIONAL_HEADER,
    STANDING_HEADER,
    STANDING_RULES_MAX,
    procedure_id,
)
from azure.cosmos.agent_memory.services.pipeline import PipelineService
from azure.cosmos.agent_memory.store import MemoryStore

PROCEDURAL_PARTITION = ["u1", "__procedural__"]


def procedural_rows(docs: list[dict[str, Any]], sql: str, parameters: Any) -> list[dict[str, Any]]:
    """The rows the procedure pass and the procedural context read, as Cosmos DB would return them."""
    params = {p["name"]: p["value"] for p in (parameters or [])}
    rows = [dict(doc) for doc in docs]
    if "@uid" in params:
        rows = [row for row in rows if row.get("user_id") == params["@uid"]]
    memory_type = params.get("@type", params.get("@memory_type"))
    if memory_type is not None:
        rows = [row for row in rows if row.get("type") == memory_type]
    for name, field in (("@status", "status"), ("@applies", "applies")):
        if name in params:
            rows = [row for row in rows if row.get(field) == params[name]]
    if "@today" in params:
        rows = [row for row in rows if not row.get("valid_until") or row["valid_until"] >= params["@today"]]
    if "superseded_by" in sql:
        rows = [row for row in rows if not row.get("superseded_by")]
    if "@min_salience" in params:
        rows = [row for row in rows if _is_behavioral(row, params["@min_salience"])]
    if "@after_ts" in params:
        after_ts, after_ids = params["@after_ts"], set(params["@after_ids"])
        rows = [
            row
            for row in rows
            if row.get("_ts", 0) > after_ts or (row.get("_ts", 0) == after_ts and row["id"] not in after_ids)
        ]
        rows.sort(key=lambda row: row.get("_ts", 0))
    if "IS_DEFINED(c.lessons)" in sql:
        rows = [row for row in rows if row.get("lessons")]
    if "VectorDistance" in sql:
        rows = [row for row in rows if row.get("embedding")]
        rows.sort(key=lambda row: cosine_similarity(params["@embedding"], row["embedding"]), reverse=True)
    if top := re.match(r"SELECT TOP (\d+) ", sql):
        rows = rows[: int(top.group(1))]
    return rows


def _is_behavioral(row: dict[str, Any], min_salience: float) -> bool:
    return (
        (row.get("metadata") or {}).get("category") in ("preference", "requirement")
        or row.get("salience", 0) >= min_salience
        or row.get("assertion") in ("directive", "correction")
    )


class Memories:
    """The memories container as the procedure pass sees it: a query names the partition it reads (or opts in to
    reading them all), and a conditional write must name the version it read."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = [dict(doc) for doc in docs]
        self._versions = itertools.count(1)

    @staticmethod
    def _partition(doc: dict[str, Any]) -> list[Any]:
        return [doc.get("user_id"), doc.get("thread_id")]

    def query_items(
        self,
        *,
        query: str,
        parameters: Any = None,
        partition_key: Any = None,
        enable_cross_partition_query: bool = False,
    ) -> list[dict[str, Any]]:
        if partition_key is None and not enable_cross_partition_query:
            raise AssertionError("a query over every partition must say so")
        rows = procedural_rows(self.docs, query, parameters)
        if partition_key is not None:
            rows = [row for row in rows if self._partition(row) == list(partition_key)]
        return rows

    def read_item(self, *, item: str, partition_key: Any) -> dict[str, Any]:
        for doc in self.docs:
            if doc.get("id") == item and self._partition(doc) == list(partition_key):
                return dict(doc)
        raise CosmosResourceNotFoundError(status_code=404, message=item)

    def create_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        if any(doc.get("id") == body.get("id") and self._partition(doc) == self._partition(body) for doc in self.docs):
            raise CosmosResourceExistsError(status_code=409, message="conflict")
        created = {**body, "_etag": f"etag-{next(self._versions)}"}
        self.docs.append(created)
        return dict(created)

    def replace_item(self, *, item: str, body: dict[str, Any], etag: Any = None, match_condition: Any = None) -> dict:
        del match_condition
        for index, doc in enumerate(self.docs):
            if doc.get("id") == item:
                if etag is not None and doc.get("_etag") != etag:
                    raise CosmosAccessConditionFailedError(status_code=412, message=f"{item} changed")
                self.docs[index] = {**body, "_etag": f"etag-{next(self._versions)}"}
                return dict(self.docs[index])
        raise CosmosResourceNotFoundError(status_code=404, message=item)


class Chat:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.messages: list[list[dict[str, Any]]] = []

    @property
    def calls(self) -> int:
        return len(self.messages)

    def generate(self, messages: list[dict[str, Any]], **opts: Any) -> str:
        del opts
        self.messages.append(messages)
        response = self.responses.pop(0)
        return response if isinstance(response, str) else json.dumps(response)


class Embeddings:
    """Embeds every text as ``[1.0]`` unless it is given a vector for it."""

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.vectors = vectors or {}

    def generate(self, text: str) -> list[float]:
        self.calls.append([text])
        return self.vectors.get(text, [1.0])


def containers_for(memories: Any) -> dict[ContainerKey, Any]:
    return {ContainerKey.MEMORIES: memories, ContainerKey.TURNS: MagicMock(), ContainerKey.SUMMARIES: MagicMock()}


def _service(
    memories: Memories, responses: list[Any], vectors: dict[str, list[float]] | None = None
) -> tuple[PipelineService, Chat]:
    chat, embeddings, containers = Chat(responses), Embeddings(vectors), containers_for(memories)
    store = MemoryStore(containers=containers, embeddings_client=embeddings)
    return PipelineService(store, chat, embeddings, containers=containers), chat


def prompt_text(chat: Any, call: int) -> str:
    return "\n".join(str(message.get("content")) for message in chat.messages[call])


def procedures_by_name(memories: Memories) -> dict[str, dict[str, Any]]:
    return {doc["name"]: doc for doc in memories.docs if doc.get("type") == "procedural"}


def fact_doc(memories: Memories, fid: str = "fact-raw-1") -> dict[str, Any]:
    return next(doc for doc in memories.docs if doc["id"] == fid)


def cursor_doc(memories: Memories) -> dict[str, Any]:
    return next(doc for doc in memories.docs if doc.get("type") == "procedure_cursor")


def cursor_moved(memories: Memories) -> bool:
    return any(doc.get("type") == "procedure_cursor" for doc in memories.docs)


def deferred(**counts: int) -> dict[str, Any]:
    """The result of a run that stopped early: what it had written before it stopped, which is nothing by default."""
    written = {"created": 0, "updated": 0, "reinforced": 0, "skipped": 0, "facts_retired": 0, **counts}
    return {
        "status": "deferred",
        "procedures_created": written["created"],
        "procedures_updated": written["updated"],
        "procedures_reinforced": written["reinforced"],
        "procedures_skipped": written["skipped"],
        "facts_retired": written["facts_retired"],
    }


def fact(
    fid: str = "fact-raw-1",
    content: str = "The user said to always run targeted tests before reporting success.",
    *,
    ts: int = 1,
    day: str = "2025-01-01",
    agent: bool = False,
    assertion: str | None = None,
    user_id: str = "u1",
) -> dict[str, Any]:
    doc = {
        "id": fid,
        "user_id": user_id,
        "thread_id": "t1",
        "type": "fact",
        "content": content,
        "metadata": {"category": "preference"},
        "salience": 0.9,
        "tags": ["sys:agent-fact"] if agent else [],
        "created_at": f"{day}T00:00:00+00:00",
        "embedding": [1.0],
        "_ts": ts,
        "_etag": f"etag-{fid}",
    }
    if assertion:
        doc["assertion"] = assertion
    return doc


def directive(fid: str = "fact-raw-1", *, assertion: str = "directive", **kwargs: Any) -> dict[str, Any]:
    """A fact extraction recorded as the user's directive (or other ``assertion``)."""
    content = "The user wants the assistant to run targeted tests before reporting success."
    return fact(fid, content, assertion=assertion, **kwargs)


def episode(eid: str = "episode-raw-1", *, ts: int = 2, day: str = "2025-01-01", status: str = "successful") -> dict:
    return {
        "id": eid,
        "user_id": "u1",
        "thread_id": "t1",
        "type": "episodic",
        "title": "Fixed flaky tests",
        "content": "The assistant reset the database fixture per test and the suite passed.",
        "outcome": {"status": status, "description": "The suite passed."},
        "lessons": ["Reset shared database fixtures per test when integration tests are flaky."],
        "created_at": f"{day}T00:00:00+00:00",
        "embedding": [1.0],
        "_ts": ts,
        "_etag": f"etag-{eid}",
    }


STEPS = [
    {
        "sequence": 1,
        "instruction": "Check for shared state.",
        "expected_result": None,
        "on_failure": None,
        "tool_name": None,
    },
    {
        "sequence": 2,
        "instruction": "Reset the fixture per test.",
        "expected_result": None,
        "on_failure": None,
        "tool_name": None,
    },
]


def answer(
    name: str = "Run targeted tests",
    *,
    kind: str = "instruction",
    evidence: str = "taught",
    grounded_in: tuple[str, ...] = ("fact-1",),
    action: str = "add",
    updates: str | None = None,
    summary: str = "Run targeted tests before reporting success.",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "action": action,
        "updates": updates,
        "kind": kind,
        "evidence": evidence,
        "name": name,
        "summary": summary,
        "applies": "when",
        "activation_conditions": ["reporting test results"],
        "preconditions": [],
        "steps": STEPS if kind == "skill" else [],
        "success_conditions": [],
        "failure_conditions": [],
        "safety_constraints": [],
        "retrieval_text": summary,
        "valid_until": None,
        "grounded_in": list(grounded_in),
        "confidence": 0.8,
        **extra,
    }


def procedure(name: str = "Run targeted tests", *, kind: str = "instruction", **extra: Any) -> dict[str, Any]:
    """An existing procedure the pass may show, revise or reinforce."""
    return {
        "id": procedure_id("u1", kind, name),
        "user_id": "u1",
        "type": "procedural",
        "thread_id": "__procedural__",
        "name": name,
        "summary": "Run targeted tests before reporting success.",
        "content": "Run targeted tests before reporting success.",
        "retrieval_text": "targeted tests success",
        "procedure_kind": kind,
        "applies": "when",
        "activation_conditions": ["reporting test results"],
        "steps": STEPS if kind == "skill" else [],
        "status": "active",
        "evidence": ["taught"],
        "source_fact_ids": ["fact-old"],
        "source_episodic_ids": [],
        "support_dates": ["2025-01-01"],
        "user_support_count": 1,
        "practiced_dates": [],
        "version": 1,
        "created_at": "2025-01-01T00:00:00+00:00",
        "embedding": [1.0],
        "_ts": 0,
        "_etag": "etag-existing",
        **extra,
    }


# ---------------------------------------------------------------------------
# Learning procedures
# ---------------------------------------------------------------------------


def test_a_taught_instruction_is_active_with_its_support_recorded() -> None:
    memories = Memories([fact()])
    service, chat = _service(memories, [{"procedures": [answer()]}])

    result = service.synthesize_procedural("u1")

    assert result == {
        "status": "synthesized",
        "procedures_created": 1,
        "procedures_updated": 0,
        "procedures_reinforced": 0,
        "procedures_skipped": 0,
        "facts_retired": 0,
    }
    created = procedures_by_name(memories)["Run targeted tests"]
    assert (created["procedure_kind"], created["status"], created["applies"]) == ("instruction", "active", "when")
    assert (created["evidence"], created["user_support_count"], created["support_dates"]) == (
        ["taught"],
        1,
        ["2025-01-01"],
    )
    assert (created["source_fact_ids"], created["source_episodic_ids"]) == (["fact-raw-1"], [])
    assert (created["source_kind"], created["source_authority"]) == ("explicit_user_instruction", "high")
    assert created["id"] == procedure_id("u1", "instruction", "Run targeted tests")
    assert (created["user_id"], created["thread_id"], created["role"]) == ("u1", "__procedural__", "system")
    assert (created["embedding"], created["utility_score"], created["version"]) == ([1.0], 0.8, 1)
    assert (created["prompt_id"], created["prompt_version"]) == ("extract_procedure.prompty", "v6")
    assert created["tags"] == ["sys:auto-extracted", "sys:procedural"]
    assert "fact-1 [preference · user · 2025-01-01]: The user said to always run targeted tests" in prompt_text(chat, 0)


def test_the_model_is_told_never_to_write_a_secret_into_a_procedure() -> None:
    memories = Memories([fact()])
    service, chat = _service(memories, [{"procedures": []}])

    service.synthesize_procedural("u1")

    assert "Never write a literal secret" in prompt_text(chat, 0)


@pytest.mark.parametrize(
    ("sources", "item", "status"),
    [
        pytest.param([fact(agent=True)], answer(), "candidate", id="instruction-stated-by-the-assistant"),
        pytest.param(
            [fact()], answer("Use the logger", kind="habit", evidence="repeated"), "candidate", id="habit-seen-once"
        ),
        pytest.param(
            [fact()], answer("Use the logger", kind="habit", evidence="corrected"), "active", id="habit-corrected"
        ),
        pytest.param(
            [episode()],
            answer("Stabilize flaky tests", kind="skill", evidence="practiced", grounded_in=("ep-1",)),
            "candidate",
            id="skill-practiced-once",
        ),
        pytest.param(
            [fact()], answer("Stabilize flaky tests", kind="skill", evidence="endorsed"), "active", id="skill-endorsed"
        ),
    ],
)
def test_status_follows_the_evidence_each_kind_needs(sources, item, status) -> None:
    memories = Memories(sources)
    service, _ = _service(memories, [{"procedures": [item]}])

    service.synthesize_procedural("u1")

    [created] = procedures_by_name(memories).values()
    assert created["status"] == status
    if status == "candidate":
        assert (created["source_kind"], created["source_authority"]) == ("agent_inference", "low")


def test_a_skill_practiced_again_on_another_day_becomes_active() -> None:
    skill = procedure(
        "Stabilize flaky tests",
        kind="skill",
        status="candidate",
        evidence=["practiced"],
        source_fact_ids=[],
        source_episodic_ids=["ep-old"],
        user_support_count=0,
        practiced_dates=["2025-01-01"],
    )
    memories = Memories([skill, episode("ep-new", ts=5, day="2025-02-01")])
    reinforce = answer(
        "Stabilize flaky tests",
        kind="skill",
        evidence="practiced",
        grounded_in=("ep-1",),
        action="reinforce",
        updates="proc-1",
    )
    service, chat = _service(memories, [{"procedures": [reinforce]}])

    result = service.synthesize_procedural("u1")

    assert result["procedures_reinforced"] == 1
    assert "proc-1 [skill · candidate · practiced · 1 source]: Stabilize flaky tests:" in prompt_text(chat, 0)
    reinforced = procedures_by_name(memories)["Stabilize flaky tests"]
    assert (reinforced["status"], reinforced["version"]) == ("active", 1)
    assert reinforced["practiced_dates"] == ["2025-01-01", "2025-02-01"]
    assert reinforced["source_episodic_ids"] == ["ep-new", "ep-old"]
    assert (reinforced["source_kind"], reinforced["source_authority"]) == ("episode_distillation", "medium")


def test_an_add_that_names_an_existing_procedure_reinforces_it() -> None:
    habit = procedure("Use the logger", kind="habit", status="candidate", evidence=["repeated"])
    later = fact("fact-new", "The user again asked for logger calls instead of print.", ts=5, day="2025-02-01")
    memories = Memories([habit, later])
    service, _ = _service(memories, [{"procedures": [answer("Use the logger", kind="habit", evidence="repeated")]}])

    result = service.synthesize_procedural("u1")

    assert (result["procedures_created"], result["procedures_reinforced"]) == (0, 1)
    reinforced = procedures_by_name(memories)["Use the logger"]
    assert (reinforced["status"], reinforced["support_dates"]) == ("active", ["2025-01-01", "2025-02-01"])


def test_an_answer_that_cites_only_known_sources_changes_nothing() -> None:
    memories = Memories([procedure(source_fact_ids=["fact-raw-1"]), fact()])
    service, _ = _service(memories, [{"procedures": [answer(action="reinforce", updates="proc-1")]}])

    result = service.synthesize_procedural("u1")

    assert result["procedures_skipped"] == 1
    assert "updated_at" not in procedures_by_name(memories)["Run targeted tests"]


def test_an_update_revises_a_procedure_in_place() -> None:
    existing = procedure()
    memories = Memories([existing, fact("fact-new", "The user wants the test command shown.", ts=5)])
    revision = answer(
        action="update", updates="proc-1", summary="Run targeted tests before reporting success; show the command."
    )
    service, chat = _service(memories, [{"procedures": [revision]}])

    result = service.synthesize_procedural("u1")

    assert result["procedures_updated"] == 1
    assert "proc-1 [instruction · active · taught · 1 source]: Run targeted tests:" in prompt_text(chat, 0)
    revised = procedures_by_name(memories)["Run targeted tests"]
    assert (revised["id"], revised["version"], revised["created_at"]) == (
        existing["id"],
        2,
        existing["created_at"],
    )
    assert revised["summary"] == "Run targeted tests before reporting success; show the command."
    assert (revised["source_fact_ids"], revised["status"]) == (["fact-new", "fact-old"], "active")
    assert (revised["prompt_id"], revised["prompt_version"]) == ("extract_procedure.prompty", "v6")
    assert revised["embedding"] == [1.0]


@pytest.mark.parametrize(
    ("source", "evidence", "revised"),
    [
        pytest.param(
            fact("fact-new", "The user likes brief answers.", ts=5), "taught", False, id="rewrite-loses-the-number"
        ),
        pytest.param(
            fact("fact-new", "The assistant shortened its replies.", ts=5, agent=True),
            "corrected",
            False,
            id="correction-the-user-never-made",
        ),
        pytest.param(
            fact("fact-new", "The user said replies must stay under 100 words, not 200.", ts=5),
            "corrected",
            True,
            id="the-users-correction",
        ),
    ],
)
def test_a_revision_keeps_numbers_and_negations_unless_the_user_corrected_them(source, evidence, revised) -> None:
    existing = procedure("Keep replies short", summary="Keep replies under 200 words.", status="candidate")
    existing["user_support_count"] = 0
    memories = Memories([existing, source])
    revision = answer(
        "Keep replies short",
        evidence=evidence,
        action="update",
        updates="proc-1",
        summary="Keep replies under 100 words.",
    )
    service, _ = _service(memories, [{"procedures": [revision]}])

    result = service.synthesize_procedural("u1")

    kept = procedures_by_name(memories)["Keep replies short"]
    if revised:
        assert result["procedures_updated"] == 1
        assert (kept["version"], kept["summary"], kept["status"]) == (2, "Keep replies under 100 words.", "active")
    else:
        # Nothing is recorded: the evidence was for a rule the procedure would not state.
        assert (result["procedures_updated"], result["procedures_skipped"]) == (0, 1)
        assert (kept["version"], kept["summary"], kept["status"]) == (1, "Keep replies under 200 words.", "candidate")
        assert kept["source_fact_ids"] == ["fact-old"]


def test_a_correction_under_an_existing_name_revises_the_procedure() -> None:
    existing = procedure("Keep replies short", summary="Keep replies under 200 words.")
    memories = Memories([existing, fact("fact-new", "The user said: under 100 words, not 200.", ts=5)])
    correction = answer("Keep replies short", evidence="corrected", summary="Keep replies under 100 words.")
    service, _ = _service(memories, [{"procedures": [correction]}])

    result = service.synthesize_procedural("u1")

    assert (result["procedures_created"], result["procedures_updated"]) == (0, 1)
    assert procedures_by_name(memories)["Keep replies short"]["summary"] == "Keep replies under 100 words."


def test_two_answers_for_the_same_procedure_in_one_batch_both_count() -> None:
    candidate = procedure(status="candidate", evidence=["repeated"])
    candidate["user_support_count"] = 0
    memories = Memories(
        [
            candidate,
            fact("fact-a", "The assistant ran the targeted tests first.", ts=5, agent=True),
            fact("fact-b", "The user said to always run the targeted tests first.", ts=6),
        ]
    )
    service, _ = _service(
        memories,
        [
            {
                "procedures": [
                    answer(evidence="repeated", grounded_in=("fact-1",), action="reinforce", updates="proc-1"),
                    answer(evidence="taught", grounded_in=("fact-2",), action="reinforce", updates="proc-1"),
                ]
            }
        ],
    )

    result = service.synthesize_procedural("u1")

    assert result["procedures_reinforced"] == 2
    reinforced = procedures_by_name(memories)["Run targeted tests"]
    assert (reinforced["source_fact_ids"], reinforced["status"]) == (["fact-a", "fact-b", "fact-old"], "active")


def test_a_procedure_changed_since_it_was_read_keeps_the_other_writers_version(monkeypatch) -> None:
    memories = Memories([procedure(), fact("fact-new", "The user wants the test command shown.", ts=5)])
    service, _ = _service(
        memories, [{"procedures": [answer(action="update", updates="proc-1", summary="Run tests; show the command.")]}]
    )
    read = memories.read_item

    def another_writer_wins(*, item: str, partition_key: Any) -> dict[str, Any]:
        doc = read(item=item, partition_key=partition_key)
        if item.startswith("proc_"):
            memories.docs = [{**d, "_etag": "etag-other"} if d["id"] == item else d for d in memories.docs]
        return doc

    monkeypatch.setattr(memories, "read_item", another_writer_wins)

    result = service.synthesize_procedural("u1")

    assert (result["procedures_updated"], result["procedures_skipped"]) == (0, 1)
    kept = procedures_by_name(memories)["Run targeted tests"]
    assert (kept["version"], kept["_etag"], kept["summary"]) == (1, "etag-other", procedure()["summary"])


def test_an_update_naming_no_shown_procedure_is_an_add() -> None:
    memories = Memories([fact()])
    service, _ = _service(memories, [{"procedures": [answer(action="update", updates="proc-9")]}])

    assert service.synthesize_procedural("u1")["procedures_created"] == 1


@pytest.mark.parametrize(
    "unusable",
    [
        pytest.param(answer(kind="policy"), id="unknown-kind"),
        pytest.param(answer(kind="workflow"), id="retired-kind"),
        pytest.param(answer(evidence="inferred"), id="unknown-evidence"),
        pytest.param(answer(grounded_in=("fact-9",)), id="ungrounded"),
        pytest.param(
            answer("Flaky tests", kind="skill", evidence="endorsed", steps=STEPS[:1]), id="skill-with-one-step"
        ),
        pytest.param("not an object", id="not-an-answer"),
    ],
)
def test_unusable_answers_are_skipped_beside_a_valid_one(unusable) -> None:
    memories = Memories([fact()])
    service, _ = _service(memories, [{"procedures": [unusable, answer("Valid instruction")]}])

    result = service.synthesize_procedural("u1")

    assert (result["procedures_created"], result["procedures_skipped"]) == (1, 1)
    assert list(procedures_by_name(memories)) == ["Valid instruction"]


def test_a_procedure_that_already_exists_is_never_created_twice() -> None:
    memories = Memories([fact()])
    service, _ = _service(memories, [{"procedures": [answer()]}])
    original = service._create_memory

    def lose_the_race(doc: dict[str, Any]) -> dict[str, Any]:
        original(doc)
        raise CosmosResourceExistsError(status_code=409, message="conflict")

    service._create_memory = lose_the_race  # type: ignore[method-assign]

    result = service.synthesize_procedural("u1")

    assert (result["procedures_created"], result["procedures_skipped"]) == (0, 1)
    assert len(procedures_by_name(memories)) == 1


# ---------------------------------------------------------------------------
# What the model reads
# ---------------------------------------------------------------------------


def test_sources_reach_the_model_oldest_first_in_batches_with_speaker_date_and_outcome() -> None:
    facts = [fact(f"fact-{i}", f"The user stated rule {i}.", ts=i, day=f"2025-01-{i + 10:02d}") for i in range(11)]
    facts[0]["tags"] = ["sys:agent-fact"]
    memories = Memories([*facts, episode(ts=20, day="2025-02-01")])
    service, chat = _service(memories, [{"procedures": []}, {"procedures": []}])

    service.synthesize_procedural("u1")

    assert chat.calls == 2
    first, second = prompt_text(chat, 0), prompt_text(chat, 1)
    assert "fact-1 [preference · assistant · 2025-01-10]: The user stated rule 0." in first
    assert "rule 9." in first and "rule 10." not in first
    assert "fact-1 [preference · user · 2025-01-20]: The user stated rule 10." in second
    assert "ep-1 [2025-02-01] Fixed flaky tests (outcome: successful - The suite passed.). Summary:" in second


def test_only_behavioral_facts_and_the_users_directives_are_read() -> None:
    plain = {
        **fact("fact-plain", "The user lives in Lisbon.", ts=1),
        "metadata": {"category": "other"},
        "salience": 0.2,
    }
    salient = {**plain, "id": "fact-salient", "content": "The user is allergic to penicillin.", "salience": 0.95}
    corrected = {**plain, "id": "fact-corrected", "content": "The user said to stop apologizing.", "_ts": 3}
    corrected["assertion"] = "correction"
    memories = Memories(
        [
            plain,
            salient,
            corrected,
            fact("fact-retired", "The user wanted tabs.", ts=4),
            fact("fact-other-user", "The user of another account.", ts=5, user_id="u2"),
        ]
    )
    memories.docs[3]["superseded_by"] = "fact-newer"
    service, chat = _service(memories, [{"procedures": []}])

    service.synthesize_procedural("u1")

    text = prompt_text(chat, 0)
    assert "allergic to penicillin" in text and "stop apologizing" in text
    assert "Lisbon" not in text and "wanted tabs" not in text and "another account" not in text
    assert "fact-2 [other · user · correction · 2025-01-01]: The user said to stop apologizing." in text


def test_the_users_nearest_existing_procedures_are_shown_so_the_model_can_revise_one() -> None:
    near = [procedure(f"Near {i}", embedding=[1.0, 0.1 * i], summary=f"Near rule {i}.", _ts=0) for i in range(1, 4)]
    far = [procedure(f"Far {i}", embedding=[0.0, 1.0], summary=f"Far rule {i}.") for i in range(1, 3)]
    retired = procedure("Retired", embedding=[1.0, 0.0], superseded_by="proc-newer")
    memories = Memories([*near, *far, retired, {**fact(), "embedding": [1.0, 0.0]}])
    service, chat = _service(memories, [{"procedures": []}])

    service.synthesize_procedural("u1")

    lines = [line for line in prompt_text(chat, 0).splitlines() if line.startswith("proc-")]
    assert [line.split("]: ")[1].split(":")[0] for line in lines] == ["Near 1", "Near 2", "Near 3"]
    assert "Far 1" not in prompt_text(chat, 0) and "Retired" not in prompt_text(chat, 0)
    assert "(none)" not in prompt_text(chat, 0).split("## Existing procedures")[1]


def test_procedures_and_the_cursor_live_in_the_users_procedural_partition() -> None:
    memories = Memories([fact(), fact("fact-theirs", "Their rule.", user_id="u2")])
    service, _ = _service(memories, [{"procedures": [answer()]}])

    service.synthesize_procedural("u1")

    created = procedures_by_name(memories)["Run targeted tests"]
    assert memories.read_item(item=created["id"], partition_key=PROCEDURAL_PARTITION)["name"] == created["name"]
    cursor = memories.read_item(item="procedure_cursor", partition_key=PROCEDURAL_PARTITION)
    assert (cursor["type"], cursor["user_id"], cursor["thread_id"]) == ("procedure_cursor", "u1", "__procedural__")
    assert not any(doc["user_id"] == "u2" and doc.get("type") != "fact" for doc in memories.docs)


# ---------------------------------------------------------------------------
# The cursor
# ---------------------------------------------------------------------------


def test_synthesize_procedural_reads_each_source_once() -> None:
    memories = Memories([fact("fact-a", "The user wants tests run before every commit.", ts=3)])
    service, chat = _service(memories, [{"procedures": [answer("Test before commit")]}, {"procedures": []}])

    first = service.synthesize_procedural("u1")
    second = service.synthesize_procedural("u1")
    memories.docs.append(fact("fact-b", "The user wants commit messages in the imperative mood.", ts=7))
    third = service.synthesize_procedural("u1")

    assert first["procedures_created"] == 1
    assert second == {"status": "unchanged", "procedures_created": 0}
    assert third["status"] == "synthesized"
    assert chat.calls == 2
    assert "imperative mood" in prompt_text(chat, 1)
    assert "tests run before every commit" not in prompt_text(chat, 1)
    assert cursor_doc(memories)["facts"] == {"last_ts": 7, "ids_at_last_ts": ["fact-b"]}


def test_memories_written_in_the_same_second_are_not_read_twice() -> None:
    memories = Memories([fact("fact-a", "The user wants tests run first.", ts=3)])
    service, chat = _service(memories, [{"procedures": []}, {"procedures": []}])

    service.synthesize_procedural("u1")
    memories.docs.append(fact("fact-b", "The user wants commit messages short.", ts=3))
    service.synthesize_procedural("u1")

    assert "commit messages short" in prompt_text(chat, 1)
    assert "tests run first" not in prompt_text(chat, 1)
    assert cursor_doc(memories)["facts"] == {"last_ts": 3, "ids_at_last_ts": ["fact-a", "fact-b"]}


@pytest.mark.parametrize(
    ("error", "expected", "moved"),
    [("rate limit", deferred(), False), ("content_filter", {"status": "skipped", "procedures_created": 0}, True)],
    ids=["retryable-keeps-the-sources", "non-retryable-skips-them"],
)
def test_synthesize_procedural_moves_its_cursor_only_past_sources_it_is_done_with(error, expected, moved) -> None:
    memories = Memories([fact()])
    service, _ = _service(memories, [])

    def fail(filename: str, inputs: dict[str, Any]) -> str:
        del filename, inputs
        raise RuntimeError(error)

    service._run_prompty = fail  # type: ignore[method-assign]

    assert service.synthesize_procedural("u1") == expected
    assert cursor_moved(memories) is moved


def test_synthesize_procedural_skips_an_unusable_answer_and_moves_on() -> None:
    memories = Memories([fact()])
    service, _ = _service(memories, ["not json"])

    assert service.synthesize_procedural("u1") == {"status": "skipped", "procedures_created": 0}
    assert any(doc.get("type") == "procedure_cursor" for doc in memories.docs)


@pytest.mark.parametrize(
    ("failure", "status", "moved"),
    [
        (CosmosHttpResponseError(status_code=429, message="throttled"), "deferred", False),
        (CosmosHttpResponseError(status_code=503, message="unavailable"), "deferred", False),
        (CosmosHttpResponseError(status_code=400, message="bad request"), "synthesized", True),
    ],
    ids=["throttled", "unavailable", "bad-request"],
)
def test_a_failed_procedure_write_moves_the_cursor_only_when_a_retry_could_succeed(failure, status, moved) -> None:
    memories = Memories([fact()])
    service, _ = _service(memories, [{"procedures": [answer()]}])
    service._embedded_procedure = MagicMock(side_effect=failure)  # type: ignore[method-assign]

    result = service.synthesize_procedural("u1")

    assert result["status"] == status
    assert procedures_by_name(memories) == {}
    assert cursor_moved(memories) is moved


# The failures above are injected into the pipeline; these come from the clients it calls.
_EMBEDDINGS_REQUEST = httpx.Request("POST", "https://example.invalid/embeddings")


@pytest.mark.parametrize(
    "failure",
    [
        openai.RateLimitError("rate limited", response=httpx.Response(429, request=_EMBEDDINGS_REQUEST), body=None),
        openai.APITimeoutError(request=_EMBEDDINGS_REQUEST),
    ],
    ids=["embeddings-throttled", "embeddings-timeout"],
)
def test_an_embeddings_failure_defers_the_batch_and_a_later_run_writes_the_procedure(failure) -> None:
    memories = Memories([fact()])
    service, _ = _service(memories, [{"procedures": [answer()]}, {"procedures": [answer()]}])
    embed = service._embeddings.generate
    calls: list[str] = []

    def fails_once(text: str) -> list[float]:
        calls.append(text)
        if len(calls) == 1:
            raise failure
        return embed(text)

    service._embeddings.generate = fails_once  # type: ignore[method-assign]

    assert service.synthesize_procedural("u1")["status"] == "deferred"
    assert procedures_by_name(memories) == {} and not cursor_moved(memories)

    second = service.synthesize_procedural("u1")

    assert (second["status"], second["procedures_created"]) == ("synthesized", 1)
    assert "Run targeted tests" in procedures_by_name(memories) and cursor_moved(memories)


@pytest.mark.parametrize(
    ("status", "outcome", "moved"),
    [(408, "deferred", False), (429, "deferred", False), (503, "deferred", False), (400, "synthesized", True)],
    ids=["timeout", "throttled", "unavailable", "bad-request"],
)
def test_a_create_that_fails_in_cosmos_moves_the_cursor_only_when_a_retry_could_succeed(status, outcome, moved) -> None:
    memories = Memories([fact()])
    service, _ = _service(memories, [{"procedures": [answer()]}])
    create = memories.create_item

    def failing(*, body: dict[str, Any]) -> dict[str, Any]:
        if body.get("type") == "procedural":
            raise CosmosHttpResponseError(status_code=status, message="failed")
        return create(body=body)

    memories.create_item = failing  # type: ignore[method-assign]

    result = service.synthesize_procedural("u1")

    assert result["status"] == outcome
    assert procedures_by_name(memories) == {}
    assert cursor_moved(memories) is moved


def test_a_replace_that_fails_in_cosmos_defers_the_batch_that_reinforces_a_procedure() -> None:
    habit = procedure("Use the logger", kind="habit", status="candidate", evidence=["repeated"])
    later = fact("fact-new", "The user again asked for logger calls instead of print.", ts=5, day="2025-02-01")
    memories = Memories([habit, later])
    service, _ = _service(memories, [{"procedures": [answer("Use the logger", kind="habit", evidence="repeated")]}])
    replace = memories.replace_item

    def failing(**kwargs: Any) -> dict[str, Any]:
        if kwargs["body"].get("type") == "procedural":
            raise CosmosHttpResponseError(status_code=503, message="unavailable")
        return replace(**kwargs)

    memories.replace_item = failing  # type: ignore[method-assign]

    assert service.synthesize_procedural("u1")["status"] == "deferred"
    assert not cursor_moved(memories)
    assert procedures_by_name(memories)["Use the logger"]["source_fact_ids"] == ["fact-old"]


def _second_source_and_answer() -> tuple[dict[str, Any], dict[str, Any]]:
    return fact("fact-next", "The user wants commit messages short.", ts=2), answer(
        "Short commit messages", summary="Keep commit messages short.", grounded_in=("fact-2",)
    )


def _fail_creating(memories: Memories, name: str) -> dict[str, bool]:
    """Make creating the procedure ``name`` fail with a throttle until ``state["fail"]`` is cleared."""
    state = {"fail": True}
    create = memories.create_item

    def throttled(*, body: dict[str, Any]) -> dict[str, Any]:
        if state["fail"] and body.get("name") == name:
            raise CosmosHttpResponseError(status_code=429, message="throttled")
        return create(body=body)

    memories.create_item = throttled  # type: ignore[method-assign]
    return state


def test_a_batch_stopped_by_a_transient_failure_reports_the_answers_it_had_already_written() -> None:
    source, second = _second_source_and_answer()
    memories = Memories([directive(), source])
    # The first answer is a taught directive: its procedure is created and the directive is retired into it.
    service, chat = _service(memories, [{"procedures": [answer(), second]}])
    state = _fail_creating(memories, "Short commit messages")

    result = service.synthesize_procedural("u1")

    assert result == deferred(created=1, facts_retired=1)
    assert set(procedures_by_name(memories)) == {"Run targeted tests"}
    assert fact_doc(memories)["supersede_reason"] == "procedure"
    assert not cursor_moved(memories), "the batch is tried again"

    # The directive is retired, so the run that tries the batch again shows only the other fact.
    state["fail"] = False
    chat.responses.append({"procedures": [{**second, "grounded_in": ["fact-1"]}]})
    rerun = service.synthesize_procedural("u1")

    assert (rerun["status"], rerun["procedures_created"], rerun["facts_retired"]) == ("synthesized", 1, 0)
    assert set(procedures_by_name(memories)) == {"Run targeted tests", "Short commit messages"}
    assert cursor_moved(memories)


def test_the_run_that_tries_a_deferred_batch_again_skips_the_answer_it_already_wrote() -> None:
    source, second = _second_source_and_answer()
    memories = Memories([fact(), source])
    service, _ = _service(memories, [{"procedures": [answer(), second]}, {"procedures": [answer(), second]}])
    state = _fail_creating(memories, "Short commit messages")

    assert service.synthesize_procedural("u1") == deferred(created=1)

    state["fail"] = False
    rerun = service.synthesize_procedural("u1")

    # Its procedure's ID follows from its kind and name, so the same answer now finds it, and cites nothing new.
    assert (rerun["status"], rerun["procedures_created"], rerun["procedures_skipped"]) == ("synthesized", 1, 1)
    assert len(procedures_by_name(memories)) == 2 and cursor_moved(memories)


def test_a_revision_written_before_a_batch_was_stopped_is_counted() -> None:
    source, second = _second_source_and_answer()
    shown = fact("fact-new", "The user wants the test command shown.", ts=1)
    memories = Memories([procedure(), shown, source])
    revision = answer(
        action="update", updates="proc-1", summary="Run targeted tests before reporting success; show the command."
    )
    service, _ = _service(memories, [{"procedures": [revision, second]}])
    _fail_creating(memories, "Short commit messages")

    assert service.synthesize_procedural("u1") == deferred(updated=1)
    assert procedures_by_name(memories)["Run targeted tests"]["version"] == 2
    assert not cursor_moved(memories)


def test_a_run_stopped_in_a_later_batch_reports_what_its_earlier_batches_wrote() -> None:
    first_directive = directive("fact-00", ts=1)
    facts = [first_directive, *(fact(f"fact-{i:02d}", f"The user stated rule {i}.", ts=i + 1) for i in range(1, 12))]
    memories = Memories(facts)
    service, _ = _service(memories, [])
    replies = iter([json.dumps({"procedures": [answer()]})])

    def first_batch_only(filename: str, inputs: dict[str, Any]) -> str:
        del filename, inputs
        reply = next(replies, None)
        if reply is None:
            raise RuntimeError("rate limit")
        return reply

    service._run_prompty = first_batch_only  # type: ignore[method-assign]

    assert service.synthesize_procedural("u1") == deferred(created=1, facts_retired=1)
    assert cursor_doc(memories)["facts"]["last_ts"] == 10


def _twelve_facts() -> list[dict[str, Any]]:
    return [fact(f"fact-{i:02d}", f"The user stated rule {i}.", ts=i + 1) for i in range(12)]


def test_a_stopped_run_keeps_the_batches_it_finished() -> None:
    memories = Memories(_twelve_facts())
    service, _ = _service(memories, [])
    replies = iter([json.dumps({"procedures": []})])

    def first_batch_only(filename: str, inputs: dict[str, Any]) -> str:
        del filename, inputs
        reply = next(replies, None)
        if reply is None:
            raise RuntimeError("rate limit")
        return reply

    service._run_prompty = first_batch_only  # type: ignore[method-assign]

    assert service.synthesize_procedural("u1") == deferred()
    assert cursor_doc(memories)["facts"] == {"last_ts": 10, "ids_at_last_ts": ["fact-09"]}

    shown: list[str] = []
    service._run_prompty = lambda filename, inputs: shown.append(inputs["behavioral_facts"]) or '{"procedures": []}'  # type: ignore[method-assign]
    service.synthesize_procedural("u1")

    assert len(shown) == 1
    assert "rule 10." in shown[0] and "rule 11." in shown[0] and "rule 9." not in shown[0]


def test_a_deadline_stops_the_pass_between_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    memories = Memories(_twelve_facts())
    service, _ = _service(memories, [])
    clock = {"now": 0.0}
    monkeypatch.setattr(pipeline_module.time, "monotonic", lambda: clock["now"])

    def slow_batch(filename: str, inputs: dict[str, Any]) -> str:
        del filename, inputs
        clock["now"] = 100.0
        return json.dumps({"procedures": []})

    service._run_prompty = slow_batch  # type: ignore[method-assign]

    result = service.synthesize_procedural("u1", deadline=50.0)

    assert (result["status"], result["procedures_created"]) == ("synthesized", 0)
    assert cursor_doc(memories)["facts"]["last_ts"] == 10
    assert service.synthesize_procedural("u1", deadline=50.0) == deferred()


@pytest.mark.parametrize(
    "docs",
    [
        pytest.param([], id="no-memories"),
        pytest.param([{**fact(), "content": "   "}, {**episode(), "lessons": ["", "   "]}], id="only-blank-sources"),
    ],
)
def test_synthesize_procedural_is_unchanged_without_a_model_call_when_nothing_qualifies(docs) -> None:
    memories = Memories(docs)
    service, chat = _service(memories, [])

    assert service.synthesize_procedural("u1") == {"status": "unchanged", "procedures_created": 0}
    assert chat.calls == 0
    assert procedures_by_name(memories) == {}


def test_synthesize_procedural_requires_a_user() -> None:
    service, _ = _service(Memories([]), [])

    with pytest.raises(Exception, match="user_id is required"):
        service.synthesize_procedural("")


# ---------------------------------------------------------------------------
# Procedural context
# ---------------------------------------------------------------------------


def rule(name: str, *, applies: str = "always", evidence: tuple[str, ...] = ("taught",), **extra: Any) -> dict:
    return {
        "id": f"proc-{name}",
        "user_id": "u1",
        "type": "procedural",
        "thread_id": "__procedural__",
        "name": name,
        "summary": f"{name} rule.",
        "procedure_kind": "instruction",
        "applies": applies,
        "status": "active",
        "evidence": list(evidence),
        "activation_conditions": [] if applies == "always" else [f"working on {name}"],
        "updated_at": "2025-01-01T00:00:00+00:00",
        "embedding": [1.0],
        **extra,
    }


def test_build_procedural_context_lists_the_standing_rules_in_force() -> None:
    memories = Memories(
        [
            rule("Repeated", evidence=("repeated",)),
            rule("Taught-old", updated_at="2024-01-01T00:00:00+00:00"),
            rule("Taught-new", updated_at="2025-06-01T00:00:00+00:00"),
            rule("Corrected", evidence=("corrected",)),
            rule("Situational", applies="when"),
            rule("Candidate", status="candidate"),
            rule("Expired", valid_until="2000-01-01"),
            rule("Retired", superseded_by="proc-Taught-new"),
            {**rule("Theirs"), "user_id": "u2"},
        ]
    )
    service, _ = _service(memories, [])

    assert service.build_procedural_context("u1").splitlines() == [
        STANDING_HEADER,
        "- Taught-new rule.",
        "- Taught-old rule.",
        "- Corrected rule.",
        "- Repeated rule.",
    ]
    assert service.build_procedural_context("missing-user") == ""


def test_build_procedural_context_leaves_out_rules_written_before_procedures_were_learned_from_evidence() -> None:
    legacy = {
        key: value for key, value in rule("Legacy", procedure_kind="behavioral_policy").items() if key != "applies"
    }
    memories = Memories([legacy, rule("Current")])
    service, _ = _service(memories, [])

    assert service.build_procedural_context("u1").splitlines() == [STANDING_HEADER, "- Current rule."]


def test_build_procedural_context_shows_a_rule_through_its_last_valid_day(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_module, "utc_today", lambda: "2025-06-01")
    memories = Memories(
        [
            rule("Through-today", valid_until="2025-06-01"),
            rule("Ended-yesterday", valid_until="2025-05-31"),
            rule("No-end", valid_until=None),
        ]
    )
    service, _ = _service(memories, [])

    assert service.build_procedural_context("u1").splitlines() == [
        STANDING_HEADER,
        "- Through-today rule.",
        "- No-end rule.",
    ]


def test_build_procedural_context_caps_the_standing_rules() -> None:
    memories = Memories([rule(f"Rule-{i:02d}") for i in range(STANDING_RULES_MAX + 2)])
    service, _ = _service(memories, [])

    assert len(service.build_procedural_context("u1").splitlines()) == 1 + STANDING_RULES_MAX


def test_build_procedural_context_adds_the_situational_rule_nearest_a_task() -> None:
    memories = Memories(
        [
            rule("Always"),
            rule("Reviews", applies="when", embedding=[0.0, 1.0]),
            rule("Cooking", applies="when", embedding=[1.0, 0.0]),
            rule("Draft", applies="when", status="candidate", embedding=[0.0, 1.0]),
            rule("Expired", applies="when", embedding=[0.0, 1.0], valid_until="2000-01-01"),
        ]
    )
    service, _ = _service(memories, [], vectors={"review this pull request": [0.0, 1.0]})

    context = service.build_procedural_context("u1", task="review this pull request")

    assert context.splitlines() == [
        STANDING_HEADER,
        "- Always rule.",
        "",
        SITUATIONAL_HEADER,
        "- When working on Reviews: Reviews rule.",
    ]
    assert service._embeddings.calls == [["review this pull request"]]
    assert service.build_procedural_context("u1").splitlines() == [STANDING_HEADER, "- Always rule."]
    assert service._embeddings.calls == [["review this pull request"]]


def test_build_procedural_context_is_empty_when_a_task_finds_nothing_and_there_are_no_standing_rules() -> None:
    service, _ = _service(Memories([rule("Draft", applies="when", status="candidate")]), [])

    assert service.build_procedural_context("u1", task="anything") == ""


def test_build_procedural_context_keeps_the_standing_rules_when_the_situational_lookup_fails(caplog) -> None:
    service, _ = _service(Memories([rule("Always"), rule("Reviews", applies="when", embedding=[0.0, 1.0])]), [])
    service._embeddings.generate = MagicMock(side_effect=RuntimeError("quoted-model-text"))  # type: ignore[method-assign]

    context = service.build_procedural_context("u1", task="review this pull request")

    assert context.splitlines() == [STANDING_HEADER, "- Always rule."]
    assert "RuntimeError" in caplog.text and "quoted-model-text" not in caplog.text


def test_build_procedural_context_never_lists_the_steps_of_a_skill() -> None:
    skill = rule("Flaky", applies="when", procedure_kind="skill", steps=STEPS)
    service, _ = _service(Memories([skill]), [])

    context = service.build_procedural_context("u1", task="flaky tests")

    assert context.splitlines()[-1] == "- When working on Flaky: Flaky rule."
    assert "Check for shared state" not in context


# ---------------------------------------------------------------------------
# Directives and corrections
# ---------------------------------------------------------------------------


def test_a_directive_is_held_by_the_active_situational_procedure_it_teaches() -> None:
    memories = Memories([directive()])
    service, chat = _service(memories, [{"procedures": [answer()]}])

    result = service.synthesize_procedural("u1")

    created = procedures_by_name(memories)["Run targeted tests"]
    assert (created["status"], created["evidence"]) == ("active", ["taught"])
    retired = fact_doc(memories)
    assert (retired["superseded_by"], retired["supersede_reason"]) == (created["id"], "procedure")
    assert retired["superseded_at"]
    assert result["facts_retired"] == 1
    assert "fact-1 [preference · user · directive · 2025-01-01]" in prompt_text(chat, 0)


def test_a_standing_rule_leaves_its_directive_searchable() -> None:
    """Search shows only situational procedures; a standing rule reaches the agent another way."""
    memories = Memories([directive()])
    service, _ = _service(memories, [{"procedures": [answer(applies="always", activation_conditions=[])]}])

    result = service.synthesize_procedural("u1")

    assert procedures_by_name(memories)["Run targeted tests"]["status"] == "active"
    assert not fact_doc(memories).get("superseded_by")
    assert result["facts_retired"] == 0


def test_a_stated_preference_taken_for_an_instruction_stays_a_hidden_candidate() -> None:
    memories = Memories([directive(assertion="stated")])
    service, _ = _service(memories, [{"procedures": [answer()]}])

    result = service.synthesize_procedural("u1")

    created = procedures_by_name(memories)["Run targeted tests"]
    assert (created["status"], created["evidence"], created["user_support_count"]) == ("candidate", [], 1)
    assert not fact_doc(memories).get("superseded_by")
    assert result["facts_retired"] == 0


def test_a_fact_without_an_assertion_teaches_but_is_never_retired() -> None:
    """Facts extracted before assertions were recorded still count as teaching, so they stay searchable."""
    memories = Memories([fact()])
    service, _ = _service(memories, [{"procedures": [answer()]}])

    result = service.synthesize_procedural("u1")

    assert procedures_by_name(memories)["Run targeted tests"]["status"] == "active"
    assert not fact_doc(memories).get("superseded_by")
    assert result["facts_retired"] == 0


def test_a_directive_from_the_assistant_is_never_retired() -> None:
    memories = Memories([directive(agent=True)])
    service, _ = _service(memories, [{"procedures": [answer()]}])

    result = service.synthesize_procedural("u1")

    assert not fact_doc(memories).get("superseded_by")
    assert result["facts_retired"] == 0


def test_a_directive_supporting_two_procedures_is_retired_once() -> None:
    memories = Memories([directive()])
    second = answer("Report targeted test results", summary="Report which targeted tests ran.")
    service, _ = _service(memories, [{"procedures": [answer(), second]}])
    retirements: list[str] = []
    store_retire = service._store.mark_superseded

    def count(old_doc: dict[str, Any], superseder_id: str, *, reason: str) -> bool:
        retirements.append(old_doc["id"])
        return store_retire(old_doc, superseder_id, reason=reason)

    service._store.mark_superseded = count  # type: ignore[method-assign]

    result = service.synthesize_procedural("u1")

    assert result["procedures_created"] == 2
    assert result["facts_retired"] == 1
    assert retirements == ["fact-raw-1"]
    assert fact_doc(memories)["superseded_by"] == procedures_by_name(memories)["Run targeted tests"]["id"]


def test_a_directive_changed_since_it_was_read_stays_beside_its_procedure(monkeypatch: pytest.MonkeyPatch) -> None:
    from azure.core import MatchConditions

    memories = Memories([directive()])
    service, _ = _service(memories, [{"procedures": [answer()]}])
    replace = memories.replace_item
    conditions: list[tuple[Any, Any]] = []

    def lose_the_race_on_facts(**kwargs: Any) -> Any:
        if kwargs["item"] == "fact-raw-1":
            conditions.append((kwargs.get("etag"), kwargs.get("match_condition")))
            raise CosmosAccessConditionFailedError(status_code=412, message="precondition failed")
        return replace(**kwargs)

    monkeypatch.setattr(memories, "replace_item", lose_the_race_on_facts)

    result = service.synthesize_procedural("u1")

    assert conditions == [("etag-fact-raw-1", MatchConditions.IfNotModified)]
    assert procedures_by_name(memories)["Run targeted tests"]["status"] == "active"
    assert not fact_doc(memories).get("superseded_by")
    assert result["facts_retired"] == 0


def test_a_directive_that_cannot_be_retired_does_not_stop_the_run() -> None:
    memories = Memories([directive(), fact("fact-next", "The user wants commit messages short.", ts=7)])
    service, _ = _service(memories, [{"procedures": [answer()]}])

    def broken(old_doc: dict[str, Any], superseder_id: str, *, reason: str) -> bool:
        raise RuntimeError("store unavailable")

    service._store.mark_superseded = broken  # type: ignore[method-assign]

    result = service.synthesize_procedural("u1")

    assert (result["status"], result["procedures_created"], result["facts_retired"]) == ("synthesized", 1, 0)
    assert cursor_doc(memories)["facts"]["last_ts"] == 7


def test_a_correction_retires_into_the_habit_it_corrects() -> None:
    memories = Memories([directive("fact-fix", assertion="correction")])
    item = answer("Use the logger", kind="habit", evidence="corrected", grounded_in=("fact-1",))
    service, _ = _service(memories, [{"procedures": [item]}])

    result = service.synthesize_procedural("u1")

    retired = fact_doc(memories, "fact-fix")
    assert (retired["supersede_reason"], result["facts_retired"]) == ("procedure", 1)
