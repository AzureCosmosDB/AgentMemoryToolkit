from __future__ import annotations

import json
from typing import Any

import pytest

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService, _AsyncStoreContainerAdapter
from azure.cosmos.agent_memory.services.pipeline import PipelineService, _StoreContainerAdapter


class _SyncChat:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, messages: list[dict[str, Any]], **opts: Any) -> str:
        del messages, opts
        self.calls += 1
        return json.dumps(self.responses.pop(0))


class _SyncEmbeddings:
    def __init__(self):
        self.calls: list[list[str]] = []

    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0] for _ in texts]

    def generate(self, text: str) -> list[float]:
        self.calls.append([text])
        return [1.0]


class _AsyncChat:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls = 0

    async def generate(self, messages: list[dict[str, Any]], **opts: Any) -> str:
        del messages, opts
        self.calls += 1
        return json.dumps(self.responses.pop(0))


class _AsyncEmbeddings(_SyncEmbeddings):
    async def generate_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0] for _ in texts]

    async def generate(self, text: str) -> list[float]:
        self.calls.append([text])
        return [1.0]


class _Store:
    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = [dict(doc) for doc in docs]

    def query(self, sql: str, parameters=None, partition_key=None, cross_partition: bool = False):
        del partition_key, cross_partition
        params = {p["name"]: p["value"] for p in (parameters or [])}
        docs = [dict(doc) for doc in self.docs]
        if "@user_id" in params:
            docs = [doc for doc in docs if doc.get("user_id") == params["@user_id"]]
        if "@thread_id" in params:
            docs = [doc for doc in docs if doc.get("thread_id") == params["@thread_id"]]
        if "c.type IN" in sql:
            types = {value for name, value in params.items() if name.startswith("@mtype")}
            docs = [doc for doc in docs if doc.get("type") in types]
        if "superseded_by" in sql:
            docs = [doc for doc in docs if not doc.get("superseded_by")]
        if "extracted_at" in sql:
            docs = [doc for doc in docs if not doc.get("extracted_at")]
        return docs

    def upsert_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        body = dict(body)
        doc_id = body.get("id")
        for i, doc in enumerate(self.docs):
            if doc.get("id") == doc_id:
                self.docs[i] = body
                return body
        self.docs.append(body)
        return body

    def read_item(self, item_id: str, partition_key: Any):
        del partition_key
        for doc in self.docs:
            if doc.get("id") == item_id:
                return dict(doc)
        raise KeyError(item_id)

    def add_cosmos(self, record: dict[str, Any]) -> dict[str, Any]:
        self.docs.append(dict(record))
        return record

    def mark_superseded(self, old_doc: dict[str, Any], superseder_id: str, *, reason: str) -> bool:
        del old_doc, superseder_id, reason
        return True


class _AsyncStore(_Store):
    async def query(self, sql: str, parameters=None, partition_key=None, cross_partition: bool = False):
        return super().query(sql, parameters=parameters, partition_key=partition_key, cross_partition=cross_partition)

    async def read_item(self, item_id: str, partition_key: Any):
        return super().read_item(item_id, partition_key)

    async def add_cosmos(self, record: dict[str, Any]) -> dict[str, Any]:
        return super().add_cosmos(record)

    async def mark_superseded(self, old_doc: dict[str, Any], superseder_id: str, *, reason: str) -> bool:
        return super().mark_superseded(old_doc, superseder_id, reason=reason)


def _containers_for_store(
    memories_store: _Store,
    *,
    turns_store: _Store | None = None,
    summaries_store: _Store | None = None,
) -> dict[ContainerKey, _StoreContainerAdapter]:
    turns_store = turns_store or _Store([])
    summaries_store = summaries_store or _Store([])
    return {
        ContainerKey.TURNS: _StoreContainerAdapter(turns_store, ContainerKey.TURNS),
        ContainerKey.MEMORIES: _StoreContainerAdapter(memories_store, ContainerKey.MEMORIES),
        ContainerKey.SUMMARIES: _StoreContainerAdapter(summaries_store, ContainerKey.SUMMARIES),
    }


def _async_containers_for_store(
    memories_store: _AsyncStore,
    *,
    turns_store: _AsyncStore | None = None,
    summaries_store: _AsyncStore | None = None,
) -> dict[ContainerKey, _AsyncStoreContainerAdapter]:
    turns_store = turns_store or _AsyncStore([])
    summaries_store = summaries_store or _AsyncStore([])
    return {
        ContainerKey.TURNS: _AsyncStoreContainerAdapter(turns_store, ContainerKey.TURNS),
        ContainerKey.MEMORIES: _AsyncStoreContainerAdapter(memories_store, ContainerKey.MEMORIES),
        ContainerKey.SUMMARIES: _AsyncStoreContainerAdapter(summaries_store, ContainerKey.SUMMARIES),
    }


def _turn(i: int) -> dict[str, Any]:
    return {
        "id": f"turn-{i}",
        "user_id": "u1",
        "thread_id": "t1",
        "role": "user",
        "type": "turn",
        "content": f"Turn {i}: I prefer dark mode and stable retries.",
        "created_at": f"2025-01-01T00:{i:02d}:00+00:00",
    }


def _response() -> dict[str, Any]:
    return {
        "facts": [
            {
                "text": "The user prefers dark mode.",
                "action": "ADD",
                "category": "preference",
                "confidence": 0.9,
                "salience": 0.8,
                "tags": ["ui"],
            }
        ],
        "episodic": [
            {
                "scope_type": "project",
                "scope_value": "CI",
                "text": "CI retries resolved flaky tests.",
                "lesson": "Use retries for flaky CI tests.",
                "confidence": 0.8,
            }
        ],
    }


def test_extract_memories_dry_shape_is_small_and_has_no_embeddings() -> None:
    chat = _SyncChat([_response()])
    embeddings = _SyncEmbeddings()
    memories_store = _Store([])
    turns_store = _Store([_turn(i) for i in range(50)])
    service = PipelineService(
        memories_store,
        chat,
        embeddings,
        containers=_containers_for_store(memories_store, turns_store=turns_store),
    )

    output = service.extract_memories_dry("u1", "t1")

    assert set(output) == {"facts", "episodic", "updates", "processed_turn_docs"}
    assert len(json.dumps(output)) < 32 * 1024
    assert output["facts"] and output["episodic"]
    assert all("embedding" not in doc for docs in (output["facts"], output["episodic"]) for doc in docs)
    assert embeddings.calls == []


def test_extract_memories_dry_is_byte_deterministic_for_same_llm_response() -> None:
    store = _Store([])
    turns_store = _Store([_turn(1)])
    service = PipelineService(
        store,
        _SyncChat([_response(), _response()]),
        _SyncEmbeddings(),
        containers=_containers_for_store(store, turns_store=turns_store),
    )

    first = service.extract_memories_dry("u1", "t1")
    second = service.extract_memories_dry("u1", "t1")

    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )


@pytest.mark.asyncio
async def test_async_extract_memories_dry_shape_is_small_and_has_no_embeddings() -> None:
    chat = _AsyncChat([_response()])
    embeddings = _AsyncEmbeddings()
    memories_store = _AsyncStore([])
    turns_store = _AsyncStore([_turn(i) for i in range(50)])
    service = AsyncPipelineService(
        memories_store,
        chat,
        embeddings,
        containers=_async_containers_for_store(memories_store, turns_store=turns_store),
    )

    output = await service.extract_memories_dry("u1", "t1")

    assert set(output) == {"facts", "episodic", "updates", "processed_turn_docs"}
    assert len(json.dumps(output)) < 32 * 1024
    assert all("embedding" not in doc for docs in (output["facts"], output["episodic"]) for doc in docs)
    assert embeddings.calls == []


@pytest.mark.asyncio
async def test_async_extract_memories_dry_is_byte_deterministic_for_same_llm_response() -> None:
    store = _AsyncStore([])
    turns_store = _AsyncStore([_turn(1)])
    service = AsyncPipelineService(
        store,
        _AsyncChat([_response(), _response()]),
        _AsyncEmbeddings(),
        containers=_async_containers_for_store(store, turns_store=turns_store),
    )

    first = await service.extract_memories_dry("u1", "t1")
    second = await service.extract_memories_dry("u1", "t1")

    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )


def test_dry_returns_processed_turn_docs_for_watermarking() -> None:
    """``extract_memories_dry`` must surface the turn docs it processed so
    ``persist_extracted_memories`` can stamp ``extracted_at`` on them and
    the next extraction call doesn't reprocess them."""
    chat = _SyncChat([_response()])
    memories_store = _Store([])
    turns_store = _Store([_turn(i) for i in range(3)])
    service = PipelineService(
        memories_store,
        chat,
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=turns_store),
    )

    output = service.extract_memories_dry("u1", "t1")

    assert "processed_turn_docs" in output
    assert {d["id"] for d in output["processed_turn_docs"]} == {"turn-0", "turn-1", "turn-2"}


def test_dry_alone_does_not_mark_turns_as_extracted() -> None:
    """A dry run is read-only: it must NOT stamp ``extracted_at`` on any
    turn (the wet ``extract_memories`` orchestrator handles marking only
    after a successful persist)."""
    chat = _SyncChat([_response()])
    memories_store = _Store([])
    turns_store = _Store([_turn(i) for i in range(3)])
    service = PipelineService(
        memories_store,
        chat,
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=turns_store),
    )

    service.extract_memories_dry("u1", "t1")

    for doc in turns_store.docs:
        assert "extracted_at" not in doc or doc["extracted_at"] is None


def test_extract_memories_marks_turns_after_successful_persist() -> None:
    """The wet ``extract_memories`` must stamp ``extracted_at`` on each
    turn it processed. Without this, the next extraction call re-loads
    the same turns and the LLM re-decides UPDATE/CONTRADICT — which is
    the runaway-extraction bug this fix is designed to prevent."""
    chat = _SyncChat([_response()])
    memories_store = _Store([])
    turns_store = _Store([_turn(i) for i in range(3)])
    service = PipelineService(
        memories_store,
        chat,
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=turns_store),
    )

    service.extract_memories("u1", "t1")

    marked_turns = [doc for doc in turns_store.docs if doc.get("extracted_at")]
    assert len(marked_turns) == 3
    marked_ids = {doc["id"] for doc in marked_turns}
    assert marked_ids == {"turn-0", "turn-1", "turn-2"}


def test_second_extract_call_does_not_reprocess_already_extracted_turns() -> None:
    """End-to-end watermarking proof: after a first ``extract_memories``
    marks the turns, a second call with no NEW turns must produce zero
    work — no second LLM call, no second persist. This is the property
    that prevents reversed-supersede / hallucinated-meta-fact bugs."""
    chat = _SyncChat([_response(), _response()])  # second response should never be consumed
    memories_store = _Store([])
    turns_store = _Store([_turn(i) for i in range(3)])
    service = PipelineService(
        memories_store,
        chat,
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=turns_store),
    )

    service.extract_memories("u1", "t1")
    calls_after_first = chat.calls

    # Second invocation with no new turns: watermarked turns are filtered
    # out by the query and the dry early-returns with empty items.
    service.extract_memories("u1", "t1")
    assert chat.calls == calls_after_first


@pytest.mark.asyncio
async def test_async_extract_memories_marks_turns_after_successful_persist() -> None:
    chat = _AsyncChat([_response()])
    memories_store = _AsyncStore([])
    turns_store = _AsyncStore([_turn(i) for i in range(3)])
    service = AsyncPipelineService(
        memories_store,
        chat,
        _AsyncEmbeddings(),
        containers=_async_containers_for_store(memories_store, turns_store=turns_store),
    )

    await service.extract_memories("u1", "t1")

    marked_turns = [doc for doc in turns_store.docs if doc.get("extracted_at")]
    assert len(marked_turns) == 3


# ---------------------------------------------------------------------------
# Grounding-check regression tests
#
# These tests pin down the two known LLM extraction-time failure modes that
# previously corrupted the fact store and required a wheel hotfix:
#
#   1. The LLM synthesizes an ADD by paraphrase-merging 2+ existing facts
#      (e.g. "user eats meat" + "user loves steak" → "user loves steak,
#      indicating they eat meat") even though the new user turn said nothing
#      on the topic.
#   2. The LLM emits a second invented CONTRADICT fact alongside the literal
#      user statement (e.g. user says "I love steak"; LLM emits both
#      "loves steak" AND "user eats meat" — the second is a phantom
#      explicit-negation that polluted the store with claims the user
#      didn't make).
#
# The fix is a prompt change that forbids both patterns. Because we can't
# directly test prompt-following at the unit-test level (no real LLM in
# the test loop), we test the structural safety net:
# ``check_extracted_fact_grounding`` logs a WARNING when these patterns
# slip through. The four scenarios below pair a "buggy" LLM response with
# a "clean" one for each pattern, asserting the WARNING fires (or not)
# accordingly. If a future change ever regresses the prompt and the LLM
# starts emitting these patterns, the WARNING in production telemetry
# becomes the visible signal.
# ---------------------------------------------------------------------------


def _existing_fact(fid: str, content: str) -> dict[str, Any]:
    """Build a minimal existing-fact doc shaped like what
    ``_load_existing_memories`` returns from Cosmos."""
    return {
        "id": fid,
        "user_id": "u1",
        "type": "fact",
        "content": content,
        "content_hash": fid,
        "salience": 0.8,
        "confidence": 0.9,
        "metadata": {"category": "preference"},
        "tags": ["sys:fact"],
    }


def _moderate_hotels_turn() -> dict[str, Any]:
    return {
        "id": "turn-new",
        "user_id": "u1",
        "thread_id": "t1",
        "role": "user",
        "type": "turn",
        "content": "Normally, I prefer moderate hotels.",
        "created_at": "2026-06-02T19:00:00+00:00",
    }


def _steak_seafood_turn() -> dict[str, Any]:
    return {
        "id": "turn-new",
        "user_id": "u1",
        "thread_id": "t1",
        "role": "user",
        "type": "turn",
        "content": "Actually, I love steak and seafood.",
        "created_at": "2026-06-02T18:00:00+00:00",
    }


def test_grounding_check_warns_when_add_synthesizes_from_multiple_existing_facts(caplog) -> None:
    """Scenario 1 (buggy): the LLM emits a synthesized ADD whose tokens come
    from 2+ existing facts but not from the new user turn. The grounding
    check must emit a WARNING naming the offending fact."""
    existing = [
        _existing_fact("fact_meat", "The user eats meat."),
        _existing_fact("fact_steak", "The user loves steak and seafood."),
    ]
    buggy_response = {
        "facts": [
            {
                "text": "The user normally prefers moderate hotels.",
                "action": "ADD",
                "category": "preference",
                "confidence": 0.9,
                "salience": 0.7,
            },
            {
                # synthesized — tokens come from existing fact_steak + fact_meat,
                # not from the new "moderate hotels" turn
                "text": "The user loves steak and seafood, indicating they eat meat.",
                "action": "ADD",
                "category": "preference",
                "confidence": 0.9,
                "salience": 0.7,
            },
        ],
        "episodic": [],
    }
    chat = _SyncChat([buggy_response])
    memories_store = _Store(existing)
    turns_store = _Store([_moderate_hotels_turn()])
    service = PipelineService(
        memories_store,
        chat,
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=turns_store),
    )

    with caplog.at_level("WARNING", logger="azure.cosmos.agent_memory.pipeline"):
        service.extract_memories_dry("u1", "t1")

    synthesis_warnings = [
        rec
        for rec in caplog.records
        if "synthesized from" in rec.getMessage() and "steak and seafood, indicating they eat meat" in rec.getMessage()
    ]
    assert synthesis_warnings, (
        f"expected a WARNING flagging the synthesized fact; got: {[rec.getMessage() for rec in caplog.records]}"
    )
    # The grounded "moderate hotels" fact must NOT trigger a warning.
    assert not any(
        "moderate hotels" in rec.getMessage() and "synthesized from" in rec.getMessage() for rec in caplog.records
    )


def test_grounding_check_silent_when_add_is_grounded_in_user_turn(caplog) -> None:
    """Scenario 2 (clean): with the same existing-facts context, a single
    grounded ADD (only the new "moderate hotels" claim) must NOT trigger
    any synthesis WARNING. This is the post-fix expected behaviour."""
    existing = [
        _existing_fact("fact_meat", "The user eats meat."),
        _existing_fact("fact_steak", "The user loves steak and seafood."),
    ]
    clean_response = {
        "facts": [
            {
                "text": "The user normally prefers moderate hotels.",
                "action": "ADD",
                "category": "preference",
                "confidence": 0.9,
                "salience": 0.7,
            }
        ],
        "episodic": [],
    }
    chat = _SyncChat([clean_response])
    memories_store = _Store(existing)
    turns_store = _Store([_moderate_hotels_turn()])
    service = PipelineService(
        memories_store,
        chat,
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=turns_store),
    )

    with caplog.at_level("WARNING", logger="azure.cosmos.agent_memory.pipeline"):
        service.extract_memories_dry("u1", "t1")

    grounding_warnings = [
        rec
        for rec in caplog.records
        if "synthesized from" in rec.getMessage() or "phantom-negation/restatement" in rec.getMessage()
    ]
    assert grounding_warnings == [], (
        f"clean output must not emit grounding warnings; got: {[rec.getMessage() for rec in grounding_warnings]}"
    )


def test_grounding_check_warns_on_phantom_explicit_negation_fact(caplog) -> None:
    """Scenario 3 (buggy): user said "Actually, I love steak and seafood";
    LLM emits both a literal-paraphrase fact AND a phantom "user eats meat"
    fact (an invented explicit-negation of the prior vegetarian fact).
    The phantom fact's tokens are NOT in the user turn and they overlap
    a single existing fact ("does not eat meat") — the single-contributor
    branch of the grounding heuristic must fire a WARNING."""
    existing = [_existing_fact("fact_veg", "The user does not eat meat.")]
    buggy_response = {
        "facts": [
            {
                # legitimate literal paraphrase of the user turn
                "text": "The user loves steak and seafood.",
                "action": "CONTRADICT",
                "supersedes_id": "fact_veg",
                "category": "preference",
                "confidence": 0.95,
                "salience": 0.8,
            },
            {
                # phantom — user never said this; tokens come from existing fact_veg
                "text": "The user eats meat.",
                "action": "CONTRADICT",
                "supersedes_id": "fact_veg",
                "category": "preference",
                "confidence": 0.95,
                "salience": 0.8,
            },
        ],
        "episodic": [],
    }
    chat = _SyncChat([buggy_response])
    memories_store = _Store(existing)
    turns_store = _Store([_steak_seafood_turn()])
    service = PipelineService(
        memories_store,
        chat,
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=turns_store),
    )

    with caplog.at_level("WARNING", logger="azure.cosmos.agent_memory.pipeline"):
        service.extract_memories_dry("u1", "t1")

    phantom_warnings = [
        rec for rec in caplog.records if "phantom-negation" in rec.getMessage() and "eats meat" in rec.getMessage()
    ]
    assert phantom_warnings, (
        f"expected a WARNING flagging the phantom-negation fact; got: {[rec.getMessage() for rec in caplog.records]}"
    )
    # The legitimate "loves steak and seafood" fact must NOT trigger a warning;
    # its tokens are grounded in the user turn.
    assert not any(
        "loves steak and seafood" in rec.getMessage()
        and ("phantom-negation" in rec.getMessage() or "synthesized from" in rec.getMessage())
        for rec in caplog.records
    )


def test_grounding_check_silent_on_clean_implicit_contradict(caplog) -> None:
    """Scenario 4 (clean): the post-fix expected behaviour for an implicit
    contradiction — ONE fact with literal user text and a CONTRADICT
    supersedes_id. No phantom-negation fact, no WARNING."""
    existing = [_existing_fact("fact_veg", "The user does not eat meat.")]
    clean_response = {
        "facts": [
            {
                "text": "The user loves steak and seafood.",
                "action": "CONTRADICT",
                "supersedes_id": "fact_veg",
                "category": "preference",
                "confidence": 0.95,
                "salience": 0.8,
            }
        ],
        "episodic": [],
    }
    chat = _SyncChat([clean_response])
    memories_store = _Store(existing)
    turns_store = _Store([_steak_seafood_turn()])
    service = PipelineService(
        memories_store,
        chat,
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=turns_store),
    )

    with caplog.at_level("WARNING", logger="azure.cosmos.agent_memory.pipeline"):
        service.extract_memories_dry("u1", "t1")

    grounding_warnings = [
        rec
        for rec in caplog.records
        if "synthesized from" in rec.getMessage() or "phantom-negation/restatement" in rec.getMessage()
    ]
    assert grounding_warnings == [], (
        "clean implicit-contradict must not emit grounding warnings; got: "
        f"{[rec.getMessage() for rec in grounding_warnings]}"
    )


@pytest.mark.asyncio
async def test_async_grounding_check_warns_on_synthesis(caplog) -> None:
    """Async-path mirror of scenario 1: confirms the grounding heuristic
    is wired into both sync and async extract pipelines."""
    existing = [
        _existing_fact("fact_meat", "The user eats meat."),
        _existing_fact("fact_steak", "The user loves steak and seafood."),
    ]
    buggy_response = {
        "facts": [
            {
                "text": "The user normally prefers moderate hotels.",
                "action": "ADD",
                "category": "preference",
                "confidence": 0.9,
                "salience": 0.7,
            },
            {
                "text": "The user loves steak and seafood, indicating they eat meat.",
                "action": "ADD",
                "category": "preference",
                "confidence": 0.9,
                "salience": 0.7,
            },
        ],
        "episodic": [],
    }
    chat = _AsyncChat([buggy_response])
    memories_store = _AsyncStore(existing)
    turns_store = _AsyncStore([_moderate_hotels_turn()])
    service = AsyncPipelineService(
        memories_store,
        chat,
        _AsyncEmbeddings(),
        containers=_async_containers_for_store(memories_store, turns_store=turns_store),
    )

    with caplog.at_level("WARNING", logger="azure.cosmos.agent_memory.pipeline.aio"):
        await service.extract_memories_dry("u1", "t1")

    synthesis_warnings = [
        rec
        for rec in caplog.records
        if "synthesized from" in rec.getMessage() and "steak and seafood, indicating they eat meat" in rec.getMessage()
    ]
    assert synthesis_warnings, (
        f"expected an async WARNING flagging the synthesized fact; got: {[rec.getMessage() for rec in caplog.records]}"
    )
