"""Async typed result slots in search.

See tests/unit/test_search_slots.py for the slot planning, composition and filter tests that apply to both clients.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.aio.cosmos_memory_client import AsyncCosmosMemoryClient
from azure.cosmos.agent_memory.aio.store import AsyncMemoryStore
from azure.cosmos.agent_memory.store._search_helpers import PROCEDURES_IN_FORCE

RANKED = [{"id": f"fact_{i}", "content": f"memory {i}"} for i in range(5)]
PREFERENCE = {"id": "pref_a", "content": "The user prefers pnpm.", "metadata": {"category": "preference"}}
PROCEDURE = {"id": "proc_a", "content": "Run targeted tests before reporting success.", "type": "procedural"}
CLIENT_LOGGER = "azure.cosmos.agent_memory.aio.cosmos_memory_client"


class _Rows:
    def __init__(self, items=()):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


# --- store ---------------------------------------------------------------------------------------


def _store() -> tuple[AsyncMemoryStore, MagicMock, MagicMock]:
    memories = MagicMock()
    memories.query_items.side_effect = lambda **_kwargs: _Rows()
    embeddings = MagicMock()
    embeddings.generate = AsyncMock(return_value=[0.1, 0.2])
    store = AsyncMemoryStore(
        containers={key: memories for key in (ContainerKey.TURNS, ContainerKey.MEMORIES, ContainerKey.SUMMARIES)},
        embeddings_client=embeddings,
    )
    return store, memories, embeddings


@pytest.mark.parametrize(("vector_only", "order_by"), [(False, "RANK RRF"), (True, "ORDER BY VectorDistance")])
async def test_a_vector_only_search_ranks_by_similarity_alone(vector_only, order_by):
    store, memories, _ = _store()

    await store.search(search_terms="Hotels in Lisbon", user_id="u1", vector_only=vector_only)

    query = memories.query_items.call_args.kwargs["query"]
    assert order_by in query
    assert ("FullTextScore" in query) is not vector_only
    assert PROCEDURES_IN_FORCE not in query


@pytest.mark.parametrize(
    ("memory_types", "hides_inactive_procedures"),
    [
        pytest.param(None, True, id="any-type"),
        pytest.param(["procedural"], True, id="procedures"),
        pytest.param(["fact", "procedural"], True, id="facts-and-procedures"),
        pytest.param(["fact"], False, id="facts"),
        pytest.param(["episodic"], False, id="episodes"),
    ],
)
async def test_search_hides_procedures_out_of_force_where_procedures_can_match(memory_types, hides_inactive_procedures):
    store, memories, _ = _store()

    await store.search(search_terms="hello", user_id="u1", memory_types=memory_types, active_procedures_only=True)

    call = memories.query_items.call_args.kwargs
    assert (PROCEDURES_IN_FORCE in call["query"]) is hides_inactive_procedures
    assert ("@procedure_status" in {p["name"] for p in call["parameters"]}) is hides_inactive_procedures


async def test_the_procedure_slot_search_is_a_vector_search_of_the_users_situational_procedures():
    store, memories, _ = _store()

    await store.search(
        search_terms="Hotels in Lisbon",
        user_id="u1",
        thread_id="t1",
        memory_types=["procedural"],
        applies="when",
        active_procedures_only=True,
        vector_only=True,
        top_k=1,
    )

    call = memories.query_items.call_args.kwargs
    params = {p["name"]: p["value"] for p in call["parameters"]}
    assert "ORDER BY VectorDistance" in call["query"] and "FullTextScore" not in call["query"]
    assert "c.applies = @applies" in call["query"]
    assert PROCEDURES_IN_FORCE in call["query"]
    assert (params["@user_id"], params["@applies"]) == ("u1", "when")
    assert "partition_key" not in call, "procedures live under a sentinel thread, so a thread cannot key the query"


async def test_a_supplied_query_embedding_is_not_embedded_again():
    store, memories, embeddings = _store()

    await store.search(search_terms="Hotels in Lisbon", user_id="u1", query_embedding=[0.5, 0.6])

    embeddings.generate.assert_not_called()
    params = {p["name"]: p["value"] for p in memories.query_items.call_args.kwargs["parameters"]}
    assert params["@embedding"] == [0.5, 0.6]


async def test_search_keeps_facts_of_the_requested_categories():
    store, memories, _ = _store()

    await store.search(search_terms="hello", user_id="u1", memory_types=["fact"], categories=["preference", "habit"])

    call = memories.query_items.call_args.kwargs
    params = {p["name"]: p["value"] for p in call["parameters"]}
    assert "c.metadata.category IN (@category_0, @category_1)" in call["query"]
    assert (params["@category_0"], params["@category_1"]) == ("preference", "habit")


# --- search_cosmos -------------------------------------------------------------------------------


def _slot_store(store: MagicMock) -> None:
    def search(**kwargs):
        if kwargs.get("categories") == ["preference"]:
            return [dict(PREFERENCE)]
        if kwargs.get("memory_types") == ["procedural"]:
            return [dict(PROCEDURE)]
        return [dict(doc) for doc in RANKED[: kwargs["top_k"]]]

    store.search = AsyncMock(side_effect=search)


def _async_client() -> tuple[AsyncCosmosMemoryClient, MagicMock]:
    client = AsyncCosmosMemoryClient(use_default_credential=False)
    store = MagicMock()
    store._embed = AsyncMock(return_value=[0.1, 0.2])
    store.search_summaries = AsyncMock(return_value=[])
    store.search_turns = AsyncMock(return_value=[])
    _slot_store(store)
    client._get_store = MagicMock(return_value=store)
    return client, store


async def test_slots_hold_a_preference_and_a_procedure_and_procedures_leave_the_ranked_search():
    client, store = _async_client()

    out = await client.search_cosmos(
        "q", user_id="u1", top_k=5, memory_types=["fact", "procedural"], preference_slot=True, procedure_slot=True
    )

    assert [doc["id"] for doc in out] == ["fact_0", "fact_1", "fact_2", "pref_a", "proc_a"]
    ranked, preference, procedure = (call.kwargs for call in store.search.call_args_list)
    assert (ranked["memory_types"], ranked["top_k"], ranked["active_procedures_only"]) == (["fact"], 5, True)
    assert (preference["memory_types"], preference["categories"], preference["vector_only"], preference["top_k"]) == (
        ["fact"],
        ["preference"],
        True,
        4,
    )
    assert (procedure["memory_types"], procedure["applies"], procedure["active_procedures_only"]) == (
        ["procedural"],
        "when",
        True,
    )
    assert (procedure["vector_only"], procedure["top_k"]) == (True, 1)


async def test_slotted_results_carry_their_slot_and_nothing_internal_leaks():
    client, _ = _async_client()

    out = await client.search_cosmos("q", user_id="u1", top_k=5, preference_slot=True, procedure_slot=True)

    assert [doc.get("retrieval") for doc in out] == [
        None,
        None,
        None,
        {"rank": 4, "slot": "preference"},
        {"rank": 5, "slot": "procedure"},
    ]
    assert not any(key.startswith("_") for doc in out for key in doc)


async def test_a_preference_slot_alone_leaves_procedures_in_the_ranked_search():
    client, store = _async_client()

    out = await client.search_cosmos(
        "q", user_id="u1", top_k=5, memory_types=["fact", "procedural"], preference_slot=True
    )

    assert [doc["id"] for doc in out] == ["fact_0", "fact_1", "fact_2", "fact_3", "pref_a"]
    ranked, preference = (call.kwargs for call in store.search.call_args_list)
    assert ranked["memory_types"] == ["fact", "procedural"]
    assert preference["memory_types"] == ["fact"]


async def test_slot_searches_read_only_what_the_search_itself_may_read():
    client, store = _async_client()

    await client.search_cosmos(
        "q",
        user_id="u1",
        thread_id="t1",
        role="user",
        top_k=5,
        tags_all=["work"],
        tags_any=["travel"],
        exclude_tags=["private"],
        include_superseded=True,
        min_salience=0.2,
        min_confidence=0.5,
        created_after="2026-01-01T00:00:00+00:00",
        created_before="2026-12-31T00:00:00+00:00",
        preference_slot=True,
        procedure_slot=True,
    )

    shared = (
        "search_terms",
        "user_id",
        "thread_id",
        "role",
        "tags_all",
        "tags_any",
        "exclude_tags",
        "include_superseded",
        "min_salience",
        "min_confidence",
        "created_after",
        "created_before",
        "active_procedures_only",
    )
    ranked, *slots = (call.kwargs for call in store.search.call_args_list)
    assert len(slots) == 2
    for slot in slots:
        assert {key: slot[key] for key in shared} == {key: ranked[key] for key in shared}


async def test_the_query_is_embedded_once_for_the_ranked_and_slot_searches():
    client, store = _async_client()

    await client.search_cosmos("q", user_id="u1", top_k=5, preference_slot=True, procedure_slot=True)

    store._embed.assert_awaited_once_with("q")
    assert [call.kwargs["query_embedding"] for call in store.search.call_args_list] == [[0.1, 0.2]] * 3


async def test_without_slots_the_client_leaves_embedding_to_the_store():
    client, store = _async_client()

    await client.search_cosmos("q", user_id="u1", top_k=5)

    store._embed.assert_not_called()
    assert "query_embedding" not in store.search.call_args.kwargs


async def test_a_failed_slot_search_leaves_the_slot_to_the_ranked_results(caplog):
    client, store = _async_client()

    def search(**kwargs):
        if kwargs.get("vector_only"):
            raise RuntimeError("cosmos unavailable")
        return [dict(doc) for doc in RANKED[: kwargs["top_k"]]]

    store.search = AsyncMock(side_effect=search)

    with caplog.at_level("WARNING", logger=CLIENT_LOGGER):
        out = await client.search_cosmos("q", user_id="u1", top_k=5, preference_slot=True, procedure_slot=True)

    assert [doc["id"] for doc in out] == ["fact_0", "fact_1", "fact_2", "fact_3", "fact_4"]
    assert [doc["retrieval"]["slot"] for doc in out[3:]] == ["backfill", "backfill"]
    assert "slot search failed" in caplog.text
    assert "cosmos unavailable" not in caplog.text, "a log names the failure, never its message"


@pytest.mark.parametrize(
    ("top_k", "slots"), [pytest.param(5, False, id="no-slots-asked"), pytest.param(2, True, id="top-k-too-small")]
)
async def test_without_slots_procedures_compete_but_only_active_ones(top_k, slots):
    client, store = _async_client()

    await client.search_cosmos(
        "q", user_id="u1", top_k=top_k, memory_types=["fact", "procedural"], preference_slot=slots, procedure_slot=slots
    )

    [call] = store.search.call_args_list
    assert (call.kwargs["memory_types"], call.kwargs["active_procedures_only"]) == (["fact", "procedural"], True)


async def test_a_search_for_procedures_alone_ranks_them_and_the_procedure_slot_has_no_effect():
    client, store = _async_client()

    out = await client.search_cosmos(
        "q", user_id="u1", top_k=5, memory_types=["procedural"], preference_slot=True, procedure_slot=True
    )

    [call] = store.search.call_args_list
    assert (call.kwargs["memory_types"], call.kwargs["active_procedures_only"]) == (["procedural"], True)
    assert [doc["id"] for doc in out] == ["proc_a"]
    assert all("retrieval" not in doc for doc in out)


@pytest.mark.parametrize(
    "search",
    [pytest.param({}, id="no-user"), pytest.param({"user_id": "u1", "memory_id": "m1"}, id="one-memory")],
)
async def test_slots_need_a_user_and_a_search_across_memories(search):
    client, store = _async_client()

    out = await client.search_cosmos("q", top_k=5, preference_slot=True, procedure_slot=True, **search)

    [call] = store.search.call_args_list
    assert call.kwargs["memory_types"] == ["fact"]
    assert "query_embedding" not in call.kwargs
    assert all("retrieval" not in doc for doc in out)


async def test_episodes_stay_in_the_ranked_results_and_never_take_a_slot():
    client, store = _async_client()

    await client.search_cosmos(
        "q", user_id="u1", top_k=5, include_episodes=True, preference_slot=True, procedure_slot=True
    )

    ranked, preference, procedure = (call.kwargs for call in store.search.call_args_list)
    assert ranked["memory_types"] == ["fact", "episodic"]
    assert preference["memory_types"] == ["fact"]
    assert procedure["memory_types"] == ["procedural"]


async def test_summaries_and_turns_still_follow_the_slotted_results_without_repeating_content():
    client, store = _async_client()
    store.search_summaries.return_value = [
        {"id": "sum_a", "content": PREFERENCE["content"]},
        {"id": "sum_b", "content": "A summary."},
    ]
    store.search_turns.return_value = [
        {"id": "turn_a", "content": "A summary."},
        {"id": "turn_b", "content": "A turn."},
    ]

    out = await client.search_cosmos(
        "q",
        user_id="u1",
        top_k=5,
        preference_slot=True,
        procedure_slot=True,
        include_summaries=True,
        include_turns=True,
    )

    assert [doc["id"] for doc in out] == ["fact_0", "fact_1", "fact_2", "pref_a", "proc_a", "sum_b", "turn_b"]
