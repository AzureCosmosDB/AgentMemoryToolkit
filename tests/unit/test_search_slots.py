"""Typed result slots in search: one preference fact and one situational procedure within top_k,
and what user-facing search shows of procedures."""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

from azure.cosmos.agent_memory import _slots
from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory._query_builder import _QueryBuilder
from azure.cosmos.agent_memory.cosmos_memory_client import CosmosMemoryClient
from azure.cosmos.agent_memory.store import MemoryStore
from azure.cosmos.agent_memory.store._search_helpers import PROCEDURES_IN_FORCE, add_result_filters

RANKED = [{"id": f"fact_{i}", "content": f"memory {i}"} for i in range(5)]
PREFERENCE = {"id": "pref_a", "content": "The user prefers pnpm.", "metadata": {"category": "preference"}}
PROCEDURE = {"id": "proc_a", "content": "Run targeted tests before reporting success.", "type": "procedural"}
CLIENT_LOGGER = "azure.cosmos.agent_memory.cosmos_memory_client"
# A procedure has no salience or confidence, its role is "system" and its tags are the pipeline's own, so
# these filters could never match one.
FACT_ONLY_FILTERS = {
    "role": "user",
    "tags_all": ["work"],
    "tags_any": ["travel"],
    "min_salience": 0.2,
    "min_confidence": 0.5,
}
# Filters that read fields every memory has.
SCOPE_FILTERS = {
    "thread_id": "t1",
    "exclude_tags": ["sys:secret"],
    "include_superseded": True,
    "created_after": "2026-01-01T00:00:00+00:00",
    "created_before": "2026-12-31T00:00:00+00:00",
}


# --- planning and composition --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("top_k", "preference", "procedure", "pool", "expected"),
    [
        pytest.param(5, True, True, ["fact", "episodic"], (True, True), id="both"),
        pytest.param(5, True, True, ["episodic"], (False, True), id="no-fact-pool-no-preference"),
        pytest.param(5, True, True, [], (False, False), id="a-search-for-procedures-alone-has-no-slots"),
        pytest.param(2, True, True, ["fact"], (False, False), id="no-ranked-result-would-remain"),
        pytest.param(2, False, True, ["fact"], (False, True), id="procedure-only"),
        pytest.param(5, False, False, ["fact"], (False, False), id="none-asked"),
    ],
)
def test_plan_slots(top_k, preference, procedure, pool, expected):
    plan = _slots.plan_slots(top_k, preference_slot=preference, procedure_slot=procedure, pool_types=pool)

    assert (plan.preference, plan.procedure) == expected


def test_compose_fills_the_slots_after_the_ranked_results_skipping_what_is_shown():
    out = _slots.compose(
        RANKED,
        top_k=5,
        plan=_slots.SlotPlan(preference=True, procedure=True),
        preferences=[RANKED[1], PREFERENCE],
        procedures=[PROCEDURE],
    )

    assert [doc["id"] for doc in out] == ["fact_0", "fact_1", "fact_2", "pref_a", "proc_a"]
    assert all("retrieval" not in doc for doc in out[:3])
    assert [doc["retrieval"] for doc in out[3:]] == [
        {"rank": 4, "slot": "preference"},
        {"rank": 5, "slot": "procedure"},
    ]
    assert "retrieval" not in PREFERENCE and "retrieval" not in PROCEDURE, "the candidates are copied, not edited"


def test_compose_fills_an_empty_slot_with_the_next_ranked_result():
    repeated_text = {"id": "other", "content": "memory 0"}
    out = _slots.compose(
        RANKED,
        top_k=5,
        plan=_slots.SlotPlan(preference=True, procedure=True),
        preferences=[repeated_text],
        procedures=[],
    )

    assert [doc["id"] for doc in out] == ["fact_0", "fact_1", "fact_2", "fact_3", "fact_4"]
    assert [doc["retrieval"]["slot"] for doc in out[3:]] == ["backfill", "backfill"]


def test_compose_never_exceeds_top_k_or_invents_results():
    out = _slots.compose(
        RANKED[:3],
        top_k=5,
        plan=_slots.SlotPlan(preference=True, procedure=True),
        preferences=[],
        procedures=[],
    )

    assert [doc["id"] for doc in out] == ["fact_0", "fact_1", "fact_2"]
    assert all("retrieval" not in doc for doc in out)


def test_the_slot_searches_reuse_the_searchs_arguments():
    base = {"search_terms": "q", "user_id": "u1", "thread_id": "t1", "top_k": 5, "memory_types": ["fact"]}

    preference = _slots.preference_search(base, 3)
    procedure = _slots.procedure_search(base)

    assert preference == {**base, "categories": ["preference"], "vector_only": True, "top_k": 4}
    assert procedure == {
        **base,
        "memory_types": ["procedural"],
        "applies": "when",
        "active_procedures_only": True,
        "vector_only": True,
        "top_k": 1,
    }
    assert base["top_k"] == 5 and base["memory_types"] == ["fact"]


def test_the_procedure_slot_drops_the_filters_no_procedure_can_meet():
    base = {
        "search_terms": "q",
        "user_id": "u1",
        "top_k": 5,
        "memory_types": ["fact"],
        "query_embedding": [0.1, 0.2],
        **SCOPE_FILTERS,
        **FACT_ONLY_FILTERS,
    }

    procedure = _slots.procedure_search(base)
    preference = _slots.preference_search(base, 3)

    assert set(FACT_ONLY_FILTERS).isdisjoint(procedure)
    assert {key: procedure[key] for key in (*SCOPE_FILTERS, "user_id", "query_embedding")} == {
        **SCOPE_FILTERS,
        "user_id": "u1",
        "query_embedding": [0.1, 0.2],
    }
    assert {key: preference[key] for key in FACT_ONLY_FILTERS} == FACT_ONLY_FILTERS, "facts keep every filter"
    assert set(FACT_ONLY_FILTERS) <= set(base), "the search's own arguments are not edited"


# --- store filters -------------------------------------------------------------------------------


def test_result_filters_add_nothing_unless_asked():
    qb = _QueryBuilder()

    add_result_filters(qb)

    assert qb.build_where() == ""
    assert qb.get_parameters() == []


def test_result_filters_limit_procedures_categories_and_applies():
    qb = _QueryBuilder()
    add_result_filters(qb, active_procedures_only=True, categories=["preference"], applies="when")

    where = qb.build_where()
    params = {p["name"]: p["value"] for p in qb.get_parameters()}
    assert PROCEDURES_IN_FORCE in where
    assert "c.metadata.category IN (@category_0)" in where
    assert "c.applies = @applies" in where
    assert (params["@procedure_type"], params["@procedure_status"], params["@category_0"], params["@applies"]) == (
        "procedural",
        "active",
        "preference",
        "when",
    )
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", params["@procedure_today"])


def _store() -> tuple[MemoryStore, MagicMock, MagicMock]:
    memories = MagicMock()
    memories.query_items.return_value = []
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    store = MemoryStore(
        containers={key: memories for key in (ContainerKey.TURNS, ContainerKey.MEMORIES, ContainerKey.SUMMARIES)},
        embeddings_client=embeddings,
    )
    return store, memories, embeddings


@pytest.mark.parametrize(("vector_only", "order_by"), [(False, "RANK RRF"), (True, "ORDER BY VectorDistance")])
def test_a_vector_only_search_ranks_by_similarity_alone(vector_only, order_by):
    store, memories, _ = _store()

    store.search(search_terms="Hotels in Lisbon", user_id="u1", vector_only=vector_only)

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
def test_search_hides_procedures_out_of_force_where_procedures_can_match(memory_types, hides_inactive_procedures):
    store, memories, _ = _store()

    store.search(search_terms="hello", user_id="u1", memory_types=memory_types, active_procedures_only=True)

    call = memories.query_items.call_args.kwargs
    assert (PROCEDURES_IN_FORCE in call["query"]) is hides_inactive_procedures
    assert ("@procedure_status" in {p["name"] for p in call["parameters"]}) is hides_inactive_procedures


def test_search_shows_every_procedure_unless_asked_for_those_in_force():
    store, memories, _ = _store()

    store.search(search_terms="hello", user_id="u1", memory_types=["procedural"])

    assert PROCEDURES_IN_FORCE not in memories.query_items.call_args.kwargs["query"]


def test_the_procedure_slot_search_is_a_vector_search_of_the_users_situational_procedures():
    store, memories, _ = _store()

    store.search(
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


def test_the_procedure_slot_query_asks_for_nothing_a_procedure_lacks():
    store, memories, _ = _store()
    search = {"search_terms": "Hotels in Lisbon", "user_id": "u1", "top_k": 5, "memory_types": ["fact"]}

    store.search(**_slots.procedure_search({**search, **SCOPE_FILTERS, **FACT_ONLY_FILTERS}))

    where = memories.query_items.call_args.kwargs["query"].split(" WHERE ", 1)[1]
    for fact_only in (
        "c.role =",
        "c.salience >=",
        "c.confidence >=",
        "ARRAY_CONTAINS(c.tags, @tag_0)",
        "ARRAY_CONTAINS(c.tags, @any_tag_0)",
    ):
        assert fact_only not in where
    for scope in ("c.user_id = @user_id", "NOT ARRAY_CONTAINS(c.tags, @exc_tag_0)", "c.created_at >= @created_after"):
        assert scope in where


def test_a_supplied_query_embedding_is_not_embedded_again():
    store, memories, embeddings = _store()

    store.search(search_terms="Hotels in Lisbon", user_id="u1", query_embedding=[0.5, 0.6])

    embeddings.generate.assert_not_called()
    params = {p["name"]: p["value"] for p in memories.query_items.call_args.kwargs["parameters"]}
    assert params["@embedding"] == [0.5, 0.6]


def test_search_keeps_facts_of_the_requested_categories():
    store, memories, _ = _store()

    store.search(search_terms="hello", user_id="u1", memory_types=["fact"], categories=["preference", "habit"])

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

    store.search.side_effect = search


def _sync_client() -> tuple[CosmosMemoryClient, MagicMock]:
    client = CosmosMemoryClient(use_default_credential=False)
    store = MagicMock()
    _slot_store(store)
    client._get_store = MagicMock(return_value=store)
    return client, store


def test_slots_hold_a_preference_and_a_procedure_and_procedures_leave_the_ranked_search():
    client, store = _sync_client()

    out = client.search_cosmos(
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


def test_slotted_results_carry_their_slot_and_nothing_internal_leaks():
    client, _ = _sync_client()

    out = client.search_cosmos("q", user_id="u1", top_k=5, preference_slot=True, procedure_slot=True)

    assert [doc.get("retrieval") for doc in out] == [
        None,
        None,
        None,
        {"rank": 4, "slot": "preference"},
        {"rank": 5, "slot": "procedure"},
    ]
    assert not any(key.startswith("_") for doc in out for key in doc)


def test_a_preference_slot_alone_leaves_procedures_in_the_ranked_search():
    client, store = _sync_client()

    out = client.search_cosmos("q", user_id="u1", top_k=5, memory_types=["fact", "procedural"], preference_slot=True)

    assert [doc["id"] for doc in out] == ["fact_0", "fact_1", "fact_2", "fact_3", "pref_a"]
    ranked, preference = (call.kwargs for call in store.search.call_args_list)
    assert ranked["memory_types"] == ["fact", "procedural"]
    assert preference["memory_types"] == ["fact"]


def test_slot_searches_read_only_what_the_search_itself_may_read():
    client, store = _sync_client()

    client.search_cosmos(
        "q",
        user_id="u1",
        top_k=5,
        preference_slot=True,
        procedure_slot=True,
        **SCOPE_FILTERS,
        **FACT_ONLY_FILTERS,
    )

    shared = ("search_terms", "user_id", "active_procedures_only", *SCOPE_FILTERS)
    ranked, preference, procedure = (call.kwargs for call in store.search.call_args_list)
    # Facts can meet every filter. A procedure can meet only those every memory has a field for.
    assert {key: preference[key] for key in (*shared, *FACT_ONLY_FILTERS)} == {
        key: ranked[key] for key in (*shared, *FACT_ONLY_FILTERS)
    }
    assert {key: procedure[key] for key in shared} == {key: ranked[key] for key in shared}
    assert all(procedure.get(key) is None for key in FACT_ONLY_FILTERS)


def test_the_query_is_embedded_once_for_the_ranked_and_slot_searches():
    client, store = _sync_client()
    store._embed.return_value = [0.1, 0.2]

    client.search_cosmos("q", user_id="u1", top_k=5, preference_slot=True, procedure_slot=True)

    store._embed.assert_called_once_with("q")
    assert [call.kwargs["query_embedding"] for call in store.search.call_args_list] == [[0.1, 0.2]] * 3


def test_without_slots_the_client_leaves_embedding_to_the_store():
    client, store = _sync_client()

    client.search_cosmos("q", user_id="u1", top_k=5)

    store._embed.assert_not_called()
    assert "query_embedding" not in store.search.call_args.kwargs


def test_a_failed_slot_search_leaves_the_slot_to_the_ranked_results(caplog):
    client, store = _sync_client()

    def search(**kwargs):
        if kwargs.get("vector_only"):
            raise RuntimeError("cosmos unavailable")
        return [dict(doc) for doc in RANKED[: kwargs["top_k"]]]

    store.search.side_effect = search

    with caplog.at_level("WARNING", logger=CLIENT_LOGGER):
        out = client.search_cosmos("q", user_id="u1", top_k=5, preference_slot=True, procedure_slot=True)

    assert [doc["id"] for doc in out] == ["fact_0", "fact_1", "fact_2", "fact_3", "fact_4"]
    assert [doc["retrieval"]["slot"] for doc in out[3:]] == ["backfill", "backfill"]
    assert "slot search failed" in caplog.text
    assert "cosmos unavailable" not in caplog.text, "a log names the failure, never its message"


@pytest.mark.parametrize(
    ("top_k", "slots"), [pytest.param(5, False, id="no-slots-asked"), pytest.param(2, True, id="top-k-too-small")]
)
def test_without_slots_procedures_compete_but_only_active_ones(top_k, slots):
    client, store = _sync_client()

    client.search_cosmos(
        "q", user_id="u1", top_k=top_k, memory_types=["fact", "procedural"], preference_slot=slots, procedure_slot=slots
    )

    [call] = store.search.call_args_list
    assert (call.kwargs["memory_types"], call.kwargs["active_procedures_only"]) == (["fact", "procedural"], True)


def test_a_search_for_procedures_alone_ranks_them_and_the_procedure_slot_has_no_effect():
    client, store = _sync_client()

    out = client.search_cosmos(
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
def test_slots_need_a_user_and_a_search_across_memories(search):
    client, store = _sync_client()

    out = client.search_cosmos("q", top_k=5, preference_slot=True, procedure_slot=True, **search)

    [call] = store.search.call_args_list
    assert call.kwargs["memory_types"] == ["fact"]
    assert "query_embedding" not in call.kwargs
    assert all("retrieval" not in doc for doc in out)


def test_episodes_stay_in_the_ranked_results_and_never_take_a_slot():
    client, store = _sync_client()

    client.search_cosmos("q", user_id="u1", top_k=5, include_episodes=True, preference_slot=True, procedure_slot=True)

    ranked, preference, procedure = (call.kwargs for call in store.search.call_args_list)
    assert ranked["memory_types"] == ["fact", "episodic"]
    assert preference["memory_types"] == ["fact"]
    assert procedure["memory_types"] == ["procedural"]


def test_summaries_and_turns_still_follow_the_slotted_results_without_repeating_content():
    client, store = _sync_client()
    store.search_summaries.return_value = [
        {"id": "sum_a", "content": PREFERENCE["content"]},
        {"id": "sum_b", "content": "A summary."},
    ]
    store.search_turns.return_value = [
        {"id": "turn_a", "content": "A summary."},
        {"id": "turn_b", "content": "A turn."},
    ]

    out = client.search_cosmos(
        "q",
        user_id="u1",
        top_k=5,
        preference_slot=True,
        procedure_slot=True,
        include_summaries=True,
        include_turns=True,
    )

    assert [doc["id"] for doc in out] == ["fact_0", "fact_1", "fact_2", "pref_a", "proc_a", "sum_b", "turn_b"]
