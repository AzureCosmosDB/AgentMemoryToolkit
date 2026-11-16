"""Async mirror of ``tests/unit/services/test_conflict_links.py``.

``AsyncPipelineService.reconcile_memories`` links conflicts the same way as the sync pipeline: both facts stay active
and one link document names the pair, a conflicting fact that is combined or folded is linked as the fact that took its
place, and a link that cannot be written neither fails the pass nor holds the cursor back. The fake container, the
scripted model and the scenarios are the sync tests'; here the container is behind awaitable methods as the async
Cosmos client exposes it and, like that client, it rejects ``enable_cross_partition_query`` (a query without a
partition reads every partition without being asked).
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
from azure.cosmos.exceptions import CosmosHttpResponseError

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService
from azure.cosmos.agent_memory.aio.store import AsyncMemoryStore
from tests.unit.services.test_conflict_links import (
    CONFLICTS,
    CURSOR_ID,
    DENVER,
    FLIGHT,
    GATE,
    LAKE,
    LEAVES,
    OLD_FLIGHT,
    PORTLAND,
    RESULT_KEYS,
    RETIREMENT_FIELDS,
    SEATTLE,
    USER,
    V_A,
    V_B,
    V_C,
    V_LAKE,
    V_LEAVES,
    V_OLD,
    V_PORTLAND,
    VECTORS,
    Account,
    Chat,
    Embeddings,
    assert_no_retirement,
    between,
    conflicting_facts,
    cursor_of,
    fact,
    flight_chat,
    lake_verify,
    link_fields,
)


class AsyncRows:
    def __init__(self, rows: Any) -> None:
        self._rows = iter(rows)

    def __aiter__(self) -> AsyncRows:
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return next(self._rows)
        except StopIteration:
            raise StopAsyncIteration


class AsyncAccount:
    """``Account`` behind awaitable methods. Passing ``enable_cross_partition_query`` is a ``TypeError``, as it is for
    the async client."""

    def __init__(self, inner: Account) -> None:
        self.inner = inner

    def query_items(self, *, query: str, parameters: Any = None, partition_key: Any = None) -> AsyncRows:
        return AsyncRows(
            self.inner.query_items(
                query=query,
                parameters=parameters,
                partition_key=partition_key,
                enable_cross_partition_query=partition_key is None,
            )
        )

    async def read_item(self, *, item: str, partition_key: Any) -> dict[str, Any]:
        return self.inner.read_item(item, partition_key)

    async def create_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        return self.inner.create_item(body)

    async def upsert_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        return self.inner.upsert_item(body)

    async def replace_item(self, *, item: str, body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self.inner.replace_item(item, body, **kwargs)


class AsyncEmbeddings(Embeddings):
    async def generate(self, text: str) -> list[float]:  # type: ignore[override]
        return super().generate(text)


class AsyncChat:
    """The scripted chat model behind an awaitable ``generate``."""

    def __init__(self, chat: Chat) -> None:
        self.chat = chat

    @property
    def calls(self) -> list[str]:
        return self.chat.calls

    async def generate(self, messages: list[dict[str, str]], **options: Any) -> str:
        return self.chat.generate(messages, **options)


def service_over(account: Account, chat: Chat) -> tuple[AsyncPipelineService, AsyncChat]:
    embeddings = AsyncEmbeddings(VECTORS)
    containers = {
        ContainerKey.MEMORIES: AsyncAccount(account),
        ContainerKey.TURNS: MagicMock(),
        ContainerKey.SUMMARIES: MagicMock(),
    }
    store = AsyncMemoryStore(containers=containers, embeddings_client=embeddings)
    scripted = AsyncChat(chat)
    return AsyncPipelineService(store, scripted, embeddings, containers=containers), scripted


def a_conflict(kind: str) -> tuple[Account, AsyncChat, AsyncPipelineService]:
    older, newer = CONFLICTS[kind]
    account = Account(conflicting_facts(older, newer))
    service, chat = service_over(account, Chat(between({(older, newer): kind})))
    return account, chat, service


class TestAConflictIsLinkedNotRetired:
    @pytest.mark.parametrize("entry", ["reconcile_memories", "consolidate_memories"])
    @pytest.mark.parametrize("kind", ["update", "inconsistent"])
    async def test_both_facts_stay_active_and_one_link_names_the_pair(self, kind, entry):
        account, chat, service = a_conflict(kind)

        result = await getattr(service, entry)(USER)

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

    async def test_the_result_has_no_contradicted_count_and_the_model_is_asked_only_the_duplicate_check(self):
        account, chat, service = a_conflict("update")

        result = await service.reconcile_memories(USER)

        assert set(result) == RESULT_KEYS and "contradicted" not in result
        assert chat.calls == ["dedup_verify"], "one call judged the pair; nothing judges the facts again"
        assert [call for call in account.queries if "ORDER BY c.created_at DESC" in call] == []

    async def test_the_link_is_written_before_the_cursor_moves_past_the_facts(self):
        account, _, service = a_conflict("update")
        last = account.get("f2")["_ts"]

        await service.reconcile_memories(USER)

        assert cursor_of(account) == (last, ["f2"])
        [link] = account.links()
        assert link["_ts"] < account.get(CURSOR_ID)["_ts"]

    async def test_judging_the_pair_again_in_a_later_pass_rewrites_the_same_document(self):
        account, chat, service = a_conflict("update")
        await service.reconcile_memories(USER)
        [first] = account.links()
        account.delete(CURSOR_ID)  # a pass that checks the facts again, as one whose cursor write lost a race does

        second = await service.reconcile_memories(USER)

        assert second["conflicts"] == 1, "a pair whose link was written again counts"
        [again] = account.links()
        assert again["id"] == first["id"] and link_fields(again) == link_fields(first)
        assert again["_etag"] != first["_etag"], "it was written again, not left as it was"
        assert chat.calls == ["dedup_verify", "dedup_verify"]

    async def test_a_later_verdict_on_the_pair_replaces_the_kind(self):
        older, newer = CONFLICTS["update"]
        account, relation = Account(conflicting_facts(older, newer)), {"now": "update"}
        service, _ = service_over(account, Chat(lambda seed, candidate: relation["now"]))
        await service.reconcile_memories(USER)
        relation["now"] = "inconsistent"
        account.delete(CURSOR_ID)

        await service.reconcile_memories(USER)

        [link] = account.links()
        assert link["kind"] == "inconsistent" and (link["first"], link["later"]) == ("f1", "f2")
        assert account.active_ids() == {"f1", "f2"}

    async def test_the_outcome_log_reports_the_conflicts(self, caplog):
        _, _, service = a_conflict("inconsistent")

        with caplog.at_level(logging.INFO, logger="azure.cosmos.agent_memory.pipeline.aio"):
            result = await service.reconcile_memories(USER)

        [record] = [r for r in caplog.records if r.getMessage() == "reconcile.outcome"]
        assert record.conflicts == result["conflicts"] == 1
        assert not hasattr(record, "contradicted")
        assert (record.facts_checked, record.candidates_considered) == (2, 2)
        assert (record.prompt_id, record.prompt_version) == ("dedup_verify.prompty", "v1")

    @pytest.mark.parametrize("relation", ["related", "unrelated"])
    async def test_a_pair_that_is_not_a_conflict_is_not_linked(self, relation):
        older, newer = CONFLICTS["update"]
        account = Account(conflicting_facts(older, newer))
        service, _ = service_over(account, Chat(between({(older, newer): relation})))

        result = await service.reconcile_memories(USER)

        assert account.links() == [] and result["conflicts"] == 0
        assert account.active_ids() == {"f1", "f2"}


class TestALinkFollowsAFactThatTookAnotherPlace:
    async def test_a_conflicting_fact_that_is_combined_is_linked_as_the_combined_fact(self):
        account = Account([fact("old", OLD_FLIGHT, V_OLD, "2026-03-01T10:00:00Z", thread_id="t0")])
        service, chat = service_over(account, flight_chat())
        await service.reconcile_memories(USER)
        account.put(fact("g", GATE, V_A, "2026-05-01T10:00:00Z", thread_id="t1"))
        account.put(fact("l", LEAVES, V_LEAVES, "2026-05-01T10:05:00Z", thread_id="t2"))

        result = await service.reconcile_memories(USER)

        combined = account.combined()
        assert combined["content"] == FLIGHT
        assert account.retirements() == {"g": (combined["id"], "combined"), "l": (combined["id"], "combined")}
        assert account.active_ids() == {"old", combined["id"]}, "the old departure is not retired for the conflict"
        assert not set(RETIREMENT_FIELDS) & set(account.get("old"))
        [link] = account.links()
        assert (link["first"], link["later"], link["kind"]) == ("old", combined["id"], "update")
        assert (result["conflicts"], result["combined"], result["synthesized"]) == (1, 2, 1)
        assert chat.calls == ["dedup_verify", "combine_memories", "dedup_verify"], "no model call is added"

    async def test_a_conflicting_fact_that_is_folded_as_a_duplicate_is_linked_as_the_fact_it_was_folded_into(self):
        account = Account(
            [
                fact("pdx", PORTLAND, V_PORTLAND, "2026-03-01T10:00:00Z", thread_id="t1"),
                fact("lake", LAKE, V_LAKE, "2026-04-01T10:00:00Z", thread_id="t2"),
                fact("sea", SEATTLE, V_A, "2026-04-10T10:00:00Z", thread_id="t3"),
            ]
        )
        service, chat = service_over(account, Chat(lake_verify))

        result = await service.reconcile_memories(USER)

        assert account.retirements() == {"sea": ("lake", "duplicate")}, "Seattle is folded into the fuller fact"
        assert account.active_ids() == {"pdx", "lake"}, "Portland is not retired for conflicting with it"
        [link] = account.links()
        assert (link["first"], link["later"], link["kind"]) == ("pdx", "lake", "update")
        assert (result["conflicts"], result["merged"], result["kept"]) == (1, 1, 2)
        assert chat.calls == ["dedup_verify", "dedup_verify"]


class TestAPairWithAnAssistantFactIsNotLinked:
    @pytest.mark.parametrize("kind", ["update", "inconsistent"])
    @pytest.mark.parametrize("assistant_is_newer", [True, False], ids=["newer", "older"])
    async def test_the_pair_is_judged_and_both_facts_stay_unlinked(self, kind, assistant_is_newer):
        said = "The assistant noted that the user lives in Seattle."
        user_fact = fact("f1", PORTLAND, V_A, "2026-03-01T10:00:00Z", thread_id="t1")
        agent_fact = fact("f2", said, V_B, "2026-04-01T10:00:00Z", thread_id="t2", agent=True)
        if not assistant_is_newer:
            user_fact["created_at"], agent_fact["created_at"] = agent_fact["created_at"], user_fact["created_at"]
        account = Account([user_fact, agent_fact])
        service, chat = service_over(account, Chat(between({(PORTLAND, said): kind})))

        result = await service.reconcile_memories(USER)

        assert chat.calls == ["dedup_verify"] and result["compared"] == 1, "the pair was judged"
        assert account.links() == [] and result["conflicts"] == 0
        assert_no_retirement(account, "f1", "f2")
        assert account.active_ids() == {"f1", "f2"}


class TestALinkThatCannotBeWrittenNeitherFailsThePassNorHoldsTheCursorBack:
    async def test_the_pass_goes_on_and_the_cursor_moves_past_the_facts(self, caplog):
        account, chat, service = a_conflict("update")
        last = account.get("f2")["_ts"]
        account.link_write_errors.append(CosmosHttpResponseError(status_code=503, message=f"unavailable: {PORTLAND}"))

        with caplog.at_level(logging.INFO, logger="azure.cosmos.agent_memory"):
            result = await service.reconcile_memories(USER)

        assert (result["conflicts"], result["checked"], result["kept"]) == (0, 2, 2)
        assert account.links() == [] and account.active_ids() == {"f1", "f2"}
        assert cursor_of(account) == (last, ["f2"]), "the cursor moved past both facts"
        warning = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
        assert "CosmosHttpResponseError" in warning and "status=503" in warning, "what failed is described"
        assert not [text for text in (PORTLAND, SEATTLE, "unavailable") if text in caplog.text], "no text is logged"

    async def test_the_other_pairs_are_still_written(self):
        account = Account(
            [
                fact("f1", PORTLAND, V_A, "2026-03-01T10:00:00Z", thread_id="t1"),
                fact("f2", DENVER, V_C, "2026-03-15T10:00:00Z", thread_id="t2"),
                fact("f3", SEATTLE, V_B, "2026-04-01T10:00:00Z", thread_id="t3"),
            ]
        )
        service, _ = service_over(account, Chat(lambda seed, candidate: "inconsistent"))
        account.link_write_errors.append(CosmosHttpResponseError(status_code=429, message="throttled"))

        result = await service.reconcile_memories(USER)

        assert result["conflicts"] == 2 and len(account.links()) == 2, "three pairs were judged; one write failed"
        assert account.active_ids() == {"f1", "f2", "f3"}
        assert cursor_of(account)[1] == ["f3"]

    async def test_a_failed_link_is_written_when_the_pair_is_judged_again(self):
        account, _, service = a_conflict("inconsistent")
        account.link_write_errors.append(RuntimeError("the host was recycled"))
        await service.reconcile_memories(USER)
        assert account.links() == []
        account.delete(CURSOR_ID)

        result = await service.reconcile_memories(USER)

        assert result["conflicts"] == 1 and len(account.links()) == 1


class TestTheNoOpTypesReportNoConflicts:
    @pytest.mark.parametrize("entry", ["reconcile_memories", "consolidate_memories"])
    @pytest.mark.parametrize("memory_type", ["episodic", "procedural"])
    async def test_the_result_has_the_full_shape_with_no_conflicts_and_nothing_is_read_or_written(
        self, memory_type, entry
    ):
        account, chat, service = a_conflict("update")
        before = account.get("f1")

        result = await getattr(service, entry)(USER, memory_type=memory_type)

        assert set(result) == RESULT_KEYS and "contradicted" not in result
        assert result["conflicts"] == 0 and sum(result.values()) == 0
        assert account.queries == [] and account.links() == [] and chat.calls == []
        assert account.get("f1") == before
