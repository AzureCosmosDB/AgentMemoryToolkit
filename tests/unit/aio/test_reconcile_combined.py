"""Async mirror of ``tests/unit/services/test_reconcile_combined.py``.

``AsyncPipelineService.reconcile_memories`` runs the same passes, so a combined fact is in its contradiction pool and
its members come back when it loses. The fake container, the scripted model and the facts are the sync tests'; here the
container is behind awaitable methods as the async Cosmos client exposes it, and, like that client, it rejects
``enable_cross_partition_query`` (a query without a partition reads every partition without being asked).
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
from azure.core import MatchConditions
from azure.cosmos.exceptions import CosmosHttpResponseError

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService
from azure.cosmos.agent_memory.aio.store import AsyncMemoryStore
from azure.cosmos.agent_memory.services._combine import COMBINED_TAG
from tests.unit.services.test_reconcile_combined import (
    AT_EIGHT,
    BOTH,
    CONCURRENT_CHANGES,
    CORRECTION,
    CRASH,
    DEPARTURE_AND_SEAT,
    FLIGHT,
    GATE,
    GYM,
    INTERRUPTIONS,
    LEAVES,
    LINE_BREAKS,
    MOVED,
    OLD_FLIGHT,
    PLACE,
    RETIREMENT_FIELDS,
    TIME,
    USER,
    V_MOVED,
    V_PLACE,
    V_TIME,
    VECTORS,
    Account,
    Embeddings,
    Model,
    assert_left_as_the_other_writer_left_it,
    correction_fact,
    departure_fact,
    dinner_facts,
    dinner_model,
    fact,
    flight_model,
    hostile_pool,
    moved_fact,
    raise_unavailable,
    seat_fact,
    seat_model,
    without_system_fields,
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

    async def delete_item(self, *, item: str, partition_key: Any) -> None:
        self.inner.delete_item(item, partition_key)


class AsyncEmbeddings(Embeddings):
    async def generate(self, text: str) -> list[float]:  # type: ignore[override]
        return super().generate(text)


class AsyncChat:
    """Keeps the messages it is sent and answers that nothing contradicts."""

    def __init__(self) -> None:
        self.messages: list[list[dict[str, str]]] = []

    async def generate(self, messages: list[dict[str, str]], **options: Any) -> str:
        del options
        self.messages.append(messages)
        return json.dumps({"contradicted_pairs": [], "kept_ids": []})


def service_over(account: Account, model: Model | None, *, chat: Any = None) -> AsyncPipelineService:
    embeddings = AsyncEmbeddings(VECTORS)
    containers = {
        ContainerKey.MEMORIES: AsyncAccount(account),
        ContainerKey.TURNS: MagicMock(),
        ContainerKey.SUMMARIES: MagicMock(),
    }
    store = AsyncMemoryStore(containers=containers, embeddings_client=embeddings)
    service = AsyncPipelineService(store, chat, embeddings, containers=containers)
    if chat is None:

        async def run_prompty(filename: str, inputs: dict[str, Any]) -> str:
            return model(filename, inputs)  # type: ignore[misc]

        service._run_prompty = run_prompty  # type: ignore[method-assign]
    return service


async def combined_dinner() -> tuple[Account, AsyncPipelineService, Model, dict[str, Any]]:
    account, model = Account(dinner_facts()), dinner_model()
    service = service_over(account, model)
    first = await service.reconcile_memories(USER)
    assert (first["synthesized"], first["combined"]) == (1, 2)
    return account, service, model, account.combined()


async def first_departure(n: int = 50) -> tuple[Account, AsyncPipelineService, Model]:
    """The old departure alone, checked by a first run."""
    account, model = Account([departure_fact()]), seat_model()
    service = service_over(account, model)
    await service.reconcile_memories(USER, n)
    return account, service, model


class TestACombinedFactCanBeCorrected:
    async def test_a_newer_contradicting_fact_retires_the_combined_fact_and_hands_back_its_members(self):
        account, service, model, combined = await combined_dinner()
        originals = {doc["id"]: doc for doc in dinner_facts()}
        assert account.active_ids() == {combined["id"]} and model.pools == []
        assert account.retirements() == {"f1": (combined["id"], "combined"), "f2": (combined["id"], "combined")}
        account.put(moved_fact())

        result = await service.reconcile_memories(USER)

        assert model.pooled() == {MOVED, BOTH}
        assert (result["contradicted"], result["kept"], result["merged"]) == (1, 1, 0)
        assert (result["combined"], result["synthesized"], result["combine_failed"]) == (0, 0, 0)
        assert account.retirements() == {combined["id"]: ("n1", "contradict")}
        assert account.active_ids() == {"f1", "f2", "n1"}
        for fid, original in originals.items():
            restored = account.get(fid)
            assert not set(RETIREMENT_FIELDS) & set(restored), "nothing of the retirement is left on the record"
            assert without_system_fields(restored) == original

    async def test_a_later_pass_retires_only_the_member_the_new_fact_contradicts(self):
        account, service, model, combined = await combined_dinner()
        account.put(moved_fact())
        await service.reconcile_memories(USER)

        third = await service.reconcile_memories(USER)

        assert model.pooled() == {MOVED, TIME, PLACE}
        assert third["contradicted"] == 1
        assert account.active_ids() == {"f2", "n1"}
        assert account.retirements() == {"f1": ("n1", "contradict"), combined["id"]: ("n1", "contradict")}
        assert (await service.reconcile_memories(USER))["contradicted"] == 0 and account.active_ids() == {"f2", "n1"}

    async def test_members_that_form_the_same_group_again_are_not_hidden_by_the_combined_fact_they_name(self):
        account, service, _, combined = await combined_dinner()
        account.put(moved_fact())
        await service.reconcile_memories(USER)

        third = await service.reconcile_memories(USER)

        counts = (third["combine_groups"], third["combine_failed"], third["synthesized"], third["combined"])
        assert counts == (1, 1, 0, 0)
        assert COMBINED_TAG in account.combined()["tags"] and account.combined()["id"] == combined["id"]
        assert account.retirements().get("f2") is None, "a member is not retired into the retired combined fact"
        assert account.retirements()["f1"] == ("n1", "contradict"), "retired as contradicted, not hidden as combined"
        assert "f2" in account.active_ids()

    async def test_an_older_fact_is_retired_by_newer_facts_combined_in_the_same_run(self):
        account, model = Account([fact("old", OLD_FLIGHT, V_MOVED, "2026-03-01T10:00:00Z")]), flight_model()
        service = service_over(account, model)
        await service.reconcile_memories(USER)
        account.put(fact("g", GATE, V_PLACE, "2026-05-01T10:00:00Z"))
        account.put(fact("l", LEAVES, V_TIME, "2026-05-01T10:05:00Z"))

        result = await service.reconcile_memories(USER)

        combined = account.combined()
        assert combined["content"] == FLIGHT
        assert (result["synthesized"], result["combined"], result["contradicted"]) == (1, 2, 1)
        assert model.pooled() == {FLIGHT, OLD_FLIGHT}
        assert account.retirements() == {
            "g": (combined["id"], "combined"),
            "l": (combined["id"], "combined"),
            "old": (combined["id"], "contradict"),
        }
        assert account.active_ids() == {combined["id"]}

    async def test_the_pool_holds_combined_facts_but_not_agent_facts(self):
        account, service, model, combined = await combined_dinner()
        agent_says = "The user's dinner with Alice and Bob was suggested for May 3 by the assistant."
        agent = {"metadata": {"source": "agent", "category": "other"}, "tags": ["sys:fact", "sys:agent-fact"]}
        account.put(fact("a1", agent_says, [0.0, 0.0, 1.0], "2026-04-21T10:00:00Z", **agent))
        account.put(moved_fact())

        await service.reconcile_memories(USER)

        assert model.pooled() == {MOVED, BOTH}, "the combined fact is in; the agent's own fact is not"
        assert account.retirements() == {combined["id"]: ("n1", "contradict")}


class TestACombinedFactIsDatedByItsOldestMember:
    async def test_a_correction_dated_between_its_members_beats_it_and_the_members_come_back(self):
        account, service, model = await first_departure()
        account.put(correction_fact("2026-05-20T09:00:00Z"))
        account.put(seat_fact())

        result = await service.reconcile_memories(USER)

        combined = account.combined()
        assert combined["content"] == DEPARTURE_AND_SEAT and (result["synthesized"], result["combined"]) == (1, 2)
        assert combined["created_at"] == "2026-05-25T09:00:00Z", "it stays among the newest facts, so the pool reads it"
        assert model.shown() == {combined["id"]: "2026-05-01T09:00:00Z", "n1": "2026-05-20T09:00:00Z"}
        assert (result["contradicted"], result["kept"]) == (1, 1)
        assert account.retirements() == {combined["id"]: ("n1", "contradict")}
        assert account.active_ids() == {"m1", "m2", "n1"}, "the correction is not lost, and the members are back"

        await service.reconcile_memories(USER)

        assert account.retirements() == {combined["id"]: ("n1", "contradict"), "m1": ("n1", "contradict")}
        assert account.active_ids() == {"m2", "n1"}, "the old departure is retired once it is judged by itself"

    async def test_a_correction_judged_without_the_old_departure_is_not_lost_when_combining_pulls_it_back_in(self):
        account, service, model = await first_departure(n=2)
        account.put(fact("gym", GYM, [0.0, 0.0, 1.0], "2026-05-10T09:00:00Z", thread_id="t4"))
        await service.reconcile_memories(USER, 2)
        account.put(correction_fact("2026-05-20T09:00:00Z"))
        await service.reconcile_memories(USER, 2)
        assert model.pooled() == {CORRECTION, GYM}, "the pool holds two facts: the old departure was left out of it"
        assert account.active_ids() == {"m1", "gym", "n1"}, "so both departures stayed active"
        account.put(seat_fact())

        result = await service.reconcile_memories(USER, 2)  # combining pulls the old departure back in

        combined = account.combined()
        assert (result["synthesized"], result["combined"], result["contradicted"]) == (1, 2, 1)
        assert model.pooled() == {DEPARTURE_AND_SEAT, CORRECTION}
        assert account.retirements() == {combined["id"]: ("n1", "contradict")}
        assert account.active_ids() == {"m1", "m2", "n1", "gym"}, "the correction is not retired; the members are back"

    async def test_a_fact_older_than_every_member_loses_to_the_combined_fact(self):
        account, service, _ = await first_departure()
        account.put(correction_fact("2026-04-20T09:00:00Z"))  # said before the old departure
        account.put(seat_fact())

        result = await service.reconcile_memories(USER)

        combined = account.combined()
        assert (result["synthesized"], result["combined"], result["contradicted"]) == (1, 2, 1)
        assert account.retirements() == {
            "m1": (combined["id"], "combined"),
            "m2": (combined["id"], "combined"),
            "n1": (combined["id"], "contradict"),
        }
        assert account.active_ids() == {combined["id"]}


class TestARestoreThatIsRaced:
    @pytest.mark.parametrize("change", list(CONCURRENT_CHANGES))
    async def test_a_member_that_changed_since_it_was_read_is_left_as_the_other_writer_left_it(self, change):
        account, service, _, combined = await combined_dinner()
        account.put(moved_fact())
        account.before_replace["f1"] = lambda: CONCURRENT_CHANGES[change](account)

        result = await service.reconcile_memories(USER)

        assert result["contradicted"] == 1 and account.retirements()[combined["id"]] == ("n1", "contradict")
        assert "f2" in account.active_ids(), "the member nothing happened to is restored"
        assert_left_as_the_other_writer_left_it(change, account.get("f1"), combined["id"])

    async def test_a_combined_fact_that_changed_while_the_model_judged_is_left_for_a_later_pass(self):
        account, service, model, combined = await combined_dinner()
        account.put(moved_fact())
        judged = model.judge

        def judge_while_another_writer_writes(pool):
            account.put(account.get(combined["id"]))  # another client's write: a new ETag, the same text
            return judged(pool)

        model.judge = judge_while_another_writer_writes

        result = await service.reconcile_memories(USER)

        assert (result["contradicted"], result["kept"]) == (0, 2)
        assert account.retirements() == {"f1": (combined["id"], "combined"), "f2": (combined["id"], "combined")}
        assert account.active_ids() == {combined["id"], "n1"}, "no member was written back"

        model.judge = judged
        later = await service.reconcile_memories(USER)

        assert later["contradicted"] == 1
        assert account.retirements() == {combined["id"]: ("n1", "contradict")}
        assert account.active_ids() == {"f1", "f2", "n1"}

    async def test_a_retirement_that_fails_leaves_the_restored_members_active_and_uncounted(self):
        account, service, _, combined = await combined_dinner()
        account.put(moved_fact())
        account.before_replace[combined["id"]] = lambda: account.put(account.get(combined["id"]))

        result = await service.reconcile_memories(USER)

        assert (result["contradicted"], result["kept"], result["merged"]) == (0, 2, 0)
        assert account.retirements() == {}
        assert account.active_ids() == {"f1", "f2", "n1", combined["id"]}, "nothing is lost"

    async def test_a_member_that_cannot_be_restored_keeps_the_combined_fact_and_its_pair_for_a_later_pass(self):
        account, service, _, combined = await combined_dinner()
        account.put(moved_fact())

        def unavailable() -> None:
            raise CosmosHttpResponseError(status_code=503, message="Service unavailable")

        account.before_replace["f2"] = unavailable

        result = await service.reconcile_memories(USER)

        assert (result["contradicted"], result["kept"]) == (0, 2)
        assert account.retirements() == {"f2": (combined["id"], "combined")}, "the combined fact is not retired"
        assert account.active_ids() == {"f1", "n1", combined["id"]}

    async def test_what_was_folded_into_the_combined_fact_comes_back_with_its_members_but_earlier_versions_stay_retired(
        self,
    ):
        account, service, _, combined = await combined_dinner()
        retirement = {"superseded_by": combined["id"], "superseded_at": "2026-04-05T10:00:00Z"}
        duplicate = fact("d1", "The user dines with Alice and Bob on May 3.", [0.0, 0.7, 0.7], "2026-04-05T10:00:00Z")
        older = fact("o1", "The user's dinner with Alice and Bob is on May 1.", [0.0, 0.7, 0.7], "2026-03-01T10:00:00Z")
        earlier = fact("e1", "The user's dinner with Alice and Bob is May 2.", [0.0, 0.7, 0.7], "2026-03-02T10:00:00Z")
        account.put({**duplicate, **retirement, "supersede_reason": "duplicate"})
        account.put({**older, **retirement, "supersede_reason": "contradict"})
        account.put({**earlier, **retirement, "supersede_reason": "update"})
        account.put(moved_fact())

        await service.reconcile_memories(USER)

        assert account.retirements() == {
            combined["id"]: ("n1", "contradict"),
            "o1": (combined["id"], "contradict"),
            "e1": (combined["id"], "update"),
        }
        assert account.active_ids() == {"f1", "f2", "d1", "n1"}, "a duplicate comes back; an earlier version does not"

    async def test_only_the_members_of_the_fact_that_lost_are_restored(self):
        account, service, _, combined = await combined_dinner()
        elsewhere = fact("c2", "The user's gym membership is paid until June.", [0.0, 0.0, 1.0], "2026-04-10T10:00:00Z")
        elsewhere["tags"] = ["sys:fact", COMBINED_TAG]
        account.put(elsewhere)
        retired = {"superseded_by": "c2", "supersede_reason": "combined", "superseded_at": "2026-04-10T10:00:00Z"}
        member = fact("m1", "The user pays for the gym monthly.", [0.0, 0.7, 0.7], "2026-04-09T10:00:00Z", **retired)
        account.put(member)
        stranger = fact("s1", "Someone else's fact.", [0.0, 0.7, 0.7], "2026-04-09T10:00:00Z", user_id="u2")
        account.put({**stranger, **retired, "superseded_by": combined["id"]})
        account.put(moved_fact())

        await service.reconcile_memories(USER)

        assert account.retirements() == {
            combined["id"]: ("n1", "contradict"),
            "m1": ("c2", "combined"),
            "s1": (combined["id"], "combined"),
        }
        assert account.active_ids() == {"f1", "f2", "n1", "c2"}

    async def test_the_pass_reads_members_across_partitions_and_writes_each_back_conditionally(self):
        account, service, _, combined = await combined_dinner()
        account.put(moved_fact())
        written: list[tuple[str, Any, Any]] = []
        replace = account.replace_item

        def watch(item: str, body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
            written.append((item, kwargs.get("etag"), kwargs.get("match_condition")))
            return replace(item, body, **kwargs)

        account.replace_item = watch  # type: ignore[method-assign]
        before = {fid: account.get(fid)["_etag"] for fid in ("f1", "f2")}

        await service.reconcile_memories(USER)

        restores = [entry for entry in written if entry[0] in before]
        assert restores == [(fid, before[fid], MatchConditions.IfNotModified) for fid in ("f1", "f2")]
        members_query = next(q for q in account.queries if "supersede_reason IN" in q)
        assert "c.user_id = @user_id" in members_query and "c.superseded_by = @memory_id" in members_query


class TestNothingIsLostWhenARestoreIsInterrupted:
    @pytest.mark.parametrize("interruption", list(INTERRUPTIONS))
    async def test_the_next_pass_folds_the_members_back_in_and_they_come_back_when_it_loses_again(self, interruption):
        account, service, _, combined = await combined_dinner()
        account.put(moved_fact())
        INTERRUPTIONS[interruption](account, combined["id"])

        with pytest.raises(RuntimeError) if interruption == CRASH else contextlib.nullcontext():
            await service.reconcile_memories(USER)
        assert combined["id"] in account.active_ids(), "the combined fact was not retired"

        second = await service.reconcile_memories(USER)

        assert second["merged"] >= 1, "the members that were written back were folded into the combined fact"
        assert account.retirements() == {combined["id"]: ("n1", "contradict")}
        assert account.active_ids() == {"f1", "f2", "n1"}, "nothing was lost: both dinner details and the move"

        await service.reconcile_memories(USER)

        assert account.retirements() == {combined["id"]: ("n1", "contradict"), "f1": ("n1", "contradict")}
        assert account.active_ids() == {"f2", "n1"}, "7pm and May 5 stay; the date that was moved is retired"

    async def test_a_member_whose_retirement_failed_while_combining_comes_back_when_the_combined_fact_loses(self):
        account, model = Account(dinner_facts()), dinner_model()
        service = service_over(account, model)
        account.before_replace["f2"] = raise_unavailable  # the second member is not retired into the new fact

        first = await service.reconcile_memories(USER)

        combined = account.combined()
        assert (first["synthesized"], first["combined"]) == (1, 1)
        assert account.retirements() == {"f1": (combined["id"], "combined")}, "f2 is still active beside the fact"

        second = await service.reconcile_memories(USER)  # a later pass folds it into the fact that holds its detail

        assert second["merged"] == 1
        assert account.retirements() == {"f1": (combined["id"], "combined"), "f2": (combined["id"], "duplicate")}
        account.put(moved_fact())

        await service.reconcile_memories(USER)

        assert account.retirements() == {combined["id"]: ("n1", "contradict")}
        assert account.active_ids() == {"f1", "f2", "n1"}, "the member folded in comes back with the other"


async def update_through_the_store(account: Account, record: dict[str, Any], **changes: Any) -> None:
    """What an app does with ``update_cosmos``: the async memory store's own update of a record. It edits only that
    record, and leaves its ``content_hash`` as it was."""
    containers = {
        ContainerKey.MEMORIES: AsyncAccount(account),
        ContainerKey.TURNS: MagicMock(),
        ContainerKey.SUMMARIES: MagicMock(),
    }
    store = AsyncMemoryStore(containers=containers, embeddings_client=AsyncEmbeddings(VECTORS))
    await store.update(
        record["id"], user_id=record["user_id"], thread_id=record["thread_id"], memory_type="fact", **changes
    )


class TestAnEditedCombinedFactIsRetiredAlone:
    async def test_the_members_stay_retired_and_nothing_is_written_back(self, caplog):
        account, service, _, combined = await combined_dinner()
        await update_through_the_store(account, combined, content=AT_EIGHT)
        account.put(moved_fact())
        members = {fid: account.get(fid) for fid in ("f1", "f2")}

        with caplog.at_level(logging.INFO, logger="azure.cosmos.agent_memory"):
            result = await service.reconcile_memories(USER)

        assert (result["contradicted"], result["kept"]) == (1, 1)
        assert account.retirements() == {
            "f1": (combined["id"], "combined"),
            "f2": (combined["id"], "combined"),
            combined["id"]: ("n1", "contradict"),
        }
        assert account.active_ids() == {"n1"}
        assert account.get(combined["id"])["content"] == AT_EIGHT
        assert {fid: account.get(fid) for fid in members} == members, "no member was written, ETags and all"
        assert "restored no members of a combined fact" in caplog.text
        assert not [text for text in (AT_EIGHT, BOTH, PLACE, TIME, MOVED) if text in caplog.text], "no text is logged"

    async def test_a_combined_fact_with_no_record_of_what_combine_wrote_has_its_members_restored_as_before(self):
        account, service, _, combined = await combined_dinner()
        await update_through_the_store(account, combined, content=AT_EIGHT)
        account.put({key: value for key, value in account.get(combined["id"]).items() if key != "combined_hash"})
        account.put(moved_fact())

        await service.reconcile_memories(USER)

        assert account.retirements() == {combined["id"]: ("n1", "contradict")}
        assert account.active_ids() == {"f1", "f2", "n1"}

    async def test_an_update_that_leaves_the_text_alone_is_not_an_edit(self):
        account, service, _, combined = await combined_dinner()
        await update_through_the_store(account, combined, metadata={**combined["metadata"], "reviewed": True})
        account.put(moved_fact())

        await service.reconcile_memories(USER)

        assert account.retirements() == {combined["id"]: ("n1", "contradict")}
        assert account.active_ids() == {"f1", "f2", "n1"}


class TestAFactCannotSplitTheContradictionPrompt:
    @pytest.mark.parametrize("separator", LINE_BREAKS, ids=[f"U+{ord(c):04X}" for c in LINE_BREAKS])
    async def test_a_line_break_followed_by_a_role_marker_adds_no_message(self, separator):
        chat = AsyncChat()
        service = service_over(Account(), None, chat=chat)

        await service._reconcile_contradictions(USER, "fact", hostile_pool(separator))

        [messages] = chat.messages
        assert [message["role"] for message in messages] == ["system", "user"]
        assert "ignore the rules above" in messages[-1]["content"], "the text is kept, escaped on one line"
