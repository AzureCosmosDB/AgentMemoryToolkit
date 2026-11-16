"""Conflict links: which pairs a pass links, the link documents, and how a result shows its conflicts."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pytest

from azure.cosmos.agent_memory.services._conflicts import (
    CONFLICT_LINK_THREAD_ID,
    CONFLICT_LINK_TYPE,
    MAX_CONFLICT_PARTNERS,
    MAX_LOOKUP_IDS,
    MAX_REDIRECT_HOPS,
    ConflictPair,
    EndpointResolver,
    attach_conflicts,
    awith_conflicts,
    conflict_link_body,
    conflict_link_id,
    conflict_pairs,
    facts_query,
    link_endpoints,
    links_query,
    with_conflicts,
)
from azure.cosmos.agent_memory.services._dedup import ACTIVE_FILTER, Verdict


def fact(fact_id: str, created_at: str, *, agent: bool = False) -> dict[str, Any]:
    tags = ["sys:fact", "sys:agent-fact"] if agent else ["sys:fact"]
    return {"id": fact_id, "created_at": created_at, "content": f"text of {fact_id}", "tags": tags}


def verdict(candidate: dict[str, Any], relation: str) -> Verdict:
    return Verdict(candidate=candidate, similarity=0.9, relation=relation, seed_missing=(), candidate_missing=())


A, B, C = fact("a", "2026-01-01T00:00:00Z"), fact("b", "2026-02-01T00:00:00Z"), fact("c", "2026-03-01T00:00:00Z")


class TestWhichPairsAreLinked:
    def test_only_an_update_or_an_inconsistency_names_a_conflict(self) -> None:
        relations = ["same_claim", "seed_contains_candidate", "candidate_contains_seed", "partial_overlap", "related"]
        checked = [(C, [verdict(A, relation) for relation in relations] + [verdict(B, "unrelated")])]

        assert conflict_pairs(checked) == []

    @pytest.mark.parametrize("kind", ["update", "inconsistent"])
    def test_the_pair_is_oriented_by_when_each_fact_was_recorded(self, kind: str) -> None:
        # The newer fact is the seed here and the older one the candidate; the link names the older one first.
        assert conflict_pairs([(B, [verdict(A, kind)])]) == [ConflictPair("a", "b", kind)]
        assert conflict_pairs([(A, [verdict(B, kind)])]) == [ConflictPair("a", "b", kind)]

    def test_a_pair_two_seeds_both_judged_is_linked_once(self) -> None:
        checked = [(B, [verdict(A, "update")]), (A, [verdict(B, "inconsistent")])]

        assert conflict_pairs(checked) == [ConflictPair("a", "b", "update")]

    def test_one_seed_can_conflict_with_several_candidates(self) -> None:
        pairs = conflict_pairs([(C, [verdict(A, "inconsistent"), verdict(B, "update")])])

        assert {pair.key for pair in pairs} == {("a", "c"), ("b", "c")}

    def test_a_fact_the_assistant_wrote_is_never_linked(self) -> None:
        said = fact("x", "2026-02-15T00:00:00Z", agent=True)

        assert conflict_pairs([(said, [verdict(A, "inconsistent")])]) == []
        assert conflict_pairs([(A, [verdict(said, "inconsistent")])]) == []

    def test_a_fact_is_not_linked_to_itself(self) -> None:
        assert conflict_pairs([(A, [verdict(A, "inconsistent")])]) == []


class TestAFactRetiredIntoAnotherKeepsItsConflicts:
    def test_the_link_goes_to_the_fact_that_took_its_place(self) -> None:
        # a was combined into the new fact "ab"; its conflict with c is now ab's.
        pairs = conflict_pairs([(C, [verdict(A, "update")])], absorbed={"a": "ab"})

        assert pairs == [ConflictPair("ab", "c", "update")]

    def test_the_orientation_is_the_original_facts(self) -> None:
        # "z" took b's place; the pair still reads b recorded first, c later.
        pairs = conflict_pairs([(C, [verdict(B, "inconsistent")])], absorbed={"b": "z"})

        assert pairs == [ConflictPair("z", "c", "inconsistent")]

    def test_a_pair_whose_sides_were_both_absorbed_into_one_fact_is_dropped(self) -> None:
        assert conflict_pairs([(B, [verdict(A, "update")])], absorbed={"a": "ab", "b": "ab"}) == []

    def test_a_fact_absorbed_in_two_steps_resolves_to_the_last_one(self) -> None:
        pairs = conflict_pairs([(C, [verdict(A, "update")])], absorbed={"a": "a2", "a2": "a3"})

        assert pairs == [ConflictPair("a3", "c", "update")]


class TestTheLinkDocument:
    def test_its_id_is_the_same_whichever_way_the_pair_is_written(self) -> None:
        forward, backward = ConflictPair("a", "b", "update"), ConflictPair("b", "a", "inconsistent")

        assert conflict_link_id("u1", forward) == conflict_link_id("u1", backward)
        assert conflict_link_id("u1", forward) != conflict_link_id("u2", forward)
        assert conflict_link_id("u1", forward).startswith("conflict_")

    def test_it_names_the_facts_and_the_kind_and_holds_no_text(self) -> None:
        now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)

        body = conflict_link_body("u1", ConflictPair("a", "b", "inconsistent"), now=now)

        assert body == {
            "id": conflict_link_id("u1", ConflictPair("a", "b", "inconsistent")),
            "type": CONFLICT_LINK_TYPE,
            "user_id": "u1",
            "thread_id": CONFLICT_LINK_THREAD_ID,
            "first": "a",
            "later": "b",
            "kind": "inconsistent",
            "created_at": "2026-05-01T12:00:00+00:00",
        }

    def test_the_queries_are_scoped_to_the_user_and_the_fact_query_reads_retired_facts_too(self) -> None:
        query, parameters = links_query("u1", ["a", "b"])
        assert "c.user_id = @user_id" in query and "c.type = @type" in query
        assert {p["name"]: p["value"] for p in parameters} == {
            "@user_id": "u1",
            "@type": CONFLICT_LINK_TYPE,
            "@ids": ["a", "b"],
        }

        query, parameters = facts_query("u1", ["a", "b"])
        assert "c.type = 'fact'" in query and "c.user_id = @user_id" in query
        # A retired fact is read too: where it went (superseded_by) is how a conflict follows it.
        assert "c.superseded_by" in query and ACTIVE_FILTER not in query
        assert {p["name"]: p["value"] for p in parameters} == {"@user_id": "u1", "@ids": ["a", "b"]}

    def test_the_endpoints_of_the_links_are_listed_once(self) -> None:
        links = [{"first": "a", "later": "b"}, {"first": "b", "later": "c"}, {"first": None, "later": "a"}]

        assert link_endpoints(links) == ["a", "b", "c"]


def row(fact_id: str, created_at: str, *, superseded_by: str | None = None) -> dict[str, Any]:
    """A fact as the fact query reads it."""
    out = {"id": fact_id, "content": f"text of {fact_id}", "created_at": created_at}
    return {**out, "superseded_by": superseded_by} if superseded_by else out


class TestWhereALinkedFactWent:
    """A link names two facts as they were when it was written; a later pass can retire either into another fact."""

    def test_an_active_fact_stands_for_itself(self) -> None:
        resolver = EndpointResolver(["a", "b"])

        assert resolver.next_ids() == ["a", "b"]
        resolver.add([row("a", "2026-01-01"), row("b", "2026-02-01")])

        assert resolver.next_ids() == []
        assert {fact_id: found["id"] for fact_id, found in resolver.resolved().items()} == {"a": "a", "b": "b"}

    def test_a_retired_fact_stands_for_the_fact_that_took_its_place(self) -> None:
        resolver = EndpointResolver(["a", "m"])
        assert resolver.next_ids() == ["a", "m"]
        resolver.add([row("a", "2026-01-01"), row("m", "2026-02-01", superseded_by="c")])

        assert resolver.next_ids() == ["c"]
        resolver.add([row("c", "2026-02-01")])

        assert resolver.next_ids() == []
        assert resolver.resolved()["m"]["id"] == "c"
        assert resolver.resolved()["a"]["id"] == "a"

    def test_a_fact_that_cannot_be_found_stands_for_nothing_and_is_asked_for_once(self) -> None:
        resolver = EndpointResolver(["a", "gone"])
        assert resolver.next_ids() == ["a", "gone"]
        resolver.add([row("a", "2026-01-01")])

        assert resolver.next_ids() == []
        assert set(resolver.resolved()) == {"a"}

    def test_a_fact_retired_into_one_that_is_gone_stands_for_nothing(self) -> None:
        resolver = EndpointResolver(["m"])
        resolver.next_ids()
        resolver.add([row("m", "2026-01-01", superseded_by="gone")])
        assert resolver.next_ids() == ["gone"]

        assert resolver.next_ids() == [] and resolver.resolved() == {}

    def test_it_follows_up_to_the_hop_limit_and_no_further(self) -> None:
        def chain(length: int) -> EndpointResolver:
            resolver = EndpointResolver(["f0"])
            rows = [row(f"f{i}", "2026-01-01", superseded_by=f"f{i + 1}") for i in range(length)]
            rows.append(row(f"f{length}", "2026-01-01"))
            while wanted := resolver.next_ids():
                resolver.add([r for r in rows if r["id"] in wanted])
            return resolver

        assert chain(MAX_REDIRECT_HOPS).resolved()["f0"]["id"] == f"f{MAX_REDIRECT_HOPS}"
        assert chain(MAX_REDIRECT_HOPS + 1).resolved() == {}

    def test_a_loop_stands_for_nothing_and_ends(self) -> None:
        resolver = EndpointResolver(["m"])
        asked = 0
        while wanted := resolver.next_ids():
            asked += len(wanted)
            resolver.add([row("m", "2026-01-01", superseded_by="n"), row("n", "2026-01-01", superseded_by="m")])

        assert resolver.resolved() == {} and asked <= 2


def fact_doc(fact_id: str, **fields: Any) -> dict[str, Any]:
    return {"id": fact_id, "type": "fact", "content": f"text of {fact_id}", **fields}


class TestAResultShowsItsConflicts:
    # The active fact each linked fact stands for, as resolved from the links.
    active = {
        "a": {"id": "a", "content": "text of a", "created_at": "2026-01-01T00:00:00Z"},
        "b": {"id": "b", "content": "text of b", "created_at": "2026-02-01T00:00:00Z"},
        "c": {"id": "c", "content": "text of c", "created_at": "2026-03-01T00:00:00Z"},
    }

    def test_each_side_of_a_link_shows_the_other(self) -> None:
        docs = [fact_doc("a"), fact_doc("b")]

        shown = attach_conflicts(docs, [{"first": "a", "later": "b", "kind": "update"}], self.active)

        assert shown[0]["conflicts_with"] == [
            {"id": "b", "content": "text of b", "created_at": "2026-02-01T00:00:00Z", "kind": "update"}
        ]
        assert shown[1]["conflicts_with"] == [
            {"id": "a", "content": "text of a", "created_at": "2026-01-01T00:00:00Z", "kind": "update"}
        ]

    def test_at_most_the_newest_partners_are_shown(self) -> None:
        links = [{"first": "a", "later": "b", "kind": "update"}, {"first": "a", "later": "c", "kind": "inconsistent"}]
        active = {**self.active, "d": {"id": "d", "content": "text of d", "created_at": "2026-04-01T00:00:00Z"}}
        links.append({"first": "a", "later": "d", "kind": "update"})

        [shown] = attach_conflicts([fact_doc("a")], links, active)

        assert MAX_CONFLICT_PARTNERS == 2
        assert [entry["id"] for entry in shown["conflicts_with"]] == ["d", "c"]

    def test_a_partner_that_stands_for_no_active_fact_is_not_shown(self) -> None:
        links = [{"first": "a", "later": "gone", "kind": "update"}]

        [shown] = attach_conflicts([fact_doc("a")], links, self.active)

        assert "conflicts_with" not in shown

    def test_a_fact_with_no_conflicts_is_returned_as_it_was(self) -> None:
        doc = fact_doc("z")

        [shown] = attach_conflicts([doc], [{"first": "a", "later": "b", "kind": "update"}], self.active)

        assert shown is doc

    def test_the_documents_passed_in_are_not_changed(self) -> None:
        doc = fact_doc("a")

        attach_conflicts([doc], [{"first": "a", "later": "b", "kind": "update"}], self.active)

        assert doc == fact_doc("a")

    def test_a_link_twice_over_shows_its_partner_once(self) -> None:
        links = [{"first": "a", "later": "b", "kind": "update"}, {"first": "b", "later": "a", "kind": "inconsistent"}]

        [shown] = attach_conflicts([fact_doc("a")], links, self.active)

        assert [entry["id"] for entry in shown["conflicts_with"]] == ["b"]

    def test_the_partner_shown_is_the_fact_that_took_the_linked_fact_s_place(self) -> None:
        # "m" was linked to "a", then a later pass combined it into "c".
        active = {"a": self.active["a"], "m": self.active["c"]}

        [shown] = attach_conflicts([fact_doc("a")], [{"first": "a", "later": "m", "kind": "update"}], active)

        assert shown["conflicts_with"] == [
            {"id": "c", "content": "text of c", "created_at": "2026-03-01T00:00:00Z", "kind": "update"}
        ]

    def test_the_fact_that_took_the_linked_fact_s_place_shows_the_conflict_too(self) -> None:
        # "c" absorbed "m" (it is in c's merged_from), and "m" was linked to "a".
        active = {"a": self.active["a"], "m": self.active["c"]}

        [shown] = attach_conflicts(
            [fact_doc("c", merged_from=["m", "other"])], [{"first": "a", "later": "m", "kind": "inconsistent"}], active
        )

        assert shown["conflicts_with"] == [
            {"id": "a", "content": "text of a", "created_at": "2026-01-01T00:00:00Z", "kind": "inconsistent"}
        ]

    def test_a_conflict_whose_two_sides_ended_up_in_one_fact_is_not_shown(self) -> None:
        active = {"m1": self.active["c"], "m2": self.active["c"]}

        [shown] = attach_conflicts(
            [fact_doc("c", merged_from=["m1", "m2"])], [{"first": "m1", "later": "m2", "kind": "update"}], active
        )

        assert "conflicts_with" not in shown

    def test_only_facts_show_what_the_facts_they_absorbed_conflict_with(self) -> None:
        active = {"a": self.active["a"], "m": self.active["c"]}
        link = {"first": "a", "later": "m", "kind": "update"}
        procedure = {"id": "p", "type": "procedural", "merged_from": ["m"]}

        assert attach_conflicts([procedure], [link], active) == [procedure]

    @pytest.mark.parametrize("merged_from", ["m", None, 7, [None, 3, ""]])
    def test_a_merged_from_that_is_not_a_list_of_ids_is_ignored(self, merged_from: Any) -> None:
        active = {"a": self.active["a"], "m": self.active["c"]}
        doc = fact_doc("c", merged_from=merged_from)

        assert attach_conflicts([doc], [{"first": "a", "later": "m", "kind": "update"}], active) == [doc]


class Recorder:
    """A stand-in for the store's ``query``: answers the link query from ``links`` and the fact query from ``facts``
    (the IDs asked for), and records the calls."""

    def __init__(self, links: list[dict[str, Any]], facts: list[dict[str, Any]], *, fail: bool = False) -> None:
        self.links, self.facts, self.fail = links, facts, fail
        self.calls: list[dict[str, Any]] = []

    def __call__(self, sql: str, parameters: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
        values = {p["name"]: p["value"] for p in parameters}
        self.calls.append({"sql": sql, "user": values["@user_id"], "ids": values["@ids"], **kwargs})
        if self.fail:
            raise RuntimeError("secret-looking failure text")
        asked = set(values["@ids"])
        if "c.type = @type" in sql:
            return [link for link in self.links if {link["first"], link["later"]} & asked]
        return [stored for stored in self.facts if stored["id"] in asked]

    @property
    def fact_reads(self) -> list[list[str]]:
        """The IDs each read of facts asked for."""
        return [call["ids"] for call in self.calls if call.get("cross_partition")]


STORED_LINKS = [{"first": "a", "later": "b", "kind": "update"}]
STORED_FACTS = [row("a", "2026-01-01T00:00:00Z"), row("b", "2026-02-01T00:00:00Z")]
# "m" was linked to "a", and a later pass retired it into "c".
RETIRED_SINCE = [
    row("a", "2026-01-01T00:00:00Z"),
    row("m", "2026-02-01T00:00:00Z", superseded_by="c"),
    row("c", "2026-02-01T00:00:00Z"),
]


def result(fact_id: str, user_id: str = "u1", kind: str = "fact", **fields: Any) -> dict[str, Any]:
    return {"id": fact_id, "user_id": user_id, "type": kind, "content": f"text of {fact_id}", **fields}


class TestLookingUpTheConflictsOfSearchResults:
    def test_a_fact_is_returned_with_the_active_facts_it_conflicts_with(self) -> None:
        query = Recorder(STORED_LINKS, STORED_FACTS)

        shown = with_conflicts([result("a"), result("z")], query)

        assert shown[0]["conflicts_with"] == [
            {"id": "b", "content": "text of b", "created_at": "2026-02-01T00:00:00Z", "kind": "update"}
        ]
        assert "conflicts_with" not in shown[1]

    def test_the_links_are_read_from_the_users_conflict_partition_and_the_facts_across_partitions(self) -> None:
        query = Recorder(STORED_LINKS, STORED_FACTS)

        with_conflicts([result("a")], query)

        links_call, facts_call = query.calls
        assert links_call["partition_key"] == ["u1", CONFLICT_LINK_THREAD_ID] and links_call["ids"] == ["a"]
        assert facts_call["cross_partition"] is True and facts_call["ids"] == ["a", "b"]

    def test_nothing_is_queried_for_results_that_are_not_facts_or_have_no_user(self) -> None:
        query = Recorder(STORED_LINKS, STORED_FACTS)
        docs = [result("t1", kind="turn"), result("e1", kind="episodic"), {"id": "f1", "type": "fact"}]

        assert with_conflicts(docs, query) == docs
        assert query.calls == []

    def test_a_search_with_no_links_costs_one_query(self) -> None:
        query = Recorder([], STORED_FACTS)

        docs = [result("a")]

        assert with_conflicts(docs, query) == docs
        assert len(query.calls) == 1

    def test_a_search_with_links_to_active_facts_costs_two_queries(self) -> None:
        query = Recorder(STORED_LINKS, STORED_FACTS)

        with_conflicts([result("a")], query)

        assert len(query.calls) == 2

    def test_each_user_is_looked_up_on_their_own(self) -> None:
        query = Recorder(STORED_LINKS, STORED_FACTS)

        with_conflicts([result("a", "u1"), result("b", "u2")], query)

        assert [(call["user"], call["ids"]) for call in query.calls if "partition_key" in call] == [
            ("u1", ["a"]),
            ("u2", ["b"]),
        ]

    def test_a_linked_fact_retired_since_is_followed_to_the_fact_that_took_its_place(self) -> None:
        query = Recorder([{"first": "a", "later": "m", "kind": "update"}], RETIRED_SINCE)

        [shown] = with_conflicts([result("a")], query)

        assert shown["conflicts_with"] == [
            {"id": "c", "content": "text of c", "created_at": "2026-02-01T00:00:00Z", "kind": "update"}
        ]
        assert query.fact_reads == [["a", "m"], ["c"]]

    def test_the_fact_that_took_its_place_shows_the_conflict_of_what_it_absorbed(self) -> None:
        query = Recorder([{"first": "a", "later": "m", "kind": "inconsistent"}], RETIRED_SINCE)

        [shown] = with_conflicts([result("c", merged_from=["m"])], query)

        assert query.calls[0]["ids"] == ["c", "m"]
        assert shown["conflicts_with"] == [
            {"id": "a", "content": "text of a", "created_at": "2026-01-01T00:00:00Z", "kind": "inconsistent"}
        ]

    def test_a_conflict_inside_one_combined_fact_is_not_shown(self) -> None:
        facts = [
            row("m1", "2026-01-01T00:00:00Z", superseded_by="c"),
            row("m2", "2026-02-01T00:00:00Z", superseded_by="c"),
            row("c", "2026-02-01T00:00:00Z"),
        ]
        query = Recorder([{"first": "m1", "later": "m2", "kind": "update"}], facts)
        docs = [result("c", merged_from=["m1", "m2"])]

        assert with_conflicts(docs, query) == docs

    def test_a_linked_fact_retired_into_one_that_is_gone_shows_no_conflict(self) -> None:
        facts = [row("a", "2026-01-01T00:00:00Z"), row("m", "2026-02-01T00:00:00Z", superseded_by="gone")]
        docs = [result("a")]

        assert with_conflicts(docs, Recorder([{"first": "a", "later": "m", "kind": "update"}], facts)) == docs

    def test_what_a_result_absorbed_is_looked_up_after_the_results_themselves_and_only_up_to_a_limit(self) -> None:
        query = Recorder([], [])

        with_conflicts([result("c", merged_from=[f"m{i}" for i in range(MAX_LOOKUP_IDS * 2)]), result("d")], query)

        ids = query.calls[0]["ids"]
        assert ids[:2] == ["c", "d"] and len(ids) == MAX_LOOKUP_IDS

    def test_a_loop_of_retirements_ends_without_a_conflict(self) -> None:
        facts = [
            row("a", "2026-01-01T00:00:00Z"),
            row("m", "2026-02-01T00:00:00Z", superseded_by="n"),
            row("n", "2026-02-01T00:00:00Z", superseded_by="m"),
        ]
        query = Recorder([{"first": "a", "later": "m", "kind": "update"}], facts)
        docs = [result("a")]

        assert with_conflicts(docs, query) == docs
        assert len(query.calls) <= 2 + MAX_REDIRECT_HOPS

    def test_a_failed_lookup_returns_the_results_as_they_were_and_logs_no_text(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        docs = [result("a")]

        with caplog.at_level(logging.WARNING):
            shown = with_conflicts(docs, Recorder([], [], fail=True))

        assert shown == docs
        assert "secret-looking failure text" not in caplog.text
        assert "conflicts_with" in caplog.text

    @pytest.mark.asyncio
    async def test_async_a_fact_is_returned_with_the_active_facts_it_conflicts_with(self) -> None:
        sync = Recorder(STORED_LINKS, STORED_FACTS)

        async def query(sql: str, parameters: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
            return sync(sql, parameters, **kwargs)

        shown = await awith_conflicts([result("a")], query)

        assert [entry["id"] for entry in shown[0]["conflicts_with"]] == ["b"]
        assert sync.calls[0]["partition_key"] == ["u1", CONFLICT_LINK_THREAD_ID]

    @pytest.mark.asyncio
    async def test_async_a_linked_fact_retired_since_is_followed_to_the_fact_that_took_its_place(self) -> None:
        sync = Recorder([{"first": "a", "later": "m", "kind": "update"}], RETIRED_SINCE)

        async def query(sql: str, parameters: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
            return sync(sql, parameters, **kwargs)

        [shown] = await awith_conflicts([result("a")], query)

        assert [entry["id"] for entry in shown["conflicts_with"]] == ["c"]
        assert sync.fact_reads == [["a", "m"], ["c"]]

    @pytest.mark.asyncio
    async def test_async_the_fact_that_took_its_place_shows_the_conflict_of_what_it_absorbed(self) -> None:
        sync = Recorder([{"first": "a", "later": "m", "kind": "inconsistent"}], RETIRED_SINCE)

        async def query(sql: str, parameters: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
            return sync(sql, parameters, **kwargs)

        [shown] = await awith_conflicts([result("c", merged_from=["m"])], query)

        assert [entry["id"] for entry in shown["conflicts_with"]] == ["a"]
        assert sync.calls[0]["ids"] == ["c", "m"]

    @pytest.mark.asyncio
    async def test_async_a_loop_of_retirements_ends_without_a_conflict(self) -> None:
        facts = [
            row("a", "2026-01-01T00:00:00Z"),
            row("m", "2026-02-01T00:00:00Z", superseded_by="n"),
            row("n", "2026-02-01T00:00:00Z", superseded_by="m"),
        ]
        sync = Recorder([{"first": "a", "later": "m", "kind": "update"}], facts)

        async def query(sql: str, parameters: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
            return sync(sql, parameters, **kwargs)

        docs = [result("a")]

        assert await awith_conflicts(docs, query) == docs
        assert len(sync.calls) <= 2 + MAX_REDIRECT_HOPS

    @pytest.mark.asyncio
    async def test_async_a_failed_lookup_returns_the_results_as_they_were(self) -> None:
        async def query(sql: str, parameters: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
            raise RuntimeError("secret-looking failure text")

        docs = [result("a")]

        assert await awith_conflicts(docs, query) == docs
