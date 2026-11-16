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
    ConflictPair,
    attach_conflicts,
    awith_conflicts,
    conflict_link_body,
    conflict_link_id,
    conflict_pairs,
    link_endpoints,
    links_query,
    partners_query,
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

    def test_the_queries_are_scoped_to_the_user_and_ask_only_for_active_facts(self) -> None:
        query, parameters = links_query("u1", ["a", "b"])
        assert "c.user_id = @user_id" in query and "c.type = @type" in query
        assert {p["name"]: p["value"] for p in parameters} == {
            "@user_id": "u1",
            "@type": CONFLICT_LINK_TYPE,
            "@ids": ["a", "b"],
        }

        query, parameters = partners_query("u1", ["a", "b"])
        assert ACTIVE_FILTER in query and "c.type = 'fact'" in query and "c.user_id = @user_id" in query
        assert {p["name"]: p["value"] for p in parameters} == {"@user_id": "u1", "@ids": ["a", "b"]}

    def test_the_endpoints_of_the_links_are_listed_once(self) -> None:
        links = [{"first": "a", "later": "b"}, {"first": "b", "later": "c"}, {"first": None, "later": "a"}]

        assert link_endpoints(links) == ["a", "b", "c"]


class TestAResultShowsItsConflicts:
    partners = {
        "a": {"id": "a", "content": "text of a", "created_at": "2026-01-01T00:00:00Z"},
        "b": {"id": "b", "content": "text of b", "created_at": "2026-02-01T00:00:00Z"},
        "c": {"id": "c", "content": "text of c", "created_at": "2026-03-01T00:00:00Z"},
    }

    def test_each_side_of_a_link_shows_the_other(self) -> None:
        docs = [{"id": "a", "content": "text of a"}, {"id": "b", "content": "text of b"}]

        shown = attach_conflicts(docs, [{"first": "a", "later": "b", "kind": "update"}], self.partners)

        assert shown[0]["conflicts_with"] == [
            {"id": "b", "content": "text of b", "created_at": "2026-02-01T00:00:00Z", "kind": "update"}
        ]
        assert shown[1]["conflicts_with"] == [
            {"id": "a", "content": "text of a", "created_at": "2026-01-01T00:00:00Z", "kind": "update"}
        ]

    def test_at_most_the_newest_partners_are_shown(self) -> None:
        links = [{"first": "a", "later": "b", "kind": "update"}, {"first": "a", "later": "c", "kind": "inconsistent"}]
        partners = {**self.partners, "d": {"id": "d", "content": "text of d", "created_at": "2026-04-01T00:00:00Z"}}
        links.append({"first": "a", "later": "d", "kind": "update"})

        [shown] = attach_conflicts([{"id": "a"}], links, partners)

        assert MAX_CONFLICT_PARTNERS == 2
        assert [entry["id"] for entry in shown["conflicts_with"]] == ["d", "c"]

    def test_a_partner_that_is_no_longer_active_is_not_shown(self) -> None:
        links = [{"first": "a", "later": "gone", "kind": "update"}]

        [shown] = attach_conflicts([{"id": "a"}], links, self.partners)

        assert "conflicts_with" not in shown

    def test_a_fact_with_no_conflicts_is_returned_as_it_was(self) -> None:
        doc = {"id": "z", "content": "text of z"}

        [shown] = attach_conflicts([doc], [{"first": "a", "later": "b", "kind": "update"}], self.partners)

        assert shown is doc

    def test_the_documents_passed_in_are_not_changed(self) -> None:
        doc = {"id": "a"}

        attach_conflicts([doc], [{"first": "a", "later": "b", "kind": "update"}], self.partners)

        assert doc == {"id": "a"}

    def test_a_link_twice_over_shows_its_partner_once(self) -> None:
        links = [{"first": "a", "later": "b", "kind": "update"}, {"first": "b", "later": "a", "kind": "inconsistent"}]

        [shown] = attach_conflicts([{"id": "a"}], links, self.partners)

        assert [entry["id"] for entry in shown["conflicts_with"]] == ["b"]


class Recorder:
    """A stand-in for the store's ``query``: answers the link query and the partner query, and records the calls."""

    def __init__(self, links: list[dict[str, Any]], partners: list[dict[str, Any]], *, fail: bool = False) -> None:
        self.links, self.partners, self.fail = links, partners, fail
        self.calls: list[dict[str, Any]] = []

    def __call__(self, sql: str, parameters: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
        values = {p["name"]: p["value"] for p in parameters}
        self.calls.append({"sql": sql, "user": values["@user_id"], "ids": values["@ids"], **kwargs})
        if self.fail:
            raise RuntimeError("secret-looking failure text")
        return self.links if "c.type = @type" in sql else self.partners


STORED_LINKS = [{"first": "a", "later": "b", "kind": "update"}]
STORED_PARTNERS = [
    {"id": "a", "content": "text of a", "created_at": "2026-01-01T00:00:00Z"},
    {"id": "b", "content": "text of b", "created_at": "2026-02-01T00:00:00Z"},
]


def result(fact_id: str, user_id: str = "u1", kind: str = "fact") -> dict[str, Any]:
    return {"id": fact_id, "user_id": user_id, "type": kind, "content": f"text of {fact_id}"}


class TestLookingUpTheConflictsOfSearchResults:
    def test_a_fact_is_returned_with_the_active_facts_it_conflicts_with(self) -> None:
        query = Recorder(STORED_LINKS, STORED_PARTNERS)

        shown = with_conflicts([result("a"), result("z")], query)

        assert shown[0]["conflicts_with"] == [
            {"id": "b", "content": "text of b", "created_at": "2026-02-01T00:00:00Z", "kind": "update"}
        ]
        assert "conflicts_with" not in shown[1]

    def test_the_links_are_read_from_the_users_conflict_partition_and_the_partners_across_partitions(self) -> None:
        query = Recorder(STORED_LINKS, STORED_PARTNERS)

        with_conflicts([result("a")], query)

        links_call, partners_call = query.calls
        assert links_call["partition_key"] == ["u1", CONFLICT_LINK_THREAD_ID] and links_call["ids"] == ["a"]
        assert partners_call["cross_partition"] is True and partners_call["ids"] == ["a", "b"]

    def test_nothing_is_queried_for_results_that_are_not_facts_or_have_no_user(self) -> None:
        query = Recorder(STORED_LINKS, STORED_PARTNERS)
        docs = [result("t1", kind="turn"), result("e1", kind="episodic"), {"id": "f1", "type": "fact"}]

        assert with_conflicts(docs, query) == docs
        assert query.calls == []

    def test_a_search_with_no_links_costs_one_query(self) -> None:
        query = Recorder([], STORED_PARTNERS)

        docs = [result("a")]

        assert with_conflicts(docs, query) == docs
        assert len(query.calls) == 1

    def test_each_user_is_looked_up_on_their_own(self) -> None:
        query = Recorder(STORED_LINKS, STORED_PARTNERS)

        with_conflicts([result("a", "u1"), result("b", "u2")], query)

        assert [(call["user"], call["ids"]) for call in query.calls if "partition_key" in call] == [
            ("u1", ["a"]),
            ("u2", ["b"]),
        ]

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
        sync = Recorder(STORED_LINKS, STORED_PARTNERS)

        async def query(sql: str, parameters: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
            return sync(sql, parameters, **kwargs)

        shown = await awith_conflicts([result("a")], query)

        assert [entry["id"] for entry in shown[0]["conflicts_with"]] == ["b"]
        assert sync.calls[0]["partition_key"] == ["u1", CONFLICT_LINK_THREAD_ID]

    @pytest.mark.asyncio
    async def test_async_a_failed_lookup_returns_the_results_as_they_were(self) -> None:
        async def query(sql: str, parameters: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
            raise RuntimeError("secret-looking failure text")

        docs = [result("a")]

        assert await awith_conflicts(docs, query) == docs
