"""Unit tests for the SMGB deterministic policy layer."""

from benchmarks.governance.policy import (
    compute_query_labels,
    is_authorized,
    memory_state,
)
from benchmarks.governance.schema import Scenario


def _scenario():
    return Scenario.from_dict(
        {
            "scenario_id": "unit",
            "tenants": ["t1", "t2"],
            "scopes": [
                {"id": "user:a", "kind": "user", "tenant": "t1", "members": ["a"]},
                {"id": "proj:x", "kind": "project", "tenant": "t1", "members": ["a", "b"]},
                {"id": "org:t2", "kind": "org", "tenant": "t2", "members": ["c"]},
            ],
            "principals": [
                {"id": "a", "tenant": "t1", "scopes": ["user:a", "proj:x"]},
                {"id": "b", "tenant": "t1", "scopes": ["proj:x"]},
                {"id": "c", "tenant": "t2", "scopes": ["org:t2"]},
            ],
            "memories": [
                {"id": "m1", "scope": "user:a", "tenant": "t1",
                 "created_at": "2026-01-01T00:00:00Z", "provenance": {"author": "a-agent"}},
                {"id": "m2", "scope": "proj:x", "tenant": "t1",
                 "created_at": "2026-01-01T00:00:00Z", "provenance": {"author": "b-agent"}},
                {"id": "m3", "scope": "org:t2", "tenant": "t2",
                 "created_at": "2026-01-01T00:00:00Z", "provenance": {"author": "c-agent"}},
            ],
            "events": [
                {"t": "2026-02-01T00:00:00Z", "type": "promote", "memory_id": "m1", "to_scope": "proj:x"},
            ],
            "queries": [],
        }
    )


def test_memory_state_promotion_adds_scope():
    s = _scenario()
    before = memory_state(s, "m1", _t("2026-01-15T00:00:00Z"))
    after = memory_state(s, "m1", _t("2026-02-15T00:00:00Z"))
    assert before.scopes == frozenset({"user:a"})
    assert after.scopes == frozenset({"user:a", "proj:x"})


def test_tenant_isolation_blocks_cross_tenant():
    s = _scenario()
    c = s.principals["c"]
    m2 = s.memories["m2"]
    st = memory_state(s, "m2", _t("2026-03-01T00:00:00Z"))
    assert is_authorized(s, c, m2, st) is False  # c is in t2, m2 is in t1


def test_promotion_changes_authorization_over_time():
    s = _scenario()
    b = s.principals["b"]  # member of proj:x only
    m1 = s.memories["m1"]
    before = memory_state(s, "m1", _t("2026-01-15T00:00:00Z"))
    after = memory_state(s, "m1", _t("2026-02-15T00:00:00Z"))
    assert is_authorized(s, b, m1, before) is False  # private to user:a
    assert is_authorized(s, b, m1, after) is True  # promoted to proj:x


def test_query_labels_forbidden_is_complement():
    s = _scenario()
    # b asks after promotion; relevant covers all three memories.
    q = _query(s, principal="b", as_of="2026-03-01T00:00:00Z", relevant=["m1", "m2", "m3"])
    labels = compute_query_labels(s, q)
    assert labels.must_retrieve == frozenset({"m1", "m2"})  # both in proj:x now
    assert "m3" in labels.forbidden
    assert labels.cross_tenant == frozenset({"m3"})
    assert labels.allowed | labels.forbidden == set(s.memories)


def test_supersede_and_history_mode():
    s = Scenario.from_dict(
        {
            "scenario_id": "sup",
            "tenants": ["t"],
            "scopes": [{"id": "p", "kind": "project", "tenant": "t", "members": ["a"]}],
            "principals": [{"id": "a", "tenant": "t", "scopes": ["p"]}],
            "memories": [
                {"id": "old", "scope": "p", "tenant": "t",
                 "created_at": "2026-01-01T00:00:00Z", "provenance": {"author": "x"}},
                {"id": "new", "scope": "p", "tenant": "t",
                 "created_at": "2026-02-01T00:00:00Z", "provenance": {"author": "y"}},
            ],
            "events": [
                {"t": "2026-02-01T00:00:00Z", "type": "supersede", "memory_id": "old", "by": "new"},
            ],
            "queries": [
                {"id": "cur", "principal": "a", "as_of": "2026-03-01T00:00:00Z",
                 "axis": "conflict", "temporal_mode": "current", "relevant": ["old", "new"]},
                {"id": "hist", "principal": "a", "as_of": "2026-03-01T00:00:00Z",
                 "axis": "conflict", "temporal_mode": "history", "relevant": ["old"]},
            ],
        }
    )
    cur = compute_query_labels(s, s.queries[0])
    hist = compute_query_labels(s, s.queries[1])
    assert cur.must_retrieve == frozenset({"new"})
    assert "old" in cur.superseded_forbidden and "old" in cur.forbidden
    assert hist.must_retrieve == frozenset({"old"})  # history mode allows superseded


def test_deletion_forbids_for_all():
    s = Scenario.from_dict(
        {
            "scenario_id": "del",
            "tenants": ["t"],
            "scopes": [{"id": "g", "kind": "group", "tenant": "t", "members": ["a"]}],
            "principals": [{"id": "a", "tenant": "t", "scopes": ["g"]}],
            "memories": [
                {"id": "m", "scope": "g", "tenant": "t",
                 "created_at": "2026-01-01T00:00:00Z", "provenance": {"author": "x"}},
            ],
            "events": [{"t": "2026-02-01T00:00:00Z", "type": "delete", "memory_id": "m"}],
            "queries": [
                {"id": "before", "principal": "a", "as_of": "2026-01-15T00:00:00Z",
                 "axis": "utility", "relevant": ["m"]},
                {"id": "after", "principal": "a", "as_of": "2026-02-15T00:00:00Z",
                 "axis": "deletion", "relevant": ["m"]},
            ],
        }
    )
    before = compute_query_labels(s, s.queries[0])
    after = compute_query_labels(s, s.queries[1])
    assert before.must_retrieve == frozenset({"m"})
    assert after.must_retrieve == frozenset()  # deleted -> not authorized
    assert "m" in after.forbidden


# -- helpers ---------------------------------------------------------------


def _t(iso):
    from benchmarks.governance.schema import parse_time

    return parse_time(iso)


def _query(scenario, *, principal, as_of, relevant, axis="utility"):
    from benchmarks.governance.schema import Query

    return Query.from_dict(
        {"id": "q", "principal": principal, "as_of": as_of, "axis": axis, "relevant": relevant}
    )
