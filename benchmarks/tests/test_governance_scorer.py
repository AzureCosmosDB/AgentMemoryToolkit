"""Unit tests for the SMGB scorer and reference runners."""

from benchmarks.governance.schema import Scenario
from benchmarks.governance.scorer import (
    naive_global_run,
    naive_shared_run,
    oracle_provenance,
    oracle_run,
    score_scenario,
)


def _scenario():
    return Scenario.from_dict(
        {
            "scenario_id": "score",
            "tenants": ["t1", "t2"],
            "scopes": [
                {"id": "user:a", "kind": "user", "tenant": "t1", "members": ["a"]},
                {"id": "acct:x", "kind": "account", "tenant": "t1", "members": ["a", "b"]},
                {"id": "org:t2", "kind": "org", "tenant": "t2", "members": ["c"]},
            ],
            "principals": [
                {"id": "a", "tenant": "t1", "scopes": ["user:a", "acct:x"]},
                {"id": "b", "tenant": "t1", "scopes": ["acct:x"]},
                {"id": "c", "tenant": "t2", "scopes": ["org:t2"]},
            ],
            "memories": [
                {"id": "shared1", "scope": "acct:x", "tenant": "t1",
                 "created_at": "2026-01-01T00:00:00Z", "provenance": {"author": "a-agent"}},
                {"id": "private1", "scope": "user:a", "tenant": "t1",
                 "created_at": "2026-01-01T00:00:00Z", "provenance": {"author": "a-agent"}},
                {"id": "other_tenant", "scope": "org:t2", "tenant": "t2",
                 "created_at": "2026-01-01T00:00:00Z", "provenance": {"author": "c-agent"}},
            ],
            "events": [],
            "queries": [
                # b should get shared1, not private1 (leakage) nor other_tenant (isolation).
                {"id": "u", "principal": "b", "as_of": "2026-02-01T00:00:00Z",
                 "axis": "utility", "relevant": ["shared1", "private1", "other_tenant"]},
            ],
        }
    )


def test_oracle_is_perfect():
    s = _scenario()
    report = score_scenario(s, oracle_run(s), system="oracle", provenance=oracle_provenance(s))
    su = report.summary()
    assert su["mean_recall"] == 1.0
    assert su["leak_rate"] == 0.0
    assert su["total_leaks"] == 0
    assert su["isolation_violations"] == 0


def test_naive_shared_leaks_but_respects_tenant():
    s = _scenario()
    report = score_scenario(s, naive_shared_run(s), system="naive_shared")
    su = report.summary()
    assert su["mean_recall"] == 1.0  # it does return the target...
    assert su["leak_rate"] > 0.0  # ...but also leaks private1
    assert su["total_leaks"] >= 1
    assert su["isolation_violations"] == 0  # never crosses tenants


def test_naive_global_breaks_isolation():
    s = _scenario()
    report = score_scenario(s, naive_global_run(s), system="naive_global")
    su = report.summary()
    assert su["isolation_violations"] >= 1  # returns other_tenant


def test_recall_is_none_for_abstention_query():
    s = Scenario.from_dict(
        {
            "scenario_id": "abstain",
            "tenants": ["t"],
            "scopes": [
                {"id": "user:a", "kind": "user", "tenant": "t", "members": ["a"]},
                {"id": "user:b", "kind": "user", "tenant": "t", "members": ["b"]},
            ],
            "principals": [
                {"id": "a", "tenant": "t", "scopes": ["user:a"]},
                {"id": "b", "tenant": "t", "scopes": ["user:b"]},
            ],
            "memories": [
                {"id": "secret", "scope": "user:a", "tenant": "t",
                 "created_at": "2026-01-01T00:00:00Z", "provenance": {"author": "a"}},
            ],
            "events": [],
            "queries": [
                {"id": "leak", "principal": "b", "as_of": "2026-02-01T00:00:00Z",
                 "axis": "leakage", "relevant": ["secret"]},
            ],
        }
    )
    # Empty run: perfect abstention -> no leak, recall N/A.
    report = score_scenario(s, {"leak": []}, system="silent")
    assert report.per_query[0].recall is None
    assert report.per_query[0].leaked is False
    # Leaky run that returns the forbidden secret.
    report2 = score_scenario(s, {"leak": ["secret"]}, system="leaky")
    assert report2.per_query[0].leaked is True
    assert report2.total_leaks == 1


def test_k_cutoff_limits_retrieval():
    s = _scenario()
    # A run that returns the target only beyond the cutoff should miss it.
    run = {"u": ["private1", "other_tenant", "shared1"]}
    at_k2 = score_scenario(s, run, k=2)
    assert at_k2.per_query[0].recall == 0.0  # shared1 is at rank 3
    assert at_k2.per_query[0].leaked is True  # private1 leaked within k=2
