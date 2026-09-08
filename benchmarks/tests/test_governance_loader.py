"""Validation tests for the SMGB loader's referential-integrity checks."""

from benchmarks.governance import Scenario, validate_scenario


def _base_dict():
    """A minimal, valid single-tenant scenario used as an editing base."""
    return {
        "scenario_id": "loader",
        "tenants": ["t1", "t2"],
        "scopes": [
            {"id": "user:a", "kind": "user", "tenant": "t1", "members": ["a"]},
            {"id": "proj:x", "kind": "project", "tenant": "t1", "members": ["a"]},
            {"id": "org:t2", "kind": "org", "tenant": "t2", "members": ["c"]},
        ],
        "principals": [
            {"id": "a", "tenant": "t1", "scopes": ["user:a", "proj:x"]},
            {"id": "c", "tenant": "t2", "scopes": ["org:t2"]},
        ],
        "memories": [
            {"id": "m1", "scope": "proj:x", "tenant": "t1",
             "created_at": "2026-01-01T00:00:00Z", "provenance": {"author": "a"}},
        ],
        "events": [],
        "queries": [
            {"id": "q1", "principal": "a", "as_of": "2026-02-01T00:00:00Z",
             "axis": "utility", "relevant": ["m1"],
             "must_retrieve": ["m1"], "must_not_retrieve": []},
        ],
    }


def test_valid_scenario_has_no_problems():
    assert validate_scenario(Scenario.from_dict(_base_dict())) == []


def test_unknown_must_not_retrieve_id_is_flagged_as_unknown():
    d = _base_dict()
    d["queries"][0]["must_not_retrieve"] = ["ghost"]
    problems = validate_scenario(Scenario.from_dict(d))
    # The unknown id is reported as unknown...
    assert any("must_not_retrieve id 'ghost' unknown" in p for p in problems)
    # ...and NOT misleadingly reported as "actually allowed".
    assert not any("actually allowed" in p for p in problems)


def test_unknown_must_retrieve_id_is_flagged_as_unknown():
    d = _base_dict()
    d["queries"][0]["must_retrieve"] = ["m1", "ghost"]
    problems = validate_scenario(Scenario.from_dict(d))
    assert any("must_retrieve id 'ghost' unknown" in p for p in problems)
    # The known part still matches policy, so no spurious mismatch is raised.
    assert not any("!= policy-derived" in p for p in problems)


def test_cross_tenant_scope_membership_is_flagged():
    d = _base_dict()
    # Principal 'a' (tenant t1) now claims membership in a t2-owned scope.
    d["principals"][0]["scopes"] = ["user:a", "proj:x", "org:t2"]
    problems = validate_scenario(Scenario.from_dict(d))
    assert any(
        "references" in p and "org:t2" in p and "tenant 't2'" in p for p in problems
    )


def test_unknown_scope_still_reported():
    d = _base_dict()
    d["principals"][0]["scopes"] = ["user:a", "nope"]
    problems = validate_scenario(Scenario.from_dict(d))
    assert any("unknown scope 'nope'" in p for p in problems)
