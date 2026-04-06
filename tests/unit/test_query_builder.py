"""Unit tests for agent_memory_toolkit._query_builder._QueryBuilder."""

from agent_memory_toolkit._query_builder import _QueryBuilder

# ---------------------------------------------------------------------------
# build_where
# ---------------------------------------------------------------------------


def test_no_filters_returns_empty_string():
    qb = _QueryBuilder()
    assert qb.build_where() == ""


def test_one_filter():
    qb = _QueryBuilder()
    qb.add_filter("c.user_id", "@user_id", "u1")
    assert qb.build_where() == " WHERE c.user_id = @user_id"
    params = qb.get_parameters()
    assert len(params) == 1
    assert params[0] == {"name": "@user_id", "value": "u1"}


def test_multiple_filters_and_joined():
    qb = _QueryBuilder()
    qb.add_filter("c.user_id", "@uid", "u1")
    qb.add_filter("c.role", "@role", "agent")
    where = qb.build_where()
    assert where == " WHERE c.user_id = @uid AND c.role = @role"
    assert len(qb.get_parameters()) == 2


# ---------------------------------------------------------------------------
# None handling
# ---------------------------------------------------------------------------


def test_none_values_skipped():
    qb = _QueryBuilder()
    qb.add_filter("c.user_id", "@uid", None)
    assert qb.build_where() == ""
    assert qb.get_parameters() == []


def test_mixed_none_and_non_none():
    qb = _QueryBuilder()
    qb.add_filter("c.user_id", "@uid", "u1")
    qb.add_filter("c.role", "@role", None)
    qb.add_filter("c.type", "@type", "turn")
    where = qb.build_where()
    assert where == " WHERE c.user_id = @uid AND c.type = @type"
    params = qb.get_parameters()
    assert len(params) == 2
    assert params[0]["value"] == "u1"
    assert params[1]["value"] == "turn"


# ---------------------------------------------------------------------------
# get_parameters returns a copy
# ---------------------------------------------------------------------------


def test_get_parameters_returns_copy():
    qb = _QueryBuilder()
    qb.add_filter("c.x", "@x", 1)
    p1 = qb.get_parameters()
    p1.append({"name": "@extra", "value": 99})
    p2 = qb.get_parameters()
    assert len(p2) == 1  # original unchanged
