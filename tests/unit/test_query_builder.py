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


# ---------------------------------------------------------------------------
# add_array_contains
# ---------------------------------------------------------------------------


def test_add_array_contains():
    qb = _QueryBuilder()
    qb.add_array_contains("c.tags", "@tag", "topic:travel")
    assert "ARRAY_CONTAINS(c.tags, @tag)" in qb.build_where()
    params = qb.get_parameters()
    assert len(params) == 1
    assert params[0] == {"name": "@tag", "value": "topic:travel"}


def test_add_array_contains_combined_with_filter():
    qb = _QueryBuilder()
    qb.add_filter("c.user_id", "@uid", "u1")
    qb.add_array_contains("c.tags", "@tag", "topic:travel")
    where = qb.build_where()
    assert "c.user_id = @uid" in where
    assert "ARRAY_CONTAINS(c.tags, @tag)" in where
    assert " AND " in where


# ---------------------------------------------------------------------------
# add_array_contains_any
# ---------------------------------------------------------------------------


def test_add_array_contains_any():
    qb = _QueryBuilder()
    qb.add_array_contains_any("c.tags", "@t_", ["a", "b"])
    where = qb.build_where()
    assert "ARRAY_CONTAINS(c.tags, @t_0)" in where
    assert "ARRAY_CONTAINS(c.tags, @t_1)" in where
    assert " OR " in where
    params = qb.get_parameters()
    assert len(params) == 2
    assert params[0] == {"name": "@t_0", "value": "a"}
    assert params[1] == {"name": "@t_1", "value": "b"}


def test_add_array_contains_any_empty_skipped():
    qb = _QueryBuilder()
    qb.add_array_contains_any("c.tags", "@t_", [])
    assert qb.build_where() == ""
    assert qb.get_parameters() == []


def test_add_array_contains_any_single_value():
    qb = _QueryBuilder()
    qb.add_array_contains_any("c.tags", "@t_", ["only"])
    where = qb.build_where()
    assert "ARRAY_CONTAINS(c.tags, @t_0)" in where
    # Single value shouldn't have OR
    assert " OR " not in where


# ---------------------------------------------------------------------------
# add_is_null_or_undefined
# ---------------------------------------------------------------------------


def test_add_is_null_or_undefined():
    qb = _QueryBuilder()
    qb.add_is_null_or_undefined("c.superseded_by")
    where = qb.build_where()
    assert "NOT IS_DEFINED(c.superseded_by)" in where
    assert "IS_NULL(c.superseded_by)" in where
    assert " OR " in where


def test_add_is_null_or_undefined_no_parameters():
    qb = _QueryBuilder()
    qb.add_is_null_or_undefined("c.superseded_by")
    assert qb.get_parameters() == []


# ---------------------------------------------------------------------------
# add_not_null
# ---------------------------------------------------------------------------


def test_add_not_null():
    qb = _QueryBuilder()
    qb.add_not_null("c.superseded_by")
    where = qb.build_where()
    assert "IS_DEFINED(c.superseded_by)" in where
    assert "NOT IS_NULL(c.superseded_by)" in where
    assert " AND " in where


def test_add_not_null_no_parameters():
    qb = _QueryBuilder()
    qb.add_not_null("c.superseded_by")
    assert qb.get_parameters() == []


# ---------------------------------------------------------------------------
# Complex combinations
# ---------------------------------------------------------------------------


def test_combined_filters_with_all_new_methods():
    qb = _QueryBuilder()
    qb.add_filter("c.user_id", "@uid", "u1")
    qb.add_array_contains("c.tags", "@tag", "topic:travel")
    qb.add_is_null_or_undefined("c.superseded_by")
    where = qb.build_where()
    assert "c.user_id = @uid" in where
    assert "ARRAY_CONTAINS(c.tags, @tag)" in where
    assert "NOT IS_DEFINED(c.superseded_by)" in where
    params = qb.get_parameters()
    assert len(params) == 2  # filter + array_contains, null/undefined adds no params


# ---------------------------------------------------------------------------
# add_in_filter
# ---------------------------------------------------------------------------


def test_add_in_filter_multiple_values():
    qb = _QueryBuilder()
    qb.add_in_filter("c.type", "@memory_type_", ["fact", "procedural", "episodic"])
    where = qb.build_where()
    assert where == " WHERE c.type IN (@memory_type_0, @memory_type_1, @memory_type_2)"
    params = qb.get_parameters()
    assert params == [
        {"name": "@memory_type_0", "value": "fact"},
        {"name": "@memory_type_1", "value": "procedural"},
        {"name": "@memory_type_2", "value": "episodic"},
    ]


def test_add_in_filter_single_value():
    qb = _QueryBuilder()
    qb.add_in_filter("c.type", "@t_", ["fact"])
    assert qb.build_where() == " WHERE c.type IN (@t_0)"
    assert qb.get_parameters() == [{"name": "@t_0", "value": "fact"}]


def test_add_in_filter_empty_list_skipped():
    qb = _QueryBuilder()
    qb.add_in_filter("c.type", "@t_", [])
    assert qb.build_where() == ""
    assert qb.get_parameters() == []


def test_add_in_filter_combined_with_other_filters():
    qb = _QueryBuilder()
    qb.add_filter("c.user_id", "@uid", "u1")
    qb.add_in_filter("c.type", "@t_", ["fact", "procedural"])
    where = qb.build_where()
    assert "c.user_id = @uid" in where
    assert "c.type IN (@t_0, @t_1)" in where
    assert " AND " in where
    assert len(qb.get_parameters()) == 3


# ---------------------------------------------------------------------------
# add_time_range
# ---------------------------------------------------------------------------


def test_add_time_range_after_only():
    qb = _QueryBuilder()
    qb.add_time_range("c.created_at", after="2026-01-01T00:00:00+00:00")
    assert qb.build_where() == " WHERE c.created_at >= @after"
    assert qb.get_parameters() == [{"name": "@after", "value": "2026-01-01T00:00:00+00:00"}]


def test_add_time_range_before_only():
    qb = _QueryBuilder()
    qb.add_time_range("c.created_at", before="2026-02-01T00:00:00+00:00")
    assert qb.build_where() == " WHERE c.created_at <= @before"
    assert qb.get_parameters() == [{"name": "@before", "value": "2026-02-01T00:00:00+00:00"}]


def test_add_time_range_both_with_custom_params():
    qb = _QueryBuilder()
    qb.add_time_range(
        "c.created_at",
        after="2026-01-01T00:00:00+00:00",
        before="2026-02-01T00:00:00+00:00",
        after_param="@created_after",
        before_param="@created_before",
    )
    assert qb.build_where() == " WHERE c.created_at >= @created_after AND c.created_at <= @created_before"
    assert qb.get_parameters() == [
        {"name": "@created_after", "value": "2026-01-01T00:00:00+00:00"},
        {"name": "@created_before", "value": "2026-02-01T00:00:00+00:00"},
    ]


def test_add_time_range_neither_is_noop():
    qb = _QueryBuilder()
    qb.add_time_range("c.created_at")
    assert qb.build_where() == ""
    assert qb.get_parameters() == []


# ---------------------------------------------------------------------------
# add_metadata_filter
# ---------------------------------------------------------------------------


def test_add_metadata_filter_supported_ops():
    for op in ["=", "!=", ">", "<", ">=", "<="]:
        qb = _QueryBuilder()
        qb.add_metadata_filter("c.metadata.score", op, 7, param_name="@score")
        assert qb.build_where() == f" WHERE c.metadata.score {op} @score"
        assert qb.get_parameters() == [{"name": "@score", "value": 7}]


def test_add_metadata_filter_auto_param_name():
    qb = _QueryBuilder()
    qb.add_filter("c.user_id", "@user_id", "u1")
    qb.add_metadata_filter("c.metadata.category", "=", "preference")
    assert "c.metadata.category = @m_1" in qb.build_where()
    assert {"name": "@m_1", "value": "preference"} in qb.get_parameters()


def test_add_metadata_filter_rejects_unsupported_op():
    qb = _QueryBuilder()
    try:
        qb.add_metadata_filter("c.metadata.category", "LIKE", "pref")
    except ValueError as exc:
        assert "unsupported op: LIKE" in str(exc)
    else:
        raise AssertionError("expected ValueError")
