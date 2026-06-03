"""Reusable query-builder for parameterized Cosmos DB queries.

The :class:`_QueryBuilder` helper eliminates duplicated
condition/parameter-building patterns across the sync and async clients.
"""

from __future__ import annotations

from typing import Any


class _QueryBuilder:
    """Accumulates optional WHERE conditions and their parameterized values.

    Usage::

        qb = _QueryBuilder()
        qb.add_filter("c.user_id", "@user_id", some_user_id)
        qb.add_filter("c.role", "@role", some_role)
        where = qb.build_where()        # " WHERE c.user_id = @user_id AND c.role = @role"
        params = qb.get_parameters()     # [{"name": "@user_id", "value": ...}, ...]
    """

    def __init__(self) -> None:
        self._conditions: list[str] = []
        self._parameters: list[dict[str, Any]] = []

    def add_filter(self, field: str, param_name: str, value: Any) -> None:
        """Add a filter only when *value* is not ``None``."""
        if value is None:
            return
        self._conditions.append(f"{field} = {param_name}")
        self._parameters.append({"name": param_name, "value": value})

    def add_thread_id_or_user_scoped(
        self,
        thread_id: Any,
        param_name: str,
        user_scoped_types: list[str],
        type_param_base: str = "@user_scoped_type_",
    ) -> None:
        """Match either ``c.thread_id = @thread_id`` OR ``c.type IN (...)``."""
        if thread_id is None:
            return
        if not user_scoped_types:
            self.add_filter("c.thread_id", param_name, thread_id)
            return
        self._parameters.append({"name": param_name, "value": thread_id})
        type_params: list[str] = []
        for i, t in enumerate(user_scoped_types):
            pname = f"{type_param_base}{i}"
            type_params.append(pname)
            self._parameters.append({"name": pname, "value": t})
        in_clause = f"c.type IN ({', '.join(type_params)})"
        self._conditions.append(f"(c.thread_id = {param_name} OR {in_clause})")

    def add_array_contains(self, field: str, param_name: str, value: Any) -> None:
        """Add an ``ARRAY_CONTAINS`` filter."""
        self._conditions.append(f"ARRAY_CONTAINS({field}, {param_name})")
        self._parameters.append({"name": param_name, "value": value})

    def add_not_array_contains(self, field: str, param_name: str, value: Any) -> None:
        """Add a ``NOT ARRAY_CONTAINS`` filter."""
        self._conditions.append(f"NOT ARRAY_CONTAINS({field}, {param_name})")
        self._parameters.append({"name": param_name, "value": value})

    def add_array_contains_any(self, field: str, param_base: str, values: list[Any]) -> None:
        """Add OR-combined ``ARRAY_CONTAINS`` filters (match any of *values*)."""
        if not values:
            return
        parts: list[str] = []
        for i, val in enumerate(values):
            pname = f"{param_base}{i}"
            parts.append(f"ARRAY_CONTAINS({field}, {pname})")
            self._parameters.append({"name": pname, "value": val})
        self._conditions.append("(" + " OR ".join(parts) + ")")

    def add_in_filter(self, field: str, param_base: str, values: list[Any]) -> None:
        """Add a ``field IN (@p_0, @p_1, ...)`` filter."""
        if not values:
            return
        parts: list[str] = []
        for i, val in enumerate(values):
            pname = f"{param_base}{i}"
            parts.append(pname)
            self._parameters.append({"name": pname, "value": val})
        self._conditions.append(f"{field} IN ({', '.join(parts)})")

    def add_is_null_or_undefined(self, field: str) -> None:
        """Add ``(NOT IS_DEFINED(field) OR IS_NULL(field))`` filter."""
        self._conditions.append(f"(NOT IS_DEFINED({field}) OR IS_NULL({field}))")

    def add_not_null(self, field: str) -> None:
        """Add ``(IS_DEFINED(field) AND NOT IS_NULL(field))`` filter."""
        self._conditions.append(f"(IS_DEFINED({field}) AND NOT IS_NULL({field}))")

    def add_gte(self, field: str, param_name: str, value: Any) -> None:
        """Add a ``field >= @param`` filter."""
        self._conditions.append(f"{field} >= {param_name}")
        self._parameters.append({"name": param_name, "value": value})

    def add_time_range(
        self,
        field: str,
        *,
        after: Any = None,
        before: Any = None,
        after_param: str = "@after",
        before_param: str = "@before",
    ) -> None:
        """Add inclusive lower/upper bounds for an ISO-sortable time field."""
        if after is not None:
            self._conditions.append(f"{field} >= {after_param}")
            self._parameters.append({"name": after_param, "value": after})
        if before is not None:
            self._conditions.append(f"{field} <= {before_param}")
            self._parameters.append({"name": before_param, "value": before})

    def add_metadata_filter(self, path: str, op: str, value: Any, *, param_name: str | None = None) -> None:
        """Add a parameterized comparison filter for a metadata path."""
        if op not in {"=", "!=", ">", "<", ">=", "<="}:
            raise ValueError(f"unsupported op: {op}")
        pname = param_name or f"@m_{len(self._parameters)}"
        self._conditions.append(f"{path} {op} {pname}")
        self._parameters.append({"name": pname, "value": value})

    def build_where(self) -> str:
        """Return the ``WHERE …`` clause (or empty string if no filters)."""
        if not self._conditions:
            return ""
        return " WHERE " + " AND ".join(self._conditions)

    def get_parameters(self) -> list[dict[str, Any]]:
        """Return a *copy* of the accumulated parameters list."""
        return list(self._parameters)
