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

    def add_is_null_or_undefined(self, field: str) -> None:
        """Add ``(NOT IS_DEFINED(field) OR IS_NULL(field))`` filter."""
        self._conditions.append(f"(NOT IS_DEFINED({field}) OR IS_NULL({field}))")

    def add_not_null(self, field: str) -> None:
        """Add ``(IS_DEFINED(field) AND NOT IS_NULL(field))`` filter."""
        self._conditions.append(f"(IS_DEFINED({field}) AND NOT IS_NULL({field}))")

    def build_where(self) -> str:
        """Return the ``WHERE …`` clause (or empty string if no filters)."""
        if not self._conditions:
            return ""
        return " WHERE " + " AND ".join(self._conditions)

    def get_parameters(self) -> list[dict[str, Any]]:
        """Return a *copy* of the accumulated parameters list."""
        return list(self._parameters)
