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

    def build_where(self) -> str:
        """Return the ``WHERE …`` clause (or empty string if no filters)."""
        if not self._conditions:
            return ""
        return " WHERE " + " AND ".join(self._conditions)

    def get_parameters(self) -> list[dict[str, Any]]:
        """Return a *copy* of the accumulated parameters list."""
        return list(self._parameters)
