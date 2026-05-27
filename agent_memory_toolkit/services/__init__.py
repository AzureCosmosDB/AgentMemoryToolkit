"""Service protocol contracts for the memory client."""

from __future__ import annotations

from typing import Any, Optional, Protocol


class MemoryStoreProtocol(Protocol):
    """Persistence primitives consumed by the pipeline service."""

    def query(
        self,
        sql: str,
        parameters: Optional[list[dict[str, Any]]] = None,
        partition_key: Any = None,
        cross_partition: bool = False,
    ) -> list[dict[str, Any]]: ...

    def read_item(self, item_id: str, partition_key: Any) -> dict[str, Any]: ...

    def add_cosmos(self, record: dict[str, Any]) -> dict[str, Any]: ...

    def mark_superseded(
        self,
        old_doc: dict[str, Any],
        superseder_id: str,
        *,
        reason: str,
    ) -> bool: ...


__all__ = ["MemoryStoreProtocol"]
