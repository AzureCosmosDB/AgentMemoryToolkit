"""Records folded into another memory, deleted with it.

Dedup retires a record that states the same claim as another into it (``supersede_reason="duplicate"``), combining
retires the records a combined memory was written from into it (``supersede_reason="combined"``), and an active
situational procedure retires the user's directive or correction it cites into itself
(``supersede_reason="procedure"``). Those records hold the same information as the memory they were folded into:
search hides them, but history still returns them. So deleting a memory deletes them too, and anything folded into
them, and what the user deleted is gone.

They are deleted before the memory itself: if that fails, the memory stays and a retry finds them again. Records
retired for another reason (``"update"`` or ``"contradict"``, which earlier versions wrote) are earlier versions, not
copies, and stay.

An owner's records all sit in one partition, so the records are looked up and deleted there.
"""

from __future__ import annotations

from typing import Any

from azure.cosmos.agent_memory._partitioning import OWNER_FILTER, owner_parameters, partition_key_for

ABSORBED_REASONS = ("duplicate", "procedure", "combined")
# The memory types records are folded into: dedup and combining fold facts into facts (episodes are left alone),
# and a procedure absorbs the statements it cites.
ABSORBING_TYPES = frozenset({"fact", "procedural"})


def absorbed_query(tenant_id: str, user_id: str, memory_id: str) -> tuple[str, list[dict[str, Any]]]:
    """The records folded directly into ``memory_id`` (run in the owner's partition)."""
    names = [f"@reason_{i}" for i in range(len(ABSORBED_REASONS))]
    query = (
        f"SELECT c.id, c.type FROM c WHERE {OWNER_FILTER} AND c.superseded_by = @memory_id "
        f"AND c.supersede_reason IN ({', '.join(names)})"
    )
    reasons = [{"name": name, "value": reason} for name, reason in zip(names, ABSORBED_REASONS)]
    return query, [*owner_parameters(tenant_id, user_id), {"name": "@memory_id", "value": memory_id}, *reasons]


def delete_absorbed(container: Any, tenant_id: str, user_id: str, memory_id: str) -> list[dict[str, Any]]:
    """Delete everything folded into ``memory_id``, directly or through another record, deepest first, and return
    it (``id`` and ``type``). Any failure but a record already gone is raised."""
    from azure.cosmos.exceptions import CosmosResourceNotFoundError

    partition_key = partition_key_for(tenant_id, user_id)
    found: list[dict[str, Any]] = []
    seen, pending = {memory_id}, [memory_id]
    while pending:
        query, parameters = absorbed_query(tenant_id, user_id, pending.pop())
        for row in container.query_items(query=query, parameters=parameters, partition_key=partition_key):
            if row["id"] not in seen:
                seen.add(row["id"])
                pending.append(row["id"])
                found.append(row)
    for row in reversed(found):
        try:
            container.delete_item(item=row["id"], partition_key=partition_key)
        except CosmosResourceNotFoundError:
            continue
    return found


async def adelete_absorbed(container: Any, tenant_id: str, user_id: str, memory_id: str) -> list[dict[str, Any]]:
    """Async mirror of ``delete_absorbed``."""
    from azure.cosmos.exceptions import CosmosResourceNotFoundError

    partition_key = partition_key_for(tenant_id, user_id)
    found: list[dict[str, Any]] = []
    seen, pending = {memory_id}, [memory_id]
    while pending:
        query, parameters = absorbed_query(tenant_id, user_id, pending.pop())
        async for row in container.query_items(query=query, parameters=parameters, partition_key=partition_key):
            if row["id"] not in seen:
                seen.add(row["id"])
                pending.append(row["id"])
                found.append(row)
    for row in reversed(found):
        try:
            await container.delete_item(item=row["id"], partition_key=partition_key)
        except CosmosResourceNotFoundError:
            continue
    return found
