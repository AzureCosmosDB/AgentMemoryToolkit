"""Records folded into another memory, deleted with it, and the members a combined fact gives back.

Dedup retires a record that states the same claim as another into it (``supersede_reason="duplicate"``), combining
retires the records a combined memory was written from into it (``supersede_reason="combined"``), and an active
situational procedure retires the user's directive or correction it cites into itself
(``supersede_reason="procedure"``). Those records hold the same information as the memory they were folded into:
search hides them, but history still returns them. So deleting a memory deletes them too, and anything folded into
them, and what the user deleted is gone.

They are deleted before the memory itself: if that fails, the memory stays and a retry finds them again. Records
replaced for another reason (``"update"``, ``"contradict"``) are earlier versions, not copies, and stay.

A combined fact that a newer fact contradicts is retired, but its members are not copies of the part that is now wrong:
each holds details the correction may not touch. So its members are restored first (:func:`restored_member`): the
records retired into it as combined, and those folded into it as duplicates (:func:`combined_members_query`). A
failed retirement, a lost race, a crash or a partial restore leaves restored members active beside the combined fact,
and the next dedup pass folds them back into it; if it loses again they must come back again, not be lost with it.
Only the member the correction contradicts is then retired, in a later pass. (Records retired into it as
``"contradict"`` or ``"update"`` are earlier versions and stay retired; nothing is retired into a combined fact as
``"procedure"``.)

A user's memories sit in one partition per thread, so the records are looked up across the user's partitions and each
is deleted, or written back, under its own thread.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

ABSORBED_REASONS = ("duplicate", "procedure", "combined")
# The memory types records are folded into: dedup and combining fold facts into facts (episodes are left alone),
# and a procedure absorbs the statements it cites.
ABSORBING_TYPES = frozenset({"fact", "procedural"})
# What retiring a record sets (``MemoryStore.mark_superseded``); restoring one takes these away again.
SUPERSEDED_FIELDS = ("superseded_by", "supersede_reason", "superseded_at")


def absorbed_query(
    user_id: str,
    memory_id: str,
    *,
    reasons: Sequence[str] = ABSORBED_REASONS,
    columns: str = "c.id, c.type, c.thread_id",
) -> tuple[str, list[dict[str, Any]]]:
    """The records folded directly into ``memory_id`` (run across the user's partitions): by default every kind of
    fold, with what deleting them needs (``columns``)."""
    names = [f"@reason_{i}" for i in range(len(reasons))]
    query = (
        f"SELECT {columns} FROM c WHERE c.user_id = @user_id AND c.superseded_by = @memory_id "
        f"AND c.supersede_reason IN ({', '.join(names)})"
    )
    values = [{"name": name, "value": reason} for name, reason in zip(names, reasons)]
    return query, [{"name": "@user_id", "value": user_id}, {"name": "@memory_id", "value": memory_id}, *values]


def combined_members_query(user_id: str, combined_id: str) -> tuple[str, list[dict[str, Any]]]:
    """The members of the combined fact ``combined_id``: the whole records retired into it as combined, and those
    folded into it as duplicates (run across the user's partitions).

    Both are members once a restore has been interrupted: the dedup pass folds a member that was written back
    into the combined fact that is still active, so a restore that took only the ``combined`` ones would lose it."""
    return absorbed_query(user_id, combined_id, reasons=("combined", "duplicate"), columns="*")


def restored_member(member: Mapping[str, Any]) -> dict[str, Any]:
    """``member`` as it was before it was retired: without the fields retiring it set, and without Cosmos DB's system
    fields (``_etag``, ``_ts``, ...), which are not part of a body to write."""
    return {k: v for k, v in member.items() if k not in SUPERSEDED_FIELDS and not k.startswith("_")}


def delete_absorbed(container: Any, user_id: str, memory_id: str) -> list[dict[str, Any]]:
    """Delete everything folded into ``memory_id``, directly or through another record, deepest first, and return
    it (``id``, ``type`` and ``thread_id``). Any failure but a record already gone is raised."""
    from azure.cosmos.exceptions import CosmosResourceNotFoundError

    found: list[dict[str, Any]] = []
    seen, pending = {memory_id}, [memory_id]
    while pending:
        query, parameters = absorbed_query(user_id, pending.pop())
        for row in container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True):
            if row["id"] not in seen:
                seen.add(row["id"])
                pending.append(row["id"])
                found.append(row)
    for row in reversed(found):
        try:
            container.delete_item(item=row["id"], partition_key=[user_id, row["thread_id"]])
        except CosmosResourceNotFoundError:
            continue
    return found


async def adelete_absorbed(container: Any, user_id: str, memory_id: str) -> list[dict[str, Any]]:
    """Async mirror of ``delete_absorbed``."""
    from azure.cosmos.exceptions import CosmosResourceNotFoundError

    found: list[dict[str, Any]] = []
    seen, pending = {memory_id}, [memory_id]
    while pending:
        query, parameters = absorbed_query(user_id, pending.pop())
        async for row in container.query_items(query=query, parameters=parameters):
            if row["id"] not in seen:
                seen.add(row["id"])
                pending.append(row["id"])
                found.append(row)
    for row in reversed(found):
        try:
            await container.delete_item(item=row["id"], partition_key=[user_id, row["thread_id"]])
        except CosmosResourceNotFoundError:
            continue
    return found
