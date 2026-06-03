"""Container routing primitive for the 3-container hard split.

Single source of truth for which Cosmos DB container holds which memory
document type. See `docs/architecture/` and the container split spec for
rationale.

* ``ContainerKey.TURNS``      → ``memories_turns``      (type=turn)
* ``ContainerKey.MEMORIES``   → ``memories``            (type ∈ {fact, episodic, procedural})
* ``ContainerKey.SUMMARIES``  → ``memories_summaries``  (type ∈ {thread_summary, user_summary})
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class ContainerKey(str, Enum):
    TURNS = "turns"
    MEMORIES = "memories"
    SUMMARIES = "summaries"


_CONTAINER_FOR_TYPE: dict[str, ContainerKey] = {
    "turn": ContainerKey.TURNS,
    "fact": ContainerKey.MEMORIES,
    "episodic": ContainerKey.MEMORIES,
    "procedural": ContainerKey.MEMORIES,
    "thread_summary": ContainerKey.SUMMARIES,
    "user_summary": ContainerKey.SUMMARIES,
}

USER_SCOPED_MEMORIES_TYPES: frozenset[str] = frozenset({"episodic", "procedural"})


def container_key_for_type(memory_type: str) -> ContainerKey:
    """Return the ``ContainerKey`` that owns documents of ``memory_type``."""
    try:
        return _CONTAINER_FOR_TYPE[memory_type]
    except KeyError as exc:
        raise ValueError(f"Unknown memory type {memory_type!r}; valid types: {sorted(_CONTAINER_FOR_TYPE)}") from exc


def container_keys_for_types(memory_types: Iterable[str]) -> list[ContainerKey]:
    """Return the distinct ``ContainerKey`` set for the given types.

    Order is deterministic (TURNS, MEMORIES, SUMMARIES).
    """
    seen: set[ContainerKey] = set()
    for t in memory_types:
        seen.add(container_key_for_type(t))
    # Deterministic order: TURNS, MEMORIES, SUMMARIES
    return [k for k in (ContainerKey.TURNS, ContainerKey.MEMORIES, ContainerKey.SUMMARIES) if k in seen]
