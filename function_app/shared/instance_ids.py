"""Instance ids for the orchestrations the change feed and the extraction orchestrator start.

Durable Functions takes an instance id of 1-100 characters without ``/``, ``\\``, ``#``, ``?`` or control characters,
and tenant, user and thread ids are free text, so an instance id never carries them: it is a readable ``kind``, a
fixed-length hash of the owner (and thread) and, for a counter threshold, the count. The same owner, thread and count
always give the same id, which is what makes a replayed change-feed batch start each orchestration once.
"""

from __future__ import annotations

from hashlib import sha256


def orchestration_instance_id(kind: str, *parts: str, count: int | None = None) -> str:
    """``{kind}:{digest}`` plus ``:{count}`` when a count is given.

    ``parts`` is ``(tenant_id, user_id)`` or ``(tenant_id, user_id, thread_id)``. They are joined with the unit
    separator before hashing, so a ``:`` inside an id cannot make two different splits give the same digest.
    """
    digest = sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]
    return f"{kind}:{digest}" + (f":{count}" if count is not None else "")
