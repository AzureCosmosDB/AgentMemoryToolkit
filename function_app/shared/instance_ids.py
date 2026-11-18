"""Instance ids for the orchestrations the change feed and the extraction orchestrator start.

Durable Functions takes an instance id of 1-100 characters without ``/``, ``\\``, ``#``, ``?`` or control characters,
and tenant, user and thread ids are free text, so an instance id never carries them: it is a readable ``kind``, a
fixed-length hash of the owner (and thread) and, for a counter threshold, the count. The same owner, thread, count and
batch always give the same id, which is what lets a replayed change-feed batch find the orchestrations it already
started and leave them alone.
"""

from __future__ import annotations

from hashlib import sha256


def orchestration_instance_id(kind: str, *parts: str, count: int | None = None, lsn: int | None = None) -> str:
    """``{kind}:{digest}`` plus ``:{count}`` when a count is given.

    ``parts`` is ``(tenant_id, user_id)`` or ``(tenant_id, user_id, thread_id)``. They are joined with the unit
    separator before hashing, so a ``:`` inside an id cannot make two different splits give the same digest.

    ``lsn`` is the highest change-feed sequence number of the batch that reached the count, and joins the hash. A
    replay of that batch has the same ``lsn`` and so the same id, while a count reached again after the counter
    expired (it lives 90 days) comes from a later batch and gets an id of its own, not that of a finished instance.
    """
    hashed = parts if lsn is None else (*parts, str(lsn))
    digest = sha256("\x1f".join(hashed).encode("utf-8")).hexdigest()[:32]
    return f"{kind}:{digest}" + (f":{count}" if count is not None else "")
