"""How records are keyed in Cosmos DB.

Every record lives in the logical partition ``[tenant_id, user_id]``. ``tenant_id`` says whose memories these are (an
application, a customer, an environment), and ``user_id`` is the owner of the memory: a user or an agent. ``thread_id``
is a plain field of the record, not part of the key, so everything one owner knows (turns, facts, episodes, procedures,
summaries, counters) sits in one partition, and a query that names the owner reads that partition only.

A client is bound to one tenant (its ``tenant_id`` argument, ``"default"`` when not given), and every read and write
it makes stays inside it. A filter on ``user_id`` alone would read the same id in every tenant, so every query also
filters on ``c.tenant_id``: build the filter with :data:`OWNER_FILTER` and :func:`owner_parameters`.

A record id is unique within its partition, that is, per owner across all of the owner's threads.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from azure.cosmos.agent_memory.exceptions import ValidationError

DEFAULT_TENANT_ID = "default"

# The partition key every keyed container is created with, and the one :func:`partition_key_problem` expects.
PARTITION_KEY_PATHS = ["/tenant_id", "/user_id"]

# Ids appear in Cosmos document ids and orchestration instance ids, which cannot hold these characters.
_UNSUPPORTED_CHARACTERS = "/\\?#"
MAX_TENANT_ID_BYTES = 256
MAX_OWNER_ID_BYTES = 256

TENANT_FILTER = "c.tenant_id = @tenant_id"
OWNER_FILTER = "c.tenant_id = @tenant_id AND c.user_id = @user_id"


def _checked_identifier(value: Any, name: str, max_bytes: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    if any(character in value for character in _UNSUPPORTED_CHARACTERS) or any(
        ord(character) < 32 for character in value
    ):
        raise ValidationError(f"{name} contains characters the memory store does not support: / \\ ? # or control")
    if len(value.encode("utf-8")) > max_bytes:
        raise ValidationError(f"{name} must not exceed {max_bytes} UTF-8 bytes")
    return value


def resolve_tenant_id(tenant_id: Optional[str] = None) -> str:
    """The tenant to use: ``DEFAULT_TENANT_ID`` when ``tenant_id`` is ``None``, else ``tenant_id`` if it is valid.

    A valid tenant id is a non-empty string of at most ``MAX_TENANT_ID_BYTES`` UTF-8 bytes without ``/``, ``\\``, ``?``,
    ``#`` or control characters. Anything else raises :class:`ValidationError`.
    """
    if tenant_id is None:
        return DEFAULT_TENANT_ID
    return _checked_identifier(tenant_id, "tenant_id", MAX_TENANT_ID_BYTES)


def validate_owner_id(user_id: Any) -> str:
    """``user_id`` if it is a valid owner id, else :class:`ValidationError`.

    An owner id follows the rules of a tenant id: a non-empty string of at most ``MAX_OWNER_ID_BYTES`` UTF-8 bytes
    without ``/``, ``\\``, ``?``, ``#`` or control characters. It is part of the partition key and of the ids of the
    records the toolkit builds for the owner, so a value outside the rules would fail later or share a partition by
    accident.
    """
    return _checked_identifier(user_id, "user_id", MAX_OWNER_ID_BYTES)


def partition_key_for(tenant_id: str, user_id: str) -> list[str]:
    """The full partition key of the records of ``user_id`` in ``tenant_id``."""
    return [tenant_id, validate_owner_id(user_id)]


def tenant_parameters(tenant_id: str) -> list[dict[str, Any]]:
    """The query parameter for :data:`TENANT_FILTER`."""
    return [{"name": "@tenant_id", "value": tenant_id}]


def owner_parameters(tenant_id: str, user_id: str) -> list[dict[str, Any]]:
    """The query parameters for :data:`OWNER_FILTER`."""
    return [{"name": "@tenant_id", "value": tenant_id}, {"name": "@user_id", "value": validate_owner_id(user_id)}]


def stamp_scope(doc: Mapping[str, Any], tenant_id: str) -> dict[str, Any]:
    """A copy of ``doc`` that carries ``tenant_id``, refusing a record that has no valid owner or names another tenant.

    A client writes only into its own tenant. Cosmos stores a record in the partition its own ``tenant_id`` and
    ``user_id`` name, so a record without a valid ``user_id`` would land where no owner-scoped read reaches it.
    """
    validate_owner_id(doc.get("user_id"))
    named = doc.get("tenant_id")
    if named is not None and named != tenant_id:
        raise ValidationError("record names a different tenant than the client it is written with")
    stamped = dict(doc)
    stamped["tenant_id"] = tenant_id
    return stamped


def partition_key_problem(container_id: Any, properties: Any) -> Optional[str]:
    """What is wrong with the partition key of a container, or ``None`` when it is :data:`PARTITION_KEY_PATHS`.

    ``properties`` is what ``ContainerProxy.read()`` returns. A partition key cannot change in place, and a container
    keyed otherwise (for example ``[/user_id, /thread_id]``) accepts writes but is read back empty, so a mismatch is
    reported up front. Anything that is not a mapping, such as a test double, is not checked.
    """
    if not isinstance(properties, Mapping):
        return None
    actual = list((properties.get("partitionKey") or {}).get("paths") or [])
    if actual == PARTITION_KEY_PATHS:
        return None
    return (
        f"container {container_id!r} has partition key {actual or 'none'} but this SDK requires {PARTITION_KEY_PATHS}. "
        "A partition key cannot be changed in place: recreate the container, and the leases container with it, "
        "because existing data is not migrated"
    )
