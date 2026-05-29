"""Cached Cosmos container clients (MI auth via DefaultAzureCredential).

Both sync and async clients are exposed:

* ``get_memories_container()`` returns a *sync* ContainerProxy used by the
  ``PipelineService`` activities (the pipeline is sync today).
* ``get_counter_container_async()`` returns an *async* AsyncContainerProxy
  used by the change-feed trigger to update counters.

Clients are lazily constructed and cached at module level — Azure Functions
re-uses the same Python worker across invocations, so we want one client
per worker instead of one per invocation.
"""

from __future__ import annotations

import atexit
from typing import Any

from . import config

# Sync clients (for activities that call the sync PipelineService)
_sync_cosmos_client: Any | None = None
_sync_memories_container: Any | None = None
_sync_turns_container: Any | None = None

# Async clients (for the change-feed trigger)
_async_cosmos_client: Any | None = None
_async_counter_container: Any | None = None
_async_credential: Any | None = None


def _credential():
    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential()


def _get_sync_database():
    global _sync_cosmos_client

    from azure.cosmos import CosmosClient

    if _sync_cosmos_client is None:
        _sync_cosmos_client = CosmosClient(config.get_cosmos_endpoint(), credential=_credential())
    return _sync_cosmos_client.get_database_client(config.CHANGE_FEED_DATABASE)


def get_memories_container():
    """Return the sync ContainerProxy for the memories container."""
    global _sync_memories_container
    if _sync_memories_container is not None:
        return _sync_memories_container

    db = _get_sync_database()
    _sync_memories_container = db.get_container_client(config.CHANGE_FEED_CONTAINER)
    return _sync_memories_container


def get_turns_container():
    """Return the sync ContainerProxy for the optional turns container."""
    global _sync_turns_container
    if not config.COSMOS_TURNS_CONTAINER:
        return None
    if _sync_turns_container is not None:
        return _sync_turns_container

    db = _get_sync_database()
    _sync_turns_container = db.get_container_client(config.COSMOS_TURNS_CONTAINER)
    return _sync_turns_container


async def get_counter_container_async():
    """Return the async AsyncContainerProxy for the counter container."""
    global _async_cosmos_client, _async_counter_container, _async_credential
    if _async_counter_container is not None:
        return _async_counter_container

    from azure.cosmos.aio import CosmosClient as AsyncCosmosClient
    from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential

    if _async_credential is None:
        _async_credential = AsyncDefaultAzureCredential()

    if _async_cosmos_client is None:
        _async_cosmos_client = AsyncCosmosClient(config.get_cosmos_endpoint(), credential=_async_credential)

    db = _async_cosmos_client.get_database_client(config.CHANGE_FEED_DATABASE)
    _async_counter_container = db.get_container_client(config.COUNTERS_CONTAINER)
    return _async_counter_container


async def close_async_clients() -> None:
    """Close the cached async Cosmos client and credential.

    Idempotent — safe to call multiple times. Exposed so tests and explicit
    shutdown hooks can release the underlying ``aiohttp`` session cleanly.
    """
    global _async_cosmos_client, _async_counter_container, _async_credential
    if _async_cosmos_client is not None:
        try:
            await _async_cosmos_client.close()
        except Exception:
            pass
        _async_cosmos_client = None
        _async_counter_container = None
    if _async_credential is not None:
        try:
            await _async_credential.close()
        except Exception:
            pass
        _async_credential = None


def _close_at_exit() -> None:
    """``atexit`` hook: run :func:`close_async_clients` if any async client is live."""
    if _async_cosmos_client is None and _async_credential is None:
        return
    import asyncio

    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(close_async_clients())
        finally:
            loop.close()
    except Exception:
        pass


atexit.register(_close_at_exit)
