"""Cached Cosmos container clients (MI auth via DefaultAzureCredential).

Both sync and async clients are exposed:

* ``get_memories_container()`` returns a *sync* ContainerProxy used by the
  ``ProcessingPipeline`` activities (the pipeline is sync today).
* ``get_counter_container_async()`` returns an *async* AsyncContainerProxy
  used by the change-feed trigger to update counters.

Clients are lazily constructed and cached at module level — Azure Functions
re-uses the same Python worker across invocations, so we want one client
per worker instead of one per invocation.
"""

from __future__ import annotations

from typing import Any

from . import config

# Sync clients (for activities that call the sync ProcessingPipeline)
_sync_cosmos_client: Any | None = None
_sync_memories_container: Any | None = None

# Async clients (for the change-feed trigger)
_async_cosmos_client: Any | None = None
_async_counter_container: Any | None = None
_async_credential: Any | None = None


def _credential():
    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential()


def get_memories_container():
    """Return the sync ContainerProxy for the memories container."""
    global _sync_cosmos_client, _sync_memories_container
    if _sync_memories_container is not None:
        return _sync_memories_container

    from azure.cosmos import CosmosClient

    if _sync_cosmos_client is None:
        _sync_cosmos_client = CosmosClient(config.get_cosmos_endpoint(), credential=_credential())

    db = _sync_cosmos_client.get_database_client(config.CHANGE_FEED_DATABASE)
    _sync_memories_container = db.get_container_client(config.CHANGE_FEED_CONTAINER)
    return _sync_memories_container


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
