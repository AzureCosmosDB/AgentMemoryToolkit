"""Async variants of the Agent Memory Toolkit clients.

This subpackage mirrors the sync API surface at ``agent_memory_toolkit``
and follows the ``azure.cosmos`` / ``azure.cosmos.aio`` convention.
"""

from agent_memory_toolkit.aio.cosmos_memory_client import AsyncCosmosMemoryStore
from agent_memory_toolkit.aio.embeddings import AsyncEmbeddingsClient
from agent_memory_toolkit.aio.memory import AsyncAgentMemory
from agent_memory_toolkit.aio.processing import AsyncProcessingClient

__all__ = [
    "AsyncAgentMemory",
    "AsyncCosmosMemoryStore",
    "AsyncEmbeddingsClient",
    "AsyncProcessingClient",
]
