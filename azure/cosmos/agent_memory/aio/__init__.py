"""Async variants of the Agent Memory Toolkit clients.

This subpackage mirrors the sync API surface at ``azure.cosmos.agent_memory``
and follows the ``azure.cosmos`` / ``azure.cosmos.aio`` convention.
"""

from azure.cosmos.agent_memory.aio.cosmos_memory_client import AsyncCosmosMemoryClient
from azure.cosmos.agent_memory.aio.embeddings import AsyncEmbeddingsClient
from azure.cosmos.agent_memory.aio.processors import (
    AsyncDurableFunctionProcessor,
    AsyncInProcessProcessor,
    AsyncMemoryProcessor,
    ProcessThreadResult,
    UserSummaryResult,
)

__all__ = [
    "AsyncCosmosMemoryClient",
    "AsyncEmbeddingsClient",
    "AsyncMemoryProcessor",
    "AsyncInProcessProcessor",
    "AsyncDurableFunctionProcessor",
    "ProcessThreadResult",
    "UserSummaryResult",
]
