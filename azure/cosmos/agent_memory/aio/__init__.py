"""Async variants of the Agent Memory Toolkit clients.

Exports :class:`AsyncCosmosMemoryClient`, :class:`AsyncEmbeddingsClient` and the async processors
(:class:`AsyncMemoryProcessor`, :class:`AsyncInProcessProcessor`, :class:`AsyncDurableFunctionProcessor`),
plus the shared :class:`ProcessThreadResult` and :class:`UserSummaryResult`. The async chat client is
:class:`azure.cosmos.agent_memory.aio.chat.AsyncChatClient`; it is not re-exported here. The layout
follows the ``azure.cosmos`` / ``azure.cosmos.aio`` convention.
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
