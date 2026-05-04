"""Agent Memory Toolkit – local and cloud agent memory management."""

from agent_memory_toolkit.aio import AsyncCosmosMemoryClient
from agent_memory_toolkit.chat import ChatClient
from agent_memory_toolkit.cosmos_memory_client import CosmosMemoryClient
from agent_memory_toolkit.exceptions import (
    AgentMemoryError,
    AuthenticationError,
    ConfigurationError,
    CosmosNotConnectedError,
    CosmosOperationError,
    DuplicateMemoryError,
    EmbeddingError,
    LLMError,
    MemoryNotFoundError,
    OrchestrationTimeoutError,
    ProcessingError,
    ValidationError,
)
from agent_memory_toolkit.models import MemoryRecord, MemoryRole, MemoryType, SearchResult
from agent_memory_toolkit.processors import (
    DurableFunctionProcessor,
    InProcessProcessor,
    MemoryProcessor,
    ProcessThreadResult,
    UserSummaryResult,
)

__all__ = [
    "CosmosMemoryClient",
    "AsyncCosmosMemoryClient",
    "ChatClient",
    "MemoryRecord",
    "MemoryRole",
    "MemoryType",
    "SearchResult",
    "MemoryProcessor",
    "InProcessProcessor",
    "DurableFunctionProcessor",
    "ProcessThreadResult",
    "UserSummaryResult",
    "AgentMemoryError",
    "AuthenticationError",
    "ConfigurationError",
    "CosmosNotConnectedError",
    "CosmosOperationError",
    "DuplicateMemoryError",
    "EmbeddingError",
    "LLMError",
    "MemoryNotFoundError",
    "OrchestrationTimeoutError",
    "ProcessingError",
    "ValidationError",
]
