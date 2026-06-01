"""Agent Memory Toolkit – local and cloud agent memory management."""

from azure.cosmos.agent_memory.aio import AsyncCosmosMemoryClient
from azure.cosmos.agent_memory.chat import ChatClient
from azure.cosmos.agent_memory.cosmos_memory_client import CosmosMemoryClient
from azure.cosmos.agent_memory.exceptions import (
    AgentMemoryError,
    ConfigurationError,
    CosmosNotConnectedError,
    CosmosOperationError,
    LLMError,
    MemoryConflictError,
    MemoryNotFoundError,
    MemoryTypeMismatchError,
    ValidationError,
)
from azure.cosmos.agent_memory.models import MemoryRecord, MemoryRole, MemoryType, SearchResult
from azure.cosmos.agent_memory.processors import (
    DurableFunctionProcessor,
    InProcessProcessor,
    MemoryProcessor,
    ProcessThreadResult,
    UserSummaryResult,
)
from azure.cosmos.agent_memory.thresholds import (
    DEFAULT_FACT_EXTRACTION_EVERY_N,
    DEFAULT_THREAD_SUMMARY_EVERY_N,
    DEFAULT_USER_SUMMARY_EVERY_N,
    PROCESSOR_OWNER_DURABLE,
    PROCESSOR_OWNER_INPROCESS,
    get_fact_extraction_every_n,
    get_processor_owner,
    get_thread_summary_every_n,
    get_user_summary_every_n,
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
    "ConfigurationError",
    "CosmosNotConnectedError",
    "CosmosOperationError",
    "LLMError",
    "MemoryConflictError",
    "MemoryNotFoundError",
    "MemoryTypeMismatchError",
    "ValidationError",
    "DEFAULT_FACT_EXTRACTION_EVERY_N",
    "DEFAULT_THREAD_SUMMARY_EVERY_N",
    "DEFAULT_USER_SUMMARY_EVERY_N",
    "PROCESSOR_OWNER_DURABLE",
    "PROCESSOR_OWNER_INPROCESS",
    "get_fact_extraction_every_n",
    "get_processor_owner",
    "get_thread_summary_every_n",
    "get_user_summary_every_n",
]
