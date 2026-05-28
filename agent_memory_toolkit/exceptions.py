"""Custom exception hierarchy for the Agent Memory Toolkit.

All exceptions inherit from :class:`AgentMemoryError` so callers can catch
a single base class or handle specific failure modes individually.
"""


class AgentMemoryError(Exception):
    """Base exception for all Agent Memory Toolkit errors."""

    error_code: str = ""


class ConfigurationError(AgentMemoryError):
    """Raised when required configuration is missing or invalid.

    Examples: missing cosmos_endpoint, missing ai_foundry_endpoint,
    missing credentials.

    Attributes:
        parameter: The name of the missing or invalid configuration parameter.
    """

    error_code = "configuration"

    def __init__(self, message: str | None = None, *, parameter: str | None = None):
        self.parameter = parameter
        if message is None and parameter:
            message = f"Missing or invalid configuration: {parameter}"
        super().__init__(message or "Missing or invalid configuration")


class ValidationError(AgentMemoryError):
    """Raised when input validation fails.

    Examples: invalid role, invalid memory_type, empty user_id.
    """

    error_code = "validation"


class CosmosNotConnectedError(AgentMemoryError):
    """Raised when a Cosmos DB operation is attempted without an active connection."""

    error_code = "cosmos_not_connected"

    def __init__(self, message: str | None = None):
        super().__init__(message or "Cosmos DB is not connected. Call connect_cosmos() first.")


class CosmosOperationError(AgentMemoryError):
    """Raised when a Cosmos DB operation fails.

    Covers connection issues, query failures, and other Cosmos DB errors.
    """

    error_code = "cosmos_operation"


class MemoryConflictError(AgentMemoryError):
    """Raised when an optimistic-concurrency guarded memory update conflicts."""

    error_code = "memory_conflict"


class MemoryNotFoundError(AgentMemoryError):
    """Raised when a memory document is not found.

    Attributes:
        memory_id: The ID of the memory that was not found.
        user_id: The user ID used in the lookup.
        thread_id: The thread ID used in the lookup.
    """

    error_code = "memory_not_found"

    def __init__(
        self,
        message: str | None = None,
        *,
        memory_id: str | None = None,
        user_id: str | None = None,
        thread_id: str | None = None,
    ):
        self.memory_id = memory_id
        self.user_id = user_id
        self.thread_id = thread_id
        if message is None:
            message = self._build_message()
        super().__init__(message)

    def _build_message(self) -> str:
        parts: list[str] = []
        if self.memory_id:
            parts.append(f"memory_id={self.memory_id!r}")
        if self.user_id:
            parts.append(f"user_id={self.user_id!r}")
        if self.thread_id:
            parts.append(f"thread_id={self.thread_id!r}")
        detail = ", ".join(parts)
        if detail:
            return f"Memory not found ({detail})"
        return "Memory not found"


class LLMError(AgentMemoryError):
    """Raised when the LLM returns a response shape the SDK does not surface
    as an exception (no choices, empty content, invalid JSON)."""

    error_code = "llm"
