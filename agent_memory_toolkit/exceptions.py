"""Custom exception hierarchy for the Agent Memory Toolkit.

All exceptions inherit from :class:`AgentMemoryError` so callers can catch
a single base class or handle specific failure modes individually.
"""


class AgentMemoryError(Exception):
    """Base exception for all Agent Memory Toolkit errors."""


class ConfigurationError(AgentMemoryError):
    """Raised when required configuration is missing or invalid.

    Examples: missing cosmos_endpoint, missing ai_foundry_endpoint,
    missing credentials.

    Attributes:
        parameter: The name of the missing or invalid configuration parameter.
    """

    def __init__(self, message: str | None = None, *, parameter: str | None = None):
        self.parameter = parameter
        if message is None and parameter:
            message = f"Missing or invalid configuration: {parameter}"
        super().__init__(message or "Missing or invalid configuration")


class ValidationError(AgentMemoryError):
    """Raised when input validation fails.

    Examples: invalid role, invalid memory_type, empty user_id.
    """


class CosmosNotConnectedError(AgentMemoryError):
    """Raised when a Cosmos DB operation is attempted without an active connection."""

    def __init__(self, message: str | None = None):
        super().__init__(
            message or "Cosmos DB is not connected. Call connect_cosmos() first."
        )


class CosmosOperationError(AgentMemoryError):
    """Raised when a Cosmos DB operation fails.

    Covers connection issues, query failures, and other Cosmos DB errors.
    """


class MemoryNotFoundError(AgentMemoryError):
    """Raised when a memory document is not found.

    Attributes:
        memory_id: The ID of the memory that was not found.
        user_id: The user ID used in the lookup.
        thread_id: The thread ID used in the lookup.
    """

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


class EmbeddingError(AgentMemoryError):
    """Raised when embedding generation fails."""


class ProcessingError(AgentMemoryError):
    """Raised when the processing pipeline (Azure Durable Functions) returns an error."""


class OrchestrationTimeoutError(AgentMemoryError):
    """Raised when polling for an orchestration result exceeds the timeout.

    Attributes:
        timeout: The timeout value in seconds that was exceeded.
        status_url: The URL to check the orchestration status.
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        timeout: float | None = None,
        status_url: str | None = None,
    ):
        self.timeout = timeout
        self.status_url = status_url
        if message is None:
            message = self._build_message()
        super().__init__(message)

    def _build_message(self) -> str:
        msg = "Orchestration timed out"
        if self.timeout is not None:
            msg = f"Orchestration did not complete within {self.timeout}s"
        if self.status_url:
            msg += f". Check status at: {self.status_url}"
        return msg


class AuthenticationError(AgentMemoryError):
    """Raised when authentication to Azure services fails."""
