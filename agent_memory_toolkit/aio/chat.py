"""Async LLM chat completion client for the Agent Memory Toolkit.

Provides :class:`AsyncChatClient` that lazily initialises an
``openai.AsyncAzureOpenAI`` connection and generates chat completions via
the OpenAI API.  Includes built-in retry logic with exponential backoff for
rate-limit and transient errors.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_memory_toolkit.chat import (
    RETRYABLE_STATUS_CODES,
    TOKEN_SCOPE,
    extract_content,
    resolve_api_version,
    unsupported_param,
)
from agent_memory_toolkit.exceptions import ConfigurationError
from agent_memory_toolkit.logging import get_logger

logger = get_logger(__name__)


def _is_async_credential(credential: Any) -> bool:
    """Return True if *credential* is an azure.identity *async* TokenCredential.

    The async variants (e.g. ``azure.identity.aio.DefaultAzureCredential``)
    expose ``get_token`` as a coroutine function. Sync variants expose it as
    a regular method. We detect with :func:`inspect.iscoroutinefunction` so
    we don't have to import the async identity package just to do an
    isinstance check.
    """
    import inspect

    get_token = getattr(credential, "get_token", None)
    return get_token is not None and inspect.iscoroutinefunction(get_token)


def _make_sync_token_provider_for_async(credential: Any, scope: str):
    """Return an async token provider that wraps a *sync* TokenCredential.

    ``AsyncAzureOpenAI`` expects ``azure_ad_token_provider`` to return an
    awaitable yielding the bearer token. When the caller supplied a sync
    ``azure.identity.DefaultAzureCredential`` (the common case), we cannot
    use ``azure.identity.aio.get_bearer_token_provider`` because it
    ``await``s the credential's ``get_token`` — which is not a coroutine on
    sync credentials and would raise at runtime.

    Instead we wrap the sync ``get_token`` call in :func:`asyncio.to_thread`
    so we don't block the event loop, and return the token string directly.
    """

    async def _provider() -> str:
        return (await asyncio.to_thread(credential.get_token, scope)).token

    return _provider


class AsyncChatClient:
    """Asynchronous LLM chat completion client backed by Azure OpenAI.

    Mirrors :class:`agent_memory_toolkit.chat.ChatClient` but uses the
    ``openai.AsyncAzureOpenAI`` client and ``asyncio``-aware retry sleeps.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        credential: Any = None,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        api_version: str | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._credential = credential
        self._api_key = api_key
        self._model = model
        self._api_version = resolve_api_version(api_version)
        self._client: Any = None

    async def __aenter__(self) -> AsyncChatClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def _ensure_client(self) -> Any:
        """Lazily create the ``AsyncAzureOpenAI`` client on first use."""
        if self._client is not None:
            return self._client

        if not self._endpoint:
            raise ConfigurationError("An LLM endpoint is required", parameter="endpoint")

        from openai import AsyncAzureOpenAI

        if self._api_key:
            self._client = AsyncAzureOpenAI(
                api_version=self._api_version,
                azure_endpoint=self._endpoint,
                api_key=self._api_key,
            )
        else:
            if self._credential is None:
                raise ConfigurationError(
                    "Either api_key or a TokenCredential is required for LLM calls",
                    parameter="credential",
                )

            # Detect sync vs async credential. Callers commonly pass a sync
            # ``DefaultAzureCredential`` (from ``azure.identity``) — its
            # ``get_token`` method is *not* awaitable and will hang the async
            # client if used with ``azure.identity.aio.get_bearer_token_provider``.
            # When the credential exposes an ``async def get_token`` we use the
            # async helper directly; otherwise we adapt the sync credential by
            # offloading token acquisition to a worker thread.
            if _is_async_credential(self._credential):
                from azure.identity.aio import get_bearer_token_provider

                token_provider = get_bearer_token_provider(self._credential, TOKEN_SCOPE)
            else:
                token_provider = _make_sync_token_provider_for_async(self._credential, TOKEN_SCOPE)

            self._client = AsyncAzureOpenAI(
                api_version=self._api_version,
                azure_endpoint=self._endpoint,
                azure_ad_token_provider=token_provider,
            )

        return self._client

    def _build_kwargs(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        logger.debug(
            "Chat completion request: model=%s, messages=%d",
            self._model,
            len(messages),
        )
        kwargs: dict[str, Any] = {"model": self._model, "messages": messages}
        if response_format is not None:
            kwargs["response_format"] = response_format
        kwargs.update(extra)
        kwargs["temperature"] = 1.0
        return kwargs

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict | None = None,
        max_retries: int = 3,
        base_delay: float = 2.0,
        **extra: Any,
    ) -> str:
        """Call chat completions and return the response content string.

        Retries on rate limit (429) and transient errors (500, 503) with
        exponential backoff using ``asyncio.sleep``.

        Any additional keyword arguments are forwarded directly to the OpenAI
        client — typically sourced from a prompty file's ``model.parameters``.

        Raises
        ------
        ConfigurationError
            If the endpoint or credentials are missing.
        LLMError
            If the response has no choices or no content (model-side issue
            the SDK does not surface as an exception).
            openai.RateLimitError, openai.APIError
            Propagated from the SDK after retries are exhausted.
        """
        import openai

        client = self._ensure_client()
        kwargs = self._build_kwargs(
            messages,
            response_format=response_format,
            **extra,
        )

        attempt = 0
        unsupported_strips = 0
        max_unsupported_strips = 5
        while True:
            try:
                response = await client.chat.completions.create(**kwargs)
                usage = response.usage
                if usage:
                    logger.info(
                        "LLM usage (model=%s): prompt=%d, completion=%d, total=%d",
                        self._model,
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        usage.total_tokens,
                    )
                return extract_content(response, self._model)
            except openai.RateLimitError as exc:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "LLM rate-limited (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                raise
            except openai.APIError as exc:
                status = getattr(exc, "status_code", None)
                bad_param = unsupported_param(exc) if status == 400 else None
                if bad_param and bad_param in kwargs and unsupported_strips < max_unsupported_strips:
                    logger.warning(
                        "LLM model=%s rejected '%s'; retrying without it.",
                        self._model,
                        bad_param,
                    )
                    kwargs.pop(bad_param, None)
                    unsupported_strips += 1
                    continue
                if status in RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "LLM API error %s (attempt %d/%d), retrying in %.1fs: %s",
                        status,
                        attempt + 1,
                        max_retries,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                raise

    async def close(self) -> None:
        """Close the underlying async HTTP client, if one has been created."""
        if self._client is not None:
            await self._client.close()
            self._client = None
