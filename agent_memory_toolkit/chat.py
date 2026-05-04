"""LLM chat completion client for the Agent Memory Toolkit.

Provides :class:`ChatClient` that lazily initialises an Azure OpenAI
connection and generates chat completions via the OpenAI API.  Includes
built-in retry logic with exponential backoff for rate-limit and transient
errors, mirroring the patterns used in ``activities.py``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .exceptions import ConfigurationError, LLMError

logger = logging.getLogger(__name__)

_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"
_RETRYABLE_STATUS_CODES = (429, 500, 503)
# Sampling parameters that some reasoning models (gpt-5, o-series) reject.
# When the API returns 400 with one of these in the message, we strip it and retry once.
_SAMPLING_PARAMS = ("temperature", "top_p", "frequency_penalty", "presence_penalty")


def _unsupported_param(exc: Exception) -> str | None:
    """If *exc* is a 400 about an unsupported sampling param, return its name."""
    msg = str(exc).lower()
    if "400" not in msg and "unsupported" not in msg and "does not support" not in msg:
        return None
    for p in _SAMPLING_PARAMS:
        if p in msg:
            return p
    return None


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class ChatClient:
    """Synchronous LLM chat completion client backed by Azure OpenAI.

    Parameters
    ----------
    endpoint:
        Azure OpenAI resource endpoint URL.
    credential:
        Optional Azure ``TokenCredential``.  Used when *api_key* is not set
        to obtain bearer tokens for the OpenAI service.
    api_key:
        Optional API key for the Azure OpenAI resource.
    model:
        Deployment / model name.  Defaults to ``"gpt-4o-mini"``.
    api_version:
        Azure OpenAI API version.  Defaults to ``"2024-12-01-preview"``.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        credential: Any = None,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        api_version: str = "2024-12-01-preview",
    ) -> None:
        self._endpoint = endpoint
        self._credential = credential
        self._api_key = api_key
        self._model = model
        self._api_version = api_version
        self._client: Any = None  # openai.AzureOpenAI (lazy)
        self._async_client: Any = None  # openai.AsyncAzureOpenAI (lazy)

    # -- internal helpers ---------------------------------------------------

    def _ensure_client(self) -> Any:
        """Lazily create the ``AzureOpenAI`` client on first use."""
        if self._client is not None:
            return self._client

        if not self._endpoint:
            raise ConfigurationError("An LLM endpoint is required", parameter="endpoint")

        from openai import AzureOpenAI

        if self._api_key:
            self._client = AzureOpenAI(
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
            from azure.identity import get_bearer_token_provider

            token_provider = get_bearer_token_provider(self._credential, _TOKEN_SCOPE)
            self._client = AzureOpenAI(
                api_version=self._api_version,
                azure_endpoint=self._endpoint,
                azure_ad_token_provider=token_provider,
            )

        return self._client

    def _ensure_async_client(self) -> Any:
        """Lazily create the ``AsyncAzureOpenAI`` client on first use."""
        if self._async_client is not None:
            return self._async_client

        if not self._endpoint:
            raise ConfigurationError("An LLM endpoint is required", parameter="endpoint")

        from openai import AsyncAzureOpenAI

        if self._api_key:
            self._async_client = AsyncAzureOpenAI(
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
            from azure.identity.aio import get_bearer_token_provider

            token_provider = get_bearer_token_provider(self._credential, _TOKEN_SCOPE)
            self._async_client = AsyncAzureOpenAI(
                api_version=self._api_version,
                azure_endpoint=self._endpoint,
                azure_ad_token_provider=token_provider,
            )

        return self._async_client

    def _build_kwargs(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        response_format: dict | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        logger.debug(
            "Chat completion request: model=%s, messages=%d",
            self._model,
            len(messages),
        )
        kwargs: dict[str, Any] = {"model": self._model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if response_format is not None:
            kwargs["response_format"] = response_format
        # Pass through any additional model parameters (e.g. top_p, seed)
        # supplied by callers — typically sourced from a prompty file's
        # ``model.parameters`` block.
        kwargs.update(extra)
        return kwargs

    # -- public API ---------------------------------------------------------

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        response_format: dict | None = None,
        max_retries: int = 3,
        base_delay: float = 2.0,
        **extra: Any,
    ) -> str:
        """Call chat completions and return the response content string.

        Retries on rate limit (429) and transient errors (500, 503) with
        exponential backoff, same as the existing retry logic in activities.py.

        Any additional keyword arguments (e.g. ``top_p``, ``seed``) are
        forwarded directly to ``client.chat.completions.create`` — this lets
        callers pass through ``model.parameters`` from a prompty file without
        modification.

        Raises
        ------
        ConfigurationError
            If the endpoint or credentials are missing.
        LLMError
            If the chat completion call fails after all retries.
        """
        import openai

        client = self._ensure_client()
        kwargs = self._build_kwargs(
            messages,
            temperature=temperature,
            response_format=response_format,
            **extra,
        )

        attempt = 0
        while True:
            try:
                response = client.chat.completions.create(**kwargs)
                usage = response.usage
                if usage:
                    logger.info(
                        "LLM usage (model=%s): prompt=%d, completion=%d, total=%d",
                        self._model,
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        usage.total_tokens,
                    )
                return response.choices[0].message.content
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
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise LLMError(f"LLM rate-limited after {max_retries} attempts: {exc}") from exc
            except openai.APIError as exc:
                status = getattr(exc, "status_code", None)
                # Reasoning models (gpt-5, o-series) reject custom sampling
                # parameters with 400. Strip the offending param and retry —
                # this does NOT consume a retry slot since it's a request-shape
                # repair, not a transient failure.
                bad_param = _unsupported_param(exc) if status == 400 else None
                if bad_param and bad_param in kwargs:
                    logger.warning(
                        "LLM model=%s rejected '%s'; retrying without it.",
                        self._model,
                        bad_param,
                    )
                    kwargs.pop(bad_param, None)
                    continue
                if status in _RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "LLM API error %s (attempt %d/%d), retrying in %.1fs: %s",
                        status,
                        attempt + 1,
                        max_retries,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise LLMError(f"LLM chat completion failed (status={status}): {exc}") from exc
            except Exception as exc:
                raise LLMError(f"LLM chat completion failed: {exc}") from exc

        # Should not be reached, but satisfy type checkers.
        raise LLMError("LLM chat completion failed after all retries")  # pragma: no cover

    async def agenerate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        response_format: dict | None = None,
        max_retries: int = 3,
        base_delay: float = 2.0,
        **extra: Any,
    ) -> str:
        """Async version of :meth:`generate`.

        Retries on rate limit (429) and transient errors (500, 503) with
        exponential backoff using ``asyncio.sleep``.

        Any additional keyword arguments are forwarded directly to the OpenAI
        client — typically sourced from a prompty file's ``model.parameters``.

        Raises
        ------
        ConfigurationError
            If the endpoint or credentials are missing.
        LLMError
            If the chat completion call fails after all retries.
        """
        import openai

        client = self._ensure_async_client()
        kwargs = self._build_kwargs(
            messages,
            temperature=temperature,
            response_format=response_format,
            **extra,
        )

        attempt = 0
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
                return response.choices[0].message.content
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
                raise LLMError(f"LLM rate-limited after {max_retries} attempts: {exc}") from exc
            except openai.APIError as exc:
                status = getattr(exc, "status_code", None)
                # Strip-unsupported-param: request-shape repair, not a transient
                # failure — does NOT consume a retry slot.
                bad_param = _unsupported_param(exc) if status == 400 else None
                if bad_param and bad_param in kwargs:
                    logger.warning(
                        "LLM model=%s rejected '%s'; retrying without it.",
                        self._model,
                        bad_param,
                    )
                    kwargs.pop(bad_param, None)
                    continue
                if status in _RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
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
                raise LLMError(f"LLM chat completion failed (status={status}): {exc}") from exc
            except Exception as exc:
                raise LLMError(f"LLM chat completion failed: {exc}") from exc

        raise LLMError("LLM chat completion failed after all retries")  # pragma: no cover

    async def close(self) -> None:
        """Close the underlying async HTTP client, if one has been created."""
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None
