"""Synchronous LLM chat completion client for the Agent Memory Toolkit.

Provides :class:`ChatClient` that lazily initialises an ``openai.AzureOpenAI``
connection and generates chat completions via the OpenAI API.  Includes
built-in retry logic with exponential backoff for rate-limit and transient
errors.

The async counterpart lives in :mod:`azure.cosmos.agent_memory.aio.chat` as
:class:`AsyncChatClient`.
"""

from __future__ import annotations

import os
import random
import re
import time
from typing import Any

from azure.cosmos.agent_memory.logging import describe_exception, get_logger

from .exceptions import ConfigurationError, LLMError

logger = get_logger(__name__)

TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"
RETRYABLE_STATUS_CODES = (429, 500, 503)
DEFAULT_AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
SAMPLING_PARAMS = ("temperature", "top_p", "frequency_penalty", "presence_penalty")
MAX_RETRY_AFTER_DELAY = 60.0
RETRY_AFTER_FLOOR = 0.1


def resolve_api_version(explicit: str | None) -> str:
    """Resolve the Azure OpenAI API version to use.

    Precedence: explicit constructor arg → ``AZURE_OPENAI_API_VERSION`` env
    var → built-in default (``2024-12-01-preview``). The env-var hook lets
    ``azd`` deployments and CI environments pin a single version across the
    SDK + function-app without code changes.
    """
    if explicit:
        return explicit
    return os.environ.get("AZURE_OPENAI_API_VERSION") or DEFAULT_AZURE_OPENAI_API_VERSION


def unsupported_param(exc: Exception) -> str | None:
    """If *exc* is a 400 about an unsupported sampling param, return its name."""
    msg = str(exc).lower()
    if "400" not in msg:
        return None
    if not (
        "does not support" in msg
        or "is not supported" in msg
        or "unsupported parameter" in msg
        or "unsupported value" in msg
    ):
        return None
    for p in SAMPLING_PARAMS:
        pattern = rf"(?<![a-z_]){re.escape(p)}(?![a-z_])"
        if re.search(pattern, msg):
            return p
    return None


def retry_after_delay(exc: Exception) -> float | None:
    """Return a Retry-After delay from an OpenAI exception, if present."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    get_header = getattr(headers, "get", None)
    if not callable(get_header):
        return None

    for name, divisor in (("retry-after", 1.0), ("retry-after-ms", 1000.0)):
        value = get_header(name)
        if value is None:
            continue
        try:
            delay = float(value) / divisor
        except (TypeError, ValueError):
            continue
        if delay >= 0:
            return min(max(delay, RETRY_AFTER_FLOOR), MAX_RETRY_AFTER_DELAY)
    return None


def retry_delay(exc: Exception | None, attempt: int, base_delay: float) -> float:
    """Compute retry delay with Retry-After support and jitter."""
    header_delay = retry_after_delay(exc) if exc is not None else None
    if header_delay is not None:
        return min(header_delay * (1.0 + 0.2 * random.random()), MAX_RETRY_AFTER_DELAY)
    return base_delay * (2**attempt) * (0.8 + 0.4 * random.random())


def extract_content(response: Any, model: str) -> str:
    """Pull the assistant content out of a chat-completions response.

    An empty reply (no choices, or no content) returns ``""`` so the caller can
    retry it. A content-filtered reply, a refusal, a tool-only reply (tool calls
    with no text, or only blank text) and a blank reply cut off at the output
    limit (``finish_reason="length"``, as when a reasoning model spends its whole
    budget thinking) raise :class:`LLMError` naming the cause (``content_filter``,
    ``model_refusal``, ``unexpected_tool_response``, ``output_truncated``): asking
    again gets the same answer, so the pipeline classifies them as non-retryable.
    """
    if not response.choices:
        return ""
    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "content_filter":
        raise LLMError(f"LLM content_filter (model={model})")
    refusal = getattr(choice.message, "refusal", None)
    if isinstance(refusal, str) and refusal.strip():
        raise LLMError(f"LLM model_refusal (model={model})")
    content = getattr(choice.message, "content", None)
    blank = content is None or (isinstance(content, str) and not content.strip())
    if blank and finish_reason in {"tool_calls", "function_call"}:
        raise LLMError(f"LLM unexpected_tool_response (model={model}, finish_reason={finish_reason})")
    if blank and finish_reason == "length":
        raise LLMError(f"LLM output_truncated (model={model}, finish_reason={finish_reason})")
    return "" if content is None else content


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
        Azure OpenAI API version.  When ``None`` (default), reads
        ``AZURE_OPENAI_API_VERSION`` from the environment, falling back to
        ``"2024-12-01-preview"``.
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

            token_provider = get_bearer_token_provider(self._credential, TOKEN_SCOPE)
            self._client = AzureOpenAI(
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

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict | None = None,
        max_retries: int = 6,
        base_delay: float = 2.0,
        **extra: Any,
    ) -> str:
        """Call chat completions and return the response content string.

        Retries on rate limit (429), transient errors (500, 503) and empty
        replies, with exponential backoff.

        Any additional keyword arguments (e.g. ``top_p``, ``seed``) are
        forwarded directly to ``client.chat.completions.create`` - this lets
        callers pass through ``model.parameters`` from a prompty file without
        modification.

        Raises
        ------
        ConfigurationError
            If the endpoint or credentials are missing.
        LLMError
            If the reply is still empty after the last retry, or the model returns
            content filtering, a refusal, an unexpected tool-only response, or an
            empty reply cut off at the output limit (``finish_reason="length"``).
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
                content = extract_content(response, self._model)
                if content.strip():
                    return content

                finish_reason = getattr(response.choices[0] if response.choices else None, "finish_reason", "unknown")
                if attempt < max_retries - 1:
                    delay = retry_delay(None, attempt, base_delay)
                    logger.warning(
                        "LLM returned empty content (model=%s, finish_reason=%s); retrying in %.1fs.",
                        self._model,
                        finish_reason,
                        delay,
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise LLMError(
                    f"LLM returned empty content after {attempt + 1} attempts "
                    f"(model={self._model}, finish_reason={finish_reason})"
                )
            except openai.RateLimitError as exc:
                if attempt < max_retries - 1:
                    delay = retry_delay(exc, attempt, base_delay)
                    logger.warning(
                        "LLM rate-limited (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        delay,
                        describe_exception(exc),
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise
            except openai.APIError as exc:
                status = getattr(exc, "status_code", None)
                # Reasoning models (gpt-5, o-series) reject custom sampling
                # parameters with 400. Strip the offending param and retry -
                # this does NOT consume a retry slot since it's a request-shape
                # repair, not a transient failure.
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
                    delay = retry_delay(exc, attempt, base_delay)
                    logger.warning(
                        "LLM API error %s (attempt %d/%d), retrying in %.1fs: %s",
                        status,
                        attempt + 1,
                        max_retries,
                        delay,
                        describe_exception(exc),
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise

    def close(self) -> None:
        """Close the underlying sync HTTP client, if one has been created.

        ``openai.AzureOpenAI`` owns an httpx connection pool that leaks
        across ``with`` blocks unless closed explicitly. Sync callers should
        invoke this from their own ``close()`` to drain the pool.
        """
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            self._client = None
