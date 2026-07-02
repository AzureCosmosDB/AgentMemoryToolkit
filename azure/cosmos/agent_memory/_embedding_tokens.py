"""Token-budget helpers for embedding inputs.

Azure OpenAI embedding models (``text-embedding-3-*``) reject any single
input that exceeds their 8192-token context window. Callers in this toolkit
embed user-supplied and document-derived text of unbounded size — the
dedup-context query in extraction, ``search_turns`` queries, and stored turn
vectors when ``enable_turn_embeddings`` is on — so an oversized string would
otherwise crash the embed call or, at the write sites that swallow embed
errors, silently persist a record with no vector.

Every embedding input is therefore capped to a safe token budget before the
API call. Truncation keeps the *head* of the text; preserving the tail of a
large document is a chunking concern handled at the ingestion layer, not
here (the embedder contract is one string -> one vector, never pooled).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from azure.cosmos.agent_memory.logging import get_logger

logger = get_logger(__name__)

# ``text-embedding-3-*`` accept up to 8192 tokens per input. Leave headroom
# below the hard cap for tokenizer drift across model/deployment names and
# any special tokens the service may add server-side.
EMBEDDING_MAX_INPUT_TOKENS = 8000

# Defensive fallback used only when tiktoken cannot be imported (it is a
# declared dependency). English text averages ~4 characters per token.
_FALLBACK_CHARS_PER_TOKEN = 4


@lru_cache(maxsize=8)
def _encoding_for_model(model: str) -> Any:
    """Return a cached tiktoken encoding for *model*.

    Falls back to the ``cl100k_base`` encoding used by ``text-embedding-3-*``
    when the model name is unknown, and to ``None`` when tiktoken itself is
    unavailable so the caller can apply a character-based estimate.
    """
    try:
        import tiktoken
    except Exception:  # pragma: no cover - tiktoken is a declared dependency
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:  # pragma: no cover - defensive
            return None


def truncate_text_to_token_budget(
    text: str,
    model: str,
    max_tokens: int = EMBEDDING_MAX_INPUT_TOKENS,
) -> str:
    """Truncate *text* so it encodes to at most *max_tokens* tokens.

    Returns *text* unchanged when it is already within budget. Truncation
    keeps the leading tokens and logs a warning so oversized inputs remain
    observable — that warning is also the signal for where ingestion-time
    chunking is needed.
    """
    if not text:
        return text

    # Fast path: every token spans at least one character, so any string
    # whose character length is within the token budget cannot exceed it.
    # This skips tokenization for the overwhelming majority of inputs
    # (short turns and queries).
    if len(text) <= max_tokens:
        return text

    encoding = _encoding_for_model(model)
    if encoding is None:
        max_chars = max_tokens * _FALLBACK_CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return text
        logger.warning(
            "Embedding input exceeds ~%d tokens; tiktoken unavailable, "
            "truncating to %d chars (model=%s, original chars=%d)",
            max_tokens,
            max_chars,
            model,
            len(text),
        )
        return text[:max_chars]

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated = encoding.decode(tokens[:max_tokens])
    logger.warning(
        "Embedding input truncated from %d to %d tokens (model=%s) to stay within the embedding context window",
        len(tokens),
        max_tokens,
        model,
    )
    return truncated
