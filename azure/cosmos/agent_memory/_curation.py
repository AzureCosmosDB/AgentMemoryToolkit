"""Secret redaction at the write boundary."""

from __future__ import annotations

import re
from functools import partial
from typing import Any, Callable, Optional
from urllib.parse import unquote

from azure.cosmos.agent_memory._utils import compute_content_hash

SECRET_CLASSIFICATION = "secret"
SECRET_TAG = "sys:secret"
SECRET_REDACTION = "[REDACTED-SECRET]"

# The write guard redacts credentials whose format cannot be mistaken for anything else: private
# keys, Azure account keys and connection strings, SAS signatures, JWTs, Bearer and Basic
# authorization values, OpenAI, GitHub and AWS keys, and passwords inside URLs. It scans every
# non-structural field of a record before the record is embedded or written, which covers raw
# turns and records callers write directly, none of which a model reviews. A format is required
# because length alone proves nothing: a base64 SHA-512 digest looks exactly like an Azure
# account key. Credentials without a fixed format (``password: hunter2`` typed into a
# conversation) are left to the prompts that write memories, each of which forbids storing one;
# a pattern cannot tell such a value from ordinary text without corrupting transcripts.
_PEM_LABEL = r"(?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?"
# Real or JSON-escaped line breaks.
_PEM_NEWLINE = r"(?:\r?\n|(?:\\r)?\\n)"
_PEM_ARMOR_HEADER = r"(?:Proc-Type|DEK-Info|Version|Comment|Hash|Charset|MessageID):[^\r\n\\]*"
_PEM_BODY_LINE = rf"[A-Za-z0-9+/=]+(?=[ \t]*(?:{_PEM_NEWLINE}|[\"']|\Z))"
# A key is a long base64 run. Prose that names the BEGIN and END markers has no word that long.
_PEM_KEY_MATERIAL = re.compile(r"[A-Za-z0-9+/=]{16}")
_TEMPLATE_MARKERS = ("${", "{{", "}}", "#{", "%(", "<%")
# Whole values that stand in for a password: the words, masks (xxxx, ****), and bracketed, printf or
# command-substitution templates. Anything else, however it starts, is taken as a literal password.
_URL_PLACEHOLDER = re.compile(
    r"(?i:password|pass|passwd|pwd|secret)|[*xX.]+|<[^<>]*>|\[[^\[\]]*\]|\{[^{}]*\}|%(?:\d+\$)?[A-Za-z]|\$\([^()]*\)"
)


def _pem_has_key_material(match: re.Match[str]) -> bool:
    return _PEM_KEY_MATERIAL.search(match.group(0)) is not None


def _is_variable_reference(password: str) -> bool:
    """``$DB_PASSWORD`` is a variable; ``$Up3rS3cret`` is a password: its digits sit among both letter cases."""
    name = password[1:]
    if not password.startswith("$") or not name.isidentifier():
        return False
    return not (any(char.isdigit() for char in name) and not (name.islower() or name.isupper()))


def _url_password_is_literal(match: re.Match[str]) -> bool:
    raw = match.group("secret")
    if re.fullmatch(r"%\w+%", raw):  # %VARIABLE%, which decoding would misread ("%DB" is a byte)
        return False
    # Decode first: an encoded key can start "%2B", and "%3Cpassword%3E" is still a template.
    password = unquote(raw)
    if _URL_PLACEHOLDER.fullmatch(password) or any(marker in password for marker in _TEMPLATE_MARKERS):
        return False
    return not _is_variable_reference(password)


_LOWERCASE_PHRASE = re.compile(r"[a-z]+(?:[-_][a-z]+)+")


def _openai_key_is_literal(match: re.Match[str]) -> bool:
    # "sk-learn-extensions-for-forecasting" is a hyphenated phrase of lowercase words; a key is random and has none.
    return _LOWERCASE_PHRASE.fullmatch(match.group("body")) is None


_SecretValidator = Callable[[re.Match[str]], bool]

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str], Optional[_SecretValidator]], ...] = (
    (
        # A block ends at its END marker. Without one before the next BEGIN, it runs through
        # every following armor-header or base64 line, however short, and stops at the first
        # line of anything else. A block without key material, such as prose that names the
        # markers, is left alone.
        "private_key_block",
        re.compile(
            rf"-----BEGIN {_PEM_LABEL}-----"
            rf"(?:(?:(?!-----BEGIN )[\s\S])*?-----END {_PEM_LABEL}-----"
            rf"|(?:[ \t]+[A-Za-z0-9+/=]{{16,}})*"
            rf"(?:(?:[ \t]*{_PEM_NEWLINE})+[ \t]*(?:{_PEM_ARMOR_HEADER}|{_PEM_BODY_LINE}))*)"
        ),
        _pem_has_key_material,
    ),
    ("cosmos_connection_string", re.compile(r"AccountKey\s*=\s*[A-Za-z0-9+/]{40,}={0,2}", re.I), None),
    (
        "azure_key",
        re.compile(
            r"(?<![A-Za-z0-9])(?:SharedAccessKey|primary(?:Readonly)?(?:Master)?Key|secondary(?:Readonly)?(?:Master)?Key"
            r"|(?:account|storage|master)[_-]?key)['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9+/]{40,}={0,2}",
            re.I,
        ),
        None,
    ),
    ("sas_connection", re.compile(r"SharedAccessSignature\s*=\s*[^;\s\"']+", re.I), None),
    ("sas_signature", re.compile(r"[?&]sig=[A-Za-z0-9%+/]{20,}"), None),
    (
        # Starts only where a base64url run starts, so a long run is scanned once, not once per "-".
        "bearer_jwt",
        re.compile(r"(?<![A-Za-z0-9_-])(?:Bearer\s+)?eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
        None,
    ),
    (
        # An opaque token after an explicit Bearer scheme: 20+ token characters with a digit,
        # or 32+ without one, so "Bearer authentication" and similar prose never match.
        "bearer_token",
        re.compile(
            r"(?i:\bbearer)[ \t]+(?:(?=[A-Za-z0-9._~+/-]{0,4096}\d)[A-Za-z0-9._~+/-]{20,}|[A-Za-z0-9._~+/-]{32,})=*"
        ),
        None,
    ),
    (
        "basic_auth",
        re.compile(r"(?i:\bauthorization)['\"]?\s*[:=]\s*['\"]?(?i:basic)[ \t]+[A-Za-z0-9+/]{8,}={0,2}"),
        None,
    ),
    (
        "openai_key",
        re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?(?P<body>[A-Za-z0-9][A-Za-z0-9_-]{31,})"),
        _openai_key_is_literal,
    ),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"), None),
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), None),
    (
        # scheme://user:password@host keeps everything but the password. Matching from "://"
        # rather than the scheme keeps a long dotted or hyphenated word from being rescanned.
        # A "?" or "#" ends the host, so the user part stops at one, and a port followed by "?"
        # (host:3000?email=a@b.com) is a query, not a login. The password may hold a raw "#" or "?".
        "url_credentials",
        re.compile(r"://[^\s/?#:@]*:(?!\d{1,5}\?)(?P<secret>[^\s/@]+)@"),
        _url_password_is_literal,
    ),
)

# Fields that address or route a record, at any depth. They are never rewritten: changing one
# would break the record rather than protect it.
_STRUCTURAL_FIELDS = frozenset(
    {
        "id",
        "type",
        "role",
        "status",
        "user_id",
        "thread_id",
        "agent_id",
        "scope_type",
        "embedding",
        "content_hash",
        "ttl",
        "version",
        "priority",
        "merged_from",
        "superseded_by",
        # The redaction's own markers; rewriting them would erase the record of what was found.
        "classification",
        "secret_patterns",
    }
)
_STRUCTURAL_SUFFIXES = ("_id", "_ids", "_at", "_hash", "_digest", "_ts")


def _is_structural(key: str) -> bool:
    return key.startswith("_") or key in _STRUCTURAL_FIELDS or key.endswith(_STRUCTURAL_SUFFIXES)


def _redaction(validator: Optional[_SecretValidator], match: re.Match[str]) -> str:
    """The replacement for ``match``: the marker, over only its ``secret`` group when it has one."""
    if validator is not None and not validator(match):
        return match.group(0)
    if "secret" not in match.re.groupindex:
        return SECRET_REDACTION
    whole, start = match.group(0), match.start()
    return whole[: match.start("secret") - start] + SECRET_REDACTION + whole[match.end("secret") - start :]


def detect_secrets(text: Any) -> list[str]:
    """Return the names of the credential formats present in ``text``."""
    value = str(text or "")
    return [
        name
        for name, pattern, validator in _SECRET_PATTERNS
        if any(validator is None or validator(match) for match in pattern.finditer(value))
    ]


def redact_secrets(text: Any) -> str:
    """Replace every recognized credential in ``text`` with a redaction marker."""
    value = str(text or "")
    for _name, pattern, validator in _SECRET_PATTERNS:
        value = pattern.sub(partial(_redaction, validator), value)
    return value


def sanitize_text(text: Any) -> tuple[Any, list[str]]:
    """Return ``(redacted_text, matched_pattern_names)``; non-strings pass through untouched."""
    if not isinstance(text, str) or not text:
        return text, []
    matched = detect_secrets(text)
    if not matched:
        return text, []
    return redact_secrets(text), matched


def mark_secret(doc: dict[str, Any], patterns: list[str]) -> None:
    """Flag ``doc`` as having held a credential: ``metadata.classification`` and the ``sys:secret`` tag."""
    if not patterns:
        return
    metadata = dict(doc.get("metadata") or {}) if isinstance(doc.get("metadata"), dict) else {}
    metadata["classification"] = SECRET_CLASSIFICATION
    metadata["secret_patterns"] = sorted(set(metadata.get("secret_patterns") or []) | set(patterns))
    doc["metadata"] = metadata
    tags = doc.get("tags") if isinstance(doc.get("tags"), list) else []
    doc["tags"] = tags if SECRET_TAG in tags else [*tags, SECRET_TAG]


def drop_secret_tags(tags: list[Any]) -> tuple[list[Any], list[str]]:
    """Return ``tags`` without the credential-bearing ones, and the format names found in the dropped ones.

    A tag cannot hold the redaction marker, so it is dropped rather than redacted.
    """
    kept: list[Any] = []
    matched: set[str] = set()
    for tag in tags:
        found = detect_secrets(tag) if isinstance(tag, str) else []
        if found:
            matched.update(found)
        else:
            kept.append(tag)
    return kept, sorted(matched)


def embedding_text(doc: dict[str, Any]) -> str:
    """The text a record's vector is built from: ``retrieval_text`` when it has one (procedures), else ``content``."""
    return str(doc.get("retrieval_text") or doc.get("content") or "")


def _scrub(value: Any, key: Optional[str], matched: set[str]) -> Any:
    if key is not None and _is_structural(key):
        return value
    if isinstance(value, str):
        clean, found = sanitize_text(value)
        matched.update(found)
        return clean
    if isinstance(value, list):
        return [_scrub(item, key, matched) for item in value]
    if isinstance(value, dict):
        return {item_key: _scrub(item, str(item_key), matched) for item_key, item in value.items()}
    return value


def sanitize_record(doc: dict[str, Any]) -> list[str]:
    """Redact recognized credentials from every non-structural field of ``doc`` in place.

    Every string is scanned at any depth, including metadata. A credential-bearing tag is
    dropped, since a tag cannot hold the marker. Derived state follows the redacted text:
    ``content_hash`` is recomputed when ``content`` changes, and the ``embedding`` is dropped
    when the text it was built from (``content`` or ``retrieval_text``) changed, because it
    would still encode the secret; a writer that can embed rebuilds it from the redacted text.
    Returns the matched pattern names; a non-empty result also flags the record via
    ``mark_secret``.
    """
    matched: set[str] = set()
    original_content = doc.get("content")
    embedded_text = (original_content, doc.get("retrieval_text"))
    for key in list(doc):
        if key != "tags":
            doc[key] = _scrub(doc[key], key, matched)
    tags = doc.get("tags")
    if isinstance(tags, list):
        kept, found = drop_secret_tags(tags)
        matched.update(found)
        if len(kept) != len(tags):
            doc["tags"] = kept
    if not matched:
        return []
    if doc.get("content") != original_content and doc.get("content_hash"):
        doc["content_hash"] = compute_content_hash(str(doc.get("content") or ""))
    if (doc.get("content"), doc.get("retrieval_text")) != embedded_text:
        doc.pop("embedding", None)
    patterns = sorted(matched)
    mark_secret(doc, patterns)
    return patterns
