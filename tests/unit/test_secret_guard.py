"""Secret redaction at the write boundary.

The guard redacts only credentials whose format is unmistakable: private keys, Azure keys and
connection strings, SAS signatures, JWTs, Bearer and Basic authorization values, OpenAI, GitHub
and AWS keys, and passwords inside URLs. It runs on every write, including raw turns, where a
false positive corrupts the source transcript, so the ordinary technical text it must leave alone
is covered as closely as the formats. Credentials without a fixed format, such as
``password: hunter2``, are the prompts' job (see test_memory_writing_prompts_forbid_literal_secrets).
"""

from __future__ import annotations

import base64
import hashlib
import time
from urllib.parse import quote

import pytest

from azure.cosmos.agent_memory._curation import (
    SECRET_REDACTION,
    detect_secrets,
    mark_secret,
    redact_secrets,
    sanitize_record,
    sanitize_text,
)
from azure.cosmos.agent_memory._utils import compute_content_hash

# A base64 SHA-512 digest has exactly the shape of an Azure account key: 88 chars ending "==".
KEY = base64.b64encode(hashlib.sha512(b"account").digest()).decode()
# A GitHub token, built here so the source holds no token literal.
TOKEN = "ghp_" + "A1b2" * 9
PEM_BODY = (
    "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKj\n"
    "MzEfYyjiWA4R4/M2bS1GB4t7NXp98C3SC6dVMvDuictGeurT8jNbvJZHtCSuYEvu"
)


# One credential of each recognized format. None is real. Every token is built from parts
# because push protection blocks those formats as literals.
JWT = ".".join(["eyJhbGciOiJIUzI1NiJ9", "eyJzdWIiOiIxMjM0NTY3ODkwIn0", "dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gF"])
BEARER_TOKEN = "8f2c1a9e4b7d6f3a0c5e9b2d8a4f1c7e"
CURL_BEARER_TOKEN = "xox" + "b-1234567890-abcdefghijklmnop"
DB_PASSWORD = "s3cretpass"
OPENAI_KEY = "sk-" + "abcdefghijklmnopqrstuvwxyz0123456789ABCD"
OPENAI_PROJECT_KEY = "sk-proj-" + "AbCdEf0123456789_ghIJklMNopQRstUVwxYZ-0123456789abcd"
# Not every sk- key is mixed case: some providers issue 32 lowercase hex digits, or lowercase letters alone.
OPENAI_HEX_KEY = "sk-" + "0123456789abcdef" * 2
OPENAI_LOWERCASE_KEY = "sk-" + "abcdefghijklmnopqrstuvwxyz" * 2
GITHUB_CLASSIC_TOKEN = "ghp_" + "abcdefghijklmnopqrstuvwxyz0123456789AB"
GITHUB_FINE_GRAINED_TOKEN = "github_pat_" + "11ABCDEFG0123456789_abcdefghijklmnop"
AWS_ACCESS_KEY_ID = "AKIA" + "IOSFODNN7EXAMPLE"
BASIC_CREDENTIALS = "dXNlcjpodW50ZXIyc2VjcmV0"
SAS_SIGNATURE = "abcdefghijklmnopqrstuvwxyz0123"


def _surviving_pieces(secret: str, text: str, size: int = 5) -> list[str]:
    """The pieces of ``secret`` still in ``text``; a partly redacted credential leaves some behind."""
    return [secret[i : i + size] for i in range(len(secret) - size + 1) if secret[i : i + size] in text]


@pytest.mark.parametrize(
    "text,secret,kept,expected",
    [
        (
            f"The user's credentials are `AccountEndpoint=https://a.documents.azure.com:443/;AccountKey={KEY};`.",
            KEY,
            "The user's credentials are `AccountEndpoint=https://a.documents.azure.com:443/;",
            "cosmos_connection_string",
        ),
        (f'"primaryMasterKey": "{KEY}"', KEY, '"', "azure_key"),
        (f"storage_key = {KEY}", KEY, "", "azure_key"),
        (f"Bearer {JWT}", JWT, "", "bearer_jwt"),
        (OPENAI_KEY, OPENAI_KEY, "", "openai_key"),
        (OPENAI_PROJECT_KEY, OPENAI_PROJECT_KEY, "", "openai_key"),
        (OPENAI_HEX_KEY, OPENAI_HEX_KEY, "", "openai_key"),
        (OPENAI_LOWERCASE_KEY, OPENAI_LOWERCASE_KEY, "", "openai_key"),
        (GITHUB_CLASSIC_TOKEN, GITHUB_CLASSIC_TOKEN, "", "github_token"),
        (GITHUB_FINE_GRAINED_TOKEN, GITHUB_FINE_GRAINED_TOKEN, "", "github_token"),
        (AWS_ACCESS_KEY_ID, AWS_ACCESS_KEY_ID, "", "aws_access_key_id"),
        (f"Authorization: Bearer {BEARER_TOKEN}", BEARER_TOKEN, "Authorization: ", "bearer_token"),
        (
            f"curl -H 'Authorization: Bearer {CURL_BEARER_TOKEN}'",
            CURL_BEARER_TOKEN,
            "curl -H 'Authorization: ",
            "bearer_token",
        ),
        (f"Authorization: Basic {BASIC_CREDENTIALS}", BASIC_CREDENTIALS, "", "basic_auth"),
        (
            f"postgres://admin:{DB_PASSWORD}@db.example.com:5432/app",
            DB_PASSWORD,
            "postgres://admin:",
            "url_credentials",
        ),
        ("redis://:hunter2secret@cache:6379", "hunter2secret", "redis://:", "url_credentials"),
        (
            "postgres://admin:$Up3rS3cret!@db.example.com:5432/app",
            "$Up3rS3cret!",
            "postgres://admin:",
            "url_credentials",
        ),
        ("postgres://admin:$Up3rS3cret@db.example.com/app", "$Up3rS3cret", "postgres://admin:", "url_credentials"),
        ("mongodb://svc:*Kp9#qL2v@cluster.example.com:27017/db", "*Kp9#qL2v", "mongodb://svc:", "url_credentials"),
        ("postgres://admin:Secretx@db.example.com/app", "Secretx", "postgres://admin:", "url_credentials"),
        ("postgres://admin:Pa?ssw0rd@db.example.com/app", "Pa?ssw0rd", "postgres://admin:", "url_credentials"),
        (
            f"https://a.blob.core.windows.net/c?sv=2021&sig={SAS_SIGNATURE}",
            SAS_SIGNATURE,
            "https://a.blob.core.windows.net/c?sv=2021",
            "sas_signature",
        ),
        (
            "BlobEndpoint=https://a.blob.core.windows.net/;SharedAccessSignature=sv=2021-06-08&sp=rl",
            "sv=2021-06-08&sp=rl",
            "BlobEndpoint=https://a.blob.core.windows.net/;",
            "sas_connection",
        ),
    ],
    ids=[
        "cosmos-connection-string-in-a-sentence",
        "azure-master-key",
        "azure-storage-key",
        "bearer-jwt",
        "openai-key",
        "openai-project-key",
        "openai-hex-key",
        "openai-lowercase-key",
        "github-classic-token",
        "github-fine-grained-token",
        "aws-access-key-id",
        "bearer-token-header",
        "bearer-token-in-curl",
        "basic-auth",
        "postgres-url-password",
        "redis-url-password",
        "url-password-after-dollar-with-symbol",
        "url-password-after-dollar",
        "url-password-with-leading-asterisk-and-hash",
        "url-password-ending-in-x",
        "url-password-with-question-mark",
        "sas-signature",
        "sas-connection-string",
    ],
)
def test_detects_and_removes_literal_credentials(text, secret, kept, expected):
    assert secret in text and text.startswith(kept)

    out = redact_secrets(text)

    assert expected in detect_secrets(text)
    assert not _surviving_pieces(secret, out), "no part of the secret is stored"
    assert SECRET_REDACTION in out
    assert out.startswith(kept), "the text before the credential is kept"


@pytest.mark.parametrize(
    "pem",
    [
        f"-----BEGIN PRIVATE KEY-----\n{PEM_BODY}\n-----END PRIVATE KEY-----",
        f"-----BEGIN RSA PRIVATE KEY-----\n{PEM_BODY}",
        f"-----BEGIN OPENSSH PRIVATE KEY----- {PEM_BODY.replace(chr(10), ' ')} -----END OPENSSH PRIVATE KEY-----",
        f"-----BEGIN PGP PRIVATE KEY BLOCK-----\n{PEM_BODY}\n-----END PGP PRIVATE KEY BLOCK-----",
        f"-----BEGIN PRIVATE KEY-----\n{PEM_BODY}\n-----BEGIN PRIVATE KEY-----\n{PEM_BODY}\n-----END PRIVATE KEY-----",
        f"> -----BEGIN PRIVATE KEY-----\n> {PEM_BODY.replace(chr(10), chr(10) + '> ')}\n> -----END PRIVATE KEY-----",
    ],
    ids=["complete", "truncated", "single-line", "pgp", "unterminated-then-complete", "quoted-lines"],
)
def test_private_key_block_is_redacted_whole(pem):
    """Replacing only the BEGIN header left the entire key body stored."""
    out = redact_secrets(f"key follows:\n{pem}\nthat was it")

    assert "MIIEvQIBADANBgkqhkiG9w0" not in out
    assert "C3SC6dVMvDuictGeurT8jNbvJZHtCSuYEvu" not in out
    assert "PRIVATE KEY" not in out, "the BEGIN and END lines go with the body"
    assert out.startswith("key follows:")


@pytest.mark.parametrize(
    "text",
    [
        f'"integrity": "sha512-{KEY}"',
        "h1:abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWX=",
        'api_key = os.getenv("OPENAI_API_KEY")',
        'secret = config.get("storage")',
        "password: ${{ secrets.DB_PASSWORD }}",
        "api_key = settings.API_KEY_V2",
        "api_key: Optional[str] = None",
        "api_key: SecretStr",
        "token = get_access_token()",
        "password=$DB_PASSWORD",
        'api_key = "YOUR_API_KEY_HERE"',
        'secret = "config/secret-name"',
        "the password must be rotated every 90 days",
        "pushed commit 1196a3ac2e to Azure/azure-sdk-for-rust#5366",
        "continuation tokens are not supported for ORDER BY",
        "",
        "connect(host, password=password)",
        "password: str",
        '"password": "string"',
        '"password": null',
        "api_key: PasswordType",
        "password: !vault |",
        'password: "must be at least 8 characters"',
        "PWD=/Users/me/project",
        "PWD=C:\\Users\\me",
        "DB_PASSWORD_FILE=/run/secrets/db",
        "PASSWORD=${DB_PASSWORD}",
        "login(user=username, password=entered)",
        "self.password = password",
        'api_key = "OPENAI_API_KEY"',
        "apiKey: Promise<string>",
        "password: Password",
        "Password: I reset it this morning",
        "Use Bearer authentication for the API",
        "postgres://user:<password>@host/db",
        "https://user:password@host/docs-example",
        "mongodb://user:%DB_PASSWORD%@host:10255/",
        '"postgresql://%s:%s@%s/%s" % (user, password, host, db)',
        "postgres://user:%3Cpassword%3E@host/db",
        "postgres://user:%24%7BDB_PASS%7D@host/db",
        "http://localhost:3000?email=jane@contoso.com",
        "postgres://user:$DB_PASSWORD@host/db",
        "postgres://user:${DB_PASSWORD}@host/db",
        "postgresql://postgres:[YOUR-PASSWORD]@db.example.com:5432/postgres",
        "mongodb://svc:{password}@cluster.example.com:27017/db",
        "postgres://user:xxxxxxxx@host/db",
        "postgres://user:********@host/db",
        "postgres://user:%1$s@host/db",
        "A PEM file starts with -----BEGIN RSA PRIVATE KEY----- and ends with -----END RSA PRIVATE KEY-----.",
        "Never paste -----BEGIN RSA PRIVATE KEY----- into a chat",
        "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----",
        "-----BEGIN PRIVATE KEY-----\n<your key here>\n-----END PRIVATE KEY-----",
        'PEM_START = b"-----BEGIN OPENSSH PRIVATE KEY-----"\nPEM_END = b"-----END OPENSSH PRIVATE KEY-----"',
        "sk-learn-extensions-for-time-series-forecasting",
        "sk-proj-roadmap-for-the-next-quarter-planning_notes",
        "123e4567-e89b-12d3-a456-426614174000",
        "1196a3ac2e4b7d6f3a0c5e9b2d8a4f1c7e9b0a2d",
    ],
    ids=[
        "sha512-digest",
        "go-sum-hash",
        "os-getenv",
        "config-get",
        "actions-secret-ref",
        "settings-attr",
        "type-hint",
        "secretstr-annotation",
        "function-call",
        "env-var-ref",
        "placeholder",
        "path",
        "prose",
        "commit-sha",
        "continuation-token",
        "empty",
        "kwarg-identifier",
        "type-name",
        "json-schema-type",
        "json-null",
        "pascalcase-annotation",
        "yaml-tag",
        "quoted-prose",
        "shell-pwd-path",
        "windows-pwd-path",
        "secret-file-path",
        "env-var-ref-assignment",
        "call-argument",
        "attribute-passthrough",
        "env-var-name",
        "generic-type",
        "type-named-like-key",
        "line-prose",
        "bearer-prose",
        "url-placeholder",
        "url-example",
        "url-windows-env-var",
        "url-printf-format",
        "url-encoded-placeholder",
        "url-encoded-template",
        "url-query-with-an-email",
        "url-shell-variable",
        "url-brace-variable",
        "url-bracket-placeholder",
        "url-curly-placeholder",
        "url-mask-of-x",
        "url-mask-of-asterisks",
        "url-positional-format",
        "pem-markers-in-prose",
        "pem-begin-marker-in-prose",
        "pem-elided-body",
        "pem-placeholder-body",
        "pem-markers-as-constants",
        "sk-hyphenated-phrase",
        "sk-project-phrase",
        "uuid",
        "git-sha",
    ],
)
def test_ordinary_technical_text_is_left_alone(text):
    assert detect_secrets(text) == []
    assert redact_secrets(text) == text


PEM_TRUNCATED_BODY = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7"


@pytest.mark.parametrize(
    "text,kept",
    [
        (
            "key:\n-----BEGIN RSA PRIVATE KEY-----\nProc-Type: 4,ENCRYPTED\nDEK-Info: AES-128-CBC,ABCDEF0123456789"
            f"\n\n{PEM_TRUNCATED_BODY}\nABCDEF0123",
            "key:\n",
        ),
        (f"-----BEGIN PRIVATE KEY-----\n{PEM_TRUNCATED_BODY}\nABCDEF0123\nthen we deployed", "\nthen we deployed"),
        (f"-----BEGIN PRIVATE KEY-----\n{PEM_TRUNCATED_BODY}\n\nThe deploy failed", "\n\nThe deploy failed"),
        (f'{{"key": "-----BEGIN PRIVATE KEY-----\\n{PEM_TRUNCATED_BODY}\\nABCDEF0123"}}', '"}'),
    ],
    ids=["armor-headers", "short-last-line", "prose-after-blank-line", "json-escaped"],
)
def test_an_unterminated_private_key_is_redacted_through_its_last_line(text, kept):
    """Stopping at armor headers or at a short final line left key material stored."""
    out = redact_secrets(text)

    assert PEM_TRUNCATED_BODY not in out
    assert "ABCDEF0123" not in out
    assert out.endswith(kept) or out.startswith(kept), "the text around the key survives"


def test_url_credentials_keep_the_scheme_user_and_host():
    out = redact_secrets("DATABASE_URL=postgres://admin:s3cretpass@db.example.com:5432/app")

    assert out == f"DATABASE_URL=postgres://admin:{SECRET_REDACTION}@db.example.com:5432/app"


@pytest.mark.parametrize("first", ["+", "/"])
def test_a_percent_encoded_url_password_is_redacted(first):
    """URL encoding turns a key's leading "+" or "/" into "%2B" or "%2F", which used to read as a placeholder."""
    url = f"mongodb://acct:{quote(first + KEY[1:], safe='')}@acct.mongo.cosmos.azure.com:10255/"

    assert redact_secrets(url) == f"mongodb://acct:{SECRET_REDACTION}@acct.mongo.cosmos.azure.com:10255/"


def test_redaction_is_idempotent():
    """Records are sanitized again at every write boundary; a second pass must find nothing."""
    text = (
        f"{TOKEN}\nAccountKey={KEY};\n-----BEGIN PRIVATE KEY-----\n{PEM_BODY}\n-----END PRIVATE KEY-----\n"
        "Authorization: Bearer 8f2c1a9e4b7d6f3a0c5e9b2d8a4f1c7e and postgres://a:s3cretpass@h/db"
    )
    once = redact_secrets(text)

    assert detect_secrets(once) == []
    assert redact_secrets(once) == once


@pytest.mark.parametrize(
    "text",
    ['{"password": "correct horse battery staple"}', '{"password": "abc"}', "password: abcde"],
)
def test_credentials_without_a_fixed_format_are_left_to_the_prompts(text):
    """A pattern cannot tell a bare password from ordinary text without corrupting transcripts, so
    the guard does not try; every memory-writing prompt forbids storing one instead."""
    doc = {"content": text, "data": {"password": "abc"}}

    assert detect_secrets(text) == []
    assert sanitize_record(doc) == []
    assert doc == {"content": text, "data": {"password": "abc"}}


@pytest.mark.parametrize(
    "text",
    [
        "a." * 50_000,
        "eyJ-" * 50_000,
        "-----BEGIN PRIVATE KEY-----" * 5_000,
        "password=x://" * 20_000,
        "sk-" * 50_000,
        "-----BEGIN PRIVATE KEY----- and ends with " * 5_000,
    ],
    ids=[
        "dotted-word",
        "base64url-run",
        "unterminated-key-markers",
        "scheme-separators",
        "sk-prefix-run",
        "markers-in-prose",
    ],
)
def test_redaction_time_grows_linearly_with_input(text):
    """Each input took from seconds to minutes when a pattern rescanned a run once per start."""
    start = time.perf_counter()
    sanitize_record({"content": text})
    assert time.perf_counter() - start < 2.0


def test_sanitize_text_passes_non_strings_and_clean_text_through():
    assert sanitize_text(None) == (None, [])
    assert sanitize_text("") == ("", [])
    assert sanitize_text("Prefers Ruff.") == ("Prefers Ruff.", [])
    assert sanitize_text(f"token {TOKEN}") == (f"token {SECRET_REDACTION}", ["github_token"])


def test_sanitize_record_reaches_every_free_text_field():
    """Derived and nested fields carry the same text as content and must be cleaned too.

    Every field is scanned except structural ones (ids, routing keys), which are never rewritten.
    """
    secret = f"token {TOKEN}"
    doc = {
        "id": "proc_1",
        "type": "procedural",
        "thread_id": secret,
        "content": f"Deploy with {secret}",
        "name": f"Deploy {secret}",
        "summary": f"Use {secret}",
        "retrieval_text": f"deploy {secret}",
        "steps": [{"sequence": 1, "instruction": f"login {secret}", "tool_name": f"az --{secret}"}],
        "safety_constraints": [f"never log {secret}"],
        "events": [{"sequence": 1, "description": f"ran {secret}"}],
        "outcome": {"status": "successful", "description": f"done {secret}"},
        "lessons": [f"rotate {secret}"],
        "user_id": "u1",
        "source_turn_ids": [secret],
    }

    assert sanitize_record(doc) == ["github_token"]

    structural = {"thread_id", "source_turn_ids"}
    assert TOKEN not in repr({key: value for key, value in doc.items() if key not in structural})
    assert doc["steps"][0]["tool_name"] == f"az --token {SECRET_REDACTION}"
    assert doc["thread_id"] == secret, "structural fields are never rewritten"
    assert doc["source_turn_ids"] == [secret]
    assert doc["user_id"] == "u1"
    assert doc["metadata"]["classification"] == "secret"
    assert "sys:secret" in doc["tags"]


def test_sanitize_record_scans_the_metadata_model_output_lands_in():
    """A fact's temporal_context and a summary's structured_summary are model text too."""
    doc = {
        "content": "Rotated the staging credentials.",
        "embedding": [0.1],
        "metadata": {
            "category": "other",
            "temporal_context": f"token {TOKEN} until Friday",
            "structured_summary": {"overview": "ok", "key_facts": [f"used token {TOKEN}"]},
            "secret_patterns": ["github_token"],
        },
    }

    assert sanitize_record(doc) == ["github_token"]

    assert TOKEN not in repr(doc)
    assert doc["metadata"]["category"] == "other"
    assert doc["metadata"]["classification"] == "secret"
    assert doc["embedding"] == [0.1], "the vector was built from content, which did not change"


@pytest.mark.parametrize(
    "doc,embedding_kept",
    [
        ({"content": f"Deploy with token {TOKEN}"}, False),
        ({"content": "Log in to staging.", "retrieval_text": f"login with token {TOKEN}"}, False),
        ({"content": "Log in to staging.", "retrieval_text": "log in to staging", "steps": [f"token {TOKEN}"]}, True),
        ({"content": "Log in to staging.", "metadata": {"temporal_context": f"token {TOKEN} until Friday"}}, True),
    ],
    ids=["content", "retrieval-text", "steps-only", "metadata-only"],
)
def test_sanitize_record_drops_the_embedding_only_when_its_source_text_changed(doc, embedding_kept):
    """The vector is built from content or retrieval_text; once either is redacted it still encodes the secret."""
    doc = {**doc, "content_hash": compute_content_hash(doc["content"]), "embedding": [0.1, 0.2]}

    assert sanitize_record(doc) == ["github_token"]

    assert TOKEN not in repr(doc)
    assert doc["content_hash"] == compute_content_hash(doc["content"])
    assert ("embedding" in doc) is embedding_kept


def test_sanitize_record_drops_credential_bearing_tags():
    """A tag cannot hold the redaction marker, so it is removed instead."""
    doc = {"content": "clean", "tags": ["topic:deploy", "db:postgres://app:S3cretPass1@db1", "ghp_" + "a" * 36]}

    assert sanitize_record(doc) == ["github_token", "url_credentials"]

    assert doc["tags"] == ["topic:deploy", "sys:secret"]


def test_sanitize_record_leaves_clean_records_untouched():
    doc = {"content": "The user prefers Ruff for linting.", "content_hash": "h", "embedding": [0.1], "tags": ["t"]}

    assert sanitize_record(doc) == []
    assert doc == {
        "content": "The user prefers Ruff for linting.",
        "content_hash": "h",
        "embedding": [0.1],
        "tags": ["t"],
    }


def test_mark_secret_does_not_mutate_the_callers_nested_objects():
    metadata = {"source": "user"}
    tags = ["sys:fact"]
    doc = {"metadata": metadata, "tags": tags}

    mark_secret(doc, ["github_token"])

    assert doc["metadata"]["classification"] == "secret"
    assert "sys:secret" in doc["tags"]
    assert metadata == {"source": "user"}
    assert tags == ["sys:fact"]


def test_mark_secret_is_a_no_op_without_patterns():
    doc = {"metadata": {}, "tags": []}
    mark_secret(doc, [])
    assert doc == {"metadata": {}, "tags": []}
