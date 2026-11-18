"""The record key: ``[tenant_id, user_id]``, the tenant a client is bound to, and the filters that confine a query."""

from __future__ import annotations

import pytest

from azure.cosmos.agent_memory._partitioning import (
    DEFAULT_TENANT_ID,
    MAX_OWNER_ID_BYTES,
    MAX_TENANT_ID_BYTES,
    MAX_THREAD_ID_BYTES,
    OWNER_FILTER,
    PARTITION_KEY_PATHS,
    TENANT_FILTER,
    owner_parameters,
    partition_key_for,
    partition_key_problem,
    resolve_tenant_id,
    stamp_scope,
    tenant_parameters,
    validate_owner_id,
    validate_thread_id,
)
from azure.cosmos.agent_memory.exceptions import ValidationError

INVALID_IDS = [
    "",
    "   ",
    "a/b",
    "a\\b",
    "what?",
    "tag#1",
    "line\nbreak",
    "tab\there",
    7,
    b"x",
    # DEL, two C1 controls (NEL and APC) and the line and paragraph separators. They sit inside the value so that they
    # are refused as control characters, not as whitespace at an end.
    "a\x7fb",
    "a\x85b",
    "a\x9fb",
    "a\u2028b",
    "a\u2029b",
    # Whitespace at either end: Cosmos DB refuses a document id that ends in it, and these ids appear in the ids of
    # counters and summaries.
    " a",
    "a ",
    "a\n",
    "a\xa0",
]

# Joined by zero-width joiners, which are format characters, not controls.
FAMILY_EMOJI = "\U0001f468\u200d\U0001f469\u200d\U0001f467"


class TestResolveTenantId:
    def test_no_tenant_is_the_default_tenant(self) -> None:
        assert resolve_tenant_id() == DEFAULT_TENANT_ID == "default"
        assert resolve_tenant_id(None) == "default"

    @pytest.mark.parametrize(
        "tenant",
        ["acme", "Acme-Corp_2", "tenant:prod", "t" * MAX_TENANT_ID_BYTES, "société", "a b", FAMILY_EMOJI],
    )
    def test_a_valid_tenant_is_returned_as_given(self, tenant: str) -> None:
        assert resolve_tenant_id(tenant) == tenant

    @pytest.mark.parametrize("tenant", [*INVALID_IDS, "t" * (MAX_TENANT_ID_BYTES + 1)])
    def test_an_invalid_tenant_is_refused(self, tenant: object) -> None:
        with pytest.raises(ValidationError):
            resolve_tenant_id(tenant)  # type: ignore[arg-type]

    def test_the_byte_limit_counts_utf8_bytes_not_characters(self) -> None:
        assert resolve_tenant_id("é" * (MAX_TENANT_ID_BYTES // 2)) == "é" * (MAX_TENANT_ID_BYTES // 2)
        with pytest.raises(ValidationError):
            resolve_tenant_id("é" * (MAX_TENANT_ID_BYTES // 2 + 1))


class TestValidateOwnerId:
    @pytest.mark.parametrize(
        "owner",
        [
            "alice",
            "agent:support-bot",
            "a@b.com",
            "user.name_1",
            "o" * MAX_OWNER_ID_BYTES,
            "société",
            "用户",
            "a b",
            FAMILY_EMOJI,
        ],
    )
    def test_a_valid_owner_is_returned_as_given(self, owner: str) -> None:
        assert validate_owner_id(owner) == owner

    @pytest.mark.parametrize("owner", [*INVALID_IDS, None, "o" * (MAX_OWNER_ID_BYTES + 1)])
    def test_an_invalid_owner_is_refused(self, owner: object) -> None:
        with pytest.raises(ValidationError, match="user_id"):
            validate_owner_id(owner)

    def test_the_byte_limit_counts_utf8_bytes_not_characters(self) -> None:
        assert validate_owner_id("é" * (MAX_OWNER_ID_BYTES // 2))
        with pytest.raises(ValidationError):
            validate_owner_id("é" * (MAX_OWNER_ID_BYTES // 2 + 1))


class TestValidateThreadId:
    @pytest.mark.parametrize(
        "thread", ["t1", "chat:2026-01-01", "société", "a b", FAMILY_EMOJI, "t" * MAX_THREAD_ID_BYTES]
    )
    def test_a_valid_thread_is_returned_as_given(self, thread: str) -> None:
        assert validate_thread_id(thread) == thread

    @pytest.mark.parametrize("thread", [*INVALID_IDS, None, "t" * (MAX_THREAD_ID_BYTES + 1)])
    def test_an_invalid_thread_is_refused(self, thread: object) -> None:
        with pytest.raises(ValidationError, match="thread_id"):
            validate_thread_id(thread)

    @pytest.mark.parametrize("thread", ["t1 ", " t1", "t1\n", "t1\xa0"])
    def test_a_thread_with_whitespace_at_an_end_is_refused_with_that_reason(self, thread: str) -> None:
        with pytest.raises(ValidationError, match="thread_id must not start or end with whitespace"):
            validate_thread_id(thread)


class TestTheKey:
    def test_the_partition_key_is_the_tenant_then_the_owner(self) -> None:
        assert partition_key_for("acme", "alice") == ["acme", "alice"]

    def test_an_agent_is_an_owner_like_a_user(self) -> None:
        assert partition_key_for("acme", "agent:support-bot") == ["acme", "agent:support-bot"]

    def test_the_same_owner_in_two_tenants_is_two_partitions(self) -> None:
        assert partition_key_for("acme", "alice") != partition_key_for("globex", "alice")

    def test_the_key_paths_are_the_tenant_then_the_owner(self) -> None:
        assert PARTITION_KEY_PATHS == ["/tenant_id", "/user_id"]

    @pytest.mark.parametrize("owner", ["", "a/b", None])
    def test_an_invalid_owner_has_no_key(self, owner: object) -> None:
        with pytest.raises(ValidationError):
            partition_key_for("acme", owner)  # type: ignore[arg-type]


class TestTheFilters:
    def test_the_owner_filter_names_the_tenant_and_the_owner(self) -> None:
        assert TENANT_FILTER == "c.tenant_id = @tenant_id"
        assert OWNER_FILTER == f"{TENANT_FILTER} AND c.user_id = @user_id"

    def test_the_parameters_match_the_filters(self) -> None:
        assert tenant_parameters("acme") == [{"name": "@tenant_id", "value": "acme"}]
        assert owner_parameters("acme", "alice") == [
            {"name": "@tenant_id", "value": "acme"},
            {"name": "@user_id", "value": "alice"},
        ]

    @pytest.mark.parametrize("owner", ["", "a#b", None])
    def test_an_invalid_owner_has_no_parameters(self, owner: object) -> None:
        with pytest.raises(ValidationError):
            owner_parameters("acme", owner)  # type: ignore[arg-type]


class TestStampScope:
    def test_a_record_without_a_tenant_is_given_one(self) -> None:
        assert stamp_scope({"id": "f1", "user_id": "alice"}, "acme") == {
            "id": "f1",
            "user_id": "alice",
            "tenant_id": "acme",
        }

    def test_a_record_of_the_same_tenant_is_kept(self) -> None:
        assert stamp_scope({"id": "f1", "user_id": "alice", "tenant_id": "acme"}, "acme")["tenant_id"] == "acme"

    def test_a_record_of_another_tenant_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="different tenant"):
            stamp_scope({"id": "f1", "user_id": "alice", "tenant_id": "globex"}, "acme")

    def test_the_record_passed_in_is_not_changed(self) -> None:
        record = {"id": "f1", "user_id": "alice"}

        stamp_scope(record, "acme")

        assert record == {"id": "f1", "user_id": "alice"}

    @pytest.mark.parametrize("record", [{"id": "f1"}, {"id": "f1", "user_id": None}, {"id": "f1", "user_id": ""}])
    def test_a_record_without_an_owner_is_refused(self, record: dict) -> None:
        with pytest.raises(ValidationError, match="user_id"):
            stamp_scope(record, "acme")

    def test_a_record_with_an_unsupported_owner_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="user_id"):
            stamp_scope({"id": "f1", "user_id": "a/b"}, "acme")

    @pytest.mark.parametrize("thread", ["", "t1 ", "a/b", 7])
    def test_a_record_with_an_unsupported_thread_is_refused(self, thread: object) -> None:
        with pytest.raises(ValidationError, match="thread_id"):
            stamp_scope({"id": "f1", "user_id": "alice", "thread_id": thread}, "acme")

    @pytest.mark.parametrize("fields", [{}, {"thread_id": None}, {"thread_id": "t1"}])
    def test_a_record_may_have_no_thread_or_a_supported_one(self, fields: dict) -> None:
        assert stamp_scope({"id": "f1", "user_id": "alice", **fields}, "acme") == {
            "id": "f1",
            "user_id": "alice",
            "tenant_id": "acme",
            **fields,
        }


class TestPartitionKeyProblem:
    @staticmethod
    def properties(*paths: str) -> dict:
        return {"id": "memories", "partitionKey": {"paths": list(paths), "kind": "MultiHash", "version": 2}}

    def test_the_expected_key_is_not_a_problem(self) -> None:
        assert partition_key_problem("memories", self.properties("/tenant_id", "/user_id")) is None

    def test_the_earlier_key_is_named_with_what_to_do(self) -> None:
        problem = partition_key_problem("memories", self.properties("/user_id", "/thread_id"))

        assert problem is not None
        assert "'memories'" in problem
        assert "['/user_id', '/thread_id']" in problem
        assert "['/tenant_id', '/user_id']" in problem
        assert "recreate" in problem and "leases" in problem

    @pytest.mark.parametrize(
        "paths", [("/tenant_id",), ("/user_id", "/tenant_id"), ("/tenant_id", "/user_id", "/thread_id"), ()]
    )
    def test_any_other_key_is_a_problem(self, paths: tuple[str, ...]) -> None:
        assert partition_key_problem("memories", self.properties(*paths)) is not None

    def test_a_container_without_a_partition_key_says_none(self) -> None:
        problem = partition_key_problem("memories", {"id": "memories"})

        assert problem is not None and "partition key none" in problem

    def test_what_is_not_container_metadata_is_not_checked(self) -> None:
        assert partition_key_problem("memories", object()) is None
        assert partition_key_problem("memories", None) is None
