"""Unit tests for ``function_app/shared/instance_ids.py``.

An orchestration instance id is ``{kind}:{digest}`` plus ``:{count}``. Durable Functions takes 1-100 characters without
``/``, ``\\``, ``#``, ``?`` or control characters, and tenant, user and thread ids are free text, so none of them
appears in the id.
"""

from __future__ import annotations

import re
import unicodedata

import pytest
from shared.instance_ids import orchestration_instance_id

KINDS = ("thread_summary", "extract", "episode", "user_summary", "procedural")


def _is_allowed_instance_id(instance_id: str) -> bool:
    """Durable Functions' rules for an instance id."""
    return (
        1 <= len(instance_id) <= 100
        and not instance_id.startswith("@")
        and not any(character in "/\\#?" for character in instance_id)
        and not any(unicodedata.category(character) == "Cc" for character in instance_id)
    )


def test_the_id_is_the_kind_a_hash_of_the_owner_and_the_count():
    """The digest is the first 32 hex digits of the SHA-256 of the parts joined with the unit separator."""
    assert (
        orchestration_instance_id("extract", "acme", "u1", "t1", count=4)
        == "extract:bdddaa37089972c2babf1b5dda603e63:4"
    )
    assert orchestration_instance_id("user_summary", "acme", "u1", count=20) == (
        "user_summary:16694a27e83b5de5685ed29bf499bc68:20"
    )


def test_the_count_is_an_optional_suffix_and_zero_is_a_count():
    without = orchestration_instance_id("episode", "acme", "u1", "t1")
    assert re.fullmatch(r"episode:[0-9a-f]{32}", without)
    assert orchestration_instance_id("episode", "acme", "u1", "t1", count=0) == f"{without}:0"


def test_the_batch_lsn_joins_the_hash_and_leaves_the_shape_of_the_id_alone():
    """A replay of a batch has its LSN and so its id; a later batch that reaches the same count gets another id."""
    first = orchestration_instance_id("extract", "acme", "u1", "t1", count=4, lsn=13)

    assert re.fullmatch(r"extract:[0-9a-f]{32}:4", first)
    assert first == orchestration_instance_id("extract", "acme", "u1", "t1", count=4, lsn=13)
    assert first != orchestration_instance_id("extract", "acme", "u1", "t1", count=4, lsn=503)
    assert first != orchestration_instance_id("extract", "acme", "u1", "t1", count=4)
    # Without an LSN, which a turn that has none gives, the id is the one it was before the LSN joined the hash.
    assert orchestration_instance_id("extract", "acme", "u1", "t1", count=4, lsn=None) == (
        "extract:bdddaa37089972c2babf1b5dda603e63:4"
    )


def test_the_same_inputs_give_the_same_id():
    first = orchestration_instance_id("thread_summary", "acme", "u1", "t1", count=4)
    second = orchestration_instance_id("thread_summary", "".join(["ac", "me"]), "u1", "t" + "1", count=4)

    assert first == second


def test_a_different_kind_tenant_user_thread_or_count_gives_a_different_id():
    ids = {
        orchestration_instance_id("thread_summary", "acme", "u1", "t1", count=4),
        orchestration_instance_id("extract", "acme", "u1", "t1", count=4),
        orchestration_instance_id("thread_summary", "globex", "u1", "t1", count=4),
        orchestration_instance_id("thread_summary", "acme", "u2", "t1", count=4),
        orchestration_instance_id("thread_summary", "acme", "u1", "t2", count=4),
        orchestration_instance_id("thread_summary", "acme", "u1", "t1", count=5),
    }

    assert len(ids) == 6


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(("a:b", "c", "d"), ("a", "b:c", "d"), id="thread-level-colon-between-tenant-and-user"),
        pytest.param(("a", "b:c", "d"), ("a", "b", "c:d"), id="thread-level-colon-between-user-and-thread"),
        pytest.param(("a:b", "c"), ("a", "b:c"), id="user-level-colon-between-tenant-and-user"),
        pytest.param(("ab", "c", "d"), ("a", "bc", "d"), id="thread-level-boundary-between-tenant-and-user"),
        pytest.param(("acme", "u1"), ("acme", "u1", ""), id="user-level-and-an-empty-thread"),
    ],
)
def test_a_colon_or_a_moved_boundary_cannot_make_two_owners_collide(first, second):
    first_id = orchestration_instance_id("extract", *first, count=1)
    second_id = orchestration_instance_id("extract", *second, count=1)

    assert first_id != second_id


@pytest.mark.parametrize("kind", KINDS)
def test_the_worst_case_stays_within_the_limit_and_the_allowed_characters(kind):
    """Tenant, user and thread of 200 characters, a 10-digit count and the largest LSN: the id is as long as for
    one-character ids."""
    longest = orchestration_instance_id(kind, "t" * 200, "u" * 200, "h" * 200, count=9_999_999_999, lsn=2**63 - 1)
    shortest = orchestration_instance_id(kind, "t", "u", "h", count=1_000_000_000, lsn=0)

    assert len(longest) == len(shortest) == len(kind) + 1 + 32 + 1 + 10
    assert _is_allowed_instance_id(longest)


@pytest.mark.parametrize("user_id", ["a/b", "a\\b", "a#b", "a?b", "a\nb", "a\x1fb", "@user", "ünï©ødé", "a b"])
def test_characters_a_user_id_may_hold_never_reach_the_id(user_id):
    instance_id = orchestration_instance_id("episode", "acme", user_id, "t1", count=7)

    assert re.fullmatch(r"episode:[0-9a-f]{32}:7", instance_id)
    assert _is_allowed_instance_id(instance_id)
