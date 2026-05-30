"""Unit tests for the ``_container_routing`` primitive.

The routing primitive (``ContainerKey`` enum + ``_CONTAINER_FOR_TYPE`` dict
+ helpers) is the single source of truth for which Cosmos DB container
owns documents of a given memory type. These tests pin its public contract
so the rest of the codebase can rely on it.
"""

from __future__ import annotations

import pytest

from agent_memory_toolkit._container_routing import (
    _CONTAINER_FOR_TYPE,
    ContainerKey,
    container_key_for_type,
    container_keys_for_types,
)
from agent_memory_toolkit._utils import VALID_TYPES


class TestContainerKey:
    def test_enum_has_three_members(self) -> None:
        assert {k.name for k in ContainerKey} == {"TURNS", "MEMORIES", "SUMMARIES"}

    def test_enum_values_are_lowercase_container_short_names(self) -> None:
        assert ContainerKey.TURNS.value == "turns"
        assert ContainerKey.MEMORIES.value == "memories"
        assert ContainerKey.SUMMARIES.value == "summaries"

    def test_enum_is_string_subclass(self) -> None:
        # ``str, Enum`` mixin keeps members usable as plain dict keys
        # alongside their literal string values.
        assert isinstance(ContainerKey.TURNS, str)


class TestContainerForTypeMapping:
    def test_routing_covers_every_valid_type(self) -> None:
        """Every type accepted by the SDK must route to a container.

        Using ``VALID_TYPES`` as the source of truth means adding a new type
        without wiring routing for it will fail this test.
        """
        missing = VALID_TYPES - set(_CONTAINER_FOR_TYPE)
        assert missing == set(), f"_CONTAINER_FOR_TYPE missing routing for: {missing}"

    def test_routing_targets_are_valid_container_keys(self) -> None:
        assert all(isinstance(v, ContainerKey) for v in _CONTAINER_FOR_TYPE.values())

    @pytest.mark.parametrize(
        ("memory_type", "expected_key"),
        [
            ("turn", ContainerKey.TURNS),
            ("fact", ContainerKey.MEMORIES),
            ("episodic", ContainerKey.MEMORIES),
            ("procedural", ContainerKey.MEMORIES),
            ("thread_summary", ContainerKey.SUMMARIES),
            ("user_summary", ContainerKey.SUMMARIES),
        ],
    )
    def test_each_type_routes_to_expected_container(self, memory_type: str, expected_key: ContainerKey) -> None:
        assert container_key_for_type(memory_type) is expected_key


class TestContainerKeyForType:
    def test_unknown_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            container_key_for_type("unknown_type")
        assert "unknown_type" in str(exc_info.value)

    def test_error_message_lists_valid_types(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            container_key_for_type("not_a_type")
        msg = str(exc_info.value)
        # Spot-check a couple of well-known types appear in the error.
        assert "turn" in msg
        assert "fact" in msg


class TestContainerKeysForTypes:
    def test_empty_iterable_returns_empty_list(self) -> None:
        assert container_keys_for_types([]) == []

    def test_single_type_returns_single_container(self) -> None:
        assert container_keys_for_types(["fact"]) == [ContainerKey.MEMORIES]

    def test_deduplicates_when_types_share_container(self) -> None:
        result = container_keys_for_types(["fact", "episodic", "procedural"])
        assert result == [ContainerKey.MEMORIES]

    def test_returns_deterministic_turns_memories_summaries_order(self) -> None:
        """Order is fixed regardless of input order to avoid flaky fan-out tests."""
        result = container_keys_for_types(["user_summary", "fact", "turn"])
        assert result == [
            ContainerKey.TURNS,
            ContainerKey.MEMORIES,
            ContainerKey.SUMMARIES,
        ]

    def test_order_stable_with_alternate_input_order(self) -> None:
        assert container_keys_for_types(["turn", "fact", "thread_summary"]) == (
            container_keys_for_types(["thread_summary", "turn", "fact"])
        )

    def test_unknown_type_in_list_raises(self) -> None:
        with pytest.raises(ValueError):
            container_keys_for_types(["fact", "totally_made_up"])

    def test_summary_types_both_route_to_summaries(self) -> None:
        assert container_keys_for_types(["thread_summary", "user_summary"]) == [
            ContainerKey.SUMMARIES,
        ]
