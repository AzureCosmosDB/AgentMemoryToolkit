from __future__ import annotations

from azure.cosmos.agent_memory import thresholds


def test_the_similarity_floors_are_fixed_and_the_combine_floor_is_the_lower() -> None:
    assert (thresholds.DEDUP_SIMILARITY_FLOOR, thresholds.COMBINE_SIMILARITY_FLOOR) == (0.85, 0.65)
    assert (thresholds.DEDUP_MAX_NEIGHBORS, thresholds.COMBINE_MAX_SOURCES, thresholds.COMBINE_MAX_CHARS) == (5, 4, 800)
