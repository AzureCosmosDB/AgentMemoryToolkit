from __future__ import annotations

import pytest
from pydantic import ValidationError

from azure.cosmos.agent_memory.models import FactRecord, MemoryRecordBase
from azure.cosmos.agent_memory.store._search_helpers import MEMORY_PROJECTION


def _fact_doc(**extra) -> dict:
    return {
        "id": "fact_1",
        "user_id": "u1",
        "thread_id": "t1",
        "type": "fact",
        "content": "The user likes coffee",
        "content_hash": "0" * 32,
        "prompt_id": "extract_memories.prompty",
        "metadata": {"category": "preference"},
        **extra,
    }


def test_a_fact_that_absorbed_others_lists_them_in_merged_from():
    record = FactRecord.model_validate(_fact_doc(merged_from=["fact_0", "fact_2"]))

    assert record.merged_from == ["fact_0", "fact_2"]
    assert record.to_doc()["merged_from"] == ["fact_0", "fact_2"]


def test_a_fact_that_absorbed_nothing_keeps_its_wire_shape():
    record = FactRecord.model_validate(_fact_doc())

    assert record.merged_from == []
    assert "merged_from" not in record.to_doc()


def test_merged_from_survives_a_round_trip_through_a_stored_document():
    stored = FactRecord.model_validate(_fact_doc(merged_from=["fact_0"], _etag="e1")).to_doc()

    assert MemoryRecordBase.from_doc(stored).merged_from == ["fact_0"]


def test_search_and_history_results_carry_merged_from():
    assert "c.merged_from" in MEMORY_PROJECTION


@pytest.mark.parametrize("reason", ["duplicate", "contradict", "update", "combined", "procedure"])
def test_every_supersede_reason_the_pipeline_writes_is_accepted(reason):
    record = FactRecord.model_validate(_fact_doc(superseded_by="fact_2", supersede_reason=reason))

    assert record.supersede_reason == reason


def test_an_unknown_supersede_reason_is_refused():
    with pytest.raises(ValidationError):
        FactRecord.model_validate(_fact_doc(superseded_by="fact_2", supersede_reason="merged"))
