"""Log-safe descriptions of exceptions and model output.

An exception's message and arguments can quote a search query, memory text or model output. Each case drives a
distinctive string through a failing path and checks it is absent from what :func:`describe_exception`,
:func:`describe_shape` and :func:`describe_turn_refs` return, while the diagnostic that replaced it is present.
"""

from __future__ import annotations

import json
import logging
import sys

import pytest
from azure.cosmos.exceptions import CosmosHttpResponseError
from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from azure.cosmos.agent_memory.exceptions import LLMError
from azure.cosmos.agent_memory.logging import (
    JsonFormatter,
    describe_exception,
    describe_shape,
    describe_turn_refs,
)

QUERY = "zx-query-lisbon-anniversary-surprise-7731"
MEMORY = "zx-memory-ana-allergic-to-cashews-4410"
MODEL = "zx-model-ana-prefers-window-seats-9902"


class _Fact(BaseModel):
    salience: float
    tags: list[str]


def _raise_chained() -> None:
    answer = MODEL  # noqa: F841 - a local in the failing frame
    try:
        json.loads(f'{{"facts": [{MODEL}')
    except json.JSONDecodeError as exc:
        raise LLMError(f"LLM returned invalid JSON (preview={MODEL!r})") from exc


@pytest.mark.parametrize("stack", [False, True])
def test_an_exception_is_described_by_type_code_position_and_location_never_its_text(stack: bool) -> None:
    with pytest.raises(LLMError) as caught:
        _raise_chained()

    text = describe_exception(caught.value, stack=stack)

    assert MODEL not in text
    assert text.startswith("agent_memory.LLMError(code=llm) at ")
    assert "_raise_chained" in text
    assert "<- json.JSONDecodeError(line=1, col=12, pos=11) at " in text


def test_a_validation_error_is_described_by_model_and_fields_never_the_input() -> None:
    with pytest.raises(PydanticValidationError) as caught:
        _Fact(salience=MEMORY, tags=MODEL)

    text = describe_exception(caught.value)

    assert MEMORY not in text and MODEL not in text
    assert "pydantic_core.ValidationError(model=_Fact, errors=2, fields=salience:float_parsing tags:list_type)" in text


class _Seat(BaseModel):
    note: str


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scores: dict[str, float]
    seats: list[_Seat]


def test_a_validation_path_never_names_a_key_the_data_chose() -> None:
    with pytest.raises(PydanticValidationError) as caught:
        _Strict(scores={"cashews": "high", MODEL: "low"}, seats=[{"note": 1}], **{"zx_key_ana_window": 1})

    text = describe_exception(caught.value)

    assert [key for key in ("cashews", MODEL, "zx_key_ana_window", "note") if key in text] == []
    assert "fields=scores.*:float_parsing scores.*:float_parsing seats.0.*:string_type *:extra_forbidden" in text


def test_describing_an_unusual_exception_falls_back_to_its_type() -> None:
    class OddStatusError(Exception):
        @property
        def status_code(self) -> int:
            raise RuntimeError(MODEL)

    assert describe_exception(OddStatusError(MEMORY)).endswith("OddStatusError")


def test_a_dependency_error_keeps_its_status_and_a_safe_code_only() -> None:
    class ServiceError(Exception):
        status_code, sub_status, code = 400, 1001, "BadRequest"

    class OddError(Exception):
        code = f"{MODEL} said no"

    text = describe_exception(ServiceError(f"query {QUERY!r} rejected for document {MEMORY!r}"))

    assert QUERY not in text and MEMORY not in text
    assert text.endswith("ServiceError(code=BadRequest, status=400, sub_status=1001)")
    assert MODEL not in describe_exception(OddError(MODEL))


def test_an_azure_sdk_error_is_labelled_by_its_sdk_package_with_its_status() -> None:
    text = describe_exception(CosmosHttpResponseError(status_code=404, message=f"{MEMORY} not found"))

    assert MEMORY not in text
    assert text.startswith("azure.cosmos.CosmosHttpResponseError(")
    assert "status=404" in text


def test_model_output_is_described_by_shape_never_its_text() -> None:
    assert describe_shape({"summary": MODEL, "title": ""}) == "object(keys=2)"
    assert describe_shape(MODEL) == f"string(chars={len(MODEL)})"
    assert describe_shape("   ") == "string(chars=3, blank)"
    assert describe_shape([MODEL]) == "array(len=1)"
    assert describe_shape(None) == "null"
    assert describe_turn_refs(["turn-12", MODEL, " Turn-3.1 ", 7]) == (
        f"4 [turn-12, string(chars={len(MODEL)}), Turn-3.1, number]"
    )


def test_the_json_formatter_writes_an_exception_by_description_not_by_message() -> None:
    try:
        raise LLMError(f"the model answered {MODEL!r}")
    except LLMError:
        record = logging.LogRecord(
            "azure.cosmos.agent_memory.t", logging.ERROR, __file__, 1, "boom", (), sys.exc_info()
        )

    payload = json.loads(JsonFormatter().format(record))

    assert MODEL not in json.dumps(payload)
    assert payload["exc_type"] == "LLMError"
    assert payload["exc_detail"].startswith("agent_memory.LLMError(code=llm) at ")
