from __future__ import annotations

import pytest
from shared import config


@pytest.mark.parametrize(("raw", "expected"), [("", 240), ("600", 600), ("5", 30), ("-1", 240), ("soon", 240)])
def test_an_activity_time_budget_leaves_room_for_at_least_one_model_call(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
) -> None:
    monkeypatch.setenv("ACTIVITY_TIME_BUDGET_SECONDS", raw)

    assert config.get_activity_time_budget_seconds() == expected


def test_an_activity_time_budget_defaults_to_four_minutes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ACTIVITY_TIME_BUDGET_SECONDS", raising=False)

    assert config.get_activity_time_budget_seconds() == config.DEFAULT_ACTIVITY_TIME_BUDGET_SECONDS == 240
