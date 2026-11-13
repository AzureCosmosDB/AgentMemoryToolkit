from __future__ import annotations

import pytest
from shared import config


@pytest.mark.parametrize(("raw", "expected"), [("7", 7), ("1", 1), ("0", 20), ("-5", 20), ("", 20), ("many", 20)])
def test_the_batch_window_is_never_empty(monkeypatch: pytest.MonkeyPatch, raw: str, expected: int) -> None:
    """A window below 1 reads no turns, so the backlog would be silently skipped."""
    monkeypatch.setenv("MAX_BATCH_SIZE", raw)

    assert config.get_max_batch_size() == expected
