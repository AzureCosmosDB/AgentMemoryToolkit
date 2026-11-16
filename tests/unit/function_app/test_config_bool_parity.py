from __future__ import annotations

import pytest
from shared import config

from azure.cosmos.agent_memory import thresholds


@pytest.mark.parametrize(
    "raw", ["true", "1", "yes", "on", "t", "y", "false", "0", "no", "off", "f", "n", " N ", "maybe", ""]
)
def test_the_function_app_reads_on_off_settings_like_the_sdk(monkeypatch, raw):
    """One setting must mean the same thing in-process and in the Durable Function app."""
    monkeypatch.setenv("PROCEDURAL_SYNTHESIS_AUTO", raw)

    assert config.get_procedural_synthesis_auto() == thresholds.get_procedural_synthesis_auto()


def test_a_short_off_token_turns_chained_procedural_synthesis_off(monkeypatch):
    monkeypatch.setenv("PROCEDURAL_SYNTHESIS_AUTO", "n")

    assert config.get_procedural_synthesis_auto() is False


@pytest.mark.parametrize("raw", ["t", "y", " Y "])
def test_a_short_on_token_is_not_mistaken_for_an_unrecognized_value(monkeypatch, raw):
    """An unrecognized value takes the default, so an on token is only proven against a default of off."""
    monkeypatch.setenv("AMT_TEST_SWITCH", raw)

    assert config._parse_bool("AMT_TEST_SWITCH", False) is True
