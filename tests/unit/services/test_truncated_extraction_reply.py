"""A reply cut off at the output limit comes back cut off again, so the batch that drew it is quarantined.

Asking again sends the same turns and gets a reply cut off at the same place. Deferring the batch would send it again,
with every turn that arrived since, on each later run, and nothing would be extracted until the backlog outgrew the
batch. The error code ``output_truncated`` is what a log shows of it.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService
from azure.cosmos.agent_memory.services.pipeline import PipelineService
from tests.unit.services.test_extract_dry import (
    _async_containers_for_store,
    _AsyncEmbeddings,
    _AsyncStore,
    _containers_for_store,
    _Store,
    _SyncEmbeddings,
    _turn,
)

# What a model that runs out of output tokens mid-object returns; its text must never reach a log.
MODEL_TEXT = "zx-model-text-4417"
CUT_OFF_REPLY = f'{{"facts": [{{"text": "{MODEL_TEXT} prefers dark mode", "category": "pref'


class _Reply:
    """A model that answers every call with the same text, as the chat client hands it over."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.messages: list[list[dict[str, Any]]] = []

    def generate(self, messages: list[dict[str, Any]], **opts: Any) -> str:
        del opts
        self.messages.append(messages)
        return self.text


class _AsyncReply(_Reply):
    async def generate(self, messages: list[dict[str, Any]], **opts: Any) -> str:
        return super().generate(messages, **opts)


def _sync_service(chat: _Reply, turns: list[dict[str, Any]]) -> tuple[PipelineService, _Store]:
    memories_store, turns_store = _Store([]), _Store(turns)
    service = PipelineService(
        memories_store,
        chat,
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=turns_store),
    )
    return service, turns_store


def _async_service(chat: _AsyncReply, turns: list[dict[str, Any]]) -> tuple[AsyncPipelineService, _AsyncStore]:
    memories_store, turns_store = _AsyncStore([]), _AsyncStore(turns)
    service = AsyncPipelineService(
        memories_store,
        chat,
        _AsyncEmbeddings(),
        containers=_async_containers_for_store(memories_store, turns_store=turns_store),
    )
    return service, turns_store


def _new_line_labels(messages: list[dict[str, Any]]) -> list[str]:
    lines = messages[-1]["content"].split("## New lines\n", 1)[1].strip().splitlines()
    return [line.split("Turn ", 1)[1].split(":", 1)[0] for line in lines]


def _assert_quarantined(out: dict[str, Any], turns: list[dict[str, Any]], log: str) -> None:
    assert out["facts"] == []
    assert [turn["id"] for turn in out["processed_turn_docs"]] == [turn["id"] for turn in turns]
    [stats] = [update for update in out["updates"] if update.get("op") == "stats"]
    assert (stats["quarantined_turn_count"], stats["deferred_turn_count"]) == (len(turns), 0)
    assert f"quarantining {len(turns)} turns" in log
    assert "code=output_truncated" in log, "the log names the cause; its message is never logged"
    assert MODEL_TEXT not in log


def test_a_reply_cut_off_at_the_output_limit_quarantines_its_batch(caplog: pytest.LogCaptureFixture) -> None:
    turns = [_turn(i) for i in range(3)]
    chat = _Reply(CUT_OFF_REPLY)
    service, _ = _sync_service(chat, turns)

    with caplog.at_level(logging.WARNING):
        out = service.extract_memories_durable("u1", "t1")

    _assert_quarantined(out, turns, caplog.text)
    assert len(chat.messages) == 1, "a cut-off reply is not asked for again"


def test_a_cut_off_batch_is_not_sent_again_with_the_turns_that_arrive_after_it() -> None:
    chat = _Reply(CUT_OFF_REPLY)
    service, turns_store = _sync_service(chat, [_turn(i) for i in range(3)])

    first = service.extract_memories("u1", "t1")

    assert (first["quarantined_turn_count"], first["deferred_turn_count"]) == (3, 0)
    assert all(turn.get("extracted_at") for turn in turns_store.docs), "quarantined turns are marked extracted"
    turns_store.docs.append(_turn(3))
    service.extract_memories("u1", "t1")
    assert _new_line_labels(chat.messages[1]) == ["turn-3"]


@pytest.mark.asyncio
async def test_async_a_reply_cut_off_at_the_output_limit_quarantines_its_batch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    turns = [_turn(i) for i in range(3)]
    chat = _AsyncReply(CUT_OFF_REPLY)
    service, _ = _async_service(chat, turns)

    with caplog.at_level(logging.WARNING):
        out = await service.extract_memories_durable("u1", "t1")

    _assert_quarantined(out, turns, caplog.text)
    assert len(chat.messages) == 1, "a cut-off reply is not asked for again"


@pytest.mark.asyncio
async def test_async_a_cut_off_batch_is_not_sent_again_with_the_turns_that_arrive_after_it() -> None:
    chat = _AsyncReply(CUT_OFF_REPLY)
    service, turns_store = _async_service(chat, [_turn(i) for i in range(3)])

    first = await service.extract_memories("u1", "t1")

    assert (first["quarantined_turn_count"], first["deferred_turn_count"]) == (3, 0)
    assert all(turn.get("extracted_at") for turn in turns_store.docs), "quarantined turns are marked extracted"
    turns_store.docs.append(_turn(3))
    await service.extract_memories("u1", "t1")
    assert _new_line_labels(chat.messages[1]) == ["turn-3"]


# Episode extraction reads its reply the same way, so a cut-off reply quarantines the segment it was asked about.


def test_a_segment_whose_episode_reply_was_cut_off_is_quarantined_not_asked_for_again(monkeypatch) -> None:
    monkeypatch.setenv("EPISODE_TOPIC_DRIFT", "0")
    chat = _Reply(CUT_OFF_REPLY)
    service, _ = _sync_service(chat, [_turn(1), _turn(2)])

    assert service.extract_episodes("u1", "t1", flush=True) == {"episodes": 0}
    assert service.extract_episodes("u1", "t1", flush=True) == {"episodes": 0}

    assert len(chat.messages) == 1, "the episode watermark moved past the segment, so its turns are not read again"


@pytest.mark.asyncio
async def test_async_a_segment_whose_episode_reply_was_cut_off_is_quarantined_not_asked_for_again(monkeypatch) -> None:
    monkeypatch.setenv("EPISODE_TOPIC_DRIFT", "0")
    chat = _AsyncReply(CUT_OFF_REPLY)
    service, _ = _async_service(chat, [_turn(1), _turn(2)])

    assert await service.extract_episodes("u1", "t1", flush=True) == {"episodes": 0}
    assert await service.extract_episodes("u1", "t1", flush=True) == {"episodes": 0}

    assert len(chat.messages) == 1, "the episode watermark moved past the segment, so its turns are not read again"
