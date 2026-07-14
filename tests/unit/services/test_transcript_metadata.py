"""Transcript-building behavior: default + metadata allow-list."""

from __future__ import annotations

import json

import pytest

from azure.cosmos.agent_memory.services._pipeline_helpers import build_transcript


def _turn(role: str, content: str, **meta: object) -> dict[str, object]:
    return {"role": role, "content": content, "metadata": dict(meta)}


class TestDefaultExcludesMetadata:
    """Default behavior must NEVER serialize metadata into the transcript.

    Coding-agent frameworks routinely stash full tool-call payloads, IDE
    schema fragments, and raw model responses under ``TurnRecord.metadata``.
    Dumping that blob into every extraction/summarization prompt was
    drowning out the actual dialogue (issue: user reported 24-turn session
    where metadata was 20-25x the size of the conversation content).
    """

    def test_metadata_omitted_by_default_flat(self) -> None:
        items = [
            _turn("user", "Hello", raw_response={"huge": "blob"}, tool_calls=[1, 2, 3]),
            _turn("assistant", "Hi", model_id="gpt-4o", token_count=42),
        ]
        out = build_transcript(items)
        assert out == "[user]: Hello\n[agent]: Hi"
        assert "metadata" not in out
        assert "raw_response" not in out
        assert "tool_calls" not in out

    def test_metadata_omitted_by_default_grouped(self) -> None:
        items = [
            {"role": "user", "content": "Q", "metadata": {"raw_response": "..."}, "thread_id": "t1"},
            {"role": "assistant", "content": "A", "metadata": {"raw_response": "..."}, "thread_id": "t1"},
        ]
        out = build_transcript(items, group_by_thread=True)
        assert "raw_response" not in out
        assert "metadata" not in out
        assert "=== Thread t1 ===" in out
        assert "[user]: Q" in out
        assert "[agent]: A" in out

    def test_empty_allowlist_is_same_as_default(self) -> None:
        items = [_turn("user", "x", a=1, b=2)]
        assert build_transcript(items, metadata_keys=[]) == "[user]: x"
        assert build_transcript(items, metadata_keys=None) == "[user]: x"

    def test_no_metadata_field_does_not_crash(self) -> None:
        items = [{"role": "user", "content": "no meta key here"}]
        assert build_transcript(items) == "[user]: no meta key here"


class TestAllowlist:
    """``metadata_keys`` is an opt-in allow-list, not a denylist."""

    def test_only_allowed_keys_surface(self) -> None:
        items = [_turn("user", "Hi", agent_id="copilot", raw_response={"huge": "blob"}, timestamp="2026-01-01")]
        out = build_transcript(items, metadata_keys=["agent_id", "timestamp"])
        # The line still includes the metadata segment, but only the
        # allow-listed keys made it through.
        assert "[user]: Hi" in out
        assert "agent_id" in out
        assert "timestamp" in out
        assert "raw_response" not in out
        assert "huge" not in out

    def test_allowlist_ordering_is_iteration_order(self) -> None:
        items = [_turn("user", "Hi", agent_id="copilot", timestamp="2026-01-01")]
        out_a = build_transcript(items, metadata_keys=["agent_id", "timestamp"])
        out_b = build_transcript(items, metadata_keys=["timestamp", "agent_id"])
        # JSON serialization must reflect the iteration order of the
        # allow-list so callers can pin ordering for deterministic prompts.
        assert out_a.index("agent_id") < out_a.index("timestamp")
        assert out_b.index("timestamp") < out_b.index("agent_id")

    def test_missing_allowed_keys_silently_skipped(self) -> None:
        items = [_turn("user", "Hi", agent_id="copilot")]
        out = build_transcript(items, metadata_keys=["agent_id", "model_id", "not_present"])
        assert "agent_id" in out
        # Absent keys do not appear as ``null`` or empty entries.
        assert "model_id" not in out
        assert "not_present" not in out
        assert "null" not in out

    def test_segment_dropped_when_no_allowed_key_present(self) -> None:
        items = [_turn("user", "Hi", raw_response="big")]
        out = build_transcript(items, metadata_keys=["agent_id"])
        # None of the allow-listed keys exist on this turn's metadata,
        # so the ``[metadata: ...]`` segment is suppressed entirely.
        assert out == "[user]: Hi"

    def test_works_with_grouped_layout(self) -> None:
        items = [
            {"role": "user", "content": "Q", "metadata": {"agent_id": "x", "raw": "big"}, "thread_id": "t1"},
            {"role": "assistant", "content": "A", "metadata": {"agent_id": "x", "raw": "big"}, "thread_id": "t1"},
        ]
        out = build_transcript(items, group_by_thread=True, metadata_keys=["agent_id"])
        assert "agent_id" in out
        assert "raw" not in out

    def test_non_dict_metadata_is_safe(self) -> None:
        items = [{"role": "user", "content": "x", "metadata": "not a dict"}]
        # Should not crash and should not invent metadata content.
        assert build_transcript(items, metadata_keys=["agent_id"]) == "[user]: x"

    def test_serialized_json_is_valid(self) -> None:
        items = [_turn("user", "Hi", agent_id="copilot", count=7)]
        out = build_transcript(items, metadata_keys=["agent_id", "count"])
        start = out.index("{")
        end = out.rindex("}") + 1
        parsed = json.loads(out[start:end])
        assert parsed == {"agent_id": "copilot", "count": 7}


class TestPipelineServicePlumbing:
    """``PipelineService.__init__`` must accept the allow-list and thread it
    through ``_build_transcript``."""

    def test_sync_pipeline_propagates_allowlist(self) -> None:
        from azure.cosmos.agent_memory.services.pipeline import PipelineService

        service = PipelineService.__new__(PipelineService)
        service._transcript_metadata_keys = ("agent_id",)
        items = [_turn("user", "Hi", agent_id="copilot", raw="big")]
        out = service._build_transcript(items)
        assert "agent_id" in out
        assert "raw" not in out

    def test_sync_pipeline_default_empty(self) -> None:
        from azure.cosmos.agent_memory.services.pipeline import PipelineService

        service = PipelineService.__new__(PipelineService)
        service._transcript_metadata_keys = None
        items = [_turn("user", "Hi", agent_id="copilot")]
        assert service._build_transcript(items) == "[user]: Hi"

    @pytest.mark.asyncio
    async def test_async_pipeline_propagates_allowlist(self) -> None:
        from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService

        service = AsyncPipelineService.__new__(AsyncPipelineService)
        service._transcript_metadata_keys = ("agent_id",)
        items = [_turn("user", "Hi", agent_id="copilot", raw="big")]
        out = service._build_transcript(items)
        assert "agent_id" in out
        assert "raw" not in out


class TestStringInputRejected:
    """A bare ``str`` is iterable char-by-char - passing it would silently
    produce a per-character allow-list. Reject with a clear ``TypeError``."""

    def test_build_transcript_rejects_str(self) -> None:
        with pytest.raises(TypeError, match="not a single str"):
            build_transcript([_turn("user", "Hi", agent_id="x")], metadata_keys="agent_id")

    def test_pipeline_service_ctor_rejects_str(self) -> None:
        from unittest.mock import MagicMock

        from azure.cosmos.agent_memory._container_routing import ContainerKey
        from azure.cosmos.agent_memory.services.pipeline import PipelineService

        containers = {
            ContainerKey.MEMORIES: MagicMock(),
            ContainerKey.TURNS: MagicMock(),
            ContainerKey.SUMMARIES: MagicMock(),
        }
        with pytest.raises(TypeError, match="not a single str"):
            PipelineService(
                MagicMock(),
                MagicMock(),
                MagicMock(),
                containers=containers,
                transcript_metadata_keys="agent_id",
            )

    def test_async_pipeline_service_ctor_rejects_str(self) -> None:
        from unittest.mock import MagicMock

        from azure.cosmos.agent_memory._container_routing import ContainerKey
        from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService

        containers = {
            ContainerKey.MEMORIES: MagicMock(),
            ContainerKey.TURNS: MagicMock(),
            ContainerKey.SUMMARIES: MagicMock(),
        }
        with pytest.raises(TypeError, match="not a single str"):
            AsyncPipelineService(
                MagicMock(),
                MagicMock(),
                MagicMock(),
                containers=containers,
                transcript_metadata_keys="agent_id",
            )

    def test_client_ctor_rejects_str(self) -> None:
        from azure.cosmos.agent_memory import CosmosMemoryClient

        with pytest.raises(TypeError, match="not a single str"):
            CosmosMemoryClient(
                cosmos_endpoint="",
                transcript_metadata_keys="agent_id",
            )


class TestCompactJsonSerialization:
    """Token-reduction is the whole point - emit compact JSON, not pretty."""

    def test_no_whitespace_between_separators(self) -> None:
        items = [_turn("user", "Hi", a=1, b=2)]
        out = build_transcript(items, metadata_keys=["a", "b"])
        # Default json.dumps would produce ``{"a": 1, "b": 2}`` (spaces
        # after `:` and `,`). Compact form drops both.
        assert '{"a":1,"b":2}' in out
        assert '"a": 1' not in out
        assert '"b": 2' not in out

    def test_non_ascii_preserved(self) -> None:
        items = [_turn("user", "Hi", note="café")]
        out = build_transcript(items, metadata_keys=["note"])
        # ensure_ascii=False keeps human-readable utf-8 in the prompt.
        assert "café" in out
        assert "\\u00e9" not in out


class TestNonSerializableMetadataCoerced:
    """Real-world metadata carries datetimes, UUIDs, etc. - ``default=str``
    keeps a single bad value from blowing up the entire extraction."""

    def test_datetime_value_does_not_crash(self) -> None:
        from datetime import datetime, timezone

        items = [_turn("user", "Hi", timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))]
        out = build_transcript(items, metadata_keys=["timestamp"])
        assert "timestamp" in out
        assert "2026-01-01" in out

    def test_uuid_value_does_not_crash(self) -> None:
        import uuid

        u = uuid.uuid4()
        items = [_turn("user", "Hi", agent_id=u)]
        out = build_transcript(items, metadata_keys=["agent_id"])
        assert str(u) in out


class TestGeneratorCoercion:
    """A generator passed for ``metadata_keys`` would be exhausted after the
    first turn; ``build_transcript`` must coerce it to a sequence up front."""

    def test_generator_works_across_multiple_turns(self) -> None:
        items = [
            _turn("user", "T1", agent_id="x"),
            _turn("user", "T2", agent_id="y"),
            _turn("user", "T3", agent_id="z"),
        ]
        out = build_transcript(items, metadata_keys=(k for k in ["agent_id"]))
        assert out.count("agent_id") == 3


class TestClientToPipelineWiring:
    """A typo in ``_build_pipeline`` (e.g. ``_transcript_metadata_key`` vs
    ``_keys``) must fail a test, not silently disarm the kwarg."""

    def test_sync_client_threads_kwarg_into_pipeline(self) -> None:
        from unittest.mock import MagicMock

        from azure.cosmos.agent_memory import CosmosMemoryClient

        client = CosmosMemoryClient.__new__(CosmosMemoryClient)
        client._transcript_metadata_keys = ("agent_id", "timestamp")
        client._turns_container_client = MagicMock()
        client._memories_container_client = MagicMock()
        client._summaries_container_client = MagicMock()
        client._chat_client = MagicMock()
        client._embeddings_client = MagicMock()
        captured: dict[str, object] = {}

        from azure.cosmos.agent_memory import cosmos_memory_client as mod

        original = mod.PipelineService

        def fake_pipeline(*args: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return MagicMock()

        mod.PipelineService = fake_pipeline  # type: ignore[assignment]
        try:
            client._build_pipeline(MagicMock())
        finally:
            mod.PipelineService = original

        assert captured.get("transcript_metadata_keys") == ("agent_id", "timestamp")

    def test_async_client_threads_kwarg_into_pipeline(self) -> None:
        from unittest.mock import MagicMock

        from azure.cosmos.agent_memory.aio import AsyncCosmosMemoryClient
        from azure.cosmos.agent_memory.aio import cosmos_memory_client as mod

        client = AsyncCosmosMemoryClient.__new__(AsyncCosmosMemoryClient)
        client._transcript_metadata_keys = ("agent_id",)
        client._turns_container_client = MagicMock()
        client._memories_container_client = MagicMock()
        client._summaries_container_client = MagicMock()
        client._chat_client = MagicMock()
        client._embeddings_client = MagicMock()
        captured: dict[str, object] = {}

        original = mod.AsyncPipelineService

        def fake_pipeline(*args: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return MagicMock()

        mod.AsyncPipelineService = fake_pipeline  # type: ignore[assignment]
        try:
            client._build_pipeline(MagicMock())
        finally:
            mod.AsyncPipelineService = original

        assert captured.get("transcript_metadata_keys") == ("agent_id",)


class TestIncludeTimestamp:
    """include_timestamp prefixes lines with the turn's event time so the
    extraction LLM can resolve relative time expressions to absolute dates."""

    def test_timestamp_prefix_flat(self) -> None:
        items = [
            {"role": "user", "content": "Hello", "created_at": "2024-06-20T10:00:00+00:00"},
            {"role": "assistant", "content": "Hi", "created_at": "2024-06-20T10:01:00+00:00"},
        ]
        out = build_transcript(items, include_timestamp=True)
        assert out == ("[2024-06-20T10:00:00+00:00 | user]: Hello\n[2024-06-20T10:01:00+00:00 | agent]: Hi")

    def test_timestamp_absent_falls_back_to_plain_role(self) -> None:
        items = [{"role": "user", "content": "Hello"}]  # no created_at
        out = build_transcript(items, include_timestamp=True)
        assert out == "[user]: Hello"

    def test_timestamp_off_by_default(self) -> None:
        items = [{"role": "user", "content": "Hello", "created_at": "2024-06-20T10:00:00+00:00"}]
        out = build_transcript(items)
        assert out == "[user]: Hello"

    def test_timestamp_prefix_grouped(self) -> None:
        items = [
            {"role": "user", "content": "Q", "thread_id": "t1", "created_at": "2024-06-20T10:00:00+00:00"},
        ]
        out = build_transcript(items, group_by_thread=True, include_timestamp=True)
        assert "[2024-06-20T10:00:00+00:00 | user]: Q" in out
        assert "=== Thread t1 ===" in out


class TestCanonicalSpeaker:
    """Roles are normalized to canonical speaker labels unconditionally, so the
    extraction prompt sees a stable vocabulary regardless of the caller's role
    string. Synonyms fold onto user/agent; ``tool`` and ``system`` are kept."""

    def test_agent_synonym_normalized(self) -> None:
        # OpenAI-style "assistant" must render as the canonical "agent".
        assert build_transcript([{"role": "assistant", "content": "Hi"}]) == "[agent]: Hi"

    def test_user_synonym_normalized(self) -> None:
        assert build_transcript([{"role": "human", "content": "a"}]) == "[user]: a"

    def test_tool_and_system_preserved(self) -> None:
        items = [
            {"role": "tool", "content": "b"},
            {"role": "system", "content": "c"},
            {"role": "agent", "content": "d"},
        ]
        out = build_transcript(items)
        assert out == "[tool]: b\n[system]: c\n[agent]: d"

    def test_unknown_role_passed_through(self) -> None:
        assert build_transcript([{"role": "narrator", "content": "x"}]) == "[narrator]: x"

    def test_normalized_with_timestamp(self) -> None:
        items = [{"role": "assistant", "content": "Hi", "created_at": "2024-06-20T10:01:00+00:00"}]
        out = build_transcript(items, include_timestamp=True)
        assert out == "[2024-06-20T10:01:00+00:00 | agent]: Hi"
