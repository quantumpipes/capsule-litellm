# Copyright 2026 Quantum Pipes Technologies, LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CapsuleLogger LiteLLM callback."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from capsule_litellm.callback import (
    CapsuleLogger,
    _duration_ms,
    _extract_request,
    _hash_messages,
    _parse_response,
    _RESULT_TRUNCATE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    content: str = "Hello!",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    model: str = "gpt-4o",
) -> SimpleNamespace:
    """Build a minimal object that looks like litellm.ModelResponse."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage, model=model, id="resp-1")


def _make_kwargs(
    model: str = "gpt-4o",
    messages: list[dict] | None = None,
    call_type: str = "completion",
) -> dict:
    if messages is None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
    return {"model": model, "messages": messages, "call_type": call_type}


# ---------------------------------------------------------------------------
# Unit tests: _extract_request
# ---------------------------------------------------------------------------


class TestExtractRequest:
    def test_last_user_message(self):
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Follow-up"},
        ]
        assert _extract_request(msgs) == "Follow-up"

    def test_no_user_message(self):
        msgs = [{"role": "system", "content": "System only"}]
        assert _extract_request(msgs) == ""

    def test_empty_messages(self):
        assert _extract_request([]) == ""

    def test_single_user_message(self):
        msgs = [{"role": "user", "content": "Hello"}]
        assert _extract_request(msgs) == "Hello"

    def test_non_string_content_converted(self):
        msgs = [{"role": "user", "content": ["structured", "content"]}]
        result = _extract_request(msgs)
        assert isinstance(result, str)
        assert "structured" in result

    def test_missing_content_key(self):
        msgs = [{"role": "user"}]
        assert _extract_request(msgs) == ""

    def test_none_content(self):
        msgs = [{"role": "user", "content": None}]
        result = _extract_request(msgs)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Unit tests: _hash_messages
# ---------------------------------------------------------------------------


class TestHashMessages:
    def test_deterministic(self):
        msgs = [{"role": "user", "content": "Hello"}]
        h1 = _hash_messages(msgs)
        h2 = _hash_messages(msgs)
        assert h1 == h2
        assert len(h1) == 64

    def test_different_messages_different_hash(self):
        h1 = _hash_messages([{"role": "user", "content": "Hello"}])
        h2 = _hash_messages([{"role": "user", "content": "World"}])
        assert h1 != h2

    def test_sha3_256_length(self):
        h = _hash_messages([])
        assert len(h) == 64

    def test_order_matters(self):
        msgs_a = [{"role": "user", "content": "A"}, {"role": "user", "content": "B"}]
        msgs_b = [{"role": "user", "content": "B"}, {"role": "user", "content": "A"}]
        assert _hash_messages(msgs_a) != _hash_messages(msgs_b)

    def test_unicode_messages(self):
        msgs = [{"role": "user", "content": "café ☃ 東京"}]
        h = _hash_messages(msgs)
        assert len(h) == 64


# ---------------------------------------------------------------------------
# Unit tests: _duration_ms
# ---------------------------------------------------------------------------


class TestDurationMs:
    def test_normal(self):
        start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        end = start + timedelta(seconds=1.5)
        assert _duration_ms(start, end) == 1500

    def test_zero(self):
        t = datetime(2026, 1, 1, tzinfo=UTC)
        assert _duration_ms(t, t) == 0

    def test_negative_clamped(self):
        start = datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC)
        end = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        assert _duration_ms(start, end) == 0

    def test_sub_millisecond(self):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        end = start + timedelta(microseconds=500)
        assert _duration_ms(start, end) == 0

    def test_large_duration(self):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        end = start + timedelta(hours=1)
        assert _duration_ms(start, end) == 3_600_000

    def test_non_datetime_returns_zero(self):
        assert _duration_ms("not a datetime", "also not") == 0


# ---------------------------------------------------------------------------
# Unit tests: _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_success(self):
        resp = _make_response(content="Paris", prompt_tokens=10, completion_tokens=3)
        text, t_in, t_out, err = _parse_response(resp, success=True)
        assert text == "Paris"
        assert t_in == 10
        assert t_out == 3
        assert err is None

    def test_failure(self):
        text, t_in, t_out, err = _parse_response(
            Exception("timeout"), success=False
        )
        assert text == ""
        assert t_in == 0
        assert t_out == 0
        assert "timeout" in err

    def test_failure_none(self):
        _, _, _, err = _parse_response(None, success=False)
        assert err == "Unknown error"

    def test_success_no_choices(self):
        resp = SimpleNamespace(
            choices=[], usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0)
        )
        text, _, _, _ = _parse_response(resp, success=True)
        assert isinstance(text, str)

    def test_success_none_content(self):
        message = SimpleNamespace(content=None)
        choice = SimpleNamespace(message=message)
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=0)
        resp = SimpleNamespace(choices=[choice], usage=usage)
        text, t_in, _, _ = _parse_response(resp, success=True)
        assert text == ""
        assert t_in == 5

    def test_success_no_usage_attribute(self):
        message = SimpleNamespace(content="OK")
        choice = SimpleNamespace(message=message)
        resp = SimpleNamespace(choices=[choice])
        text, t_in, t_out, _ = _parse_response(resp, success=True)
        assert text == "OK"
        assert t_in == 0
        assert t_out == 0

    def test_success_none_response(self):
        text, t_in, t_out, err = _parse_response(None, success=True)
        assert text == ""
        assert err is None


# ---------------------------------------------------------------------------
# Integration tests: CapsuleLogger
# ---------------------------------------------------------------------------


class _RecordingChain:
    """Stand-in for ``CapsuleChain`` that records every persisted capsule.

    Mirrors the real ``seal_and_store(capsule, *, seal=...)`` async contract so
    the callback exercises the SAME code path it uses in production, while the
    test can introspect the built capsule. Use this for field-level assertions;
    use a real ``Capsules()`` for the durable-persistence integration test.
    """

    def __init__(self) -> None:
        self.captured: list[Any] = []
        self.fail_with: Exception | None = None

    async def seal_and_store(self, capsule: Any, *, seal: Any = None) -> Any:
        if self.fail_with is not None:
            raise self.fail_with
        self.captured.append(capsule)
        return capsule


class _FakeCapsules:
    """Minimal ``Capsules``-shaped object wrapping a ``_RecordingChain``."""

    def __init__(self, chain: _RecordingChain) -> None:
        self._chain = chain
        self.seal = MagicMock()

    @property
    def chain(self) -> _RecordingChain:
        return self._chain


class TestCapsuleLogger:
    @pytest.fixture()
    def chain(self) -> _RecordingChain:
        return _RecordingChain()

    @pytest.fixture()
    def mock_capsules(self, chain: _RecordingChain) -> _FakeCapsules:
        return _FakeCapsules(chain)

    def _only(self, chain: _RecordingChain) -> Any:
        assert len(chain.captured) == 1, "exactly one capsule must be persisted"
        return chain.captured[0]

    def test_success_creates_capsule(self, mock_capsules, chain):
        logger = CapsuleLogger(capsules=mock_capsules, agent_id="test-agent")

        start = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
        end = start + timedelta(seconds=2)

        logger.log_success_event(
            _make_kwargs(), _make_response(), start, end
        )

        capsule = self._only(chain)
        assert capsule.trigger.source == "litellm"
        assert capsule.trigger.request == "What is 2+2?"
        assert capsule.trigger.timestamp == start
        assert capsule.context.agent_id == "test-agent"
        assert capsule.context.environment["model"] == "gpt-4o"
        assert capsule.context.environment["call_type"] == "completion"
        assert capsule.reasoning.model == "gpt-4o"
        assert len(capsule.reasoning.prompt_hash) == 64
        assert capsule.execution.duration_ms == 2000
        assert capsule.execution.tool_calls[0].tool == "litellm.completion"
        assert capsule.execution.tool_calls[0].success is True
        assert capsule.outcome.status == "success"
        assert capsule.outcome.error is None

    def test_failure_creates_capsule(self, mock_capsules, chain):
        logger = CapsuleLogger(capsules=mock_capsules)

        start = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
        end = start + timedelta(milliseconds=100)

        logger.log_failure_event(
            _make_kwargs(), Exception("rate_limit"), start, end
        )

        capsule = self._only(chain)
        assert capsule.outcome.status == "failure"
        assert "rate_limit" in capsule.outcome.error
        assert capsule.execution.duration_ms == 100
        assert capsule.execution.tool_calls[0].success is False

    def test_sync_persists_via_seal_and_store(self, mock_capsules, chain):
        """The sync callback drives the async seal_and_store to completion."""
        logger = CapsuleLogger(capsules=mock_capsules)
        start = datetime(2026, 3, 1, tzinfo=UTC)

        logger.log_success_event(_make_kwargs(), _make_response(), start, start)
        assert len(chain.captured) == 1

    def test_swallow_errors_default(self, mock_capsules, chain):
        chain.fail_with = RuntimeError("boom")
        logger = CapsuleLogger(capsules=mock_capsules)

        start = datetime(2026, 3, 1, tzinfo=UTC)
        logger.log_success_event(_make_kwargs(), _make_response(), start, start)

    def test_swallow_errors_false(self, mock_capsules, chain):
        chain.fail_with = RuntimeError("boom")
        logger = CapsuleLogger(capsules=mock_capsules, swallow_errors=False)

        start = datetime(2026, 3, 1, tzinfo=UTC)
        with pytest.raises(RuntimeError, match="boom"):
            logger.log_success_event(
                _make_kwargs(), _make_response(), start, start
            )

    def test_custom_domain_and_type(self, mock_capsules, chain):
        from qp_capsule import CapsuleType

        logger = CapsuleLogger(
            capsules=mock_capsules,
            domain="finance",
            capsule_type=CapsuleType.AGENT,
        )

        start = datetime(2026, 3, 1, tzinfo=UTC)
        logger.log_success_event(_make_kwargs(), _make_response(), start, start)

        capsule = self._only(chain)
        assert capsule.domain == "finance"

    def test_embedding_call_type(self, mock_capsules, chain):
        logger = CapsuleLogger(capsules=mock_capsules)
        kwargs = _make_kwargs(call_type="embedding")
        start = datetime(2026, 3, 1, tzinfo=UTC)

        logger.log_success_event(kwargs, _make_response(), start, start)

        capsule = self._only(chain)
        assert capsule.context.environment["call_type"] == "embedding"
        assert capsule.execution.tool_calls[0].tool == "litellm.embedding"

    def test_result_truncation(self, mock_capsules, chain):
        long_content = "x" * (_RESULT_TRUNCATE + 500)
        logger = CapsuleLogger(capsules=mock_capsules)

        resp = _make_response(content=long_content)
        start = datetime(2026, 3, 1, tzinfo=UTC)
        logger.log_success_event(_make_kwargs(), resp, start, start)

        capsule = self._only(chain)
        assert len(capsule.outcome.result) == _RESULT_TRUNCATE
        assert len(capsule.execution.tool_calls[0].result) == _RESULT_TRUNCATE

    def test_missing_messages_in_kwargs(self, mock_capsules, chain):
        logger = CapsuleLogger(capsules=mock_capsules)
        kwargs = {"model": "gpt-4o"}
        start = datetime(2026, 3, 1, tzinfo=UTC)

        logger.log_success_event(kwargs, _make_response(), start, start)

        capsule = self._only(chain)
        assert capsule.trigger.request == ""
        assert capsule.execution.tool_calls[0].arguments["message_count"] == 0

    def test_missing_model_in_kwargs(self, mock_capsules, chain):
        logger = CapsuleLogger(capsules=mock_capsules)
        kwargs = {"messages": [{"role": "user", "content": "hi"}]}
        start = datetime(2026, 3, 1, tzinfo=UTC)

        logger.log_success_event(kwargs, _make_response(), start, start)

        capsule = self._only(chain)
        assert capsule.reasoning.model == "unknown"

    @pytest.mark.asyncio
    async def test_async_success(self, mock_capsules, chain):
        logger = CapsuleLogger(capsules=mock_capsules)

        start = datetime(2026, 3, 1, tzinfo=UTC)
        end = start + timedelta(seconds=1)

        await logger.async_log_success_event(
            _make_kwargs(), _make_response(), start, end
        )

        capsule = self._only(chain)
        assert capsule.outcome.status == "success"
        assert capsule.execution.duration_ms == 1000

    @pytest.mark.asyncio
    async def test_async_failure(self, mock_capsules, chain):
        logger = CapsuleLogger(capsules=mock_capsules)

        start = datetime(2026, 3, 1, tzinfo=UTC)
        await logger.async_log_failure_event(
            _make_kwargs(), Exception("500"), start, start
        )

        capsule = self._only(chain)
        assert capsule.outcome.status == "failure"

    @pytest.mark.asyncio
    async def test_async_swallow_errors(self, mock_capsules, chain):
        chain.fail_with = RuntimeError("async boom")
        logger = CapsuleLogger(capsules=mock_capsules)
        start = datetime(2026, 3, 1, tzinfo=UTC)

        await logger.async_log_success_event(
            _make_kwargs(), _make_response(), start, start
        )

    @pytest.mark.asyncio
    async def test_async_swallow_errors_false_propagates(self, mock_capsules, chain):
        chain.fail_with = RuntimeError("async boom")
        logger = CapsuleLogger(capsules=mock_capsules, swallow_errors=False)
        start = datetime(2026, 3, 1, tzinfo=UTC)

        with pytest.raises(RuntimeError, match="async boom"):
            await logger.async_log_success_event(
                _make_kwargs(), _make_response(), start, start
            )

    @pytest.mark.asyncio
    async def test_sync_callback_inside_running_loop_schedules_task(
        self, mock_capsules, chain
    ):
        """A sync callback fired from inside a running loop still persists.

        LiteLLM may invoke the sync hook from async code. The callback must not
        raise 'asyncio.run() cannot be called from a running event loop'; it
        schedules the coroutine on the live loop instead.
        """
        logger = CapsuleLogger(capsules=mock_capsules)
        start = datetime(2026, 3, 1, tzinfo=UTC)

        logger.log_success_event(_make_kwargs(), _make_response(), start, start)
        # Let the scheduled task run.
        await asyncio.sleep(0)
        assert len(chain.captured) == 1

    def test_token_metrics(self, mock_capsules, chain):
        logger = CapsuleLogger(capsules=mock_capsules)

        resp = _make_response(prompt_tokens=500, completion_tokens=200)
        start = datetime(2026, 3, 1, tzinfo=UTC)
        logger.log_success_event(_make_kwargs(), resp, start, start)

        capsule = self._only(chain)
        assert capsule.outcome.metrics["tokens_in"] == 500
        assert capsule.outcome.metrics["tokens_out"] == 200
        assert capsule.outcome.metrics["latency_ms"] == 0
        assert capsule.execution.resources_used["tokens_in"] == 500
        assert capsule.execution.resources_used["tokens_out"] == 200

    def test_prompt_hash_determinism(self, mock_capsules, chain):
        logger = CapsuleLogger(capsules=mock_capsules)

        kwargs = _make_kwargs()
        start = datetime(2026, 3, 1, tzinfo=UTC)

        logger.log_success_event(kwargs, _make_response(), start, start)
        logger.log_success_event(kwargs, _make_response(), start, start)

        assert len(chain.captured) == 2
        hash1 = chain.captured[0].reasoning.prompt_hash
        hash2 = chain.captured[1].reasoning.prompt_hash
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_multiple_calls_create_separate_capsules(self, mock_capsules, chain):
        logger = CapsuleLogger(capsules=mock_capsules)
        start = datetime(2026, 3, 1, tzinfo=UTC)

        logger.log_success_event(_make_kwargs(), _make_response(), start, start)
        logger.log_success_event(
            _make_kwargs(model="claude-3"), _make_response(), start, start
        )

        assert len(chain.captured) == 2


# ---------------------------------------------------------------------------
# Durable-persistence integration tests: real Capsules() + SQLite store
# ---------------------------------------------------------------------------


class TestCapsuleLoggerDurablePersistence:
    """End-to-end: a recorded LLM call must leave a sealed, verifiable capsule
    row in a real SQLite-backed Capsules store. This is the regression guard for
    the original defect where every call produced no capsule."""

    @pytest.fixture()
    def db_path(self, tmp_path):
        return tmp_path / "capsules.db"

    def test_sync_call_writes_verifiable_capsule_row(self, db_path):
        from qp_capsule import Capsules

        capsules = Capsules(str(db_path))
        try:
            logger = CapsuleLogger(capsules=capsules, agent_id="durable-test")

            start = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
            end = start + timedelta(seconds=1)
            logger.log_success_event(_make_kwargs(), _make_response(), start, end)

            async def _read() -> list:
                return list(await capsules.storage.get_all_ordered())

            rows = asyncio.run(_read())
            assert len(rows) == 1, "exactly one capsule row must exist after one call"

            capsule = rows[0]
            # Durably sealed.
            assert capsule.hash != ""
            assert capsule.signature != ""
            assert capsule.signed_at is not None
            # Genuine cryptographic verification of the persisted row.
            assert capsules.seal.verify(capsule) is True
            # Content fidelity.
            assert capsule.context.agent_id == "durable-test"
            assert capsule.outcome.status == "success"
        finally:
            asyncio.run(capsules.close())

    @pytest.mark.asyncio
    async def test_async_call_writes_verifiable_capsule_row(self, db_path):
        from qp_capsule import Capsules

        capsules = Capsules(str(db_path))
        try:
            logger = CapsuleLogger(capsules=capsules)

            start = datetime(2026, 3, 1, tzinfo=UTC)
            end = start + timedelta(milliseconds=250)
            await logger.async_log_success_event(
                _make_kwargs(), _make_response(), start, end
            )

            rows = list(await capsules.storage.get_all_ordered())
            assert len(rows) == 1
            assert capsules.seal.verify(rows[0]) is True
        finally:
            await capsules.close()

    @pytest.mark.asyncio
    async def test_multiple_calls_form_valid_chain(self, db_path):
        from qp_capsule import Capsules

        capsules = Capsules(str(db_path))
        try:
            logger = CapsuleLogger(capsules=capsules)
            start = datetime(2026, 3, 1, tzinfo=UTC)

            await logger.async_log_success_event(
                _make_kwargs(), _make_response(), start, start
            )
            await logger.async_log_failure_event(
                _make_kwargs(), Exception("rate_limit"), start, start
            )

            rows = list(await capsules.storage.get_all_ordered())
            assert len(rows) == 2
            assert rows[0].sequence == 0
            assert rows[1].sequence == 1
            assert rows[1].previous_hash == rows[0].hash

            result = await capsules.chain.verify(seal=capsules.seal)
            assert result.valid is True
            assert result.capsules_verified == 2
        finally:
            await capsules.close()

    def test_broken_storage_config_surfaces_at_startup(self):
        """A storage backend that cannot initialise must raise at construction,
        not silently swallow every capsule at call time."""
        from unittest.mock import patch

        from qp_capsule.exceptions import CapsuleError

        with (
            patch(
                "qp_capsule.audit.Capsules.__init__",
                side_effect=CapsuleError("storage unavailable"),
            ),
            pytest.raises(CapsuleError, match="storage unavailable"),
        ):
            CapsuleLogger()


class TestExtractRequestEdgeCases:
    def test_non_string_content(self):
        msgs = [{"role": "user", "content": ["text block", {"type": "image"}]}]
        result = _extract_request(msgs)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_missing_content_key(self):
        msgs = [{"role": "user"}]
        assert _extract_request(msgs) == ""

    def test_unicode_content(self):
        msgs = [{"role": "user", "content": "cafe\u0301 ☃"}]
        assert _extract_request(msgs) == "cafe\u0301 ☃"


class TestParseResponseEdgeCases:
    def test_no_usage_attribute(self):
        resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
        text, t_in, t_out, err = _parse_response(resp, success=True)
        assert text == "ok"
        assert t_in == 0
        assert t_out == 0

    def test_none_content(self):
        msg = SimpleNamespace(content=None)
        choice = SimpleNamespace(message=msg)
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
        resp = SimpleNamespace(choices=[choice], usage=usage)
        text, _, _, _ = _parse_response(resp, success=True)
        assert text == ""

    def test_success_none_response(self):
        text, t_in, t_out, err = _parse_response(None, success=True)
        assert text == ""
        assert err is None
