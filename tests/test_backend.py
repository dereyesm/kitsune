"""Tests for the unified inference backend — retry + 429 escalation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kitsune.inference import backend
from kitsune.inference.backend import RateLimitExceeded, _is_rate_limit, invoke


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeRateLimitError(Exception):
    """Mimics openai.RateLimitError for duck-typing detection."""

    def __init__(self, msg: str = "rate limit reached"):
        super().__init__(msg)
        self.status_code = 429


# ---------------------------------------------------------------------------
# _is_rate_limit — duck typing sanity
# ---------------------------------------------------------------------------


def test_is_rate_limit_detects_class_name():
    err = _FakeRateLimitError()
    assert _is_rate_limit(err) is True


def test_is_rate_limit_detects_429_in_message():
    assert _is_rate_limit(RuntimeError("HTTP 429 Too Many Requests")) is True


def test_is_rate_limit_detects_rate_limit_phrase():
    assert _is_rate_limit(RuntimeError("rate limit exceeded")) is True


def test_is_rate_limit_rejects_other_errors():
    assert _is_rate_limit(ValueError("bad input")) is False
    assert _is_rate_limit(RuntimeError("connection refused")) is False


# ---------------------------------------------------------------------------
# invoke() — retry + escalation
# ---------------------------------------------------------------------------


@pytest.fixture
def fast_backoff(monkeypatch):
    """Shrink the backoff so tests stay under a second."""
    monkeypatch.setattr(backend, "_BASE_BACKOFF", 0.0)
    yield


def test_invoke_retries_and_succeeds(fast_backoff):
    fake_llm = MagicMock()
    # First 2 calls raise rate limit, third returns a response
    fake_llm.invoke.side_effect = [
        _FakeRateLimitError(),
        _FakeRateLimitError(),
        _FakeResponse("recovered answer"),
    ]
    with patch.object(backend, "get_llm", return_value=fake_llm):
        result = invoke("sys", "user")
    assert result == "recovered answer"
    assert fake_llm.invoke.call_count == 3


def test_invoke_raises_after_retry_budget_exhausted(fast_backoff):
    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = _FakeRateLimitError()
    with patch.object(backend, "get_llm", return_value=fake_llm):
        with pytest.raises(RateLimitExceeded, match="after 3 attempts"):
            invoke("sys", "user")
    # Exactly _MAX_RETRIES attempts, no more
    assert fake_llm.invoke.call_count == backend._MAX_RETRIES


def test_invoke_non_rate_limit_errors_propagate_immediately(fast_backoff):
    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = ValueError("bad prompt")
    with patch.object(backend, "get_llm", return_value=fake_llm):
        with pytest.raises(ValueError, match="bad prompt"):
            invoke("sys", "user")
    # Should NOT retry for non-rate-limit errors
    assert fake_llm.invoke.call_count == 1


def test_invoke_strips_eos_tokens(fast_backoff):
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = _FakeResponse("hello<|im_end|> world<|endoftext|>")
    with patch.object(backend, "get_llm", return_value=fake_llm):
        result = invoke("sys", "user")
    assert result == "hello world"


def test_rate_limit_exceeded_message_mentions_provider(monkeypatch, fast_backoff):
    monkeypatch.setattr(backend.settings, "provider_name", "openrouter")
    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = _FakeRateLimitError()
    with patch.object(backend, "get_llm", return_value=fake_llm):
        with pytest.raises(RateLimitExceeded, match="openrouter"):
            invoke("sys", "user")
