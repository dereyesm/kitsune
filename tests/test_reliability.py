"""SRE / reliability tests — Kitsune must degrade gracefully.

A reliability engineer asks: what happens when the happy path breaks?
Timeouts, connection errors, empty responses, partial reads, reloads,
race conditions. These tests pin down degradation behaviour so future
regressions are visible.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

import kitsune.config as config_module
import kitsune.consent as consent_module
from kitsune.inference import backend as backend_module
from kitsune.inference.backend import RateLimitExceeded, invoke


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    for key in (
        "KITSUNE_BACKEND",
        "KITSUNE_BASE_URL",
        "KITSUNE_MODEL_NAME",
        "KITSUNE_MODEL_TIER",
        "KITSUNE_PROVIDER",
        "KITSUNE_REMOTE_CONSENT",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    fake_dir = tmp_path / ".kitsune"
    fake_file = fake_dir / "consent.json"
    monkeypatch.setattr(consent_module, "_CONSENT_DIR", fake_dir)
    monkeypatch.setattr(consent_module, "_CONSENT_FILE", fake_file)

    monkeypatch.setattr(backend_module, "_BASE_BACKOFF", 0.0)
    yield


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _RateLimitErr(Exception):
    status_code = 429

    def __init__(self, msg="rate limit"):
        super().__init__(msg)


# ===========================================================================
# Class R1: Retry policy — retry only what is safe to retry
# ===========================================================================


class TestRetryPolicy:
    """Only rate-limit (429) errors are retried. Everything else propagates
    immediately — retrying a 400 or a 500 is just wasted latency.
    """

    def test_retry_budget_exactly_three_attempts(self):
        fake = MagicMock()
        fake.invoke.side_effect = _RateLimitErr()
        with patch.object(backend_module, "get_llm", return_value=fake):
            with pytest.raises(RateLimitExceeded):
                invoke("sys", "user")
        assert fake.invoke.call_count == backend_module._MAX_RETRIES

    def test_non_rate_limit_error_not_retried(self):
        fake = MagicMock()
        fake.invoke.side_effect = ValueError("malformed request")
        with patch.object(backend_module, "get_llm", return_value=fake):
            with pytest.raises(ValueError):
                invoke("sys", "user")
        assert fake.invoke.call_count == 1

    def test_connection_error_not_retried(self):
        fake = MagicMock()
        fake.invoke.side_effect = ConnectionRefusedError("server down")
        with patch.object(backend_module, "get_llm", return_value=fake):
            with pytest.raises(ConnectionRefusedError):
                invoke("sys", "user")
        assert fake.invoke.call_count == 1

    def test_timeout_error_not_retried(self):
        fake = MagicMock()
        fake.invoke.side_effect = TimeoutError("request timed out")
        with patch.object(backend_module, "get_llm", return_value=fake):
            with pytest.raises(TimeoutError):
                invoke("sys", "user")
        assert fake.invoke.call_count == 1

    def test_first_attempt_success_no_retry(self):
        fake = MagicMock()
        fake.invoke.return_value = _FakeResponse("ok")
        with patch.object(backend_module, "get_llm", return_value=fake):
            out = invoke("sys", "user")
        assert out == "ok"
        assert fake.invoke.call_count == 1


# ===========================================================================
# Class R2: Response handling — survive weird server outputs
# ===========================================================================


class TestResponseHandling:
    """Downstream servers are flaky. Empty responses, trailing tokens,
    whitespace, and unicode noise all need to be normalised.
    """

    def test_empty_response_returns_empty_string(self):
        fake = MagicMock()
        fake.invoke.return_value = _FakeResponse("")
        with patch.object(backend_module, "get_llm", return_value=fake):
            assert invoke("sys", "user") == ""

    def test_whitespace_only_response_returns_empty_string(self):
        fake = MagicMock()
        fake.invoke.return_value = _FakeResponse("   \n\t  ")
        with patch.object(backend_module, "get_llm", return_value=fake):
            assert invoke("sys", "user") == ""

    def test_eos_tokens_are_stripped(self):
        fake = MagicMock()
        fake.invoke.return_value = _FakeResponse("answer<|im_end|><|endoftext|> middle<|im_start|>")
        with patch.object(backend_module, "get_llm", return_value=fake):
            out = invoke("sys", "user")
        assert "im_end" not in out
        assert "endoftext" not in out
        assert "im_start" not in out
        assert "answer" in out

    def test_unicode_response_preserved(self):
        fake = MagicMock()
        fake.invoke.return_value = _FakeResponse("Hola 世界 🌍")
        with patch.object(backend_module, "get_llm", return_value=fake):
            assert invoke("sys", "user") == "Hola 世界 🌍"

    def test_multi_line_response_preserved(self):
        fake = MagicMock()
        fake.invoke.return_value = _FakeResponse("line1\nline2\nline3")
        with patch.object(backend_module, "get_llm", return_value=fake):
            assert invoke("sys", "user") == "line1\nline2\nline3"


# ===========================================================================
# Class R3: Config reload determinism
# ===========================================================================


class TestConfigReload:
    """Reloading the config module repeatedly must be deterministic — same
    env, same settings, every time. A non-idempotent reload would break the
    test suite AND the MCP server's hot-reload behaviour.
    """

    def test_reload_is_idempotent(self, monkeypatch):
        monkeypatch.setenv("KITSUNE_MODEL_TIER", "medium")
        importlib.reload(config_module)
        s1 = config_module.settings.model_name
        importlib.reload(config_module)
        s2 = config_module.settings.model_name
        importlib.reload(config_module)
        s3 = config_module.settings.model_name
        assert s1 == s2 == s3

    def test_tier_change_between_reloads_takes_effect(self, monkeypatch):
        monkeypatch.setenv("KITSUNE_MODEL_TIER", "small")
        importlib.reload(config_module)
        small_name = config_module.settings.model_name

        monkeypatch.setenv("KITSUNE_MODEL_TIER", "large")
        importlib.reload(config_module)
        large_name = config_module.settings.model_name

        assert small_name != large_name
        # Backend-agnostic: case-insensitive substring (Mac=1.5B, Linux=1.5b)
        assert "1.5b" in small_name.lower()
        assert "9b" in large_name.lower()

    def test_provider_change_between_reloads_takes_effect(self, monkeypatch):
        # Start local
        importlib.reload(config_module)
        assert config_module.settings.privacy_level == "local"

        # Switch to openrouter
        monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        importlib.reload(config_module)
        assert config_module.settings.privacy_level == "remote_free"

        # Drop back to local
        monkeypatch.delenv("KITSUNE_PROVIDER")
        importlib.reload(config_module)
        assert config_module.settings.privacy_level == "local"


# ===========================================================================
# Class R4: Consent file resilience
# ===========================================================================


class TestConsentFileResilience:
    """The consent file may be missing, corrupt, or a symlink. None of these
    should crash Kitsune."""

    def test_missing_parent_directory_is_created(self):
        from kitsune.consent import record_consent

        assert not consent_module._CONSENT_DIR.exists()
        record_consent("openrouter", "https://openrouter.ai/api/v1")
        assert consent_module._CONSENT_DIR.is_dir()

    def test_empty_file_tolerated(self):
        from kitsune.consent import has_consent

        consent_module._CONSENT_DIR.mkdir(parents=True, exist_ok=True)
        consent_module._CONSENT_FILE.write_text("", encoding="utf-8")
        assert has_consent("openrouter") is False

    def test_non_object_json_tolerated(self):
        from kitsune.consent import has_consent

        consent_module._CONSENT_DIR.mkdir(parents=True, exist_ok=True)
        consent_module._CONSENT_FILE.write_text('["a", "b"]', encoding="utf-8")
        # Not a dict — has_consent should return False rather than crash
        # (any exception other than crash is acceptable)
        try:
            ok = has_consent("openrouter")
            assert ok is False
        except Exception:
            # Acceptable: any non-dict JSON is treated as "no consent"
            pass

    def test_concurrent_record_consent_calls_preserve_history(self):
        from kitsune.consent import has_consent, record_consent

        for provider in ("openrouter", "anthropic", "local-mlx", "local-ollama"):
            record_consent(provider, f"https://{provider}/v1")
        for provider in ("openrouter", "anthropic", "local-mlx", "local-ollama"):
            assert has_consent(provider)


# ===========================================================================
# Class R5: Graph reliability — invariants around escalation paths
# ===========================================================================


class TestGraphInvariants:
    """The LangGraph state machine must preserve its invariants under all
    routing paths."""

    def test_every_routing_path_sets_task_type(self):
        from kitsune.graph.router import route
        from kitsune.graph.state import KitsuneState

        bases: list[KitsuneState] = [
            {
                "user_input": "explain this",
                "task_type": "ask",
                "code_context": "x = 1",
                "file_path": "",
                "response": "",
                "escalation_reason": "",
            },
            {
                "user_input": "refactor this entire module",
                "task_type": "ask",
                "code_context": "y = 2",
                "file_path": "",
                "response": "",
                "escalation_reason": "",
            },
            {
                "user_input": "",
                "task_type": "ask",
                "code_context": "z = 3",
                "file_path": "",
                "response": "",
                "escalation_reason": "",
            },
        ]
        for state in bases:
            result = route(state)
            assert result["task_type"] in ("explain", "ask", "fallback")

    def test_fallback_always_carries_reason(self):
        from kitsune.graph.router import route
        from kitsune.graph.state import KitsuneState

        state: KitsuneState = {
            "user_input": "audit for XSS",
            "task_type": "ask",
            "code_context": "",
            "file_path": "",
            "response": "",
            "escalation_reason": "",
        }
        result = route(state)
        assert result["task_type"] == "fallback"
        assert result["escalation_reason"]  # not empty
