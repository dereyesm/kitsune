"""I18n + data integrity tests.

Two cross-cutting concerns:

1. Internationalization — Kitsune targets LatAm first, but must handle any
   unicode (Chinese/Japanese code comments, emoji, accented filenames). The
   project also advertises support for 10 programming languages. Router and
   backend must be unicode-clean.

2. Data integrity — the consent.json file is Kitsune's only persistent state.
   It must be valid JSON after every write, ISO-8601 timestamps must parse,
   and concurrent/repeated writes must be append-only (dict semantics).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

import kitsune.consent as consent_module
from kitsune.consent import build_warning_banner, record_consent
from kitsune.graph.router import route
from kitsune.graph.state import KitsuneState
from kitsune.inference import backend as backend_module
from kitsune.prompts.loader import build_system_prompt


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


def _state(**over) -> KitsuneState:
    base: KitsuneState = {
        "user_input": "",
        "task_type": "ask",
        "code_context": "",
        "file_path": "",
        "response": "",
        "escalation_reason": "",
    }
    base.update(over)
    return base  # type: ignore[return-value]


# ===========================================================================
# Class I1: Router handles non-ASCII input
# ===========================================================================


class TestRouterUnicode:
    """The router runs regexes over user_input. Regexes must be unicode-clean
    and never crash on Chinese, Arabic, emoji, or combining characters."""

    def test_spanish_explain_request_routes_to_explain(self):
        result = route(_state(user_input="explain esta función", code_context="def f(): pass"))
        assert result["task_type"] == "explain"

    def test_chinese_input_does_not_crash_router(self):
        result = route(_state(user_input="解释这段代码", code_context="x = 1"))
        assert result["task_type"] in ("explain", "ask", "fallback")

    def test_emoji_in_code_context(self):
        result = route(_state(user_input="what is this?", code_context="name = '🌍 world'"))
        assert result["task_type"] in ("explain", "ask")

    def test_rtl_arabic_does_not_crash(self):
        result = route(_state(user_input="اشرح هذا", code_context=""))
        assert result["task_type"] in ("explain", "ask", "fallback")

    def test_security_keyword_still_matches_even_with_unicode_padding(self):
        result = route(
            _state(
                user_input="检查 this SQL injection vulnerability 请",
                code_context="SELECT * FROM t",
            )
        )
        assert result["task_type"] == "fallback"


# ===========================================================================
# Class I2: Language skills — loader covers advertised languages
# ===========================================================================


class TestLanguageSkillCoverage:
    """The README advertises 10 supported languages. Every one must have a
    working skill file — otherwise the README lies."""

    EXPECTED_LANGUAGES = {
        "python",
        "javascript",
        "typescript",
        "go",
        "rust",
        "java",
        "csharp",
        "ruby",
        "php",
        "swift",
    }

    def test_all_advertised_languages_have_skill_prompts(self, tmp_path):
        """Every language should produce a non-trivial system prompt when
        loaded via the loader API."""
        # The loader uses the file extension to pick a skill. Map languages
        # to representative extensions.
        ext_by_lang = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "go": ".go",
            "rust": ".rs",
            "java": ".java",
            "csharp": ".cs",
            "ruby": ".rb",
            "php": ".php",
            "swift": ".swift",
        }
        for lang, ext in ext_by_lang.items():
            fake_path = str(tmp_path / f"sample{ext}")
            prompt = build_system_prompt("ask", fake_path)
            assert prompt, f"empty prompt for {lang}"
            assert len(prompt) > 50, f"stub prompt for {lang}"


# ===========================================================================
# Class I3: Backend preserves unicode in responses
# ===========================================================================


class TestBackendUnicode:
    """Responses from the LLM may contain any unicode. Our EOS stripping
    must not corrupt unicode characters."""

    class _FakeResponse:
        def __init__(self, content):
            self.content = content

    def test_chinese_response_preserved(self):
        fake = MagicMock()
        fake.invoke.return_value = self._FakeResponse("这是一个函数")
        with patch.object(backend_module, "get_llm", return_value=fake):
            out = backend_module.invoke("sys", "user")
        assert out == "这是一个函数"

    def test_emoji_response_preserved(self):
        fake = MagicMock()
        fake.invoke.return_value = self._FakeResponse("Looks good 👍 ship it 🚀")
        with patch.object(backend_module, "get_llm", return_value=fake):
            out = backend_module.invoke("sys", "user")
        assert "👍" in out and "🚀" in out

    def test_zero_width_joiner_preserved(self):
        # ZWJ sequences (used by emoji families) must survive intact
        fake = MagicMock()
        fake.invoke.return_value = self._FakeResponse("family: 👨‍👩‍👧")
        with patch.object(backend_module, "get_llm", return_value=fake):
            out = backend_module.invoke("sys", "user")
        assert "👨‍👩‍👧" in out

    def test_combining_characters_preserved(self):
        fake = MagicMock()
        fake.invoke.return_value = self._FakeResponse("café resumé")
        with patch.object(backend_module, "get_llm", return_value=fake):
            out = backend_module.invoke("sys", "user")
        assert out == "café resumé"


# ===========================================================================
# Class D1: Consent file schema
# ===========================================================================


class TestConsentFileSchema:
    """The consent file is Kitsune's only persistent state. Its schema must
    be stable: top-level is {provider_name: {base_url, consented_at}}."""

    def test_fresh_record_produces_expected_schema(self):
        record_consent("openrouter", "https://openrouter.ai/api/v1")
        data = json.loads(consent_module._CONSENT_FILE.read_text())
        assert isinstance(data, dict)
        assert "openrouter" in data
        entry = data["openrouter"]
        assert set(entry.keys()) == {"base_url", "consented_at"}
        assert entry["base_url"] == "https://openrouter.ai/api/v1"

    def test_consented_at_is_iso8601_parseable(self):
        record_consent("openrouter", "https://openrouter.ai/api/v1")
        data = json.loads(consent_module._CONSENT_FILE.read_text())
        # Should parse as ISO-8601 with timezone
        ts = datetime.fromisoformat(data["openrouter"]["consented_at"])
        assert ts.tzinfo is not None

    def test_consented_at_is_recent(self):
        record_consent("openrouter", "https://openrouter.ai/api/v1")
        data = json.loads(consent_module._CONSENT_FILE.read_text())
        ts = datetime.fromisoformat(data["openrouter"]["consented_at"])
        delta = datetime.now(timezone.utc) - ts
        assert delta.total_seconds() < 5  # should be within the last 5 seconds

    def test_record_consent_is_append_only_for_distinct_providers(self):
        record_consent("openrouter", "https://openrouter.ai/api/v1")
        record_consent("anthropic", "https://api.anthropic.com/v1")
        record_consent("local-mlx", "http://localhost:8008/v1")
        data = json.loads(consent_module._CONSENT_FILE.read_text())
        assert set(data.keys()) == {"openrouter", "anthropic", "local-mlx"}

    def test_second_record_for_same_provider_updates_in_place(self):
        record_consent("openrouter", "https://openrouter.ai/api/v1")
        first = json.loads(consent_module._CONSENT_FILE.read_text())
        record_consent("openrouter", "https://openrouter.ai/api/v1")
        second = json.loads(consent_module._CONSENT_FILE.read_text())
        # Same provider → exactly one entry
        assert len(second) == 1
        # Timestamp may change, base_url same
        assert second["openrouter"]["base_url"] == first["openrouter"]["base_url"]


# ===========================================================================
# Class D2: Consent banner renders unicode correctly
# ===========================================================================


class TestBannerUnicodeRender:
    """The Rich banner must render cleanly for provider names and URLs that
    contain unicode (conceivable for custom remote providers in other
    jurisdictions)."""

    def test_banner_with_unicode_provider_name(self):
        from io import StringIO

        from rich.console import Console

        panel = build_warning_banner(
            provider_name="远程服务",
            base_url="https://example.com/v1",
        )
        buf = StringIO()
        Console(file=buf, width=100).print(panel)
        out = buf.getvalue()
        assert "远程服务" in out
        assert "example.com" in out

    def test_banner_does_not_lose_https_slashes(self):
        from io import StringIO

        from rich.console import Console

        panel = build_warning_banner("x", "https://x.example/v1")
        buf = StringIO()
        Console(file=buf, width=100).print(panel)
        assert "https://x.example/v1" in buf.getvalue()
