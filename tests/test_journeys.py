"""Customer journey tests — narrative end-to-end flows.

Each test reads like a user story: "as a <persona>, I want to <goal>, so that
<outcome>". These tests stitch together config + consent + graph + backend to
verify the full path a real user would traverse.

Personas covered:
    P1 — IndieDev Mac, privacy-first (100% local)
    P2 — Linux hobbyist on Ollama upgrading to Qwen3.5
    P3 — Team Lead LatAm with $0 budget (free remote)
    P4 — Enterprise Dev blocked from remote providers
    P5 — CI/CD pipeline without TTY
    P6 — Power user mixing tiers
    P7 — Return user (consent already recorded)
    P8 — Discoverer browsing free models
    P9 — User hitting the escalation gate
    P10 — User hitting a 429 from the remote provider
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

import kitsune.config as config_module
import kitsune.consent as consent_module
from kitsune.consent import ConsentDenied, ensure_consent, has_consent, record_consent
from kitsune.graph.router import route, suggest_tiers
from kitsune.graph.state import KitsuneState
from kitsune.inference import backend as backend_module
from kitsune.inference.backend import RateLimitExceeded, invoke
from kitsune.providers import PROVIDERS, PrivacyLevel, get_provider


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Every journey starts from a blank slate: no env vars, no consent file."""
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

    # Keep retry loops instant so journeys stay fast.
    monkeypatch.setattr(backend_module, "_BASE_BACKOFF", 0.0)
    yield


def _fresh_settings() -> config_module.Settings:
    importlib.reload(config_module)
    return config_module.Settings()


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
# P1 — IndieDev Mac, privacy-first
# ===========================================================================


class TestPersonaIndieDevMac:
    """As an IndieDev on macOS, I want Kitsune to never send my code anywhere
    so that my unpublished work stays on my laptop."""

    def test_J1_default_is_100_percent_local(self):
        s = _fresh_settings()
        # Backend depends on host OS (mlx on Darwin, ollama elsewhere) — both
        # are local. The privacy contract does not depend on the backend.
        assert s.backend in ("mlx", "ollama")
        assert s.privacy_level == "local"
        assert s.provider_name == ""  # no override
        assert s.api_key == "not-needed"
        assert "localhost" in s.base_url

    def test_local_requests_never_hit_consent_flow(self):
        # ensure_consent must return instantly for local, never touching disk
        ok = ensure_consent(
            provider_name="local-mlx",
            base_url="http://localhost:8008/v1",
            privacy_level="local",
        )
        assert ok is True
        assert has_consent("local-mlx") is False  # nothing recorded


# ===========================================================================
# P2 — Linux hobbyist upgrading to Qwen3.5
# ===========================================================================


class TestPersonaLinuxHobbyist:
    """As an Ollama user on Linux, I want to try Qwen3.5-4B without rewriting
    my config so I can see if the quality bump is worth the RAM."""

    def test_J2_medium_tier_resolves_to_qwen35_4b_on_ollama(self, monkeypatch):
        monkeypatch.setattr(config_module, "_SYSTEM", "Linux")
        monkeypatch.setenv("KITSUNE_BACKEND", "ollama")
        monkeypatch.setenv("KITSUNE_MODEL_TIER", "medium")
        s = _fresh_settings()
        assert s.backend == "ollama"
        assert s.model_tier == "medium"
        assert s.model_name == "qwen3.5:4b"
        # Still 100% local
        assert s.privacy_level == "local"

    def test_large_tier_resolves_to_qwen35_9b(self, monkeypatch):
        monkeypatch.setenv("KITSUNE_MODEL_TIER", "large")
        s = _fresh_settings()
        assert "Qwen3.5-9B" in s.model_name or "qwen3.5:9b" in s.model_name


# ===========================================================================
# P3 — Team Lead LatAm with $0 budget
# ===========================================================================


class TestPersonaLatAmFreeTier:
    """As a Team Lead in LatAm with a zero budget, I want access to frontier
    models for free so my team can compete with better-funded orgs."""

    def test_openrouter_exposes_qwen3_coder_480b_and_nemotron_free(self):
        prov = get_provider("openrouter")
        model_ids = {mid for mid, _ in prov.free_models}
        assert "qwen/qwen3-coder:free" in model_ids
        assert any("nemotron" in mid for mid in model_ids)

    def test_J3_first_switch_to_openrouter_requires_consent(self, monkeypatch):
        monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
        s = _fresh_settings()
        # Settings loaded cleanly
        assert s.provider_name == "openrouter"
        assert s.privacy_level == "remote_free"
        # But consent has NOT been granted yet
        assert has_consent("openrouter") is False
        # And a non-interactive ensure_consent must refuse
        with pytest.raises(ConsentDenied):
            ensure_consent(
                provider_name=s.provider_name,
                base_url=s.base_url,
                privacy_level=s.privacy_level,
                interactive=False,
            )


# ===========================================================================
# P4 — Enterprise Dev blocked from remote providers
# ===========================================================================


class TestPersonaEnterpriseLocked:
    """As an Enterprise Dev, my company policy forbids sending source code to
    external services. I need Kitsune to never silently reach out."""

    def test_default_kitsune_never_reaches_out(self):
        s = _fresh_settings()
        # No remote state whatsoever
        assert s.privacy_level == "local"
        assert "localhost" in s.base_url
        assert s.api_key == "not-needed"

    def test_remote_provider_requires_explicit_opt_in(self, monkeypatch):
        # Just having OPENROUTER_API_KEY in env is NOT enough to flip the
        # privacy level — KITSUNE_PROVIDER must be set explicitly.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-accident")
        s = _fresh_settings()
        assert s.privacy_level == "local"
        assert s.provider_name == ""


# ===========================================================================
# P5 — CI/CD pipeline without TTY
# ===========================================================================


class TestPersonaCICD:
    """As a CI pipeline, I need to run Kitsune headlessly with a remote
    provider. I will NOT have a TTY but I will set KITSUNE_REMOTE_CONSENT=1."""

    def test_J5_env_bypass_records_consent_and_proceeds(self, monkeypatch):
        monkeypatch.setenv("KITSUNE_REMOTE_CONSENT", "1")
        ok = ensure_consent(
            provider_name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            privacy_level="remote_free",
            interactive=False,
        )
        assert ok is True
        assert has_consent("openrouter") is True

    def test_J6_ci_without_bypass_fails_fast(self):
        # No TTY, no env var → must raise so the pipeline surfaces the error
        with pytest.raises(ConsentDenied, match="interactive"):
            ensure_consent(
                provider_name="openrouter",
                base_url="https://openrouter.ai/api/v1",
                privacy_level="remote_free",
                interactive=False,
            )


# ===========================================================================
# P6 — Power user mixing tiers in a single session
# ===========================================================================


class TestPersonaPowerUser:
    """As a power user, I want to pin an arbitrary base_url and model while
    still keeping the provider's privacy label + api_key resolution."""

    def test_J13_explicit_url_and_model_win(self, monkeypatch):
        monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-pow")
        monkeypatch.setenv("KITSUNE_BASE_URL", "https://my-proxy.local/v1")
        monkeypatch.setenv("KITSUNE_MODEL_NAME", "my/private-model")
        s = _fresh_settings()
        assert s.base_url == "https://my-proxy.local/v1"
        assert s.model_name == "my/private-model"
        # Provider metadata still flows through — consent still required
        assert s.provider_name == "openrouter"
        assert s.privacy_level == "remote_free"
        assert s.api_key == "sk-or-pow"


# ===========================================================================
# P7 — Return user with consent already recorded
# ===========================================================================


class TestPersonaReturnUser:
    """As a returning user, I already said yes to openrouter last week. I want
    Kitsune to remember that and not ask again."""

    def test_J4_recorded_consent_skips_prompt(self, monkeypatch):
        record_consent("openrouter", "https://openrouter.ai/api/v1")
        assert has_consent("openrouter") is True
        # No interactive input needed, no env var needed — just proceeds
        ok = ensure_consent(
            provider_name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            privacy_level="remote_free",
            interactive=False,
        )
        assert ok is True


# ===========================================================================
# P8 — Discoverer browsing free models
# ===========================================================================


class TestPersonaDiscoverer:
    """As a curious dev, I want to see which free tiers Kitsune knows about
    even when I haven't configured any keys yet."""

    def test_suggest_tiers_lists_every_tier_even_without_keys(self):
        tiers = suggest_tiers()
        assert "Local tier up" in tiers
        assert "openrouter" in tiers
        assert "anthropic" in tiers
        # Without any keys, every remote entry should flag "set key to enable"
        assert "set key to enable" in tiers

    def test_suggest_tiers_marks_ready_providers(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        tiers = suggest_tiers()
        # OpenRouter should be ready, anthropic still not
        assert "openrouter" in tiers and "READY" in tiers
        ready_lines = [line for line in tiers.splitlines() if "openrouter" in line]
        assert any("READY" in line for line in ready_lines)

    def test_registry_has_expected_coverage(self):
        names = set(PROVIDERS.keys())
        assert {"local-mlx", "local-ollama", "openrouter", "anthropic"}.issubset(names)
        free = [p for p in PROVIDERS.values() if p.privacy_level == PrivacyLevel.REMOTE_FREE]
        assert len(free) >= 1


# ===========================================================================
# P9 — User hitting the escalation gate
# ===========================================================================


class TestPersonaEscalationUser:
    """As a user asking a task too complex for the local model, I want clear
    guidance on how to escalate — NOT a dead end."""

    def test_J9_token_overflow_routes_to_fallback(self):
        huge_code = "x = 1\n" * 2000  # way over the 1500 token threshold
        result = route(_state(user_input="explain this", code_context=huge_code))
        assert result["task_type"] == "fallback"
        assert "too large" in result["escalation_reason"]

    def test_J12_security_keyword_auto_escalates(self):
        result = route(
            _state(
                user_input="help me find SQL injection vulnerabilities",
                code_context="SELECT * FROM users",
            )
        )
        assert result["task_type"] == "fallback"
        assert "security" in result["escalation_reason"]

    def test_fallback_message_advertises_every_tier(self):
        from kitsune.graph.nodes import fallback_node

        result = fallback_node(
            _state(
                user_input="refactor this entire module",
                escalation_reason="refactoring/migration tasks are too complex",
                file_path="big.py",
            )
        )
        body = result["response"]
        assert "Local tier up" in body
        assert "Free remote" in body
        assert "openrouter" in body
        assert "anthropic" in body
        assert "refactor this entire module" in body


# ===========================================================================
# P10 — User hitting 429 on a remote provider
# ===========================================================================


class TestPersonaRateLimited:
    """As a user on OpenRouter free tier, I sometimes hit 429. I want the
    backend to retry briefly and then give me a useful error, not a crash."""

    def _build_rate_limit_error(self):
        class _RateLimit(Exception):
            status_code = 429

        return _RateLimit("rate limit reached")

    def test_J10_transient_429_is_retried_and_recovers(self, monkeypatch):
        fake = MagicMock()
        fake.invoke.side_effect = [
            self._build_rate_limit_error(),
            self._build_rate_limit_error(),
            MagicMock(content="final answer"),
        ]
        with patch.object(backend_module, "get_llm", return_value=fake):
            out = invoke("sys", "user")
        assert out == "final answer"
        assert fake.invoke.call_count == 3

    def test_J11_persistent_429_raises_rate_limit_exceeded(self, monkeypatch):
        monkeypatch.setattr(backend_module.settings, "provider_name", "openrouter")
        fake = MagicMock()
        fake.invoke.side_effect = self._build_rate_limit_error()
        with patch.object(backend_module, "get_llm", return_value=fake):
            with pytest.raises(RateLimitExceeded, match="openrouter"):
                invoke("sys", "user")

    def test_rate_limit_error_message_points_user_to_recovery(self, monkeypatch):
        monkeypatch.setattr(backend_module.settings, "provider_name", "openrouter")
        fake = MagicMock()
        fake.invoke.side_effect = self._build_rate_limit_error()
        with patch.object(backend_module, "get_llm", return_value=fake):
            try:
                invoke("sys", "user")
            except RateLimitExceeded as err:
                msg = str(err)
        # The error must mention both alternatives: local or paid
        assert "local" in msg.lower()
        assert "paid" in msg.lower()
