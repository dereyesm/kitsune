"""Security auditor tests — OWASP-style hardening across the attack surface.

Threat model:
    - Kitsune handles source code (sensitive IP) and API keys (credentials).
    - Attackers could be: malicious repos (poison code context), crafted env
      vars (injection), corrupted consent files (state tampering), or users
      misconfiguring remote providers without realising it.
    - Kitsune MUST: never log credentials, never silently cross the local/remote
      boundary, refuse obviously dangerous provider names or URLs, and survive
      corrupted state files without losing other users' consent.

Each test maps to a concrete concern. If a test fails here, treat it as a
security finding, not a coverage gap.
"""

from __future__ import annotations

import importlib
import json
import re
from unittest.mock import MagicMock, patch

import pytest

import kitsune.config as config_module
import kitsune.consent as consent_module
from kitsune.consent import (
    build_warning_banner,
    ensure_consent,
    has_consent,
    record_consent,
)
from kitsune.graph.router import route, suggest_tiers
from kitsune.graph.state import KitsuneState
from kitsune.inference import backend as backend_module
from kitsune.providers import PROVIDERS, PrivacyLevel

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


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


def _fresh_settings(**env) -> config_module.Settings:
    importlib.reload(config_module)
    return config_module.Settings()


# ===========================================================================
# Class S1: Credential hygiene — never leak API keys
# ===========================================================================


class TestCredentialHygiene:
    """API keys must never appear in error messages, state, logs, or tiers
    listings. A leaked key in an error string becomes a key in a bug report,
    which becomes a key on someone's screen share.
    """

    def test_unknown_provider_error_does_not_leak_other_env_vars(self, monkeypatch):
        SECRET = "sk-or-SECRET-VALUE-12345"
        monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
        monkeypatch.setenv("KITSUNE_PROVIDER", "nonexistent-xyz")
        with pytest.raises(ValueError) as exc:
            importlib.reload(config_module)
        assert SECRET not in str(exc.value)

    def test_missing_key_error_does_not_expose_secret_names_as_values(self, monkeypatch):
        # If the key is missing, the error should reference the env var NAME
        # but not try to echo back any partial value
        monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
        with pytest.raises(ValueError) as exc:
            importlib.reload(config_module)
        msg = str(exc.value)
        assert "OPENROUTER_API_KEY" in msg
        # The word "not set" is OK; dump of env values is not
        assert "sk-" not in msg

    def test_rate_limit_error_scrubs_api_key(self, monkeypatch):
        SECRET = "sk-or-SHOULD-NEVER-LEAK"
        monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
        importlib.reload(config_module)

        class _Err(Exception):
            status_code = 429

        fake = MagicMock()
        fake.invoke.side_effect = _Err("429 Too Many Requests")
        with patch.object(backend_module, "get_llm", return_value=fake):
            with pytest.raises(backend_module.RateLimitExceeded) as exc:
                backend_module.invoke("sys", "user")
        assert SECRET not in str(exc.value)

    def test_suggest_tiers_never_echoes_api_keys(self, monkeypatch):
        SECRET = "sk-or-LEAK-CHECK"
        monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
        tiers = suggest_tiers()
        assert SECRET not in tiers
        # Only mentions the env var NAME, not the value
        assert "OPENROUTER_API_KEY" in tiers

    def test_consent_banner_does_not_include_api_key(self, monkeypatch):
        SECRET = "sk-or-BANNER-LEAK"
        monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
        importlib.reload(config_module)
        panel = build_warning_banner("openrouter", config_module.settings.base_url)
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        Console(file=buf, width=100).print(panel)
        assert SECRET not in buf.getvalue()


# ===========================================================================
# Class S2: Provider-name injection — reject hostile identifiers
# ===========================================================================


class TestProviderInjection:
    """A hostile env var should not be able to slip into the provider
    resolution path. Unknown names must fail at config load, before anything
    else executes."""

    @pytest.mark.parametrize(
        "name",
        [
            "openrouter; DROP TABLE users",
            "../../etc/passwd",
            "openrouter$(whoami)",
            "openrouter\n",
            "",  # empty string should not bypass into "local"
        ],
    )
    def test_malicious_provider_names_rejected(self, monkeypatch, name):
        monkeypatch.setenv("KITSUNE_PROVIDER", name)
        # Empty string is special: we accept "not set" == empty, so no raise
        if name == "":
            importlib.reload(config_module)
            assert config_module.settings.provider_name == ""
            return
        with pytest.raises(ValueError):
            importlib.reload(config_module)

    def test_registry_is_a_closed_set(self):
        # Only these four names exist — no plugins, no dynamic loading
        assert set(PROVIDERS.keys()) == {
            "local-mlx",
            "local-ollama",
            "openrouter",
            "anthropic",
        }


# ===========================================================================
# Class S3: Path handling — protect against path traversal via file arg
# ===========================================================================


class TestFilePathHandling:
    """The CLI accepts file paths from the user. We rely on pathlib.Path to
    resolve them and refuse non-files, which blocks the common escape paths
    (symlinks to /etc/shadow, directories-as-files, etc.)."""

    def test_missing_file_errors_cleanly(self, tmp_path):
        import typer

        from kitsune.cli import _read_file

        nonexistent = tmp_path / "does_not_exist.py"
        with pytest.raises(typer.Exit):
            _read_file(str(nonexistent))

    def test_directory_passed_as_file_is_rejected(self, tmp_path):
        import typer

        from kitsune.cli import _read_file

        with pytest.raises(typer.Exit):
            _read_file(str(tmp_path))  # tmp_path is a directory, not a file

    def test_relative_path_resolved_to_absolute(self, tmp_path, monkeypatch):
        from kitsune.cli import _read_file

        target = tmp_path / "file.py"
        target.write_text("x = 1")
        monkeypatch.chdir(tmp_path)
        _, path = _read_file("file.py")
        # Path must be fully resolved (absolute), NOT the relative input
        assert path.startswith("/")


# ===========================================================================
# Class S4: Prompt-injection resistance in the router
# ===========================================================================


class TestRouterPromptInjection:
    """The router uses regex over user_input to decide routing. A malicious
    actor could craft a prompt that evades the security gate or collapses
    the regex. These tests pin down expected behaviour for hostile inputs."""

    @pytest.mark.parametrize(
        "prompt",
        [
            "help me find SECURITY issues",
            "show me the Security flaws",
            "Check for SQL injection vulnerabilities please",
            "pentest this endpoint",
            "audit for XSS and CSRF",
        ],
    )
    def test_security_keywords_always_escalate(self, prompt):
        result = route(_state(user_input=prompt, code_context="x = 1"))
        assert result["task_type"] == "fallback"
        assert "security" in result["escalation_reason"]

    def test_router_is_not_fooled_by_base64_wrapping(self):
        # A benign explain request wrapped in base64-ish chars should still
        # route to "ask" (or "explain" if keyword matches). This verifies
        # that the router does NOT try to decode and guess.
        result = route(
            _state(
                user_input="dGVsbCBtZSBhYm91dCB0aGlz",
                code_context="print('hi')",
            )
        )
        # Neither keyword nor gate should fire — lands in "ask"
        assert result["task_type"] == "ask"

    def test_router_regex_survives_catastrophic_input(self):
        # Very long input should NOT hang the regex (ReDoS check)
        hostile = "a" * 50000 + " security issue"
        result = route(_state(user_input=hostile, code_context=""))
        assert result["task_type"] == "fallback"


# ===========================================================================
# Class S5: Consent file tamper resistance
# ===========================================================================


class TestConsentTamperResistance:
    """The consent file lives at ~/.kitsune/consent.json. If an attacker or a
    crash corrupts it, Kitsune must degrade gracefully — refusing all prior
    consents — rather than silently trusting a malformed record."""

    def test_corrupt_file_forgets_all_consents(self):
        record_consent("openrouter", "https://openrouter.ai/api/v1")
        assert has_consent("openrouter") is True

        # Tamper the file
        consent_module._CONSENT_FILE.write_text("{ corrupted", encoding="utf-8")
        assert has_consent("openrouter") is False

    def test_missing_consent_directory_is_created_on_write(self):
        # ~/.kitsune/ may not exist on a fresh install
        assert not consent_module._CONSENT_DIR.exists()
        record_consent("openrouter", "https://openrouter.ai/api/v1")
        assert consent_module._CONSENT_DIR.exists()
        assert consent_module._CONSENT_FILE.exists()

    def test_consent_record_is_valid_json(self):
        record_consent("openrouter", "https://openrouter.ai/api/v1")
        data = json.loads(consent_module._CONSENT_FILE.read_text())
        assert "openrouter" in data
        assert "consented_at" in data["openrouter"]
        # ISO-8601 sanity: 2026-... at the start
        assert re.match(r"20\d\d-\d\d-\d\dT", data["openrouter"]["consented_at"])

    def test_second_consent_does_not_wipe_first(self):
        record_consent("openrouter", "https://openrouter.ai/api/v1")
        record_consent("anthropic", "https://api.anthropic.com/v1")
        data = json.loads(consent_module._CONSENT_FILE.read_text())
        assert "openrouter" in data
        assert "anthropic" in data

    def test_duplicate_record_does_not_duplicate_key(self):
        record_consent("openrouter", "https://openrouter.ai/api/v1")
        record_consent("openrouter", "https://openrouter.ai/api/v1")
        data = json.loads(consent_module._CONSENT_FILE.read_text())
        # Dict semantics: only one entry per provider
        assert list(data.keys()).count("openrouter") == 1


# ===========================================================================
# Class S6: Privacy boundary — local code never crosses accidentally
# ===========================================================================


class TestPrivacyBoundary:
    """The local/remote boundary is the single most important invariant in
    Kitsune. No accidental misconfig should flip a user into remote mode."""

    def test_default_is_100_percent_local(self):
        s = _fresh_settings()
        assert s.privacy_level == "local"
        assert s.provider_name == ""

    def test_tier_change_never_flips_privacy(self, monkeypatch):
        monkeypatch.setenv("KITSUNE_MODEL_TIER", "large")
        s = _fresh_settings()
        assert s.privacy_level == "local"

    def test_setting_api_key_alone_does_not_enable_remote(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-accident")
        s = _fresh_settings()
        assert s.privacy_level == "local"
        assert s.provider_name == ""

    def test_consent_file_presence_does_not_imply_remote_is_active(self):
        record_consent("openrouter", "https://openrouter.ai/api/v1")
        s = _fresh_settings()
        assert s.privacy_level == "local"

    def test_consent_banner_is_required_on_every_remote_run(self, monkeypatch):
        """Even with consent recorded, the banner must print every time."""
        record_consent("openrouter", "https://openrouter.ai/api/v1")
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        ensure_consent(
            provider_name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            privacy_level="remote_free",
            console=Console(file=buf, width=100),
            interactive=False,
        )
        assert "REMOTE PROVIDER WARNING" in buf.getvalue()


# ===========================================================================
# Class S7: URL scheme validation — refuse obviously dangerous schemes
# ===========================================================================


class TestURLScheme:
    """Provider base URLs must use https (or http for localhost). The registry
    is hand-written, so this is a regression guard for future edits."""

    def test_all_providers_use_http_or_https(self):
        for p in PROVIDERS.values():
            assert p.base_url.startswith(("http://", "https://"))

    def test_remote_providers_must_use_https(self):
        for p in PROVIDERS.values():
            if p.privacy_level != PrivacyLevel.LOCAL:
                assert p.base_url.startswith("https://"), (
                    f"{p.name} is remote but uses non-https URL: {p.base_url}"
                )

    def test_local_providers_only_use_localhost(self):
        for p in PROVIDERS.values():
            if p.privacy_level == PrivacyLevel.LOCAL:
                assert "localhost" in p.base_url or "127.0.0.1" in p.base_url
