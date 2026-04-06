"""Config science + UX tests — env var precedence matrix and error UX.

A config scientist asks: given N env vars, what is the exact precedence and
can the user predict the outcome without reading source?  A UX researcher
asks: when something goes wrong, does the error tell the user how to fix it,
or just describe the failure?

These two disciplines meet here because config failures are almost always
experienced as UX failures.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import kitsune.cli
import kitsune.config as config_module
import kitsune.consent as consent_module


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

    importlib.reload(config_module)
    importlib.reload(kitsune.cli)
    yield


def _reload() -> config_module.Settings:
    importlib.reload(config_module)
    return config_module.Settings()


# ===========================================================================
# Class CS1: Env var precedence matrix
# ===========================================================================


class TestEnvVarPrecedence:
    """Pin down the precedence order explicitly:

    1. Explicit KITSUNE_MODEL_NAME always wins over tier resolution.
    2. Explicit KITSUNE_BASE_URL always wins over provider default.
    3. KITSUNE_PROVIDER overrides model_name / base_url unless those are
       explicitly pinned.
    4. KITSUNE_MODEL_TIER overrides _default_model() unless MODEL_NAME
       is also set.
    5. An empty KITSUNE_PROVIDER is treated as "not set" (back to local).
    """

    def test_explicit_model_name_wins_over_tier(self, monkeypatch):
        monkeypatch.setenv("KITSUNE_MODEL_TIER", "large")
        monkeypatch.setenv("KITSUNE_MODEL_NAME", "my/favorite-model")
        s = _reload()
        assert s.model_name == "my/favorite-model"

    def test_explicit_model_name_wins_over_provider_default(self, monkeypatch):
        monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        monkeypatch.setenv("KITSUNE_MODEL_NAME", "my/override")
        s = _reload()
        assert s.model_name == "my/override"

    def test_explicit_base_url_wins_over_provider_default(self, monkeypatch):
        monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        monkeypatch.setenv("KITSUNE_BASE_URL", "https://local-proxy/v1")
        s = _reload()
        assert s.base_url == "https://local-proxy/v1"

    def test_provider_wins_over_tier(self, monkeypatch):
        monkeypatch.setenv("KITSUNE_MODEL_TIER", "medium")
        monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        s = _reload()
        # Provider's default model should apply, not the tier's model
        assert "qwen3-coder" in s.model_name
        assert s.privacy_level == "remote_free"

    def test_tier_wins_over_default(self, monkeypatch):
        monkeypatch.setenv("KITSUNE_MODEL_TIER", "medium")
        s = _reload()
        assert "Qwen3.5-4B" in s.model_name

    def test_default_when_nothing_is_set(self):
        s = _reload()
        assert s.model_tier == "small"
        assert "Qwen2.5-Coder" in s.model_name
        assert s.privacy_level == "local"

    def test_empty_provider_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("KITSUNE_PROVIDER", "")
        s = _reload()
        # Empty string → no provider override → stays local
        assert s.provider_name == ""
        assert s.privacy_level == "local"


# ===========================================================================
# Class CS2: Tier resolution determinism
# ===========================================================================


class TestTierResolution:
    """Tier -> model resolution is deterministic and symmetric: same tier,
    same backend, always the same model identifier."""

    def test_each_tier_resolves_to_expected_model_on_mlx(self):
        from kitsune.config import resolve_model

        assert resolve_model("mlx", "small") == "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
        assert resolve_model("mlx", "medium") == "mlx-community/Qwen3.5-4B-Instruct-4bit"
        assert resolve_model("mlx", "large") == "mlx-community/Qwen3.5-9B-Instruct-4bit"

    def test_each_tier_resolves_to_expected_model_on_ollama(self):
        from kitsune.config import resolve_model

        assert resolve_model("ollama", "small") == "qwen2.5-coder:1.5b"
        assert resolve_model("ollama", "medium") == "qwen3.5:4b"
        assert resolve_model("ollama", "large") == "qwen3.5:9b"

    def test_unknown_tier_falls_back_to_small(self):
        from kitsune.config import resolve_model

        fallback = resolve_model("mlx", "gigantic")
        assert fallback == resolve_model("mlx", "small")

    def test_unknown_backend_yields_empty_string(self):
        from kitsune.config import resolve_model

        # Unknown backend can't fall back to anything meaningful
        assert resolve_model("nonexistent-backend", "small") == ""


# ===========================================================================
# Class CS3: Error UX — messages must be actionable
# ===========================================================================


class TestErrorMessageUX:
    """Every user-visible error should tell the user either what went wrong
    AND how to fix it, or at minimum mention the name of the relevant env
    var / config flag."""

    def test_unknown_provider_error_lists_valid_options(self, monkeypatch):
        monkeypatch.setenv("KITSUNE_PROVIDER", "nonexistent")
        with pytest.raises(ValueError) as exc:
            importlib.reload(config_module)
        msg = str(exc.value)
        # Must mention the bad value AND the valid options
        assert "nonexistent" in msg
        assert "openrouter" in msg
        assert "local-mlx" in msg
        assert "anthropic" in msg

    def test_missing_api_key_error_names_the_env_var(self, monkeypatch):
        monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
        with pytest.raises(ValueError) as exc:
            importlib.reload(config_module)
        msg = str(exc.value)
        assert "OPENROUTER_API_KEY" in msg
        # Should explicitly say "not set" or similar actionable phrasing
        assert "not set" in msg or "required" in msg

    def test_rate_limit_error_suggests_tier_switch(self, monkeypatch):
        monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        importlib.reload(config_module)

        from kitsune.inference import backend as backend_module
        from kitsune.inference.backend import RateLimitExceeded

        monkeypatch.setattr(backend_module, "_BASE_BACKOFF", 0.0)
        from unittest.mock import MagicMock

        class _Err(Exception):
            status_code = 429

        fake = MagicMock()
        fake.invoke.side_effect = _Err("429")
        with patch.object(backend_module, "get_llm", return_value=fake):
            try:
                backend_module.invoke("sys", "user")
            except RateLimitExceeded as exc:
                msg = str(exc)
        # Message should mention "local" and "paid" as alternatives
        assert "local" in msg.lower()
        assert "paid" in msg.lower()
        # And should tell the user how to switch (the env var name)
        assert "KITSUNE_PROVIDER" in msg

    def test_consent_denied_error_is_actionable(self):
        from kitsune.consent import ConsentDenied, ensure_consent

        try:
            ensure_consent(
                provider_name="openrouter",
                base_url="https://openrouter.ai/api/v1",
                privacy_level="remote_free",
                interactive=False,
            )
        except ConsentDenied as exc:
            msg = str(exc)
        # Must mention the env var that bypasses the prompt
        assert "KITSUNE_REMOTE_CONSENT" in msg
        # And "interactive" or "terminal" for context
        assert "interactive" in msg.lower() or "terminal" in msg.lower()


# ===========================================================================
# Class CS4: CLI status — the UX of discovery
# ===========================================================================


class TestStatusCommandUX:
    """kit status is the first thing users run. It must communicate the
    privacy state clearly and let users discover every available tier."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_status_works_when_server_is_down(self, runner):
        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("down")):
            result = runner.invoke(kitsune.cli.app, ["status"])
        # Even with server down, status should still exit 0 and inform user
        assert result.exit_code == 0
        assert "not reachable" in result.output.lower() or "down" in result.output.lower()
        # Must suggest a fix
        assert "mlx_lm.server" in result.output or "ollama" in result.output

    def test_status_shows_every_provider_in_registry(self, runner):
        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("down")):
            result = runner.invoke(kitsune.cli.app, ["status"])
        for name in ("local-mlx", "local-ollama", "openrouter", "anthropic"):
            assert name in result.output

    def test_status_indicates_which_providers_need_keys(self, runner):
        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("down")):
            result = runner.invoke(kitsune.cli.app, ["status"])
        # Without any keys set, both remote providers should show "no key"
        lines = result.output.splitlines()
        openrouter_line = next(
            line for line in lines if "openrouter" in line and "Provider:" not in line
        )
        assert "no key" in openrouter_line

    def test_status_highlights_active_provider(self, runner, monkeypatch):
        monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        importlib.reload(config_module)
        importlib.reload(kitsune.cli)

        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("down")):
            result = runner.invoke(kitsune.cli.app, ["status"])
        assert "openrouter" in result.output
        assert "remote_free" in result.output

    def test_status_tier_field_matches_env(self, runner, monkeypatch):
        monkeypatch.setenv("KITSUNE_MODEL_TIER", "large")
        importlib.reload(config_module)
        importlib.reload(kitsune.cli)

        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("down")):
            result = runner.invoke(kitsune.cli.app, ["status"])
        assert "large" in result.output
