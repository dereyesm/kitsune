"""End-to-end CLI tests using typer's CliRunner.

These cover the user-visible surface of `kit status`, `kit ask` and
`kit explain` across combinations of environment variables. Every test
stubs out the actual HTTP/LLM call so the suite runs offline.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import kitsune.cli
import kitsune.config as config_module
import kitsune.consent as consent_module


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Scrub Kitsune env vars and redirect consent file to tmp."""
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

    # Force settings to reload with a clean env
    importlib.reload(config_module)
    importlib.reload(kitsune.cli)
    yield


# ---------------------------------------------------------------------------
# kit status — the "show of power" moment
# ---------------------------------------------------------------------------


def test_status_default_local(runner):
    """J1 — first run, no env vars. Status should be pure local."""
    with patch("httpx.get", side_effect=__import__("httpx").ConnectError("down")):
        result = runner.invoke(kitsune.cli.app, ["status"])
    assert result.exit_code == 0
    assert "Backend:" in result.output
    assert "mlx" in result.output or "ollama" in result.output
    assert "Tier:" in result.output and "small" in result.output
    # Provider line must say local when no override
    assert "Provider:" in result.output
    assert "local" in result.output
    # Multi-tier block must list all 4 providers
    assert "local-mlx" in result.output
    assert "local-ollama" in result.output
    assert "openrouter" in result.output
    assert "anthropic" in result.output
    # Without keys, openrouter/anthropic should be marked "no key"
    assert "no key" in result.output


def test_status_shows_medium_tier(runner, monkeypatch):
    """J2 — user asked for medium tier, status should reflect Qwen3.5-4B."""
    monkeypatch.setenv("KITSUNE_MODEL_TIER", "medium")
    importlib.reload(config_module)
    importlib.reload(kitsune.cli)
    with patch("httpx.get", side_effect=__import__("httpx").ConnectError("down")):
        result = runner.invoke(kitsune.cli.app, ["status"])
    assert result.exit_code == 0
    assert "medium" in result.output
    assert "Qwen3.5-4B" in result.output


def test_status_shows_openrouter_ready_when_key_set(runner, monkeypatch):
    """J15 — with OPENROUTER_API_KEY set, openrouter should be marked ready."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    importlib.reload(config_module)
    importlib.reload(kitsune.cli)
    with patch("httpx.get", side_effect=__import__("httpx").ConnectError("down")):
        result = runner.invoke(kitsune.cli.app, ["status"])
    assert result.exit_code == 0
    # openrouter line must NOT say "no key" — should show the free tier marker
    assert "openrouter" in result.output
    lines = [ln for ln in result.output.splitlines() if "openrouter" in ln and "Provider" not in ln]
    assert any("no key" not in ln for ln in lines)


def test_status_shows_active_provider_privacy_level(runner, monkeypatch):
    """When KITSUNE_PROVIDER is openrouter, status should highlight remote_free."""
    monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    importlib.reload(config_module)
    importlib.reload(kitsune.cli)
    with patch("httpx.get", side_effect=__import__("httpx").ConnectError("down")):
        result = runner.invoke(kitsune.cli.app, ["status"])
    assert result.exit_code == 0
    # Provider field should display the provider name + remote_free
    assert "openrouter" in result.output
    assert "remote_free" in result.output


# ---------------------------------------------------------------------------
# kit ask — gating behaviour
# ---------------------------------------------------------------------------


def test_ask_local_runs_without_consent_prompt(runner, tmp_path, monkeypatch):
    """Local provider should never trigger the consent flow."""
    code_file = tmp_path / "sample.py"
    code_file.write_text("def add(a, b):\n    return a + b\n")

    fake_result = {
        "user_input": "what does add do?",
        "task_type": "ask",
        "code_context": "def add(a, b):\n    return a + b\n",
        "file_path": str(code_file),
        "response": "Adds two numbers.",
        "escalation_reason": "",
    }
    with patch.object(kitsune.cli, "graph_app") as fake_graph:
        fake_graph.invoke.return_value = fake_result
        result = runner.invoke(kitsune.cli.app, ["ask", "what does add do?", "-f", str(code_file)])
    assert result.exit_code == 0, result.output
    assert "Adds two numbers" in result.output
    # No consent banner should appear for local provider
    assert "REMOTE PROVIDER WARNING" not in result.output


def test_ask_remote_without_consent_fails(runner, tmp_path, monkeypatch):
    """J6 — remote provider, no TTY, no env bypass → ConsentDenied → exit 2."""
    monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    importlib.reload(config_module)
    importlib.reload(kitsune.cli)

    code_file = tmp_path / "f.py"
    code_file.write_text("x = 1")
    result = runner.invoke(kitsune.cli.app, ["ask", "explain", "-f", str(code_file)])
    # Should refuse and exit with code 2
    assert result.exit_code == 2
    assert "REMOTE PROVIDER WARNING" in result.output
    assert "interactive" in result.output.lower() or "denied" in result.output.lower()


def test_ask_remote_with_env_bypass_succeeds(runner, tmp_path, monkeypatch):
    """J5 — KITSUNE_REMOTE_CONSENT=1 bypasses the prompt and lets the run proceed."""
    monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    monkeypatch.setenv("KITSUNE_REMOTE_CONSENT", "1")
    importlib.reload(config_module)
    importlib.reload(kitsune.cli)

    code_file = tmp_path / "f.py"
    code_file.write_text("x = 1")

    fake_result = {
        "user_input": "explain",
        "task_type": "ask",
        "code_context": "x = 1",
        "file_path": str(code_file),
        "response": "It binds 1 to x.",
        "escalation_reason": "",
    }
    with patch.object(kitsune.cli, "graph_app") as fake_graph:
        fake_graph.invoke.return_value = fake_result
        result = runner.invoke(kitsune.cli.app, ["ask", "explain", "-f", str(code_file)])
    assert result.exit_code == 0, result.output
    assert "REMOTE PROVIDER WARNING" in result.output
    assert "It binds 1 to x." in result.output


def test_ask_missing_file_errors(runner):
    """Pathological input — missing file should error cleanly."""
    result = runner.invoke(kitsune.cli.app, ["ask", "what?", "-f", "/does/not/exist.py"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# kit explain
# ---------------------------------------------------------------------------


def test_explain_runs_with_file(runner, tmp_path):
    code_file = tmp_path / "hello.py"
    code_file.write_text("print('hi')")
    fake_result = {
        "user_input": "",
        "task_type": "explain",
        "code_context": "print('hi')",
        "file_path": str(code_file),
        "response": "Prints 'hi' to stdout.",
        "escalation_reason": "",
    }
    with patch.object(kitsune.cli, "graph_app") as fake_graph:
        fake_graph.invoke.return_value = fake_result
        result = runner.invoke(kitsune.cli.app, ["explain", str(code_file)])
    assert result.exit_code == 0, result.output
    assert "Prints" in result.output


def test_explain_requires_input(runner):
    """Without file and without stdin, explain should refuse."""
    result = runner.invoke(kitsune.cli.app, ["explain"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Pathological env configs surface clear errors
# ---------------------------------------------------------------------------


def test_unknown_provider_surfaces_error(runner, monkeypatch):
    """J8 — unknown provider name should fail loudly at settings load."""
    monkeypatch.setenv("KITSUNE_PROVIDER", "madeupname")
    # Reloading config should raise before CLI even dispatches
    with pytest.raises(ValueError, match="madeupname"):
        importlib.reload(config_module)


def test_openrouter_without_key_fails_at_config(monkeypatch):
    """J7 — openrouter without OPENROUTER_API_KEY must fail at settings load."""
    monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        importlib.reload(config_module)


def test_explicit_url_override_beats_provider(monkeypatch):
    """J13 — KITSUNE_BASE_URL beats the provider's default base_url."""
    monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    monkeypatch.setenv("KITSUNE_BASE_URL", "https://my-proxy.local/v1")
    monkeypatch.setenv("KITSUNE_MODEL_NAME", "my/model")
    importlib.reload(config_module)
    s = config_module.settings
    assert s.base_url == "https://my-proxy.local/v1"
    assert s.model_name == "my/model"
    assert s.provider_name == "openrouter"
    assert s.privacy_level == "remote_free"
