"""Tests for the privacy consent flow (Artemisa veto)."""

from __future__ import annotations

import json

import pytest
from rich.console import Console

import kitsune.consent as consent_module
from kitsune.consent import (
    ConsentDenied,
    build_warning_banner,
    ensure_consent,
    has_consent,
    record_consent,
)


@pytest.fixture
def tmp_consent(tmp_path, monkeypatch):
    """Redirect consent persistence to a temp file for every test."""
    fake_dir = tmp_path / ".kitsune"
    fake_file = fake_dir / "consent.json"
    monkeypatch.setattr(consent_module, "_CONSENT_DIR", fake_dir)
    monkeypatch.setattr(consent_module, "_CONSENT_FILE", fake_file)
    return fake_file


@pytest.fixture(autouse=True)
def _clean_consent_env(monkeypatch):
    monkeypatch.delenv("KITSUNE_REMOTE_CONSENT", raising=False)
    yield


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_no_consent_on_fresh_install(tmp_consent):
    assert has_consent("openrouter") is False


def test_record_and_check_consent(tmp_consent):
    record_consent("openrouter", "https://openrouter.ai/api/v1")
    assert has_consent("openrouter") is True

    data = json.loads(tmp_consent.read_text(encoding="utf-8"))
    assert "openrouter" in data
    assert data["openrouter"]["base_url"] == "https://openrouter.ai/api/v1"
    assert "consented_at" in data["openrouter"]


def test_corrupt_consent_file_is_tolerated(tmp_consent):
    tmp_consent.parent.mkdir(parents=True, exist_ok=True)
    tmp_consent.write_text("{not valid json", encoding="utf-8")
    assert has_consent("openrouter") is False


# ---------------------------------------------------------------------------
# ensure_consent — the gate
# ---------------------------------------------------------------------------


def test_local_provider_never_prompts(tmp_consent):
    # No banner, no persistence, no prompt — instant True.
    ok = ensure_consent(
        provider_name="local-mlx",
        base_url="http://localhost:8008/v1",
        privacy_level="local",
    )
    assert ok is True
    assert tmp_consent.exists() is False


def test_env_var_bypass_records_consent(tmp_consent, monkeypatch):
    monkeypatch.setenv("KITSUNE_REMOTE_CONSENT", "1")
    ok = ensure_consent(
        provider_name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        privacy_level="remote_free",
        console=Console(quiet=True),
    )
    assert ok is True
    assert has_consent("openrouter") is True


def test_previously_consented_provider_skips_prompt(tmp_consent):
    record_consent("openrouter", "https://openrouter.ai/api/v1")
    ok = ensure_consent(
        provider_name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        privacy_level="remote_free",
        console=Console(quiet=True),
        interactive=False,
    )
    assert ok is True


def test_no_tty_without_consent_raises(tmp_consent):
    with pytest.raises(ConsentDenied, match="interactive"):
        ensure_consent(
            provider_name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            privacy_level="remote_free",
            console=Console(quiet=True),
            interactive=False,
        )


def test_interactive_yes_records_consent(tmp_consent, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    ok = ensure_consent(
        provider_name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        privacy_level="remote_free",
        console=Console(quiet=True),
        interactive=True,
    )
    assert ok is True
    assert has_consent("openrouter") is True


def test_interactive_no_raises(tmp_consent, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    with pytest.raises(ConsentDenied, match="denied consent"):
        ensure_consent(
            provider_name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            privacy_level="remote_free",
            console=Console(quiet=True),
            interactive=True,
        )
    assert has_consent("openrouter") is False


def test_banner_contains_provider_and_url():
    panel = build_warning_banner("openrouter", "https://openrouter.ai/api/v1")
    # Render to a string — the banner must name the provider AND the URL
    from io import StringIO

    buf = StringIO()
    Console(file=buf, width=80, record=False).print(panel)
    rendered = buf.getvalue()
    assert "openrouter" in rendered
    assert "openrouter.ai" in rendered
    assert "REMOTE PROVIDER WARNING" in rendered
