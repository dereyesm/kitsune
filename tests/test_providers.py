"""Tests for the provider registry and config integration."""

from __future__ import annotations

import importlib

import pytest

import kitsune.config as config_module
from kitsune.providers import PROVIDERS, PrivacyLevel, get_provider, list_providers
from kitsune.providers.base import Provider

# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


def test_registry_contains_required_providers():
    assert "local-mlx" in PROVIDERS
    assert "local-ollama" in PROVIDERS
    assert "openrouter" in PROVIDERS
    assert "anthropic" in PROVIDERS


def test_local_providers_do_not_require_key():
    for name in ("local-mlx", "local-ollama"):
        prov = get_provider(name)
        assert prov is not None
        assert prov.requires_key is False
        assert prov.is_remote is False
        assert prov.privacy_level == PrivacyLevel.LOCAL


def test_openrouter_is_free_remote_and_requires_key():
    prov = get_provider("openrouter")
    assert prov is not None
    assert prov.privacy_level == PrivacyLevel.REMOTE_FREE
    assert prov.requires_key is True
    assert prov.env_key_name == "OPENROUTER_API_KEY"
    assert prov.is_remote is True
    assert any("qwen3-coder" in mid for mid, _ in prov.free_models)
    assert any("nemotron" in mid for mid, _ in prov.free_models)


def test_anthropic_is_paid_remote():
    prov = get_provider("anthropic")
    assert prov is not None
    assert prov.privacy_level == PrivacyLevel.REMOTE_PAID
    assert prov.requires_key is True


def test_get_provider_unknown_returns_none():
    assert get_provider("nonexistent") is None


def test_list_providers_filter_by_privacy():
    locals_only = list_providers(PrivacyLevel.LOCAL)
    assert all(p.privacy_level == PrivacyLevel.LOCAL for p in locals_only)
    assert len(locals_only) >= 2

    free_only = list_providers(PrivacyLevel.REMOTE_FREE)
    assert all(p.privacy_level == PrivacyLevel.REMOTE_FREE for p in free_only)


def test_provider_is_frozen_dataclass():
    prov = get_provider("openrouter")
    with pytest.raises(Exception):
        prov.name = "mutated"  # type: ignore[misc]


def test_custom_provider_construction():
    custom = Provider(
        name="fake",
        base_url="https://fake.example/v1",
        privacy_level=PrivacyLevel.REMOTE_FREE,
        default_model="fake/model",
        env_key_name="FAKE_KEY",
    )
    assert custom.requires_key is True
    assert custom.is_remote is True


# ---------------------------------------------------------------------------
# Config integration — tier + provider override
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Scrub Kitsune and provider env vars so tests are deterministic."""
    for key in (
        "KITSUNE_MODEL_TIER",
        "KITSUNE_MODEL_NAME",
        "KITSUNE_BASE_URL",
        "KITSUNE_PROVIDER",
        "KITSUNE_REMOTE_CONSENT",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    yield


def _fresh_settings():
    importlib.reload(config_module)
    return config_module.Settings()


def test_default_settings_are_local():
    s = _fresh_settings()
    assert s.model_tier == "small"
    assert s.provider_name == ""
    assert s.privacy_level == "local"
    assert s.api_key == "not-needed"
    # Backend-agnostic: Mac uses MLX (mlx-community/Qwen2.5-Coder-1.5B...),
    # Linux/Win use Ollama (qwen2.5-coder:1.5b). Match the family + size.
    name = s.model_name.lower()
    assert "qwen2.5" in name and "coder" in name and "1.5b" in name


def test_model_tier_medium_picks_qwen35_4b(monkeypatch):
    monkeypatch.setenv("KITSUNE_MODEL_TIER", "medium")
    s = _fresh_settings()
    assert s.model_tier == "medium"
    name = s.model_name.lower()
    assert "qwen3.5" in name and "4b" in name


def test_model_tier_large_picks_qwen35_9b(monkeypatch):
    monkeypatch.setenv("KITSUNE_MODEL_TIER", "large")
    s = _fresh_settings()
    name = s.model_name.lower()
    assert "qwen3.5" in name and "9b" in name


def test_explicit_model_name_beats_tier(monkeypatch):
    monkeypatch.setenv("KITSUNE_MODEL_TIER", "large")
    monkeypatch.setenv("KITSUNE_MODEL_NAME", "custom/model-xyz")
    s = _fresh_settings()
    assert s.model_name == "custom/model-xyz"


def test_openrouter_provider_loads_with_key(monkeypatch):
    monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    s = _fresh_settings()
    assert s.provider_name == "openrouter"
    assert s.privacy_level == "remote_free"
    assert s.api_key == "sk-or-test"
    assert s.base_url == "https://openrouter.ai/api/v1"
    assert "qwen3-coder" in s.model_name


def test_openrouter_without_key_raises(monkeypatch):
    monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        _fresh_settings()


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("KITSUNE_PROVIDER", "nonexistent")
    with pytest.raises(ValueError, match="Unknown KITSUNE_PROVIDER"):
        _fresh_settings()


def test_explicit_url_and_model_win_over_provider(monkeypatch):
    monkeypatch.setenv("KITSUNE_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("KITSUNE_BASE_URL", "https://my-proxy.local/v1")
    monkeypatch.setenv("KITSUNE_MODEL_NAME", "my-model")
    s = _fresh_settings()
    assert s.base_url == "https://my-proxy.local/v1"
    assert s.model_name == "my-model"
    # Provider metadata still reflects openrouter for consent purposes
    assert s.provider_name == "openrouter"
    assert s.privacy_level == "remote_free"
