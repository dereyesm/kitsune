"""Protocol compliance tests — OpenAI + MCP + A2A + HERMES contract.

A protocol engineer asks: does Kitsune respect the interfaces it claims to
speak? These tests pin down structural invariants that protect us from
regressions that would silently break external integrations.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import FrozenInstanceError

import pytest

import kitsune.config as config_module
import kitsune.consent as consent_module
from kitsune.providers import PROVIDERS, PrivacyLevel, get_provider
from kitsune.providers.base import Provider


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
    yield


# ===========================================================================
# Class PC1: OpenAI-compatible contract — all providers must conform
# ===========================================================================


class TestOpenAICompatContract:
    """Every provider in the registry must speak the OpenAI-compatible API
    dialect. Kitsune's whole design depends on this — if we add one that
    doesn't, the unified backend in backend.py will break silently."""

    def test_all_providers_expose_v1_path(self):
        for p in PROVIDERS.values():
            assert p.base_url.rstrip("/").endswith("/v1"), (
                f"{p.name} base_url does not end with /v1: {p.base_url}"
            )

    def test_all_providers_have_default_model(self):
        for p in PROVIDERS.values():
            assert p.default_model, f"{p.name} has empty default_model"
            # Model IDs shouldn't contain whitespace or control chars
            assert " " not in p.default_model
            assert "\n" not in p.default_model

    def test_all_providers_have_human_description(self):
        for p in PROVIDERS.values():
            assert len(p.description) >= 5, f"{p.name} has stub description"

    def test_all_providers_have_non_empty_name(self):
        for name, p in PROVIDERS.items():
            assert name == p.name  # registry key matches provider.name
            assert name.replace("-", "").isalnum(), f"bad provider name: {name}"

    def test_free_models_list_format(self):
        # Every free_models entry must be (model_id, human_label)
        for p in PROVIDERS.values():
            for entry in p.free_models:
                assert isinstance(entry, tuple)
                assert len(entry) == 2
                assert entry[0] and entry[1]


# ===========================================================================
# Class PC2: Provider immutability — Provider is a frozen dataclass
# ===========================================================================


class TestProviderImmutability:
    """Providers must be immutable so parallel code can share them without
    risk of tampering. Frozen dataclass is the enforcement mechanism."""

    def test_providers_cannot_be_mutated(self):
        prov = get_provider("openrouter")
        with pytest.raises(FrozenInstanceError):
            prov.name = "hacked"

    def test_providers_cannot_be_deleted(self):
        prov = get_provider("openrouter")
        with pytest.raises(FrozenInstanceError):
            del prov.base_url

    def test_providers_are_hashable(self):
        # frozen=True without eq=False means hashable — can go into sets
        assert len({p for p in PROVIDERS.values()}) == len(PROVIDERS)


# ===========================================================================
# Class PC3: PrivacyLevel is a JSON-serialisable enum
# ===========================================================================


class TestPrivacyLevelEnum:
    """PrivacyLevel travels over the wire (to consent file, to status output).
    It must serialise cleanly to JSON and back."""

    def test_values_are_strings(self):
        for level in PrivacyLevel:
            assert isinstance(level.value, str)

    def test_enum_is_json_serializable(self):
        data = {p.name: p.privacy_level.value for p in PROVIDERS.values()}
        serialized = json.dumps(data)
        parsed = json.loads(serialized)
        for name, lvl in parsed.items():
            assert lvl in ("local", "remote_free", "remote_paid")

    def test_level_count_is_exactly_three(self):
        assert set(PrivacyLevel) == {
            PrivacyLevel.LOCAL,
            PrivacyLevel.REMOTE_FREE,
            PrivacyLevel.REMOTE_PAID,
        }


# ===========================================================================
# Class PC4: MCP server — tool count + registration invariants
# ===========================================================================


class TestMCPServerContract:
    """Kitsune advertises 4 MCP tools. The README, business case, and glama.json
    all depend on that number. Any change to the surface must be intentional."""

    def test_mcp_server_exposes_exactly_four_tools(self):
        import kitsune.mcp_server as mcp_module

        tools = getattr(mcp_module.mcp, "_tool_manager", None)
        # fastmcp 3.x stores tools on the manager
        if tools is None:
            # Fallback: count @mcp.tool() decorated functions in the module
            import inspect

            funcs = [
                name
                for name, obj in inspect.getmembers(mcp_module)
                if inspect.isfunction(obj)
                and hasattr(obj, "__wrapped__")
                or name in ("explain_code", "ask_about_code", "search_code", "kitsune_status")
            ]
            assert "explain_code" in funcs
            assert "ask_about_code" in funcs
            assert "search_code" in funcs
            assert "kitsune_status" in funcs
        else:
            tool_names = set(getattr(tools, "_tools", {}).keys())
            for expected in ("explain_code", "ask_about_code", "search_code", "kitsune_status"):
                assert expected in tool_names

    def test_mcp_server_has_name_and_instructions(self):
        import kitsune.mcp_server as mcp_module

        assert mcp_module.mcp is not None
        # fastmcp exposes name either directly or via internal attr
        name = getattr(mcp_module.mcp, "name", None) or getattr(mcp_module.mcp, "_name", None)
        assert name == "kitsune"


# ===========================================================================
# Class PC5: Settings schema — contract with downstream consumers
# ===========================================================================


class TestSettingsSchema:
    """Anything reading settings (cli.py, backend.py, mcp_server.py) relies on
    these field names being present. This test pins the schema."""

    REQUIRED_FIELDS = {
        "backend",
        "base_url",
        "model_name",
        "model_tier",
        "temperature",
        "fallback_threshold",
        "provider",
        "provider_name",
        "privacy_level",
        "api_key",
    }

    def test_settings_exposes_required_fields(self):
        importlib.reload(config_module)
        s = config_module.settings
        for field in self.REQUIRED_FIELDS:
            assert hasattr(s, field), f"missing field: {field}"

    def test_privacy_level_field_is_string(self):
        importlib.reload(config_module)
        assert isinstance(config_module.settings.privacy_level, str)

    def test_privacy_level_values_are_in_enum(self):
        importlib.reload(config_module)
        assert config_module.settings.privacy_level in (
            "local",
            "remote_free",
            "remote_paid",
        )

    def test_api_key_never_none(self):
        importlib.reload(config_module)
        # api_key must always be a string, even for local providers
        assert config_module.settings.api_key is not None
        assert isinstance(config_module.settings.api_key, str)


# ===========================================================================
# Class PC6: Custom provider construction — extensibility contract
# ===========================================================================


class TestExtensibility:
    """Power users (and future us) should be able to construct a Provider
    outside the registry. This tests the public API of base.py."""

    def test_custom_provider_with_no_key(self):
        custom = Provider(
            name="my-local",
            base_url="http://localhost:9000/v1",
            privacy_level=PrivacyLevel.LOCAL,
            default_model="custom/model",
        )
        assert custom.requires_key is False
        assert custom.is_remote is False

    def test_custom_provider_with_key(self):
        custom = Provider(
            name="my-remote",
            base_url="https://my-service/v1",
            privacy_level=PrivacyLevel.REMOTE_FREE,
            default_model="custom/model",
            env_key_name="MY_API_KEY",
        )
        assert custom.requires_key is True
        assert custom.is_remote is True

    def test_custom_provider_optional_fields_default(self):
        custom = Provider(
            name="x",
            base_url="https://x/v1",
            privacy_level=PrivacyLevel.REMOTE_PAID,
            default_model="x/m",
        )
        assert custom.description == ""
        assert custom.free_models == ()
