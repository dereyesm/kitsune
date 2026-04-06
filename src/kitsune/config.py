"""Configuration via environment variables and pydantic-settings."""

import os
import platform

from pydantic_settings import BaseSettings

_SYSTEM = platform.system()

# Model tiers — progressive quality/RAM tradeoff, all local.
# Tier is selected via KITSUNE_MODEL_TIER=small|medium|large.
# Keys: (backend, tier) → model identifier string.
_MODEL_TIERS: dict[tuple[str, str], str] = {
    # small — default, fast, ~1.4 GB RAM
    ("mlx", "small"): "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    ("ollama", "small"): "qwen2.5-coder:1.5b",
    # medium — balanced, ~5 GB RAM, Qwen3.5 (Apache 2.0, Feb-Mar 2026)
    ("mlx", "medium"): "mlx-community/Qwen3.5-4B-Instruct-4bit",
    ("ollama", "medium"): "qwen3.5:4b",
    # large — quality, ~10 GB RAM, Qwen3.5
    ("mlx", "large"): "mlx-community/Qwen3.5-9B-Instruct-4bit",
    ("ollama", "large"): "qwen3.5:9b",
}


def _default_backend() -> str:
    return "mlx" if _SYSTEM == "Darwin" else "ollama"


def _default_base_url() -> str:
    if _SYSTEM == "Darwin":
        return "http://localhost:8008/v1"  # mlx_lm.server
    return "http://localhost:11434/v1"  # ollama


def _default_model_tier() -> str:
    return "small"


def _default_model() -> str:
    backend = _default_backend()
    tier = _default_model_tier()
    return _MODEL_TIERS.get((backend, tier), _MODEL_TIERS[(backend, "small")])


def resolve_model(backend: str, tier: str) -> str:
    """Resolve a (backend, tier) pair to a concrete model identifier.

    Falls back to small tier if the requested tier is unknown.
    """
    return _MODEL_TIERS.get((backend, tier), _MODEL_TIERS.get((backend, "small"), ""))


class Settings(BaseSettings):
    model_config = {"env_prefix": "KITSUNE_"}

    backend: str = _default_backend()
    base_url: str = _default_base_url()
    model_name: str = _default_model()
    model_tier: str = _default_model_tier()
    temperature: float = 0.1
    fallback_threshold: int = 2000
    anthropic_api_key: str | None = None

    # Provider override. If set, loads from the providers registry and
    # overrides base_url + model_name (unless those are set explicitly).
    provider: str | None = None

    # Active provider metadata — populated in __init__ after resolution.
    # Empty string when running in local mode.
    provider_name: str = ""
    privacy_level: str = "local"
    api_key: str = "not-needed"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Step 1: apply tier → model resolution when user asked for a tier
        # but did NOT pin an explicit model name.
        if "KITSUNE_MODEL_TIER" in os.environ and "KITSUNE_MODEL_NAME" not in os.environ:
            resolved = resolve_model(self.backend, self.model_tier)
            if resolved:
                self.model_name = resolved

        # Step 2: apply provider override. Provider settings win unless the
        # user explicitly pinned KITSUNE_BASE_URL / KITSUNE_MODEL_NAME.
        provider_name = os.environ.get("KITSUNE_PROVIDER") or self.provider
        if provider_name:
            # Lazy import to avoid circular dependency at module import time.
            from kitsune.providers import get_provider

            prov = get_provider(provider_name)
            if prov is None:
                known = ", ".join(
                    sorted(
                        [
                            "local-mlx",
                            "local-ollama",
                            "openrouter",
                            "anthropic",
                        ]
                    )
                )
                raise ValueError(
                    f"Unknown KITSUNE_PROVIDER='{provider_name}'. Known providers: {known}"
                )

            if "KITSUNE_BASE_URL" not in os.environ:
                self.base_url = prov.base_url
            if "KITSUNE_MODEL_NAME" not in os.environ:
                self.model_name = prov.default_model

            self.provider_name = prov.name
            self.privacy_level = prov.privacy_level.value

            # Resolve API key from env. Local providers don't need one.
            if prov.env_key_name:
                key = os.environ.get(prov.env_key_name)
                if not key:
                    raise ValueError(
                        f"Provider '{prov.name}' requires env var "
                        f"{prov.env_key_name} but it is not set."
                    )
                self.api_key = key
            else:
                self.api_key = "not-needed"


settings = Settings()
