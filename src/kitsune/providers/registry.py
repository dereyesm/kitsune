"""Provider registry — declarative presets for every backend Kitsune knows about.

To add a new provider: add an entry to PROVIDERS. No subclasses, no plugins.
All providers MUST expose an OpenAI-compatible /v1/chat/completions endpoint.
"""

from kitsune.providers.base import PrivacyLevel, Provider

#: All known providers keyed by short name. Short name is what users set in
#: the ``KITSUNE_PROVIDER`` env var.
PROVIDERS: dict[str, Provider] = {
    # ---- Local backends (default, privacy-preserving) -----------------------
    "local-mlx": Provider(
        name="local-mlx",
        base_url="http://localhost:8008/v1",
        privacy_level=PrivacyLevel.LOCAL,
        default_model="mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
        env_key_name=None,
        description="mlx_lm.server on macOS (Apple Silicon)",
    ),
    "local-ollama": Provider(
        name="local-ollama",
        base_url="http://localhost:11434/v1",
        privacy_level=PrivacyLevel.LOCAL,
        default_model="qwen2.5-coder:1.5b",
        env_key_name=None,
        description="Ollama server on Linux/Windows/macOS",
    ),
    # ---- Free remote backends (opt-in, requires consent) --------------------
    "openrouter": Provider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        privacy_level=PrivacyLevel.REMOTE_FREE,
        default_model="qwen/qwen3-coder:free",
        env_key_name="OPENROUTER_API_KEY",
        description="OpenRouter — free-tier access to Qwen3-Coder 480B, Nemotron 3 Super, and more",
        free_models=(
            ("qwen/qwen3-coder:free", "Qwen3-Coder 480B A35B (free)"),
            ("qwen/qwen3.5-9b:free", "Qwen3.5 9B (free)"),
            ("nvidia/nemotron-3-super-120b-a12b:free", "NVIDIA Nemotron 3 Super 120B (free)"),
            ("nvidia/nemotron-3-nano-30b-a3b:free", "NVIDIA Nemotron 3 Nano 30B (free)"),
        ),
    ),
    # ---- Paid remote backends (fallback of last resort) ---------------------
    "anthropic": Provider(
        name="anthropic",
        # Anthropic now exposes an OpenAI-compatible endpoint at this path.
        base_url="https://api.anthropic.com/v1",
        privacy_level=PrivacyLevel.REMOTE_PAID,
        default_model="claude-sonnet-4-6",
        env_key_name="ANTHROPIC_API_KEY",
        description="Anthropic Claude — paid, high quality, use as last resort",
    ),
}


def get_provider(name: str) -> Provider | None:
    """Look up a provider by short name. Returns None if not found."""
    return PROVIDERS.get(name)


def list_providers(privacy_level: PrivacyLevel | None = None) -> list[Provider]:
    """List providers, optionally filtered by privacy level."""
    values = list(PROVIDERS.values())
    if privacy_level is not None:
        values = [p for p in values if p.privacy_level == privacy_level]
    return values
