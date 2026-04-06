"""Provider registry for Kitsune — local + free remote + paid backends.

All providers expose OpenAI-compatible /v1/chat/completions endpoints.
No new SDKs, no jerarchy — just declarative config.
"""

from kitsune.providers.base import PrivacyLevel, Provider
from kitsune.providers.registry import PROVIDERS, get_provider, list_providers

__all__ = ["Provider", "PrivacyLevel", "get_provider", "list_providers", "PROVIDERS"]
