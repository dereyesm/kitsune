"""Provider dataclass — declarative config for OpenAI-compatible backends."""

from dataclasses import dataclass, field
from enum import Enum


class PrivacyLevel(str, Enum):
    """Where does the user's code go?"""

    LOCAL = "local"  # Code stays on the machine
    REMOTE_FREE = "remote_free"  # Code sent to a free-tier remote endpoint
    REMOTE_PAID = "remote_paid"  # Code sent to a paid remote endpoint


@dataclass(frozen=True)
class Provider:
    """Declarative provider config.

    Every field is data — no methods, no magic. Load a provider, override
    settings.base_url / settings.model_name, done.
    """

    name: str
    base_url: str
    privacy_level: PrivacyLevel
    default_model: str
    # If set, backend.py reads the API key from this env var.
    # If None, no key is required (local backends).
    env_key_name: str | None = None
    # Human-readable note for status / CLI.
    description: str = ""
    # Known free models at this provider (for status + router suggestions).
    # List of (model_id, human_label) tuples.
    free_models: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def requires_key(self) -> bool:
        return self.env_key_name is not None

    @property
    def is_remote(self) -> bool:
        return self.privacy_level != PrivacyLevel.LOCAL
