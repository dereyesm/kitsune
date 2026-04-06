"""Privacy consent flow for remote providers.

Veto Artemisa: whenever Kitsune would send user code to a non-local endpoint,
the user MUST see a warning AND give explicit consent at least once per
provider. This module handles the warning, the prompt, and the persistence.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

_CONSENT_DIR = Path.home() / ".kitsune"
_CONSENT_FILE = _CONSENT_DIR / "consent.json"


def _load_consent() -> dict:
    if not _CONSENT_FILE.exists():
        return {}
    try:
        return json.loads(_CONSENT_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError, OSError:
        return {}


def _save_consent(data: dict) -> None:
    _CONSENT_DIR.mkdir(parents=True, exist_ok=True)
    _CONSENT_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def has_consent(provider_name: str) -> bool:
    """Return True if the user has already consented to this provider."""
    return provider_name in _load_consent()


def record_consent(provider_name: str, base_url: str) -> None:
    """Persist that the user consented to this provider."""
    data = _load_consent()
    data[provider_name] = {
        "base_url": base_url,
        "consented_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_consent(data)


def build_warning_banner(provider_name: str, base_url: str) -> Panel:
    """Build the Rich banner shown before sending code to a remote provider."""
    body = (
        f"[bold yellow]Provider:[/bold yellow] {provider_name}\n"
        f"[bold yellow]Endpoint:[/bold yellow] {base_url}\n\n"
        "[white]Your code will leave this machine. Only continue if you trust "
        "this endpoint with the code you are about to send.[/white]"
    )
    return Panel(
        body,
        title="[bold red]⚠  REMOTE PROVIDER WARNING[/bold red]",
        border_style="red",
    )


class ConsentDenied(RuntimeError):
    """Raised when the user refuses consent or when no terminal is available."""


def ensure_consent(
    provider_name: str,
    base_url: str,
    privacy_level: str,
    *,
    console: Console | None = None,
    interactive: bool | None = None,
) -> bool:
    """Block until consent is resolved for a remote provider.

    Returns True when the request may proceed, raises :class:`ConsentDenied`
    when the user said no or when no interactive input is available.

    Local providers always return True without showing anything.
    """
    if privacy_level == "local":
        return True

    console = console or Console()

    # Always print the banner on every remote run — even post-consent.
    # Artemisa's rule: the user must never forget where their code is going.
    console.print(build_warning_banner(provider_name, base_url))

    # Already consented in a past session → proceed.
    if has_consent(provider_name):
        console.print(
            f"[dim](consent recorded for '{provider_name}' — "
            f"remove ~/.kitsune/consent.json to reset)[/dim]"
        )
        return True

    # Env-var bypass (CI, scripts, power users).
    if os.environ.get("KITSUNE_REMOTE_CONSENT") == "1":
        record_consent(provider_name, base_url)
        console.print("[green]Consent granted via KITSUNE_REMOTE_CONSENT=1.[/green]")
        return True

    # Interactive prompt — only if stdin is a TTY.
    is_tty = sys.stdin.isatty() if interactive is None else interactive
    if not is_tty:
        raise ConsentDenied(
            f"Remote provider '{provider_name}' requires consent but no "
            "interactive terminal is available. Either run interactively, "
            "set KITSUNE_REMOTE_CONSENT=1, or switch to a local provider."
        )

    try:
        answer = input("Send code to this remote provider? [y/N]: ").strip().lower()
    except EOFError:
        raise ConsentDenied("No input available to confirm remote consent.") from None

    if answer in ("y", "yes"):
        record_consent(provider_name, base_url)
        console.print(
            f"[green]Consent recorded for '{provider_name}'. "
            f"Future runs will skip the prompt (banner still shown).[/green]"
        )
        return True

    raise ConsentDenied(f"User denied consent for remote provider '{provider_name}'.")
