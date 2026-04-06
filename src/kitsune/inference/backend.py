"""Unified inference backend via OpenAI-compatible API.

Works with any server that exposes /v1/chat/completions:
- macOS: mlx_lm.server (MLX, Apple Silicon native)
- Linux/Windows: ollama serve (with OpenAI compat layer)
- Any: llama.cpp server, vLLM, LocalAI, etc.

Remote providers (OpenRouter, NVIDIA NIM, Anthropic) are rate-limited. When a
429 arrives we back off briefly then surface a clear escalation suggestion to
the caller — never fail silently.
"""

from __future__ import annotations

import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from kitsune.config import settings

#: Max retry attempts on 429. Kept small so the user is not left hanging.
_MAX_RETRIES = 3

#: Base sleep in seconds for exponential backoff (1s, 2s, 4s).
_BASE_BACKOFF = 1.0


class RateLimitExceeded(RuntimeError):
    """Raised when the provider returns 429 after the retry budget is spent."""


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.model_name,
        temperature=settings.temperature,
        max_tokens=500,
    )


def _is_rate_limit(exc: Exception) -> bool:
    """Return True if the exception looks like a provider rate limit.

    We duck-type instead of importing ``openai.RateLimitError`` directly to
    keep the backend decoupled from any specific client library.
    """
    name = type(exc).__name__.lower()
    if "ratelimit" in name:
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status == 429:
        return True
    msg = str(exc).lower()
    return "rate limit" in msg or "429" in msg


def invoke(system_prompt: str, user_content: str) -> str:
    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    response = None
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = llm.invoke(messages)
            last_exc = None
            break
        except Exception as exc:  # noqa: BLE001 — we inspect then re-raise
            if not _is_rate_limit(exc):
                raise
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BASE_BACKOFF * (2**attempt))

    if response is None:
        provider = settings.provider_name or "remote"
        raise RateLimitExceeded(
            f"Provider '{provider}' returned 429 after {_MAX_RETRIES} attempts. "
            "Either wait, switch to a local tier "
            "(`unset KITSUNE_PROVIDER`), or escalate to a paid provider."
        ) from last_exc

    text = response.content
    # Strip EOS tokens that some servers leak
    for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        text = text.replace(token, "")
    return text.strip()
