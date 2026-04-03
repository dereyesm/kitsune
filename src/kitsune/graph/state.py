"""LangGraph state schema for Kitsune."""

from typing import Literal, TypedDict


class KitsuneState(TypedDict):
    user_input: str
    task_type: Literal["explain", "ask", "fallback"]
    code_context: str
    file_path: str
    response: str
    escalation_reason: str
