"""Task-specific LangGraph nodes."""

from kitsune.graph.router import suggest_tiers
from kitsune.graph.state import KitsuneState
from kitsune.inference.backend import invoke
from kitsune.prompts.loader import build_system_prompt
from kitsune.prompts.templates import FALLBACK_MSG


def explain_node(state: KitsuneState) -> KitsuneState:
    system_prompt = build_system_prompt("explain", state.get("file_path", ""))
    code = state["code_context"]
    user_msg = state["user_input"]
    prompt = f"{user_msg}\n\n```\n{code}\n```" if user_msg.strip() else f"```\n{code}\n```"
    response = invoke(system_prompt, prompt)
    return {**state, "response": response}


def ask_node(state: KitsuneState) -> KitsuneState:
    system_prompt = build_system_prompt("ask", state.get("file_path", ""))
    code = state["code_context"]
    question = state["user_input"]
    prompt = f"Question: {question}"
    if code:
        prompt += f"\n\nCode context:\n```\n{code}\n```"
    response = invoke(system_prompt, prompt)
    return {**state, "response": response}


def fallback_node(state: KitsuneState) -> KitsuneState:
    response = FALLBACK_MSG.format(
        reason=state.get("escalation_reason", "unknown"),
        tiers=suggest_tiers(),
        prompt=state["user_input"],
        file_path=state.get("file_path", "<stdin>"),
    )
    return {**state, "response": response}
