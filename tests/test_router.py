"""Tests for the multi-gate router."""

from kitsune.graph.router import route
from kitsune.graph.state import KitsuneState


def _make_state(**overrides) -> KitsuneState:
    base: KitsuneState = {
        "user_input": "",
        "task_type": "ask",
        "code_context": "",
        "file_path": "",
        "response": "",
        "escalation_reason": "",
    }
    return {**base, **overrides}


# --- Task classification ---


def test_explain_keyword():
    state = _make_state(user_input="explain this code", code_context="x = 1")
    result = route(state)
    assert result["task_type"] == "explain"


def test_what_does_keyword():
    state = _make_state(user_input="what does this function do?", code_context="def f(): pass")
    result = route(state)
    assert result["task_type"] == "explain"


def test_empty_input_with_code_defaults_to_explain():
    state = _make_state(user_input="", code_context="print('hello')")
    result = route(state)
    assert result["task_type"] == "explain"


def test_generic_question_defaults_to_ask():
    state = _make_state(user_input="is this efficient?")
    result = route(state)
    assert result["task_type"] == "ask"


# --- Escalation gates ---


def test_security_always_escalates():
    state = _make_state(user_input="check for XSS vulnerabilities", code_context="x = 1")
    result = route(state)
    assert result["task_type"] == "fallback"
    assert "security" in result["escalation_reason"]


def test_architecture_always_escalates():
    state = _make_state(user_input="design a microservice for this")
    result = route(state)
    assert result["task_type"] == "fallback"
    assert "architecture" in result["escalation_reason"]


def test_refactor_escalates():
    state = _make_state(user_input="refactor this module to use dependency injection")
    result = route(state)
    assert result["task_type"] == "fallback"
    assert "refactoring" in result["escalation_reason"]


def test_token_gate_escalates():
    long_code = "x = 1\n" * 3000  # ~6000 words -> ~8000 tokens
    state = _make_state(user_input="what is this?", code_context=long_code)
    result = route(state)
    assert result["task_type"] == "fallback"
    assert "too large" in result["escalation_reason"]


def test_explain_stays_local():
    state = _make_state(user_input="explain this", code_context="def hello(): print('hi')")
    result = route(state)
    assert result["task_type"] == "explain"
    assert result["escalation_reason"] == ""


def test_short_ask_stays_local():
    state = _make_state(user_input="is this efficient?", code_context="x = lambda: 42")
    result = route(state)
    assert result["task_type"] == "ask"
    assert result["escalation_reason"] == ""
