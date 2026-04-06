"""Tests for the LangGraph graph structure."""

from unittest.mock import patch

from kitsune.graph.build import build_graph
from kitsune.graph.state import KitsuneState


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None


@patch("kitsune.graph.nodes.invoke", return_value="mocked explanation")
def test_explain_flow(mock_invoke):
    graph = build_graph()
    state: KitsuneState = {
        "user_input": "",
        "task_type": "explain",
        "code_context": "def hello(): pass",
        "file_path": "test.py",
        "response": "",
        "escalation_reason": "",
    }
    result = graph.invoke(state)
    assert result["response"] == "mocked explanation"
    assert result["task_type"] == "explain"
    mock_invoke.assert_called_once()


@patch("kitsune.graph.nodes.invoke", return_value="mocked answer")
def test_ask_flow(mock_invoke):
    graph = build_graph()
    state: KitsuneState = {
        "user_input": "is this fast?",
        "task_type": "ask",
        "code_context": "for i in range(n): pass",
        "file_path": "test.py",
        "response": "",
        "escalation_reason": "",
    }
    result = graph.invoke(state)
    assert result["response"] == "mocked answer"
    assert result["task_type"] == "ask"


def test_fallback_flow_security():
    graph = build_graph()
    state: KitsuneState = {
        "user_input": "check for SQL injection vulnerabilities",
        "task_type": "ask",
        "code_context": "query = f'SELECT * FROM users WHERE id={user_id}'",
        "file_path": "db.py",
        "response": "",
        "escalation_reason": "",
    }
    result = graph.invoke(state)
    assert "security" in result["escalation_reason"]
    assert result["task_type"] == "fallback"
    # The new multi-tier escalation message must advertise every escalation path
    body = result["response"]
    assert "Local tier up" in body
    assert "Free remote" in body
    assert "openrouter" in body
    assert "anthropic" in body
