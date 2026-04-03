"""Tests for the skill-based prompt loader."""

from kitsune.prompts.loader import build_system_prompt, detect_language


def test_detect_python():
    assert detect_language("app.py") == "python"


def test_detect_javascript():
    assert detect_language("index.js") == "javascript"
    assert detect_language("App.tsx") == "javascript"


def test_detect_go():
    assert detect_language("main.go") == "go"


def test_detect_rust():
    assert detect_language("lib.rs") == "rust"


def test_detect_unknown():
    assert detect_language("data.csv") is None


def test_detect_stdin():
    assert detect_language("<stdin>") is None


def test_detect_empty():
    assert detect_language("") is None


def test_build_prompt_with_python():
    prompt = build_system_prompt("explain", "app.py")
    assert "Kitsune" in prompt
    assert "Python" in prompt or "decorator" in prompt.lower()
    assert "Explain" in prompt or "explain" in prompt


def test_build_prompt_unknown_language():
    prompt = build_system_prompt("explain", "data.csv")
    assert "Kitsune" in prompt
    # Should NOT contain language-specific content
    assert "decorator" not in prompt.lower()


def test_build_prompt_ask_task():
    prompt = build_system_prompt("ask", "main.go")
    assert "Answer" in prompt or "answer" in prompt
    assert "goroutine" in prompt.lower() or "Go" in prompt
