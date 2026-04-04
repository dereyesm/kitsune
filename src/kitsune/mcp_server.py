"""Kitsune MCP Server — expose local code assistant as Claude Code tool.

Usage:
    # Add to Claude Code:
    claude mcp add kitsune -- uv run --directory ~/Dev/kitsune python -m kitsune.mcp_server

    # Or in .claude/settings.local.json:
    {
      "mcpServers": {
        "kitsune": {
          "command": "uv",
          "args": ["run", "--directory", "~/Dev/kitsune", "python", "-m", "kitsune.mcp_server"]
        }
      }
    }
"""

from pathlib import Path

from fastmcp import FastMCP

from kitsune.config import settings
from kitsune.graph.build import app as graph_app
from kitsune.graph.state import KitsuneState
from kitsune.rag.bm25_backend import BM25Backend

mcp = FastMCP(
    "kitsune",
    instructions="Local code assistant — explain, ask, search code using SLMs",
)

_rag = BM25Backend()


@mcp.tool()
def explain_code(file_path: str) -> str:
    """Explain what a code file does. Uses a local 1.5B model — fast, free, no API cost.
    Best for: understanding single files, identifying patterns, quick orientation.
    NOT for: security audits, architecture design, multi-file analysis."""
    p = Path(file_path).expanduser().resolve()
    if not p.is_file():
        return f"Error: file not found: {p}"

    code = p.read_text(encoding="utf-8", errors="replace")
    state: KitsuneState = {
        "user_input": "",
        "task_type": "explain",
        "code_context": code,
        "file_path": str(p),
        "response": "",
        "escalation_reason": "",
    }
    result = graph_app.invoke(state)
    return result["response"]


@mcp.tool()
def ask_about_code(question: str, file_path: str = "") -> str:
    """Ask a question about code. Optionally provide a file for context.
    Uses a local 1.5B model — fast, free, no API cost.
    Best for: specific questions, function behavior, pattern identification.
    NOT for: security, architecture, refactoring suggestions."""
    code = ""
    fpath = ""
    if file_path:
        p = Path(file_path).expanduser().resolve()
        if p.is_file():
            code = p.read_text(encoding="utf-8", errors="replace")
            fpath = str(p)

    state: KitsuneState = {
        "user_input": question,
        "task_type": "ask",
        "code_context": code,
        "file_path": fpath,
        "response": "",
        "escalation_reason": "",
    }
    result = graph_app.invoke(state)
    return result["response"]


@mcp.tool()
def search_code(query: str, directory: str = ".", top_k: int = 5) -> str:
    """Search a codebase for relevant code chunks using BM25 keyword search.
    Returns ranked code snippets matching the query.
    Fast (0ms search), no embeddings needed."""
    _rag.index(directory)
    results = _rag.search(query, top_k=top_k)
    if not results:
        return "No results found."

    output = []
    for i, chunk in enumerate(results, 1):
        output.append(
            f"[{i}] {chunk.file_path}:{chunk.start_line}-{chunk.end_line} "
            f"(score: {chunk.score})\n{chunk.content}"
        )
    return "\n\n---\n\n".join(output)


@mcp.tool()
def kitsune_status() -> str:
    """Check Kitsune's status — model, backend, server availability."""
    import platform

    import httpx

    lines = [
        f"Backend: {settings.backend} ({platform.system()})",
        f"Model: {settings.model_name}",
        f"Server: {settings.base_url}",
    ]
    try:
        r = httpx.get(f"{settings.base_url}/models", timeout=3)
        models = [m["id"] for m in r.json().get("data", [])]
        lines.append(f"Available models: {', '.join(models)}")
    except Exception:
        lines.append("Server: NOT RUNNING")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
