# Kitsune — Local AI Gateway

## Overview
CLI + MCP server + HERMES node. Works with any OpenAI-compatible backend — SLMs (Qwen 1.5B via MLX/Ollama) or cloud LLMs (OpenAI, Anthropic, etc.).
Repo: https://github.com/tinkuy-dev/kitsune | MIT License

## Stack
- **Inference**: MLX (Mac) / Ollama (Linux/Win) via OpenAI-compatible API
- **Orchestration**: LangGraph state graph
- **CLI**: Typer + Rich
- **RAG**: BM25 (default) + ChromaDB (optional)
- **MCP**: fastmcp server (registered globally in ~/.claude.json)

## Commands
```bash
# Start MLX server first
mlx_lm.server --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit --port 8008

# CLI
uv run kit explain <file>
uv run kit ask "<question>" -f <file>
uv run kit search "<query>" -d <directory>
uv run kit status

# Tests
uv run pytest tests/ -v                          # unit (25) + e2e (12)
uv run pytest tests/ --ignore=tests/e2e -v       # unit only (no server needed)
uv run python tests/benchmark/arena_eval.py       # quality benchmark
uv run python tests/benchmark/rag_compare.py      # RAG comparison

# Lint
uv run ruff check . && uv run ruff format --check .
```

## Architecture
```
src/kitsune/
  cli.py              — Typer entry point (kit command)
  config.py           — Auto-detect OS, set backend defaults
  __main__.py          — python -m kitsune support
  mcp_server.py       — MCP server (4 tools for Claude Code)
  hermes_node.py      — HERMES bus watcher daemon
  graph/
    state.py           — KitsuneState TypedDict
    router.py          — Multi-gate escalation router
    nodes.py           — explain, ask, fallback LangGraph nodes
    build.py           — Graph compiler
  inference/
    backend.py         — Unified OpenAI-compatible backend
  prompts/
    loader.py          — Skill-based prompt composer
    templates.py       — Fallback message template
    skills/            — 10 language skill files (.md)
  rag/
    base.py            — RAGBackend ABC + chunking utils
    bm25_backend.py    — Zero-dep keyword search
    chroma_backend.py  — Vector search (optional dep group)
```

## Key Decisions
- MLX > Ollama on macOS Tahoe (Ollama Metal shaders broken)
- 1.5B > 7B as default (same quality, 4.4x faster)
- BM25 > ChromaDB for code search (same precision, 160x faster)
- Skill enrichment > model size as quality lever

## Conventions
- Commit format: `type(kitsune): message` (see ~/.claude/CONVENTIONS.md)
- Ruff: line-length 100, py314, select E/F/I/W
- Tests: pytest + pytest-asyncio, e2e needs running server
- Deps: uv sync, optional groups: rag, mcp, fallback

## Dimension
Dev (cross-dimensional infra, like heraldo-gateway)
