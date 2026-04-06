# Kitsune

Local AI gateway powered by [LangGraph](https://langchain-ai.github.io/langgraph/). Works with any model — from 1.5B SLMs on your laptop to cloud LLMs via API.

Part of the [Tinkuy](https://github.com/tinkuy-dev/tinkuy) ecosystem.

## What It Does

Kitsune explains, searches, and answers questions about your code. It runs three ways:

| Mode | How | Best for |
|------|-----|----------|
| **CLI** | `kit explain file.py` | Quick code understanding |
| **MCP Server** | Tools inside Claude Code, Cursor, etc. | AI-assisted workflows |
| **HERMES Node** | Daemon watching a message bus | Multi-agent coordination |

## Quick Start (10 minutes)

### 1. Clone and install

```bash
git clone https://github.com/tinkuy-dev/kitsune.git
cd kitsune
uv sync
```

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

### 2. Start a model server

Pick ONE — any OpenAI-compatible server works:

**macOS (Apple Silicon) — MLX:**
```bash
uv tool install mlx-lm
mlx_lm.server --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit --port 8008
```

**Linux / Windows — Ollama:**
```bash
# Install from https://ollama.com/download
ollama pull qwen2.5-coder:1.5b
ollama serve
```

**Any platform — cloud LLM:**
```bash
# Point to any OpenAI-compatible API
export KITSUNE_BASE_URL="https://api.openai.com/v1"
export KITSUNE_MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="sk-..."
```

### 3. Try it

```bash
# Explain a file
uv run kit explain path/to/file.py

# Ask a question with context
uv run kit ask "what does this function do?" -f path/to/file.py

# Search your codebase
uv run kit search "authentication" -d ./src

# Index a project for faster search
uv run kit index ./my-project

# Check status
uv run kit status
```

That's it. Your code never leaves your machine (unless you point to a cloud API).

## How It Works

```
Input --> [Router] --> explain  --> [Model] --> Rich output
      (multi-gate)  --> ask      --> [Model] --> Rich output
                    --> search   --> [BM25]  --> Ranked results
                    --> escalate --> "Use Claude for this" + why
```

- **Router**: 4 deterministic gates (security, architecture, complexity, token budget). Zero latency — no LLM call to decide.
- **Inference**: any OpenAI-compatible server — SLMs (MLX, Ollama) or LLMs (OpenAI, Anthropic, etc.)
- **Skills**: language-specific prompt injection for 10 languages
- **RAG**: BM25 keyword search (zero dependencies, 160x faster than vector search for code)
- **Escalation**: when a task exceeds the local model's capability, Kitsune tells you WHY and suggests a prompt for Claude

## Supported Languages

| Language | Skill file | Knowledge |
|----------|-----------|-----------|
| Python | `python.md` | Async, type hints, FastAPI/Django, decorators |
| JavaScript | `javascript.md` | Promises, closures, React/Node patterns |
| TypeScript | `typescript.md` | Generics, utility types, strict mode |
| Go | `go.md` | Goroutines, channels, error handling, interfaces |
| Rust | `rust.md` | Ownership, borrowing, traits, async tokio |
| Java | `java.md` | Streams, generics, Spring patterns |
| C# | `csharp.md` | LINQ, async/await, .NET patterns |
| Ruby | `ruby.md` | Blocks, metaprogramming, Rails conventions |
| PHP | `php.md` | Laravel, type declarations, modern PHP |
| Swift | `swift.md` | Optionals, protocols, SwiftUI patterns |

Add your own: create a `.md` file in `src/kitsune/prompts/skills/` and map the extension in `loader.py`.

## Platform Support

| OS | Backend | Model Server |
|----|---------|-------------|
| **macOS** (Apple Silicon) | MLX | `mlx_lm.server` |
| **Linux** | Ollama | `ollama serve` |
| **Windows** | Ollama | `ollama serve` |
| **Any** | Cloud API | OpenAI/Anthropic/etc. |

Kitsune auto-detects your OS and configures the right backend.

## Local Model Tiers

Pick a tier based on your hardware. All tiers run **100% local** — your code never leaves your machine.

| Tier | Model | Size | RAM | Best for |
|------|-------|------|-----|----------|
| `small` (default) | Qwen2.5-Coder-1.5B | 1.5B | ~1.4 GB | Fast explanations, low-end hardware |
| `medium` | **Qwen3.5-4B** | 4B | ~5 GB | Balanced quality/speed, most laptops |
| `large` | **Qwen3.5-9B** | 9B | ~10 GB | Best local quality, workstations |

Switch tiers with one env var:

```bash
# macOS (MLX)
export KITSUNE_MODEL_TIER=medium
mlx_lm.server --model mlx-community/Qwen3.5-4B-Instruct-4bit --port 8008

# Linux / Windows (Ollama)
export KITSUNE_MODEL_TIER=medium
ollama pull qwen3.5:4b
ollama serve
```

Qwen3.5 is Apache 2.0 (commercial use OK, no attribution required).

## Configuration

All settings via environment variables (prefix `KITSUNE_`):

| Variable | Default (macOS) | Default (Linux/Win) | Description |
|----------|----------------|-------------------|-------------|
| `KITSUNE_BACKEND` | `mlx` | `ollama` | Backend type |
| `KITSUNE_BASE_URL` | `http://localhost:8008/v1` | `http://localhost:11434/v1` | Server URL |
| `KITSUNE_MODEL_TIER` | `small` | `small` | Local model tier: `small`, `medium`, `large` |
| `KITSUNE_MODEL_NAME` | auto-resolved | auto-resolved | Override model identifier (wins over tier) |
| `KITSUNE_TEMPERATURE` | `0.1` | `0.1` | Generation temperature |

## MCP Server Mode

Register Kitsune as an MCP tool provider in Claude Code or any MCP-compatible IDE:

```bash
claude mcp add kitsune -- uv run --directory /path/to/kitsune python -m kitsune.mcp_server
```

This exposes 4 tools: `explain_code`, `ask_about_code`, `search_code`, `kitsune_status`.

## Using Kitsune in VS Code

Kitsune is an OpenAI-compatible local gateway + MCP server, so it plugs into any VS Code AI extension that speaks either protocol. No new extension needed.

### Option A — Continue.dev + MCP (recommended)

[Continue.dev](https://continue.dev) is an open-source AI coding assistant for VS Code that supports MCP. This is the path most aligned with the Tinkuy ecosystem (local-first, open source, democratizing access to AI).

1. Install the [Continue.dev extension](https://marketplace.visualstudio.com/items?itemName=Continue.continue) from the VS Code marketplace.
2. Add Kitsune as an MCP server in `~/.continue/config.yaml`:

   ```yaml
   mcpServers:
     - name: kitsune
       command: uv
       args:
         - run
         - --directory
         - /path/to/kitsune
         - python
         - -m
         - kitsune.mcp_server
   ```

3. Start your local model server (`mlx_lm.server` on macOS, `ollama serve` on Linux/Windows).
4. Open the Continue chat panel in VS Code. Kitsune's 4 tools (`explain_code`, `ask_about_code`, `search_code`, `kitsune_status`) now appear as callable tools in the chat.

Ask anything like "explain what this file does" and Continue will invoke Kitsune's local model — your code never leaves the machine.

### Option B — Continue.dev pointing directly at the model server (no Kitsune routing)

If you want the raw local model without Kitsune's router / skills / RAG, configure Continue.dev to talk directly to your OpenAI-compatible server:

```yaml
# ~/.continue/config.yaml
models:
  - name: Qwen local (MLX)
    provider: openai
    model: mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
    apiBase: http://localhost:8008/v1
    apiKey: not-needed
```

**Tradeoff**: simpler setup, but you lose Kitsune's multi-gate router, language skills, BM25 RAG, and escalation logic. Use this only if you want a raw chat UI.

### Option C — VS Code tasks.json (no extension at all)

Create `.vscode/tasks.json` to run Kitsune on the current file from the Command Palette:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Kitsune: Explain current file",
      "type": "shell",
      "command": "uv run kit explain ${file}",
      "options": { "cwd": "/path/to/kitsune" },
      "presentation": { "panel": "dedicated", "reveal": "always" }
    },
    {
      "label": "Kitsune: Ask about current file",
      "type": "shell",
      "command": "uv run kit ask \"${input:question}\" -f ${file}",
      "options": { "cwd": "/path/to/kitsune" },
      "presentation": { "panel": "dedicated", "reveal": "always" }
    }
  ],
  "inputs": [
    {
      "id": "question",
      "type": "promptString",
      "description": "Question about the code"
    }
  ]
}
```

Run with `Cmd+Shift+P` → `Tasks: Run Task` → pick a Kitsune task. Zero dependencies, works today, no chat UI — output lands in the integrated terminal.

### Other MCP-compatible VS Code clients

The same MCP server also works with:
- **Cline** — Claude agent extension for VS Code
- **GitHub Copilot Chat** (agent mode, MCP support)
- **Cursor**, **Windsurf**, **Zed** — all support MCP servers

Registration details differ per client; the Kitsune MCP command is the same as shown for Continue.dev above.

## Development

```bash
uv sync
uv run pytest tests/ -v                    # 36 tests (25 unit + 12 e2e)
uv run pytest tests/ --ignore=tests/e2e    # unit only (no server needed)
uv run ruff check . && uv run ruff format --check .
```

## Tri-Protocol Architecture

Kitsune operates across three protocol layers simultaneously:

| Protocol | Role | Standard |
|----------|------|----------|
| **MCP** | Tool provider for AI IDEs | Anthropic/AAIF |
| **A2A** | Agent discovery and peering | Google (150+ orgs) |
| **HERMES** | Inter-agent state and memory | Open source (MIT) |

This means Kitsune can serve tools to Claude Code (MCP), discover other agents (A2A), and maintain state across sessions (HERMES) — all at the same time.

## License

[MIT](LICENSE) — Part of the [Tinkuy](https://github.com/tinkuy-dev) ecosystem. Made in Colombia.
