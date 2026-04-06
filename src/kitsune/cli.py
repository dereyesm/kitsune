"""Kitsune CLI — local code assistant."""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from kitsune.consent import ConsentDenied, ensure_consent
from kitsune.graph.build import app as graph_app
from kitsune.graph.state import KitsuneState

cli = typer.Typer(name="kit", help="Kitsune — local code assistant with SLMs", no_args_is_help=True)
console = Console()


def _gate_consent() -> None:
    """Enforce the remote-provider consent flow before any inference call."""
    from kitsune.config import settings

    if settings.privacy_level == "local":
        return
    try:
        ensure_consent(
            provider_name=settings.provider_name or "remote",
            base_url=settings.base_url,
            privacy_level=settings.privacy_level,
            console=console,
        )
    except ConsentDenied as err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(2) from err


def _read_file(path: str) -> tuple[str, str]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        console.print(f"[red]File not found: {p}[/red]")
        raise typer.Exit(1)
    return p.read_text(encoding="utf-8"), str(p)


def _read_stdin() -> str:
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def _run(state: KitsuneState):
    _gate_consent()
    result = graph_app.invoke(state)
    response = result["response"]
    task = result.get("task_type", "")
    title = f"Kitsune [{task}]"
    console.print(Panel(Markdown(response), title=title, border_style="cyan"))


@cli.command()
def explain(
    file: str = typer.Argument(None, help="File to explain (or pipe via stdin)"),
):
    """Explain what a piece of code does."""
    if file:
        code, fpath = _read_file(file)
    else:
        code = _read_stdin()
        fpath = "<stdin>"
        if not code:
            console.print("[yellow]Provide a file path or pipe code via stdin[/yellow]")
            raise typer.Exit(1)

    state: KitsuneState = {
        "user_input": "",
        "task_type": "explain",
        "code_context": code,
        "file_path": fpath,
        "response": "",
        "escalation_reason": "",
    }
    _run(state)


@cli.command()
def ask(
    question: str = typer.Argument(..., help="Question about the code"),
    file: str = typer.Option(None, "--file", "-f", help="File for context"),
):
    """Ask a question about code."""
    code = ""
    fpath = ""
    if file:
        code, fpath = _read_file(file)
    else:
        stdin_code = _read_stdin()
        if stdin_code:
            code = stdin_code
            fpath = "<stdin>"

    state: KitsuneState = {
        "user_input": question,
        "task_type": "ask",
        "code_context": code,
        "file_path": fpath,
        "response": "",
        "escalation_reason": "",
    }
    _run(state)


@cli.command()
def status():
    """Show Kitsune status."""
    import platform

    import httpx

    from kitsune.config import settings
    from kitsune.providers import PROVIDERS, PrivacyLevel

    console.print(f"[bold]Backend:[/bold] {settings.backend} ({platform.system()})")
    console.print(f"[bold]Tier:[/bold] {settings.model_tier}")
    console.print(f"[bold]Model:[/bold] {settings.model_name}")
    console.print(f"[bold]Server:[/bold] {settings.base_url}")

    # Privacy / provider indicator
    if settings.provider_name:
        colour = {
            "local": "green",
            "remote_free": "yellow",
            "remote_paid": "magenta",
        }.get(settings.privacy_level, "white")
        console.print(
            f"[bold]Provider:[/bold] {settings.provider_name} "
            f"[{colour}]({settings.privacy_level})[/{colour}]"
        )
    else:
        console.print("[bold]Provider:[/bold] [green]local (no override)[/green]")

    try:
        r = httpx.get(f"{settings.base_url}/models", timeout=3)
        models = [m["id"] for m in r.json().get("data", [])]
        if models:
            console.print(f"[bold]Available models:[/bold] {', '.join(models[:8])}")
        else:
            console.print("[yellow]No models loaded[/yellow]")
    except httpx.ConnectError, httpx.HTTPError:
        if settings.backend == "mlx":
            hint = (
                "mlx_lm.server --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit --port 8008"
            )
        else:
            hint = "ollama serve  # then: ollama pull qwen2.5-coder:1.5b"
        console.print(f"[red]Server not reachable. Start with:[/red]\n  {hint}")

    # Available provider tiers — the "show of power" moment.
    console.print("\n[bold cyan]Available provider tiers:[/bold cyan]")
    for p in PROVIDERS.values():
        if p.privacy_level == PrivacyLevel.LOCAL:
            marker = "[green]● local[/green]"
            detail = p.base_url
        elif p.privacy_level == PrivacyLevel.REMOTE_FREE:
            has_key = bool(p.env_key_name and __import__("os").environ.get(p.env_key_name))
            marker = (
                "[yellow]● free remote[/yellow]" if has_key else "[dim]○ free remote (no key)[/dim]"
            )
            detail = f"requires {p.env_key_name}"
        else:
            has_key = bool(p.env_key_name and __import__("os").environ.get(p.env_key_name))
            marker = "[magenta]● paid[/magenta]" if has_key else "[dim]○ paid (no key)[/dim]"
            detail = f"requires {p.env_key_name}"
        console.print(f"  {marker} [bold]{p.name}[/bold] — {detail}")


@cli.command()
def index(
    directory: str = typer.Argument(".", help="Directory to index"),
    backend: str = typer.Option("bm25", "--backend", "-b", help="bm25 or chroma"),
):
    """Index a codebase for RAG search."""
    from kitsune.rag.bm25_backend import BM25Backend

    if backend == "chroma":
        from kitsune.rag.chroma_backend import ChromaBackend

        rag = ChromaBackend()
    else:
        rag = BM25Backend()

    console.print(f"Indexing [bold]{directory}[/bold] with {rag.name}...")
    stats = rag.index(directory)
    console.print(
        f"[green]Done:[/green] {stats.total_files} files, "
        f"{stats.total_chunks} chunks, {stats.index_time_ms}ms"
    )

    # Save backend for search command
    import json

    state_path = Path(".kitsune-rag.json")
    state_path.write_text(json.dumps({"backend": backend, "directory": directory}))


@cli.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top", "-k", help="Number of results"),
    backend: str = typer.Option("bm25", "--backend", "-b", help="bm25 or chroma"),
    directory: str = typer.Option(".", "--dir", "-d", help="Directory to search"),
):
    """Search indexed codebase for relevant code."""
    from rich.syntax import Syntax

    from kitsune.rag.bm25_backend import BM25Backend

    if backend == "chroma":
        from kitsune.rag.chroma_backend import ChromaBackend

        rag = ChromaBackend()
    else:
        rag = BM25Backend()

    # Index on the fly (BM25 is fast enough)
    rag.index(directory)
    results = rag.search(query, top_k=top_k)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    for i, chunk in enumerate(results, 1):
        lang = chunk.language if chunk.language != "md" else "markdown"
        loc = f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
        title = f"[{i}] {loc} (score: {chunk.score})"
        console.print(f"\n[bold cyan]{title}[/bold cyan]")
        console.print(Syntax(chunk.content, lang, line_numbers=True, start_line=chunk.start_line))


# Typer app entry point
app = cli
