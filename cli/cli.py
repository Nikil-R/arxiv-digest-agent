"""
frontend/cli.py - Modern, Polished Terminal & Interactive CLI Interface.
Uses the `rich` library to render:
- Themed rounded Panels and Badges
- Structured Metadata Tables
- Markdown rendering with syntax highlighting
- Live step-by-step progress spinners
- Clean citation tags and user prompt styling
"""

import sys
import io
import argparse

# Ensure standard output safely handles Unicode and emojis on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text
from rich.prompt import Prompt
from rich import box

from backend.agent import run_digest_pipeline, ask_question_state

console = Console()

def render_banner():
    """Renders a modern, visually appealing application banner."""
    banner_text = Text()
    banner_text.append("Autonomous arXiv Paper Digest & QA Agent\n", style="bold cyan")
    banner_text.append("8byte AI Intern Assessment  •  Stateful Agent Graph  •  Local ChromaDB RAG", style="dim white")
    
    console.print(
        Panel(
            banner_text,
            box=box.ROUNDED,
            border_style="bright_blue",
            padding=(1, 2)
        )
    )


def render_metadata_table(paper: dict, state: dict):
    """Renders paper details and extraction status in clean structured tables."""
    table = Table(box=box.ROUNDED, border_style="cyan", show_header=True, header_style="bold bright_cyan")
    table.add_column("Property", style="bold white", width=18)
    table.add_column("Details", style="bright_white")

    table.add_row("Title", paper.get("title", "Untitled"))
    table.add_row("Authors", ", ".join(paper.get("authors", [])))
    table.add_row("arXiv ID", f"[cyan bold]{paper.get('arxiv_id')}[/cyan bold]")
    table.add_row("Published", paper.get("published", "")[:10])
    table.add_row("Primary Category", f"[magenta]{paper.get('primary_category')}[/magenta]")
    table.add_row("PDF Link", f"[link={paper.get('pdf_url')}]{paper.get('pdf_url')}[/link]")

    if "relevance_score" in paper:
        table.add_row("Rank Score", f"[green bold]{paper.get('relevance_score')}[/green bold] (Top Ranked Candidate)")

    console.print(Panel(table, title="[bold cyan] 📄 Paper Metadata [/bold cyan]", border_style="bright_blue", box=box.ROUNDED))

    # Pipeline & Vector Storage Status Card
    status_table = Table(box=box.SIMPLE, show_header=False)
    status_table.add_column("Key", style="dim white", width=20)
    status_table.add_column("Val", style="bold white")

    status_table.add_row("Local Cache Path", state.get("pdf_path", "N/A"))
    status_table.add_row(
        "PDF Parsing Status",
        "[green]✔ Parsed cleanly[/green]" if state.get("parsing_status") == "success" else f"[yellow]⚠ {state.get('parsing_status')}[/yellow]"
    )
    status_table.add_row(
        "Vector Storage",
        f"ChromaDB [dim]({state.get('index_id', 'none')})[/dim]"
    )
    status_table.add_row(
        "Indexed Evidence",
        f"[green]{len(state.get('chunks', []))} chunks[/green] ([dim]384-dim local MiniLM embeddings[/dim])"
    )

    console.print(Panel(status_table, title="[bold green] ⚡ Pipeline Engine Status [/bold green]", border_style="green", box=box.ROUNDED))


def render_executive_briefing(briefing: dict):
    """Renders the markdown executive briefing with rich formatting."""
    content_md = Markdown(briefing.get("content_markdown", ""))
    console.print(
        Panel(
            content_md,
            title="[bold yellow] 📑 Structured Executive Briefing [/bold yellow]",
            subtitle="[dim]Mandated Rubric Artifact[/dim]",
            border_style="yellow",
            box=box.ROUNDED,
            padding=(1, 2)
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous arXiv Paper Digest & QA Agent (CLI Client)"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="arXiv paper ID (e.g., '1706.03762'), arXiv URL, or research topic"
    )
    parser.add_argument(
        "--mode",
        choices=["groq", "gemini", "mock"],
        default="groq",
        help="Pipeline mode: 'groq' (primary), 'gemini' (fallback), 'mock' (offline) (default: groq)"
    )

    args = parser.parse_args()

    render_banner()

    query = args.query
    if not query:
        query = Prompt.ask("\n[bold cyan]?[/bold cyan] Enter research topic or arXiv ID/URL").strip()

    if not query:
        console.print("[bold red]✖ Error:[/bold red] No query provided. Exiting.")
        sys.exit(1)

    console.print(f"\n[dim]Input Query:[/dim] [bold white]{query}[/bold white]")
    console.print(f"[dim]Engine Mode:[/dim] [bold green]{args.mode.upper()}[/bold green]\n")

    with console.status("[bold cyan]Executing Agent State Graph...[/bold cyan]", spinner="dots") as status:
        status.update("[bold cyan]Parsing query, retrieving from arXiv, parsing PDF & indexing vector store...[/bold cyan]")
        state = run_digest_pipeline(query, mode=args.mode)

    if state.get("status") == "error":
        console.print(
            Panel(
                f"[bold red]✖ Pipeline Error:[/bold red] {state.get('error_message')}",
                border_style="red",
                box=box.ROUNDED
            )
        )
        sys.exit(1)

    paper = state.get("selected_paper")
    briefing = state.get("executive_briefing")

    if not paper or not briefing:
        console.print("[bold red]✖ Pipeline did not produce a valid briefing artifact.[/bold red]")
        sys.exit(1)

    # 1. Render Metadata & Status Cards
    render_metadata_table(paper, state)

    # 2. Render Structured Executive Briefing
    render_executive_briefing(briefing)

    # 3. Interactive Grounded QA Loop
    console.print(
        Panel(
            "[bold white]Entering Grounded QA Mode.[/bold white]\n"
            "[dim]Every answer is strictly bounded by retrieved paper chunks. Ask questions or type [bold cyan]'exit'[/bold cyan] to finish.[/dim]",
            border_style="bright_blue",
            box=box.ROUNDED
        )
    )

    if not state.get("retrieval_available", False):
        console.print("[yellow]⚠ Notice: Answering solely from abstract/metadata due to scanned or unparseable PDF layout.[/yellow]")

    turn_count = 0
    while True:
        try:
            user_question = Prompt.ask("\n[bold cyan]Ask a question[/bold cyan]").strip()
            if not user_question:
                continue
            if user_question.lower() in ["exit", "quit", "q"]:
                console.print("\n[bold cyan]👋 Exiting QA session. Thank you![/bold cyan]\n")
                break

            with console.status("[bold cyan]Retrieving vector chunks and formulating grounded response...[/bold cyan]", spinner="dots"):
                state = ask_question_state(state, user_question)

            qa_history = state.get("qa_history", [])
            if not qa_history:
                console.print(f"[red]✖ Could not generate answer: {state.get('error_message')}[/red]")
                continue

            last_turn = qa_history[-1]
            answer_text = last_turn.get("answer", "")
            evidence = last_turn.get("evidence", [])
            turn_count += 1

            answer_render = Markdown(answer_text)
            footer_text = f"[bold green]✔ Grounded in {len(evidence)} retrieved chunks[/bold green] [dim]• Turn {turn_count}[/dim]" if evidence else "[dim]No chunks retrieved[/dim]"

            console.print(
                Panel(
                    answer_render,
                    title=f"[bold green] 💬 Answer: \"{user_question}\" [/bold green]",
                    subtitle=footer_text,
                    border_style="green",
                    box=box.ROUNDED,
                    padding=(1, 2)
                )
            )

        except (KeyboardInterrupt, EOFError):
            console.print("\n\n[bold cyan]👋 Session ended.[/bold cyan]\n")
            break

if __name__ == "__main__":
    main()
