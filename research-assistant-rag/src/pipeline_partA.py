"""
RAG Pipeline — Standalone Query Runner.

Chains retrieval and generation for direct CLI testing
without needing to start the FastAPI server.

Usage:
    python3 -m src.pipeline "What are the main approaches to RAG?"
    python3 -m src.pipeline --interactive
"""

import argparse
import json
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.config.prompt_loader import load_prompt_config
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator

console = Console()


def query(question: str, top_k: int = None, verbose: bool = True) -> dict:
    """
    Run a single query through the full RAG pipeline.

    Args:
        question: The user's question.
        top_k: Number of chunks to retrieve (defaults to prompt config).
        verbose: Whether to print detailed output.

    Returns:
        QueryResponse as a dict.
    """
    config = load_prompt_config()
    top_k = top_k or config.top_k

    retriever = Retriever()
    generator = Generator(prompt_config=config)

    if verbose:
        console.print(f"\n[bold]Question:[/bold] {question}")
        console.print(f"[dim]Config: prompt {config.version}, top_k={top_k}, model={config.model}[/dim]\n")

    # Retrieve
    retrieval = retriever.retrieve(query=question, top_k=top_k)

    # Generate
    response = generator.generate(query=question, retrieval=retrieval)

    if verbose:
        # Display the answer
        console.print()
        console.print(Panel(
            response.answer,
            title="[bold green]Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))

        # Display citations
        if response.citations:
            citation_text = "\n".join(
                f"  • {c.source_file}, Page {', '.join(str(p) for p in c.page_numbers)}"
                f" (relevance: {c.relevance_score:.3f})"
                for c in response.citations
            )
            console.print(f"\n[bold]Citations:[/bold]\n{citation_text}")

        # Display performance
        console.print(f"\n[dim]Performance:[/dim]")
        console.print(f"  [dim]Retrieval: {response.retrieval_latency_ms:.0f}ms[/dim]")
        console.print(f"  [dim]Generation: {response.generation_latency_ms:.0f}ms[/dim]")
        console.print(f"  [dim]Total: {response.total_latency_ms:.0f}ms[/dim]")
        console.print(f"  [dim]Chunks retrieved: {response.chunks_retrieved}[/dim]")
        console.print(f"  [dim]Sources: {', '.join(response.sources_used)}[/dim]")

    return response.model_dump()


def interactive():
    """Run an interactive query loop in the terminal."""
    console.print("\n[bold]═══ Research Assistant — Interactive Mode ═══[/bold]")
    console.print("[dim]Type your question, or 'quit' to exit.[/dim]\n")

    while True:
        try:
            question = console.input("[bold teal]Question:[/bold teal] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        query(question)
        console.print()  # blank line between queries


def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline directly")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted text")

    args = parser.parse_args()

    if args.interactive:
        interactive()
    elif args.question:
        result = query(args.question, top_k=args.top_k, verbose=not args.json)
        if args.json:
            print(json.dumps(result, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()