"""
RAG Pipeline — Standalone Query Runner (Updated for Part B).

Supports switching between retrieval strategies:
  --strategy semantic     Part A baseline (embedding similarity only)
  --strategy hybrid       Part B (semantic + BM25 via RRF)
  --strategy reranked     Part B (semantic + BM25 + cross-encoder re-ranking)

Usage:
    python3 -m src.pipeline "What are the main approaches to RAG?" --strategy reranked
    python3 -m src.pipeline --interactive --strategy hybrid
    python3 -m src.pipeline --compare "What is dense passage retrieval?"
"""

import argparse
import json
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config.prompt_loader import load_prompt_config
from src.retrieval.retriever import Retriever
from src.retrieval.hybrid import HybridRetriever
from src.generation.generator import Generator
from src.generation.advanced_generator import AdvancedGenerator

console = Console()


def _build_retriever(strategy: str):
    """Build the appropriate retriever based on strategy name."""
    if strategy == "semantic":
        return Retriever()
    elif strategy in ("hybrid", "reranked"):
        use_reranker = strategy == "reranked"
        return HybridRetriever(use_reranker=use_reranker)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose: semantic, hybrid, reranked")


def _build_generator(prompt_version: str = None):
    """Build the generator — uses AdvancedGenerator for v2+ prompts."""
    config = load_prompt_config(prompt_version)
    if config.version >= "v2":
        return AdvancedGenerator(prompt_config=config), config
    else:
        return Generator(prompt_config=config), config


def query(
    question: str,
    strategy: str = "reranked",
    top_k: int = None,
    prompt_version: str = None,
    verbose: bool = True,
) -> dict:
    """
    Run a single query through the RAG pipeline.

    Args:
        question: The user's question.
        strategy: Retrieval strategy (semantic, hybrid, reranked).
        top_k: Number of chunks to retrieve.
        prompt_version: Prompt version to use (defaults to env config).
        verbose: Whether to print formatted output.

    Returns:
        QueryResponse as a dict.
    """
    generator, config = _build_generator(prompt_version)
    top_k = top_k or config.top_k

    retriever = _build_retriever(strategy)

    if verbose:
        console.print(f"\n[bold]Question:[/bold] {question}")
        console.print(
            f"[dim]Config: prompt {config.version}, strategy={strategy}, "
            f"top_k={top_k}, model={config.model}[/dim]\n"
        )

    # Retrieve
    retrieval = retriever.retrieve(query=question, top_k=top_k)

    # Generate
    response = generator.generate(query=question, retrieval=retrieval)

    if verbose:
        # Display answer
        console.print()
        console.print(Panel(
            response.answer,
            title="[bold green]Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))

        # Confidence
        if response.confidence is not None:
            conf_pct = int(response.confidence * 100)
            color = "green" if conf_pct >= 60 else "yellow" if conf_pct >= 40 else "red"
            console.print(f"\n[{color}]Confidence: {conf_pct}%[/{color}]")

        # Citations
        if response.citations:
            citation_text = "\n".join(
                f"  • {c.source_file}, Page {', '.join(str(p) for p in c.page_numbers)}"
                f" (relevance: {c.relevance_score:.3f})"
                for c in response.citations
            )
            console.print(f"\n[bold]Citations:[/bold]\n{citation_text}")

        # Performance
        console.print(f"\n[dim]Performance:[/dim]")
        console.print(f"  [dim]Strategy: {response.retrieval_strategy}[/dim]")
        console.print(f"  [dim]Retrieval: {response.retrieval_latency_ms:.0f}ms[/dim]")
        console.print(f"  [dim]Generation: {response.generation_latency_ms:.0f}ms[/dim]")
        console.print(f"  [dim]Total: {response.total_latency_ms:.0f}ms[/dim]")
        console.print(f"  [dim]Chunks: {response.chunks_retrieved}, Sources: {', '.join(response.sources_used)}[/dim]")

    return response.model_dump()


def compare(question: str, top_k: int = 5):
    """
    Run the same question through all retrieval strategies and compare.
    
    This is the key comparison tool for Part B analysis — it shows
    side-by-side how semantic vs hybrid vs reranked retrieval affects
    the final answer quality.
    """
    console.print(f"\n[bold]═══ Strategy Comparison ═══[/bold]")
    console.print(f"[bold]Question:[/bold] {question}\n")

    strategies = ["semantic", "hybrid", "reranked"]
    results = {}

    for strategy in strategies:
        console.print(f"\n[bold blue]── {strategy.upper()} ──[/bold blue]")
        try:
            result = query(question, strategy=strategy, top_k=top_k, verbose=False)
            results[strategy] = result

            # Brief summary
            answer_preview = result["answer"][:200] + "..." if len(result["answer"]) > 200 else result["answer"]
            conf = result.get("confidence")
            conf_str = f"{int(conf * 100)}%" if conf else "N/A"

            console.print(f"  Answer: {result['answer']}")
            console.print(f"  Confidence: {conf_str}")
            console.print(f"  Citations: {len(result.get('citations', []))}")
            console.print(f"  Latency: {result['total_latency_ms']:.0f}ms")
            console.print(f"  Sources: {', '.join(result.get('sources_used', []))}")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            results[strategy] = {"error": str(e)}

    # Summary table
    console.print(f"\n[bold]═══ Comparison Summary ═══[/bold]\n")
    table = Table(title="Retrieval Strategy Comparison")
    table.add_column("Metric", style="bold")
    for s in strategies:
        table.add_column(s.upper(), justify="center")

    metrics = ["citations", "confidence", "total_latency_ms", "chunks_retrieved"]
    labels = ["Citations", "Confidence", "Latency (ms)", "Chunks Retrieved"]

    for label, metric in zip(labels, metrics):
        row = [label]
        for s in strategies:
            r = results.get(s, {})
            if "error" in r:
                row.append("[red]Error[/red]")
            elif metric == "citations":
                row.append(str(len(r.get("citations", []))))
            elif metric == "confidence":
                c = r.get("confidence")
                row.append(f"{int(c * 100)}%" if c else "N/A")
            else:
                row.append(str(round(r.get(metric, 0))))
        table.add_row(*row)

    console.print(table)

    return results


def interactive(strategy: str = "reranked"):
    """Run an interactive query loop."""
    console.print(f"\n[bold]═══ Research Assistant — Interactive Mode ═══[/bold]")
    console.print(f"[dim]Strategy: {strategy} | Type 'quit' to exit, 'compare' to compare strategies[/dim]\n")

    while True:
        try:
            question = console.input("[bold teal]Question:[/bold teal] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        if question.lower().startswith("compare "):
            compare(question[8:])
        else:
            query(question, strategy=strategy)

        console.print()


def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--compare", "-c", type=str, help="Compare all strategies on a question")
    parser.add_argument(
        "--strategy", "-s",
        choices=["semantic", "hybrid", "reranked"],
        default="reranked",
        help="Retrieval strategy (default: reranked)"
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--prompt-version", type=str, default=None)
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    if args.compare:
        compare(args.compare, top_k=args.top_k or 5)
    elif args.interactive:
        interactive(strategy=args.strategy)
    elif args.question:
        result = query(
            args.question,
            strategy=args.strategy,
            top_k=args.top_k,
            prompt_version=args.prompt_version,
            verbose=not args.json,
        )
        if args.json:
            print(json.dumps(result, indent=2, default=str))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()