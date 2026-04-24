"""
Document Ingestion CLI.

Entry point for ingesting PDF documents into the vector store.
Orchestrates: PDF extraction → chunking → embedding → storage.

Usage:
    python3 -m src.ingestion.ingest --input-dir data/papers/
    python3 -m src.ingestion.ingest --input-dir data/papers/ --reset
"""

import argparse
import time
from pathlib import Path
from rich.console import Console

from src.config.settings import settings
from src.config.prompt_loader import load_prompt_config
from src.ingestion.pdf_extractor import extract_all_pdfs
from src.ingestion.chunker import TokenChunker
from src.ingestion.embedder import Embedder
from src.ingestion.vectorstore import VectorStore

console = Console()


def ingest(input_dir: str, reset: bool = False) -> dict:
    """
    Run the full ingestion pipeline.
    
    Steps:
    1. Load prompt config (for chunk size/overlap parameters)
    2. Extract text from all PDFs in the input directory
    3. Chunk the extracted text
    4. Generate embeddings for all chunks
    5. Store chunks + embeddings in the vector store
    
    Args:
        input_dir: Path to directory containing PDF files.
        reset: If True, clear the vector store before ingesting.
    
    Returns:
        Summary dict with counts and timing.
    """
    start_time = time.time()
    console.print("\n[bold]═══ Document Ingestion Pipeline ═══[/bold]\n")

    # 1. Load config
    prompt_config = load_prompt_config()

    # 2. Extract PDFs
    console.print("[bold]Step 1: Extracting PDFs[/bold]")
    documents = extract_all_pdfs(Path(input_dir))
    if not documents:
        console.print("[red]No documents extracted. Exiting.[/red]")
        return {"status": "no_documents"}

    # 3. Chunk
    console.print(f"\n[bold]Step 2: Chunking (size={prompt_config.chunk_size}, overlap={prompt_config.chunk_overlap})[/bold]")
    chunker = TokenChunker(
        chunk_size=prompt_config.chunk_size,
        chunk_overlap=prompt_config.chunk_overlap,
    )
    chunks = chunker.chunk_documents(documents)

    # 4. Embed
    console.print(f"\n[bold]Step 3: Generating embeddings ({prompt_config.embedding_model})[/bold]")
    embedder = Embedder(model=prompt_config.embedding_model)
    chunks_with_embeddings = embedder.embed_chunks(chunks)

    # 5. Store
    console.print(f"\n[bold]Step 4: Storing in vector database[/bold]")
    store = VectorStore()
    if reset:
        store.reset()
    stored_count = store.add_chunks(chunks_with_embeddings)

    # Summary
    elapsed = time.time() - start_time
    summary = {
        "status": "success",
        "documents_processed": len(documents),
        "total_pages": sum(doc.total_pages for doc in documents),
        "chunks_created": len(chunks),
        "chunks_stored": stored_count,
        "embedding_tokens_used": embedder.total_tokens_used,
        "prompt_version": prompt_config.version,
        "chunk_size": prompt_config.chunk_size,
        "chunk_overlap": prompt_config.chunk_overlap,
        "elapsed_seconds": round(elapsed, 2),
    }

    console.print(f"\n[bold green]═══ Ingestion Complete ═══[/bold green]")
    console.print(f"  Documents:  {summary['documents_processed']}")
    console.print(f"  Pages:      {summary['total_pages']}")
    console.print(f"  Chunks:     {summary['chunks_created']}")
    console.print(f"  Tokens:     {summary['embedding_tokens_used']:,} (embedding)")
    console.print(f"  Time:       {summary['elapsed_seconds']}s")
    console.print(f"  Config:     prompt {summary['prompt_version']}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF documents into the vector store")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/papers/",
        help="Directory containing PDF files to ingest",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the vector store before ingesting",
    )
    args = parser.parse_args()
    ingest(args.input_dir, args.reset)


if __name__ == "__main__":
    main()