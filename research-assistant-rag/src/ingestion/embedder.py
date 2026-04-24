"""
Embedding Generation.

This module wraps the OpenAI embedding API to convert text chunks into
vector representations for semantic search.

Why text-embedding-3-small?
- Good quality-to-cost ratio for development and iteration
- 1536 dimensions (same as ada-002 but better quality)
- Supports shortening (we can use fewer dimensions if needed)
- Fast enough for real-time queries

Batching:
The OpenAI embedding API accepts up to 2048 inputs per request.
We batch to minimize API calls (cost + latency), but keep batches
small enough that a single failure doesn't lose too much work.

Design note: This is a simple wrapper in Part A. In Part B, when we
introduce LangChain, we'll swap this for LangChain's Embeddings
abstraction — but the interface stays the same.
"""

from openai import OpenAI
from typing import Optional
from rich.console import Console
from rich.progress import track

from src.config.settings import settings
from src.ingestion.chunker import Chunk

console = Console()

# Maximum texts per embedding API call
BATCH_SIZE = 100


class Embedder:
    """
    Generates embeddings for text chunks using OpenAI's API.
    
    Usage:
        embedder = Embedder()
        chunks_with_embeddings = embedder.embed_chunks(chunks)
    """

    def __init__(self, model: Optional[str] = None):
        """
        Args:
            model: OpenAI embedding model name. Defaults to config value.
        """
        self.model = model or settings.embedding_model
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.total_tokens_used = 0  # Track for cost monitoring

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of text strings.
        
        Handles batching automatically. Tracks token usage for cost awareness.
        
        Args:
            texts: List of text strings to embed.
        
        Returns:
            List of embedding vectors (same order as input texts).
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]

            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
            except Exception as e:
                console.print(f"[red]Embedding API error: {e}[/red]")
                raise

            # Extract embeddings in order
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            # Track usage
            self.total_tokens_used += response.usage.total_tokens

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a single query string.
        
        Separated from embed_texts because some embedding models
        use different prefixes for queries vs. documents. OpenAI's
        text-embedding-3 doesn't need this, but the abstraction is
        useful for future model swaps.
        """
        result = self.embed_texts([query])
        return result[0]

    def embed_chunks(self, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
        """
        Generate embeddings for a list of Chunks.
        
        Returns tuples of (chunk, embedding) so the caller can
        store them together in the vector store.
        
        Args:
            chunks: List of Chunk objects to embed.
        
        Returns:
            List of (Chunk, embedding_vector) tuples.
        """
        if not chunks:
            return []

        console.print(f"\n[blue]Generating embeddings for {len(chunks)} chunks...[/blue]")

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)

        console.print(
            f"  [green]✓[/green] Generated {len(embeddings)} embeddings "
            f"({self.total_tokens_used:,} tokens used, "
            f"model: {self.model})"
        )

        return list(zip(chunks, embeddings))