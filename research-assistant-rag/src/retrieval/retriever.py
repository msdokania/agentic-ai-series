"""
Retrieval Module — Query → Relevant Chunks.

This is the core of the RAG pipeline: given a user question,
find the most relevant chunks from the vector store.

In Part A, retrieval is simple: embed the query, find nearest neighbors.
This is the BASELINE. Part B will introduce hybrid search, re-ranking,
and query transformation.

What we track for every retrieval (for evaluation in Part C):
- Which chunks were retrieved
- Their similarity scores
- Which source documents they came from
- Latency
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from rich.console import Console

from src.ingestion.embedder import Embedder
from src.ingestion.vectorstore import VectorStore

console = Console()


@dataclass
class RetrievalResult:
    """
    Complete result of a retrieval operation.
    
    Includes not just the chunks but all metadata needed for:
    - Passing context to the generator
    - Evaluating retrieval quality
    - Debugging why a particular answer was good/bad
    """
    query: str
    chunks: list[dict]              # List of chunk results from vector store
    top_k: int
    retrieval_latency_ms: float
    embedding_model: str
    strategy: str = "semantic"      # Will change in Part B: "hybrid", "reranked"

    @property
    def has_results(self) -> bool:
        return len(self.chunks) > 0

    def format_context(self) -> str:
        """
        Format retrieved chunks into a context string for the LLM prompt.
        
        Each chunk is prefixed with its source metadata so the LLM
        can reference specific sources in its citations.
        """
        if not self.chunks:
            return ""

        context_parts = []
        for i, chunk in enumerate(self.chunks, 1):
            meta = chunk["metadata"]
            pages = ", ".join(str(p) for p in meta.get("page_numbers", []))
            source = meta.get("source_file", "unknown")
            score = chunk.get("score", 0)

            context_parts.append(
                f"--- Passage {i} [Source: {source}, Page {pages}] "
                f"(relevance: {score:.3f}) ---\n"
                f"{chunk['text']}\n"
            )

        return "\n".join(context_parts)

    def get_source_summary(self) -> str:
        """Summary of which sources were retrieved, for logging."""
        sources = set()
        for chunk in self.chunks:
            meta = chunk["metadata"]
            sources.add(meta.get("source_file", "unknown"))
        return ", ".join(sorted(sources))


class Retriever:
    """
    Semantic retriever using vector similarity search.
    
    This is the Part A baseline retriever. It:
    1. Embeds the query using the same model used for document chunks
    2. Queries ChromaDB for the top-k most similar chunks
    3. Returns results with full metadata for context formatting
    
    Known limitations (to be addressed in Part B):
    - Pure semantic search — misses exact keyword matches
    - No re-ranking — relies entirely on embedding similarity
    - Single retrieval pass — can't handle multi-hop questions
    - No query transformation — takes the user's question as-is
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> RetrievalResult:
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query: The user's question.
            top_k: Number of chunks to retrieve.
            where: Optional metadata filter for scoping retrieval.
        
        Returns:
            RetrievalResult with chunks, scores, and metadata.
        """
        start_time = time.time()

        # Step 1: Embed the query
        query_embedding = self.embedder.embed_query(query)

        # Step 2: Search the vector store
        chunks = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )

        latency_ms = (time.time() - start_time) * 1000

        result = RetrievalResult(
            query=query,
            chunks=chunks,
            top_k=top_k,
            retrieval_latency_ms=round(latency_ms, 2),
            embedding_model=self.embedder.model,
            strategy="semantic",
        )

        # Log retrieval summary
        if result.has_results:
            scores = [c["score"] for c in chunks]
            console.print(
                f"  [dim]Retrieved {len(chunks)} chunks in {latency_ms:.0f}ms "
                f"(scores: {max(scores):.3f} → {min(scores):.3f}, "
                f"sources: {result.get_source_summary()})[/dim]"
            )
        else:
            console.print(f"  [yellow]No relevant chunks found for query[/yellow]")

        return result