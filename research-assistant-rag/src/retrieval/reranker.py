"""
Cross-Encoder Re-ranking (Part B).

WHY RE-RANKING?

Embedding-based retrieval (Part A) uses a BI-ENCODER: query and document
are encoded INDEPENDENTLY, then compared via cosine similarity. This is
fast (embed once, compare many) but loses information because the model
can't see query and document together.

A CROSS-ENCODER sees the (query, document) pair JOINTLY. It processes
them through the same transformer, with full cross-attention between
query tokens and document tokens. This is much more accurate because
the model can understand how specific parts of the query relate to
specific parts of the document.

The tradeoff:
  Bi-encoder:    Fast (1ms per comparison), less accurate
  Cross-encoder: Slow (5-20ms per comparison), much more accurate

That's why we use cross-encoders as a SECOND STAGE on a small candidate
set (20-40 results from Stage 1), not as primary search over thousands
of chunks.

MODEL CHOICE:
We use 'cross-encoder/ms-marco-MiniLM-L-6-v2' because:
- Trained on MS MARCO passage ranking (relevant to our task)
- Small and fast (~22M params, ~5ms per pair)
- Good accuracy for its size
- Well-supported by sentence-transformers

For production, larger models like 'cross-encoder/ms-marco-MiniLM-L-12-v2'
are more accurate but slower.
"""

import time
from sentence_transformers import CrossEncoder
from rich.console import Console

console = Console()

# Default model — small, fast, good enough for development
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Re-ranks retrieval candidates using a cross-encoder model.
    
    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, candidates, top_k=5)
    
    The reranker scores each (query, candidate_text) pair and returns
    the candidates sorted by cross-encoder score (most relevant first).
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        """
        Load the cross-encoder model.
        
        First call downloads the model (~90MB for MiniLM-L-6).
        Subsequent calls use the cached version.
        """
        console.print(f"[dim]Loading cross-encoder: {model_name}[/dim]")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Re-rank candidate chunks by cross-encoder relevance score.
        
        Args:
            query: The user's question.
            candidates: List of chunk dicts (from semantic/hybrid retrieval).
                       Each must have a 'text' key.
            top_k: Number of results to return after re-ranking.
        
        Returns:
            Top-k candidates sorted by cross-encoder score (descending).
            Each result dict gets an additional 'rerank_score' field.
        """
        if not candidates:
            return []

        start_time = time.time()

        # Build (query, document) pairs for the cross-encoder
        pairs = [(query, c["text"]) for c in candidates]

        # Score all pairs in a single batch (efficient)
        scores = self.model.predict(pairs)

        # Attach scores to candidates
        scored = []
        for candidate, score in zip(candidates, scores):
            result = candidate.copy()
            result["rerank_score"] = float(score)
            # Update the main score field so downstream code uses the reranked score
            result["score"] = float(score)
            scored.append(result)

        # Sort by cross-encoder score (descending) and take top-k
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_results = scored[:top_k]

        latency_ms = (time.time() - start_time) * 1000
        console.print(
            f"  [dim]Re-ranked {len(candidates)} candidates → top-{top_k} "
            f"({latency_ms:.0f}ms, model: {self.model_name})[/dim]"
        )

        return top_results