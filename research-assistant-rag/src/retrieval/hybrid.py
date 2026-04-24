"""
Hybrid Retrieval with Re-ranking (Part B).

This is the most important module in Part B. It combines three retrieval
techniques into a single pipeline:

1. SEMANTIC SEARCH — embedding-based similarity (from Part A)
2. BM25 KEYWORD SEARCH — term frequency matching (new in Part B)
3. CROSS-ENCODER RE-RANKING — neural relevance scoring (new in Part B)

The combination works in two stages:

STAGE 1: Candidate Generation (broad recall)
  Run both semantic and BM25 search independently, each fetching top-N
  candidates (N > final top_k, e.g., 20 each). Merge results using
  Reciprocal Rank Fusion (RRF), which combines rankings without needing
  to normalize scores across different systems.

STAGE 2: Re-ranking (precise relevance)
  Take the merged candidates and re-score each one with a cross-encoder
  model that sees the (query, chunk) pair together. This is much more
  accurate than embedding similarity because the cross-encoder can
  attend to both query and document jointly.

WHY THIS IS BETTER:
  Semantic alone:  Misses exact keyword matches
  BM25 alone:      Misses semantic similarity
  Hybrid alone:    Good recall but noisy ranking
  Hybrid + rerank: Best of all worlds — broad recall + precise ranking

PERFORMANCE NOTE:
  Cross-encoder re-ranking is slower than embedding search (~50-200ms
  for 40 candidates). This is why we use it as a SECOND stage on a
  small candidate set, not as the primary search method.
"""

import time
from typing import Optional
from dataclasses import dataclass
from rich.console import Console

from src.ingestion.embedder import Embedder
from src.ingestion.vectorstore import VectorStore
from src.retrieval.retriever import RetrievalResult
from src.retrieval.bm25_search import BM25Index
from src.retrieval.reranker import CrossEncoderReranker

console = Console()


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) — merges multiple ranked result lists.
    
    RRF is a simple but effective rank aggregation method. For each
    document, it sums 1/(k + rank) across all lists where the document
    appears. Higher RRF score = document ranked highly in more lists.
    
    Why RRF over other fusion methods?
    - No need to normalize scores across different systems
    - Robust to outlier scores
    - Simple to implement and understand
    - Widely used in production hybrid search systems
    
    Args:
        result_lists: List of ranked result lists. Each result dict
                     must have a 'chunk_id' key for deduplication.
        k: Smoothing constant (default 60). Higher k = less emphasis
           on top ranks. 60 is the standard value from the RRF paper.
    
    Returns:
        Merged and deduplicated list sorted by RRF score (descending).
    """
    rrf_scores = {}     # chunk_id → cumulative RRF score
    chunk_data = {}     # chunk_id → full chunk data (keep the best version)

    for results in result_lists:
        for rank, result in enumerate(results):
            chunk_id = result["chunk_id"]
            rrf_score = 1.0 / (k + rank + 1)  # rank is 0-indexed

            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
                chunk_data[chunk_id] = result
            rrf_scores[chunk_id] += rrf_score

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    merged = []
    for chunk_id in sorted_ids:
        result = chunk_data[chunk_id].copy()
        result["rrf_score"] = round(rrf_scores[chunk_id], 6)
        merged.append(result)

    return merged


class HybridRetriever:
    """
    Hybrid retriever combining semantic search, BM25, and cross-encoder re-ranking.
    
    This is the Part B upgrade to the simple Retriever from Part A.
    It follows the same interface (retrieve() → RetrievalResult) so
    it's a drop-in replacement in the pipeline.
    
    Configuration options:
    - semantic_weight: How many candidates from semantic search (default: 20)
    - bm25_weight: How many candidates from BM25 (default: 20)  
    - use_reranker: Whether to apply cross-encoder re-ranking (default: True)
    - final_top_k: How many results to return after all stages
    
    The pipeline:
    ┌─────────────┐     ┌─────────────┐
    │  Semantic    │     │   BM25      │
    │  top-20     │     │   top-20    │
    └──────┬──────┘     └──────┬──────┘
           │                   │
           └─────────┬─────────┘
                     │
              ┌──────▼──────┐
              │  RRF Merge  │
              │  ~30 unique │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │  Re-rank    │
              │  cross-enc  │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │  Top-K      │
              │  final      │
              └─────────────┘
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        semantic_candidates: int = 20,
        bm25_candidates: int = 20,
        use_reranker: bool = True,
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()

        # BM25 index — build if not provided
        if bm25_index is None:
            self.bm25_index = BM25Index()
            self.bm25_index.build_from_vectorstore(self.vector_store)
        else:
            self.bm25_index = bm25_index

        # Cross-encoder re-ranker
        self.use_reranker = use_reranker
        if use_reranker:
            self.reranker = reranker or CrossEncoderReranker()
        else:
            self.reranker = None

        self.semantic_candidates = semantic_candidates
        self.bm25_candidates = bm25_candidates

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> RetrievalResult:
        """
        Hybrid retrieval: semantic + BM25 → RRF merge → re-rank → top-k.
        
        Args:
            query: The user's question.
            top_k: Number of final results to return.
            where: Optional metadata filter (applied to semantic search only).
        
        Returns:
            RetrievalResult with the best chunks from the combined pipeline.
        """
        start_time = time.time()

        # ── Stage 1a: Semantic search ────────────────────────
        query_embedding = self.embedder.embed_query(query)
        semantic_results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=self.semantic_candidates,
            where=where,
        )
        # Normalize the format: ensure each has 'chunk_id'
        for r in semantic_results:
            if "chunk_id" not in r:
                r["chunk_id"] = r.get("id", "")

        # ── Stage 1b: BM25 search ───────────────────────────
        bm25_results_raw = self.bm25_index.search(query, top_k=self.bm25_candidates)
        bm25_results = [
            {
                "chunk_id": r.chunk_id,
                "text": r.text,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in bm25_results_raw
        ]

        semantic_count = len(semantic_results)
        bm25_count = len(bm25_results)

        # ── Stage 2: Reciprocal Rank Fusion ──────────────────
        merged = reciprocal_rank_fusion([semantic_results, bm25_results])

        # ── Stage 3: Cross-encoder re-ranking ────────────────
        if self.use_reranker and self.reranker and merged:
            reranked = self.reranker.rerank(query, merged, top_k=top_k)
            strategy = "hybrid+reranked"
        else:
            reranked = merged[:top_k]
            strategy = "hybrid"

        latency_ms = (time.time() - start_time) * 1000

        result = RetrievalResult(
            query=query,
            chunks=reranked,
            top_k=top_k,
            retrieval_latency_ms=round(latency_ms, 2),
            embedding_model=self.embedder.model,
            strategy=strategy,
        )

        if result.has_results:
            scores = [c.get("score", c.get("rerank_score", 0)) for c in reranked]
            console.print(
                f"  [dim]Hybrid retrieval: {semantic_count} semantic + {bm25_count} bm25 "
                f"→ {len(merged)} merged → {len(reranked)} final "
                f"({latency_ms:.0f}ms, strategy: {strategy})[/dim]"
            )
        else:
            console.print(f"  [yellow]No relevant chunks found (hybrid)[/yellow]")

        return result