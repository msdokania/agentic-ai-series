"""
BM25 Keyword Search (Part B).

BM25 (Best Matching 25) is a bag-of-words retrieval function that ranks
documents by term frequency. It's the algorithm behind traditional search
engines like Elasticsearch.

WHY ADD BM25 TO A SYSTEM THAT ALREADY HAS SEMANTIC SEARCH?

Semantic search (embeddings) is great at understanding MEANING:
  "What causes high blood pressure?" matches "Hypertension risk factors"

But it misses EXACT MATCHES that matter in technical documents:
  "What is the BLEU score in Table 3?" — semantic search might return
  chunks about evaluation metrics in general, but BM25 will find the
  chunk containing the literal string "BLEU" and "Table 3".

Real-world failure cases where BM25 saves you:
  - Acronyms and abbreviations (RAG, RLHF, PPO, DPO)
  - Version numbers and specific values ("GPT-4", "0.92 F1")
  - Table references ("Table 2", "Figure 5")
  - Author names and specific citations
  - Code identifiers (function names, variable names)

The HYBRID approach combines both: semantic search for meaning,
BM25 for exact matches, merged via Reciprocal Rank Fusion.
"""

import re
import math
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from rich.console import Console

console = Console()


@dataclass
class BM25Result:
    """Single BM25 search result."""
    chunk_id: str
    text: str
    score: float
    metadata: dict


class BM25Index:
    """
    BM25 search index over document chunks.
    
    Maintains an in-memory BM25 index that can be queried with
    keyword-based queries. The index is built from the same chunks
    stored in ChromaDB, ensuring consistent results.
    
    Tokenization is intentionally simple (whitespace + lowercasing)
    to match what BM25 expects. More sophisticated tokenization
    (stemming, lemmatization) could help but adds complexity —
    we keep it simple and measure whether it's good enough.
    """

    def __init__(self):
        self.index: BM25Okapi = None
        self.chunks: list[dict] = []  # Parallel list of chunk data
        self._is_built = False

    def build_from_vectorstore(self, vector_store) -> int:
        """
        Build the BM25 index from all chunks in the vector store.
        
        We pull all documents from ChromaDB and create a parallel
        BM25 index. This means both search methods operate over
        the exact same corpus.
        
        Args:
            vector_store: VectorStore instance to pull chunks from.
        
        Returns:
            Number of chunks indexed.
        """
        # Get all chunks from ChromaDB
        # ChromaDB's get() returns all documents when no filter is applied
        collection = vector_store.collection
        result = collection.get(include=["documents", "metadatas"])

        if not result["ids"]:
            console.print("[yellow]No chunks in vector store — BM25 index is empty[/yellow]")
            return 0

        self.chunks = []
        tokenized_corpus = []

        for i in range(len(result["ids"])):
            text = result["documents"][i]
            meta = result["metadatas"][i]

            # Parse page_numbers back from string
            page_numbers = [
                int(p) for p in meta.get("page_numbers", "0").split(",") if p
            ]

            self.chunks.append({
                "chunk_id": result["ids"][i],
                "text": text,
                "metadata": {
                    "source_file": meta.get("source_file", ""),
                    "page_numbers": page_numbers,
                    "chunk_index": meta.get("chunk_index", 0),
                    "token_count": meta.get("token_count", 0),
                    "title": meta.get("title", ""),
                    "author": meta.get("author", ""),
                },
            })

            # Tokenize for BM25 — simple whitespace tokenization
            tokens = self._tokenize(text)
            tokenized_corpus.append(tokens)

        # Build the BM25 index
        self.index = BM25Okapi(tokenized_corpus)
        self._is_built = True

        console.print(f"  [green]✓[/green] BM25 index built: {len(self.chunks)} chunks")
        return len(self.chunks)

    def search(self, query: str, top_k: int = 20) -> list[BM25Result]:
        """
        Search the BM25 index with a keyword query.
        
        Args:
            query: Search query (will be tokenized the same way as documents).
            top_k: Number of results to return.
        
        Returns:
            List of BM25Result sorted by descending score.
        """
        if not self._is_built:
            raise RuntimeError("BM25 index not built. Call build_from_vectorstore() first.")

        query_tokens = self._tokenize(query)
        scores = self.index.get_scores(query_tokens)

        # Get top-k indices sorted by score (descending)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append(BM25Result(
                    chunk_id=self.chunks[idx]["chunk_id"],
                    text=self.chunks[idx]["text"],
                    score=float(scores[idx]),
                    metadata=self.chunks[idx]["metadata"],
                ))

        return results

    def _tokenize(self, text: str) -> list[str]:
        """
        Simple tokenization for BM25.
        
        We keep this intentionally simple:
        - Lowercase everything
        - Split on non-alphanumeric characters
        - Remove very short tokens (1 char)
        
        More sophisticated tokenization (stemming, stopword removal)
        could help with recall but hurt precision on technical terms.
        For academic papers, keeping exact tokens is often better.
        """
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return [t for t in tokens if len(t) > 1]