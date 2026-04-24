"""
Vector Store — ChromaDB Integration.

Why ChromaDB over Qdrant?
For this project, ChromaDB is the better choice because:
- Runs embedded (no Docker, no external service to manage)
- Persistent storage to disk out of the box
- Simple Python-native API
- Good enough for our document corpus size (hundreds to low thousands of chunks)
- Lower barrier for anyone following this series

Qdrant would be a better choice for production scale (millions of vectors),
but for a research assistant over 10-20 papers, ChromaDB is the right tool.

This module handles:
1. Storing chunks + embeddings with full metadata
2. Retrieving nearest neighbors by query embedding
3. Managing the collection lifecycle (create, reset, stats)
"""

import chromadb
from pathlib import Path
from typing import Optional
from rich.console import Console

from src.config.settings import settings
from src.ingestion.chunker import Chunk

console = Console()


class VectorStore:
    """
    ChromaDB wrapper for chunk storage and retrieval.
    
    Each chunk is stored with:
    - Its text (for display in retrieval results)
    - Its embedding vector (for similarity search)
    - Full metadata (source file, pages, chunk index, etc.)
    
    The chunk_id serves as the unique document ID in ChromaDB,
    ensuring we don't create duplicates on re-ingestion.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize the ChromaDB client and collection.
        
        Args:
            persist_dir: Directory for persistent storage.
            collection_name: Name of the collection to use.
        """
        persist_dir = persist_dir or settings.chroma_persist_dir
        collection_name = collection_name or settings.chroma_collection_name

        # Ensure the persist directory exists
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Get or create the collection
        # We use cosine distance — standard for text embedding similarity
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

        console.print(
            f"[dim]Vector store: {collection_name} "
            f"({self.collection.count()} existing chunks)[/dim]"
        )

    def add_chunks(
        self,
        chunks_with_embeddings: list[tuple[Chunk, list[float]]],
    ) -> int:
        """
        Store chunks with their embeddings in ChromaDB.
        
        Uses chunk_id as the unique ID to prevent duplicates.
        If a chunk_id already exists, it will be updated (upsert behavior).
        
        Args:
            chunks_with_embeddings: List of (Chunk, embedding_vector) tuples.
        
        Returns:
            Number of chunks successfully stored.
        """
        if not chunks_with_embeddings:
            return 0

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for chunk, embedding in chunks_with_embeddings:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            embeddings.append(embedding)

            # ChromaDB metadata must be flat (no nested dicts/lists)
            # Convert page_numbers list to a string for storage
            metadatas.append({
                "source_file": chunk.source_file,
                "page_numbers": ",".join(str(p) for p in chunk.page_numbers),
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "title": chunk.metadata.get("title", ""),
                "author": chunk.metadata.get("author", ""),
                "chunking_strategy": chunk.metadata.get("chunking_strategy", ""),
            })

        # Upsert to handle re-ingestion gracefully
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        console.print(
            f"  [green]✓[/green] Stored {len(ids)} chunks in vector store "
            f"(total: {self.collection.count()})"
        )

        return len(ids)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Retrieve the most similar chunks to a query embedding.
        
        Args:
            query_embedding: The query's embedding vector.
            top_k: Number of results to return.
            where: Optional metadata filter (e.g., {"source_file": "paper.pdf"}).
        
        Returns:
            List of result dicts, each containing:
            - text: The chunk text
            - chunk_id: Unique identifier
            - score: Cosine similarity score (higher = more similar)
            - metadata: Full chunk metadata
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        # ChromaDB returns nested lists (one per query). We sent one query.
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                # ChromaDB returns cosine DISTANCE, not similarity.
                # Similarity = 1 - distance for cosine metric.
                distance = results["distances"][0][i]
                similarity = 1 - distance

                # Parse page_numbers back from string to list
                meta = results["metadatas"][0][i]
                page_numbers = [
                    int(p) for p in meta.get("page_numbers", "0").split(",") if p
                ]

                formatted.append({
                    "text": results["documents"][0][i],
                    "chunk_id": results["ids"][0][i],
                    "score": round(similarity, 4),
                    "metadata": {
                        "source_file": meta.get("source_file", ""),
                        "page_numbers": page_numbers,
                        "chunk_index": meta.get("chunk_index", 0),
                        "token_count": meta.get("token_count", 0),
                        "title": meta.get("title", ""),
                        "author": meta.get("author", ""),
                    },
                })

        return formatted

    def get_stats(self) -> dict:
        """Return collection statistics."""
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.collection.name,
        }

    def reset(self) -> None:
        """Delete all chunks in the collection. Use with caution."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        console.print("[yellow]Vector store reset — all chunks deleted[/yellow]")