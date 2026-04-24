"""
Text Chunking with Metadata Preservation.

Chunking is the most important decision in a RAG pipeline.
The chunk size determines:
- How much context the LLM sees per retrieval hit
- How precise the retrieval can be (smaller chunks = more precise)
- How much information can fit in the LLM's context window
- Whether structured content (tables, lists) stays intact or gets broken

This module implements fixed-size token-based chunking for Part A (baseline).
Part B will add recursive and structure-aware strategies for comparison.

TRADE-OFFS (to be measured in evaluation):
┌─────────────────┬──────────────────────┬──────────────────────┐
│                  │ Small chunks (200)   │ Large chunks (1000)  │
├─────────────────┼──────────────────────┼──────────────────────┤
│ Retrieval        │ More precise         │ Less precise         │
│ Context          │ May miss surrounding │ More complete        │
│ LLM quality      │ Less info per chunk  │ More info per chunk  │
│ Multi-hop        │ Needs more chunks    │ Fewer chunks needed  │
│ Tables/structure │ More likely to break │ More likely intact   │
└─────────────────┴──────────────────────┴──────────────────────┘
"""

import tiktoken
from dataclasses import dataclass, field
from typing import Optional
from rich.console import Console

from src.ingestion.pdf_extractor import DocumentContent, PageContent

console = Console()


@dataclass
class Chunk:
    """
    A single chunk of text with full provenance metadata.
    
    Every chunk knows exactly where it came from — which file,
    which page(s), and its position in the document. This metadata
    is critical for two things:
    1. Accurate citations in generated answers
    2. Debugging retrieval quality (did we fetch the right chunk?)
    """
    text: str
    chunk_id: str              # Unique identifier: "{filename}::chunk_{index}"
    source_file: str           # Original PDF filename
    page_numbers: list[int]    # Which page(s) this chunk spans
    chunk_index: int           # Position in the document's chunk sequence
    token_count: int           # Actual token count (not estimated)
    metadata: dict = field(default_factory=dict)  # Extensible metadata

    def to_context_string(self) -> str:
        """
        Format this chunk for inclusion in an LLM prompt.
        Includes source attribution so the LLM can cite it.
        """
        pages = ", ".join(str(p) for p in self.page_numbers)
        return (
            f"[Source: {self.source_file}, Page {pages}]\n"
            f"{self.text}"
        )


class TokenChunker:
    """
    Fixed-size token-based text chunker.
    
    Why token-based (not character-based)?
    LLM context windows are measured in tokens. A 500-token chunk is
    predictable in how much context window it consumes. Character-based
    chunks vary wildly in token count depending on content.
    
    This is the BASELINE chunker for Part A. It has known weaknesses:
    - Splits mid-sentence if the boundary falls there
    - Breaks tables and structured content
    - No awareness of document structure (headings, sections)
    
    These weaknesses are intentional — they motivate Part B improvements.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base",  # GPT-4 / text-embedding-3 tokenizer
    ):
        """
        Args:
            chunk_size: Target size of each chunk in tokens.
            chunk_overlap: Number of overlapping tokens between consecutive chunks.
                          Overlap helps when relevant content spans a chunk boundary.
            encoding_name: Tiktoken encoding to use for token counting.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding(encoding_name)

        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )

    def chunk_document(self, document: DocumentContent) -> list[Chunk]:
        """
        Split a document into token-based chunks with metadata.
        
        Process:
        1. Concatenate all page texts with page boundary markers
        2. Tokenize the full text
        3. Split into fixed-size windows with overlap
        4. Decode each window back to text
        5. Map each chunk back to its source page(s)
        
        Args:
            document: Extracted document content with pages.
        
        Returns:
            List of Chunks with full provenance metadata.
        """
        if not document.pages:
            console.print(f"[yellow]Warning: No pages in {document.source_file}[/yellow]")
            return []

        # Build a combined text with page boundary tracking.
        # We need to know which page each token came from for citations.
        combined_text = ""
        page_boundaries = []  # List of (char_start, char_end, page_number)

        for page in document.pages:
            start = len(combined_text)
            combined_text += page.text + "\n\n"
            end = len(combined_text)
            page_boundaries.append((start, end, page.page_number))

        # Tokenize
        tokens = self.encoder.encode(combined_text)

        if len(tokens) == 0:
            return []

        # Create chunks with sliding window
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        chunk_index = 0

        for start_token in range(0, len(tokens), step):
            end_token = min(start_token + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_token:end_token]

            # Decode back to text
            chunk_text = self.encoder.decode(chunk_tokens).strip()

            if not chunk_text:
                continue

            # Determine which pages this chunk spans.
            # We decode the character positions to find the overlap with page boundaries.
            chunk_char_start = len(self.encoder.decode(tokens[:start_token]))
            chunk_char_end = len(self.encoder.decode(tokens[:end_token]))
            page_numbers = _find_pages(chunk_char_start, chunk_char_end, page_boundaries)

            chunk = Chunk(
                text=chunk_text,
                chunk_id=f"{document.source_file}::chunk_{chunk_index}",
                source_file=document.source_file,
                page_numbers=page_numbers,
                chunk_index=chunk_index,
                token_count=len(chunk_tokens),
                metadata={
                    "title": document.metadata.get("title", ""),
                    "author": document.metadata.get("author", ""),
                    "chunking_strategy": "fixed_token",
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                },
            )
            chunks.append(chunk)
            chunk_index += 1

            # If we've reached the end of the document, stop
            if end_token >= len(tokens):
                break

        console.print(
            f"  [green]✓[/green] Chunked {document.source_file}: "
            f"{len(tokens):,} tokens → {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )

        return chunks

    def chunk_documents(self, documents: list[DocumentContent]) -> list[Chunk]:
        """Chunk multiple documents and return all chunks."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        console.print(
            f"\n[bold]Total: {len(all_chunks)} chunks from "
            f"{len(documents)} documents[/bold]"
        )
        return all_chunks

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.encoder.encode(text))


def _find_pages(
    char_start: int,
    char_end: int,
    page_boundaries: list[tuple[int, int, int]]
) -> list[int]:
    """
    Find which pages a character range spans.
    
    Args:
        char_start: Start character position in the combined text.
        char_end: End character position in the combined text.
        page_boundaries: List of (start, end, page_number) tuples.
    
    Returns:
        Sorted list of page numbers that the range overlaps with.
    """
    pages = set()
    for boundary_start, boundary_end, page_num in page_boundaries:
        # Check if the chunk overlaps with this page's character range
        if char_start < boundary_end and char_end > boundary_start:
            pages.add(page_num)
    return sorted(pages) if pages else [0]  # 0 = unknown page