"""
Advanced Chunking Strategies (Part B).

Part A used fixed-size token-based chunking — simple but blind to
document structure. This module adds two smarter strategies:

1. RECURSIVE — Splits at natural boundaries (paragraphs, sentences)
   before falling back to character splits. Respects text flow.

2. STRUCTURE-AWARE — Uses document headings and section markers to
   create semantically meaningful chunks. Each chunk corresponds to
   a logical section of the paper, not an arbitrary token window.

All three strategies produce the same Chunk dataclass, so the rest
of the pipeline (embedding, storage, retrieval) works identically
regardless of which chunker was used. This makes A/B comparison trivial.

COMPARISON STRATEGY:
Ingest the same corpus with each chunker into separate ChromaDB
collections, then run identical queries against each to measure
retrieval quality differences.
"""

import re
import tiktoken
from typing import Optional
from rich.console import Console

from src.ingestion.chunker import Chunk, TokenChunker
from src.ingestion.pdf_extractor import DocumentContent

console = Console()


class RecursiveChunker:
    """
    Recursive character-based chunker that respects natural text boundaries.
    
    Splitting hierarchy (tries each in order, falls back to next):
    1. Double newlines (paragraph breaks)
    2. Single newlines (line breaks)
    3. Sentence endings (". ", "? ", "! ")
    4. Spaces (word boundaries)
    5. Characters (last resort)
    
    This mirrors LangChain's RecursiveCharacterTextSplitter — we implement
    it from scratch first to understand the mechanics, then compare against
    LangChain's version.
    
    WHY THIS IS BETTER THAN FIXED-SIZE:
    - Chunks end at sentence/paragraph boundaries (no mid-sentence cuts)
    - Better semantic coherence per chunk
    - Tables and lists are more likely to stay intact
    
    WHY THIS STILL HAS LIMITATIONS:
    - No awareness of document STRUCTURE (headings, sections)
    - A section spanning 2000 tokens still gets split arbitrarily
    - Can't group related subsections together
    """

    SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding(encoding_name)

    def _token_length(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def _split_text(self, text: str, separators: Optional[list[str]] = None) -> list[str]:
        """
        Recursively split text using the separator hierarchy.
        
        Tries the first separator. If any resulting piece is still too large,
        recursively splits it with the next separator in the hierarchy.
        """
        separators = separators or self.SEPARATORS
        final_chunks = []

        # Find the appropriate separator for this level
        separator = separators[-1]  # default: character-level
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        # Split the text
        if separator:
            pieces = text.split(separator)
        else:
            pieces = list(text)

        # Merge small pieces and split large ones
        current_chunk = ""
        remaining_separators = separators[separators.index(separator) + 1:] if separator in separators else []

        for piece in pieces:
            piece_with_sep = piece + separator if separator else piece
            candidate = current_chunk + piece_with_sep

            if self._token_length(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                # Save current chunk if it has content
                if current_chunk.strip():
                    final_chunks.append(current_chunk.strip())

                # If this single piece is too large, recurse with finer separators
                if self._token_length(piece_with_sep) > self.chunk_size and remaining_separators:
                    sub_chunks = self._split_text(piece, remaining_separators)
                    final_chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = piece_with_sep

        # Don't forget the last chunk
        if current_chunk.strip():
            final_chunks.append(current_chunk.strip())

        return final_chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between consecutive chunks by prepending tail of previous chunk."""
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tokens = self.encoder.encode(chunks[i - 1])
            overlap_tokens = prev_tokens[-self.chunk_overlap:] if len(prev_tokens) > self.chunk_overlap else prev_tokens
            overlap_text = self.encoder.decode(overlap_tokens)
            overlapped.append(overlap_text.strip() + " " + chunks[i])

        return overlapped

    def chunk_document(self, document: DocumentContent) -> list[Chunk]:
        """Split a document using recursive chunking strategy."""
        if not document.pages:
            return []

        # Combine all pages (with page boundary tracking)
        combined_text = ""
        page_boundaries = []
        for page in document.pages:
            start = len(combined_text)
            combined_text += page.text + "\n\n"
            end = len(combined_text)
            page_boundaries.append((start, end, page.page_number))

        # Split recursively
        raw_chunks = self._split_text(combined_text)
        raw_chunks = self._add_overlap(raw_chunks)

        # Build Chunk objects with metadata
        chunks = []
        char_pos = 0
        for i, text in enumerate(raw_chunks):
            # Find approximate character position for page mapping
            idx = combined_text.find(text[:50], max(0, char_pos - 200))
            if idx == -1:
                idx = char_pos
            char_end = idx + len(text)

            page_numbers = self._find_pages(idx, char_end, page_boundaries)

            chunks.append(Chunk(
                text=text,
                chunk_id=f"{document.source_file}::recursive_chunk_{i}",
                source_file=document.source_file,
                page_numbers=page_numbers,
                chunk_index=i,
                token_count=self._token_length(text),
                metadata={
                    "title": document.metadata.get("title", ""),
                    "author": document.metadata.get("author", ""),
                    "chunking_strategy": "recursive",
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                },
            ))
            char_pos = char_end

        console.print(
            f"  [green]✓[/green] Recursive chunked {document.source_file}: "
            f"{len(chunks)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def chunk_documents(self, documents: list[DocumentContent]) -> list[Chunk]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        console.print(f"\n[bold]Total (recursive): {len(all_chunks)} chunks from {len(documents)} documents[/bold]")
        return all_chunks

    def _find_pages(self, char_start, char_end, page_boundaries):
        pages = set()
        for bs, be, pn in page_boundaries:
            if char_start < be and char_end > bs:
                pages.add(pn)
        return sorted(pages) if pages else [0]


class StructureAwareChunker:
    """
    Structure-aware chunker that splits on document section boundaries.
    
    Academic papers have clear structure: Abstract, Introduction, Methods,
    Results, Discussion, References. This chunker detects section headings
    and keeps each section (or subsection) as a coherent chunk.
    
    If a section is too long, it falls back to recursive splitting within
    that section — but the section boundary is always respected.
    
    WHY THIS IS BETTER:
    - Each chunk is a semantically complete unit (a section or subsection)
    - The LLM gets coherent context, not arbitrary text windows
    - Retrieval can be more precise (matching to relevant sections)
    - Section titles provide natural metadata for retrieval filtering
    
    LIMITATIONS:
    - Depends on consistent heading formatting (not all PDFs have this)
    - Very short sections produce tiny chunks (low information density)
    - Very long sections still need sub-splitting
    """

    # Common section heading patterns in academic papers
    HEADING_PATTERNS = [
        r"^#{1,3}\s+.+",                           # Markdown headings
        r"^\d+\.?\s+[A-Z][A-Za-z\s]+$",            # "1. Introduction", "2 Methods"
        r"^\d+\.\d+\.?\s+[A-Z][A-Za-z\s]+$",       # "2.1 Data Collection"
        r"^(?:Abstract|Introduction|Methods?|Results?|Discussion|Conclusion|References|Related Work|Background|Experiments?|Evaluation|Acknowledgements?)\s*$",
    ]

    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        encoding_name: str = "cl100k_base",
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.encoder = tiktoken.get_encoding(encoding_name)
        self._heading_regex = re.compile(
            "|".join(f"({p})" for p in self.HEADING_PATTERNS),
            re.MULTILINE,
        )
        # Fallback splitter for oversized sections
        self._fallback = RecursiveChunker(
            chunk_size=max_chunk_size,
            chunk_overlap=50,
            encoding_name=encoding_name,
        )

    def _token_length(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def _detect_sections(self, text: str) -> list[tuple[str, str]]:
        """
        Split text into (heading, body) pairs based on detected section headings.
        
        Returns a list of tuples: [(section_heading, section_body), ...]
        The first tuple may have an empty heading if text starts before any heading.
        """
        lines = text.split("\n")
        sections = []
        current_heading = ""
        current_body_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped and self._heading_regex.match(stripped):
                # Save previous section
                if current_body_lines or current_heading:
                    body = "\n".join(current_body_lines).strip()
                    if body:
                        sections.append((current_heading, body))
                current_heading = stripped
                current_body_lines = []
            else:
                current_body_lines.append(line)

        # Don't forget the last section
        if current_body_lines:
            body = "\n".join(current_body_lines).strip()
            if body:
                sections.append((current_heading, body))

        return sections

    def chunk_document(self, document: DocumentContent) -> list[Chunk]:
        if not document.pages:
            return []

        combined_text = ""
        page_boundaries = []
        for page in document.pages:
            start = len(combined_text)
            combined_text += page.text + "\n\n"
            end = len(combined_text)
            page_boundaries.append((start, end, page.page_number))

        sections = self._detect_sections(combined_text)

        chunks = []
        chunk_index = 0

        for heading, body in sections:
            section_text = f"{heading}\n\n{body}" if heading else body
            token_count = self._token_length(section_text)

            if token_count <= self.max_chunk_size:
                # Section fits in one chunk — great, keep it whole
                if token_count >= self.min_chunk_size:
                    idx = combined_text.find(body[:60])
                    page_numbers = self._find_pages(
                        max(0, idx), idx + len(section_text), page_boundaries
                    ) if idx >= 0 else [0]

                    chunks.append(Chunk(
                        text=section_text,
                        chunk_id=f"{document.source_file}::section_chunk_{chunk_index}",
                        source_file=document.source_file,
                        page_numbers=page_numbers,
                        chunk_index=chunk_index,
                        token_count=token_count,
                        metadata={
                            "title": document.metadata.get("title", ""),
                            "author": document.metadata.get("author", ""),
                            "section_heading": heading,
                            "chunking_strategy": "structure_aware",
                            "chunk_size": self.max_chunk_size,
                        },
                    ))
                    chunk_index += 1
            else:
                # Section is too long — use recursive splitting within it
                # but prefix each sub-chunk with the section heading for context
                sub_texts = self._fallback._split_text(body)
                for j, sub_text in enumerate(sub_texts):
                    prefixed = f"[Section: {heading}]\n\n{sub_text}" if heading else sub_text
                    idx = combined_text.find(sub_text[:60])
                    page_numbers = self._find_pages(
                        max(0, idx), idx + len(sub_text), page_boundaries
                    ) if idx >= 0 else [0]

                    chunks.append(Chunk(
                        text=prefixed,
                        chunk_id=f"{document.source_file}::section_chunk_{chunk_index}",
                        source_file=document.source_file,
                        page_numbers=page_numbers,
                        chunk_index=chunk_index,
                        token_count=self._token_length(prefixed),
                        metadata={
                            "title": document.metadata.get("title", ""),
                            "author": document.metadata.get("author", ""),
                            "section_heading": heading,
                            "chunking_strategy": "structure_aware",
                            "chunk_size": self.max_chunk_size,
                            "is_sub_chunk": True,
                        },
                    ))
                    chunk_index += 1

        console.print(
            f"  [green]✓[/green] Structure-aware chunked {document.source_file}: "
            f"{len(chunks)} chunks (detected {len(sections)} sections)"
        )
        return chunks

    def chunk_documents(self, documents: list[DocumentContent]) -> list[Chunk]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        console.print(f"\n[bold]Total (structure-aware): {len(all_chunks)} chunks from {len(documents)} documents[/bold]")
        return all_chunks

    def _find_pages(self, char_start, char_end, page_boundaries):
        pages = set()
        for bs, be, pn in page_boundaries:
            if char_start < be and char_end > bs:
                pages.add(pn)
        return sorted(pages) if pages else [0]