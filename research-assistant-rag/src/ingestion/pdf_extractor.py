"""
PDF Text Extraction with Structure Preservation.

This module handles the first step of the RAG pipeline: getting text out of PDFs.

Why PyMuPDF (fitz)?
PyMuPDF over alternatives like pypdf or pdfplumber because:
- It preserves reading order better (critical for two-column academic papers)
- It handles tables and figures more gracefully
- It extracts text with positional metadata, enabling structure-aware processing
- It's faster on large documents

What we extract for each page:
- Raw text content
- Page number (for citation tracking)
- Character count (for quality filtering — blank pages, TOC pages, etc.)

Design note: We extract at the PAGE level, not the document level.
This preserves the mapping between text and page numbers, which is
important for accurate citations later.
"""

import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass
from typing import Generator
from rich.console import Console
from rich.progress import track

console = Console()


@dataclass
class PageContent:
    """
    Represents the extracted text from a single PDF page.
    
    Keeping page-level granularity is important because:
    1. Citations need page numbers
    2. Some pages (TOC, references, blank) should be filtered out
    3. Page boundaries are natural chunk boundaries for academic papers
    """
    text: str
    page_number: int         # 1-indexed (human-readable)
    char_count: int
    source_file: str         # Original filename for citation tracking


@dataclass
class DocumentContent:
    """
    Represents a fully extracted document with all its pages.
    """
    source_file: str
    total_pages: int
    pages: list[PageContent]
    metadata: dict            # Title, author, etc. if extractable


def extract_pdf(file_path: Path) -> DocumentContent:
    """
    Extract text from a PDF file, page by page.
    
    Args:
        file_path: Path to the PDF file.
    
    Returns:
        DocumentContent with structured page-level text.
    
    Raises:
        FileNotFoundError: If the PDF doesn't exist.
        ValueError: If the file is not a valid PDF or is encrypted.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if not file_path.suffix.lower() == ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {file_path.suffix}")

    console.print(f"[blue]Extracting:[/blue] {file_path.name}")

    try:
        doc = fitz.open(str(file_path))
    except Exception as e:
        raise ValueError(f"Failed to open PDF '{file_path.name}': {e}")

    if doc.is_encrypted:
        doc.close()
        raise ValueError(f"PDF is encrypted: {file_path.name}")

    # Extract metadata from PDF properties
    metadata = _extract_metadata(doc, file_path)

    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")  # "text" mode preserves reading order

        # Clean the extracted text
        text = _clean_text(text)

        page_content = PageContent(
            text=text,
            page_number=page_num + 1,  # 1-indexed for human readability
            char_count=len(text),
            source_file=file_path.name,
        )
        pages.append(page_content)

    # doc.close()

    # Filter out nearly-empty pages (likely blank, TOC stubs, or figure-only pages)
    meaningful_pages = [p for p in pages if p.char_count > 50]

    result = DocumentContent(
        source_file=file_path.name,
        total_pages=len(doc),
        pages=meaningful_pages,
        metadata=metadata,
    )

    console.print(
        f"  [green]✓[/green] {result.total_pages} total pages, "
        f"{len(meaningful_pages)} with content, "
        f"{sum(p.char_count for p in meaningful_pages):,} characters"
    )

    return result


def extract_all_pdfs(directory: Path) -> list[DocumentContent]:
    """
    Extract text from all PDFs in a directory.
    
    Args:
        directory: Path to directory containing PDF files.
    
    Returns:
        List of DocumentContent objects, one per PDF.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[yellow]Warning:[/yellow] No PDF files found in {directory}")
        return []

    console.print(f"\n[bold]Found {len(pdf_files)} PDF files in {directory}[/bold]\n")

    documents = []
    for pdf_path in pdf_files:
        try:
            console.print(f"{pdf_path}")
            doc = extract_pdf(pdf_path)
            documents.append(doc)
        except (ValueError, Exception) as e:
            console.print(f"  [red]✗ Skipping {pdf_path.name}: {e}[/red]")

    console.print(f"\n[bold green]Successfully extracted {len(documents)}/{len(pdf_files)} documents[/bold green]")
    return documents


def _extract_metadata(doc: fitz.Document, file_path: Path) -> dict:
    """
    Extract available metadata from PDF properties.
    
    Academic papers often have title and author in PDF metadata,
    but this is inconsistent. We extract what is available and
    fall back to the filename.
    """
    meta = doc.metadata or {}
    return {
        "title": meta.get("title", "").strip() or file_path.stem,
        "author": meta.get("author", "").strip() or "Unknown",
        "subject": meta.get("subject", "").strip(),
        "creator": meta.get("creator", "").strip(),
        "filename": file_path.name,
    }


def _clean_text(text: str) -> str:
    """
    Clean extracted PDF text.
    
    PDF extraction often produces artifacts:
    - Excessive whitespace from column layouts
    - Hyphenation artifacts from line breaks
    - Header/footer repetition
    
    We clean conservatively — removing obvious artifacts without
    risking loss of meaningful content.
    """
    import re

    # Collapse multiple spaces into one (but preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Fix hyphenation artifacts: "computa-\ntional" → "computational"
    # Only when a lowercase letter precedes the hyphen and follows the newline
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    return text.strip()