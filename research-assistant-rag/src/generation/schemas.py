"""
Output Schemas for RAG Generation.

These Pydantic models define the structure of the system's responses.
Having structured output (not just raw text) enables:
- Consistent citation format
- Machine-readable confidence scores
- Programmatic evaluation
- Clean API responses

In Part A, we parse the LLM's free-text output into these structures
using simple string processing. In Part B, we'll use LangChain's
output parsers for more robust structured generation.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class Citation(BaseModel):
    """A single source citation."""
    source_file: str = Field(description="Name of the source document")
    page_numbers: list[int] = Field(default_factory=list, description="Referenced page numbers")
    chunk_id: str = Field(default="", description="Specific chunk identifier")
    relevance_score: float = Field(default=0.0, description="How relevant this source was to the answer")


class QueryResponse(BaseModel):
    """
    Complete response to a user query.
    
    This is what the API returns. It includes not just the answer
    but all the metadata needed for evaluation and debugging.
    """
    # The answer
    answer: str = Field(description="The generated answer")
    citations: list[Citation] = Field(default_factory=list, description="Sources referenced in the answer")
    confidence: Optional[float] = Field(default=None, description="Confidence score 0-1 (added in Part B)")

    # Metadata for evaluation and debugging
    query: str = Field(description="The original question")
    prompt_version: str = Field(description="Which prompt config produced this answer")
    model: str = Field(description="LLM model used")
    retrieval_strategy: str = Field(default="semantic", description="Retrieval method used")

    # Performance
    retrieval_latency_ms: float = Field(default=0, description="Time to retrieve chunks")
    generation_latency_ms: float = Field(default=0, description="Time to generate answer")
    total_latency_ms: float = Field(default=0, description="Total end-to-end time")

    # Context (for debugging)
    chunks_retrieved: int = Field(default=0, description="Number of chunks fetched")
    sources_used: list[str] = Field(default_factory=list, description="Unique source files used")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class IngestResponse(BaseModel):
    """Response from the document ingestion endpoint."""
    status: str
    documents_processed: int = 0
    chunks_created: int = 0
    chunks_stored: int = 0
    elapsed_seconds: float = 0
    prompt_version: str = ""