"""
FastAPI Application — Research Assistant API.

Endpoints:
  POST /ingest          Upload and ingest PDF documents
  POST /query           Ask a question over ingested documents
  GET  /documents       List ingested document stats
  GET  /health          Health check
  POST /reset           Clear all ingested data (use with caution)

The API is intentionally simple in Part A. Part D will add:
- SSE streaming for real-time agent reasoning
- WebSocket support
- Agent status endpoints
"""

import shutil
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rich.console import Console

from src.config.settings import settings
from src.config.prompt_loader import load_prompt_config
from src.ingestion.ingest import ingest as run_ingestion
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator
from src.generation.schemas import QueryResponse, IngestResponse
from src.ingestion.vectorstore import VectorStore

console = Console()

app = FastAPI(
    title="Research Assistant — Deep RAG",
    description="Phase 4 of the Agentic AI Series: Document research with rigorous retrieval",
    version="0.1.0 (Part A — Naive Baseline)",
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared instances ────────────────────────────────────────
# Initialized lazily on first use to avoid loading on import
_retriever: Optional[Retriever] = None
_generator: Optional[Generator] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


# ── Request Models ──────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None  # Defaults to prompt config value
    source_filter: Optional[str] = None  # Filter by source filename


# ── Endpoints ───────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check with vector store status."""
    store = VectorStore()
    stats = store.get_stats()
    return {
        "status": "healthy",
        "vector_store": stats,
        "prompt_version": settings.active_prompt_version,
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(files: list[UploadFile] = File(...)):
    """
    Upload and ingest PDF documents.
    
    Accepts one or more PDF files. Each file is:
    1. Extracted (text from PDF)
    2. Chunked (split into token-sized pieces)
    3. Embedded (converted to vectors)
    4. Stored (saved in ChromaDB)
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Save uploaded files to a temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    try:
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Only PDF files are accepted. Got: {file.filename}"
                )
            dest = temp_dir / file.filename
            with open(dest, "wb") as f:
                content = await file.read()
                f.write(content)

        # Run ingestion pipeline
        result = run_ingestion(str(temp_dir), reset=False)

        return IngestResponse(
            status=result.get("status", "unknown"),
            documents_processed=result.get("documents_processed", 0),
            chunks_created=result.get("chunks_created", 0),
            chunks_stored=result.get("chunks_stored", 0),
            elapsed_seconds=result.get("elapsed_seconds", 0),
            prompt_version=result.get("prompt_version", ""),
        )
    finally:
        # Cleanup temp files
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Ask a question over the ingested document corpus.
    
    The pipeline:
    1. Embed the question
    2. Retrieve the top-k most relevant chunks
    3. Generate an answer grounded in the retrieved context
    4. Return the answer with citations and metadata
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    retriever = get_retriever()
    generator = get_generator()

    # Load config for top_k default
    config = load_prompt_config()
    top_k = request.top_k or config.top_k

    # Optional source filter
    where = None
    if request.source_filter:
        where = {"source_file": request.source_filter}

    console.print(f"\n[bold]Query:[/bold] {request.question}")

    # Step 1: Retrieve
    retrieval = retriever.retrieve(
        query=request.question,
        top_k=top_k,
        where=where,
    )

    # Step 2: Generate
    response = generator.generate(
        query=request.question,
        retrieval=retrieval,
    )

    return response


@app.get("/documents")
async def list_documents():
    """List statistics about the ingested document corpus."""
    store = VectorStore()
    stats = store.get_stats()
    return {
        "total_chunks": stats["total_chunks"],
        "collection_name": stats["collection_name"],
    }


@app.post("/reset")
async def reset_store():
    """Clear all ingested documents. Use with caution."""
    global _retriever, _generator
    store = VectorStore()
    store.reset()
    _retriever = None
    _generator = None
    return {"status": "reset", "message": "Vector store cleared"}


# ── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )