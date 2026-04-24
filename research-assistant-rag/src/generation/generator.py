"""
Answer Generation — Chunks + Question → Cited Answer.

This module handles the generation step of RAG: given retrieved context
and a user question, produce a grounded answer with citations.

In Part A, this is straightforward: stuff the context into a prompt
template and call the LLM. But even in this simple version, we:
- Log the full prompt for debugging
- Track token usage and latency
- Parse citations from the free-text response
- Build a structured QueryResponse object

The generator is stateless — all context comes from the retriever.
This makes it easy to test, evaluate, and swap components.
"""

import time
import re
from typing import Optional
from openai import OpenAI
from rich.console import Console

from src.config.settings import settings
from src.config.prompt_loader import PromptConfig, load_prompt_config
from src.retrieval.retriever import RetrievalResult
from src.generation.schemas import QueryResponse, Citation

console = Console()


class Generator:
    """
    Generates answers from retrieved context using an LLM.
    
    The generation process:
    1. Format retrieved chunks into a context string
    2. Fill the prompt template with context + question
    3. Send to the LLM with the system prompt
    4. Parse the response into a structured QueryResponse
    5. Extract citations from the text
    """

    def __init__(self, prompt_config: Optional[PromptConfig] = None):
        """
        Args:
            prompt_config: Versioned prompt configuration to use.
                          Defaults to the active version from env config.
        """
        self.config = prompt_config or load_prompt_config()
        self.client = OpenAI(api_key=settings.openai_api_key)

    def generate(self, query: str, retrieval: RetrievalResult) -> QueryResponse:
        """
        Generate an answer for a query using retrieved context.
        
        Args:
            query: The user's original question.
            retrieval: RetrievalResult containing relevant chunks.
        
        Returns:
            QueryResponse with the answer, citations, and metadata.
        """
        start_time = time.time()

        # If no relevant chunks were found, return the no-context response
        if not retrieval.has_results:
            return QueryResponse(
                answer=self.config.no_context_response,
                citations=[],
                query=query,
                prompt_version=self.config.version,
                model=self.config.model,
                retrieval_strategy=retrieval.strategy,
                retrieval_latency_ms=retrieval.retrieval_latency_ms,
                generation_latency_ms=0,
                total_latency_ms=retrieval.retrieval_latency_ms,
                chunks_retrieved=0,
                sources_used=[],
            )

        # Step 1: Format context from retrieved chunks
        context = retrieval.format_context()

        # Step 2: Fill the prompt template
        user_message = self.config.retrieval_template.format(
            context=context,
            question=query,
        )

        # Step 3: Call the LLM
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            answer_text = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else 0
        except Exception as e:
            console.print(f"[red]LLM API error: {e}[/red]")
            answer_text = f"Error generating answer: {str(e)}"
            tokens_used = 0

        generation_latency_ms = (time.time() - start_time) * 1000

        # Step 4: Parse citations from the answer text
        citations = _extract_citations(answer_text, retrieval)

        # Step 5: Build the response
        sources_used = list(set(
            chunk["metadata"]["source_file"]
            for chunk in retrieval.chunks
        ))

        total_latency = retrieval.retrieval_latency_ms + generation_latency_ms

        result = QueryResponse(
            answer=answer_text,
            citations=citations,
            query=query,
            prompt_version=self.config.version,
            model=self.config.model,
            retrieval_strategy=retrieval.strategy,
            retrieval_latency_ms=retrieval.retrieval_latency_ms,
            generation_latency_ms=round(generation_latency_ms, 2),
            total_latency_ms=round(total_latency, 2),
            chunks_retrieved=len(retrieval.chunks),
            sources_used=sources_used,
        )

        console.print(
            f"  [green]✓[/green] Generated answer "
            f"({generation_latency_ms:.0f}ms, {tokens_used} tokens, "
            f"{len(citations)} citations, prompt {self.config.version})"
        )

        return result


def _extract_citations(answer: str, retrieval: RetrievalResult) -> list[Citation]:
    """
    Extract source citations from the LLM's answer text.
    
    Looks for patterns like:
    - [Source: filename.pdf, Page 5]
    - [Source: filename.pdf, Pages 5, 6]
    - [Source: filename.pdf, page 5]
    
    This is simple regex-based extraction for Part A.
    Part B will use structured output with LangChain's output parsers
    for more reliable citation extraction.
    
    Also creates citations from the retrieved chunks even if the LLM
    didn't explicitly cite them — this gives us the full picture of
    what context was available.
    """
    citations = []
    seen = set()

    # Pattern: [Source: filename, Page X] or [Source: filename, Pages X, Y]
    pattern = r"\[Source:\s*([^,\]]+),\s*[Pp]ages?\s*([\d,\s]+)\]"
    matches = re.findall(pattern, answer)

    for filename, pages_str in matches:
        filename = filename.strip()
        pages = [int(p.strip()) for p in pages_str.split(",") if p.strip().isdigit()]

        key = (filename, tuple(pages))
        if key not in seen:
            seen.add(key)

            # Find the relevance score from retrieved chunks
            score = 0.0
            for chunk in retrieval.chunks:
                if chunk["metadata"]["source_file"] == filename:
                    score = max(score, chunk["score"])

            citations.append(Citation(
                source_file=filename,
                page_numbers=pages,
                relevance_score=score,
            ))

    # If no citations were extracted from the text, create citations
    # from the top retrieved chunks (so we at least know what context was used)
    if not citations and retrieval.chunks:
        for chunk in retrieval.chunks[:3]:  # Top 3 chunks
            meta = chunk["metadata"]
            citations.append(Citation(
                source_file=meta["source_file"],
                page_numbers=meta.get("page_numbers", []),
                chunk_id=chunk.get("chunk_id", ""),
                relevance_score=chunk["score"],
            ))

    return citations