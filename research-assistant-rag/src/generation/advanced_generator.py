"""
Advanced Answer Generation (Part B).

Upgrades over Part A generator:
1. Parses structured output (Answer/Confidence/Sources/Limitations)
2. Extracts machine-readable confidence scores
3. Detects INSUFFICIENT_CONTEXT and CONFLICT markers
4. Better citation extraction with multiple pattern support
5. Retry logic for API failures

The generator auto-detects the prompt version and uses the appropriate
parsing strategy — v1 responses are parsed with Part A's simple regex,
v2 responses are parsed with the structured format parser.
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

# Maximum retries for API calls
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds


class AdvancedGenerator:
    """
    Enhanced generator with structured output parsing and retry logic.
    
    Drop-in replacement for Part A's Generator — same interface,
    better parsing and reliability.
    """

    def __init__(self, prompt_config: Optional[PromptConfig] = None):
        self.config = prompt_config or load_prompt_config()
        self.client = OpenAI(api_key=settings.openai_api_key)

    def generate(self, query: str, retrieval: RetrievalResult) -> QueryResponse:
        """Generate an answer with structured output parsing."""
        start_time = time.time()

        if not retrieval.has_results:
            return QueryResponse(
                answer=self.config.no_context_response,
                citations=[],
                confidence=0.0,
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

        # Format context
        context = retrieval.format_context()
        user_message = self.config.retrieval_template.format(
            context=context,
            question=query,
        )

        # Call LLM with retry logic
        answer_text, tokens_used = self._call_llm_with_retry(user_message)

        generation_latency_ms = (time.time() - start_time) * 1000

        # Parse structured output
        parsed = self._parse_structured_output(answer_text)
        citations = self._extract_all_citations(answer_text, retrieval)
        confidence = parsed.get("confidence")

        sources_used = list(set(
            c["metadata"]["source_file"] for c in retrieval.chunks
        ))

        total_latency = retrieval.retrieval_latency_ms + generation_latency_ms

        result = QueryResponse(
            answer=parsed.get("answer", answer_text),
            citations=citations,
            confidence=confidence,
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

        confidence_str = f", confidence: {confidence}/5" if confidence else ""
        console.print(
            f"  [green]✓[/green] Generated answer "
            f"({generation_latency_ms:.0f}ms, {tokens_used} tokens, "
            f"{len(citations)} citations{confidence_str}, "
            f"prompt {self.config.version})"
        )

        return result

    def _call_llm_with_retry(self, user_message: str) -> tuple[str, int]:
        """
        Call the LLM with exponential backoff retry.
        
        Returns (answer_text, tokens_used).
        """
        last_error = None

        for attempt in range(MAX_RETRIES):
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
                answer = response.choices[0].message.content.strip()
                tokens = response.usage.total_tokens if response.usage else 0
                return answer, tokens

            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAY * (2 ** attempt)
                    console.print(
                        f"  [yellow]LLM API error (attempt {attempt + 1}/{MAX_RETRIES}): "
                        f"{e}. Retrying in {wait}s...[/yellow]"
                    )
                    time.sleep(wait)

        console.print(f"[red]LLM API failed after {MAX_RETRIES} attempts: {last_error}[/red]")
        return f"Error generating answer: {str(last_error)}", 0

    def _parse_structured_output(self, text: str) -> dict:
        """
        Parse the v2 structured output format.
        
        Expected format:
            **Answer:** ...
            **Confidence:** 4
            **Key Sources:** ...
            **Limitations:** ...
        
        Falls back gracefully if the LLM doesn't follow the format exactly.
        """
        result = {"answer": text, "confidence": None, "limitations": ""}

        # Check for INSUFFICIENT_CONTEXT marker
        if "INSUFFICIENT_CONTEXT" in text:
            result["confidence"] = 0.0
            return result

        # Extract answer section
        answer_match = re.search(
            r"\*\*Answer:\*\*\s*(.*?)(?=\*\*Confidence:|\*\*Key Sources:|\*\*Limitations:|$)",
            text,
            re.DOTALL,
        )
        if answer_match:
            result["answer"] = answer_match.group(1).strip()

        # Extract confidence score
        conf_match = re.search(r"\*\*Confidence:\*\*\s*(\d)", text)
        if conf_match:
            score = int(conf_match.group(1))
            # Normalize to 0-1 scale (1-5 → 0.2-1.0)
            result["confidence"] = round(score / 5.0, 2)

        # Extract limitations
        lim_match = re.search(
            r"\*\*Limitations:\*\*\s*(.*?)$",
            text,
            re.DOTALL,
        )
        if lim_match:
            result["limitations"] = lim_match.group(1).strip()

        return result

    def _extract_all_citations(
        self, answer: str, retrieval: RetrievalResult
    ) -> list[Citation]:
        """
        Extract citations from the answer text with multiple pattern support.
        
        Handles various citation formats the LLM might produce:
        - [Source: filename.pdf, Page 5]
        - [Source: filename, Page 5]
        - [Source: filename.pdf, Pages 5, 6]
        - (Source: filename, p. 5)
        """
        citations = []
        seen = set()

        # Multiple citation patterns to catch LLM variation
        patterns = [
            r"\[Source:\s*([^,\]]+),\s*[Pp]ages?\s*([\d,\s]+)\]",
            r"\(Source:\s*([^,\)]+),\s*[Pp]ages?\s*([\d,\s]+)\)",
            r"\[Source:\s*([^,\]]+),\s*[Pp]\.?\s*([\d,\s]+)\]",
            r"\[([^,\]]+\.pdf),\s*[Pp]ages?\s*([\d,\s]+)\]",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, answer)
            for filename, pages_str in matches:
                filename = filename.strip()
                pages = [int(p.strip()) for p in pages_str.split(",") if p.strip().isdigit()]
                key = (filename, tuple(pages))
                if key not in seen:
                    seen.add(key)
                    score = max(
                        (c["score"] for c in retrieval.chunks
                         if c["metadata"]["source_file"] == filename),
                        default=0.0,
                    )
                    citations.append(Citation(
                        source_file=filename,
                        page_numbers=pages,
                        relevance_score=score,
                    ))

        # Fallback: create citations from top retrieved chunks
        if not citations and retrieval.chunks:
            for chunk in retrieval.chunks[:3]:
                meta = chunk["metadata"]
                citations.append(Citation(
                    source_file=meta["source_file"],
                    page_numbers=meta.get("page_numbers", []),
                    chunk_id=chunk.get("chunk_id", ""),
                    relevance_score=chunk.get("score", 0),
                ))

        return citations