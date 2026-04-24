# Changelog — Prompt Version 1

## v1 — Initial Baseline (Part A)

**Date:** 2026-04-08

**Changes:** Initial version.

**Rationale:** 
Establish a naive baseline with the simplest reasonable prompting strategy.
The system prompt asks the model to cite sources and stay grounded in context,
but does not enforce structured output or confidence scoring.

**Known limitations:**
- Citation format is suggested but not enforced
- No confidence scoring
- No mechanism to handle conflicting information across sources
- No special handling for "unanswerable" questions beyond a basic instruction

**To measure:**
- How often does the model hallucinate vs. admit insufficient context?
- How consistent are the citation formats?
- On what question types does retrieval fail to fetch relevant chunks?