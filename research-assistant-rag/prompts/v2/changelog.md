# Changelog — Prompt Version 2

## v2 — Advanced Retrieval (Part B)

**Date:** 2026-04-09

**Changes from v1:**

1. **Citation enforcement**: Changed from "indicate which source" (suggestion) to 
   "EVERY factual claim MUST include a citation" (requirement). Added explicit format.

2. **Confidence scoring**: Added a 1-5 confidence rubric with clear definitions for 
   each level. This enables programmatic evaluation of the model's self-assessment.

3. **Conflict detection**: Added explicit instruction to flag conflicting information 
   across sources instead of silently choosing one.

4. **Structured output format**: Required Answer/Confidence/Key Sources/Limitations 
   sections for reliable parsing.

5. **Reduced temperature**: 0.1 → 0.05 for more deterministic, reproducible output.

6. **Better no-context handling**: Added actionable suggestions (rephrase, add docs) 
   instead of just stating inability.

**Rationale:**
Part A failure analysis showed three main issues:
- Inconsistent citation format (sometimes cited, sometimes not)
- Hallucination on questions where context was adjacent but insufficient
- No way to programmatically assess answer quality

v2 addresses all three through stricter prompting and structured output.

**To measure:**
- Citation coverage: % of factual claims with a source reference
- Citation accuracy: Do the cited sources actually support the claims?
- Confidence calibration: Does confidence 5 correlate with correct answers?
- Hallucination rate on unanswerable questions: Should be lower than v1