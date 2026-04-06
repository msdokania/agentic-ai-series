# Agentic AI Series

This repository presents a step-by-step progression of AI agent implementations built from first principles. It starts with a minimal, framework-free agent loop and gradually introduces additional complexity, tooling, and system design patterns.

Each project is intentionally scoped to a single concept, keeping the implementation small so the underlying ideas remain visible and easy to understand.

---

## Motivation

Many agent tutorials begin with high-level frameworks. While this enables quick results, it often abstracts away the core mechanics of how agents actually operate.

This repository takes the opposite approach:
- Start with a raw agent loop
- Understand how the LLM interacts with tools
- Gradually introduce structure, observability, and system design
- Only then layer in frameworks and full-stack architecture

The goal is to build intuition for how agents work internally.

---

## How to Use This Repository

Each project is self-contained and builds on the concepts introduced in the previous one. The recommended approach is to explore them in order.

Code across all projects is heavily commented to make behavior explicit and easier to follow.

---

## Projects

### [`01-utility-agent/`](./utility-agent/)
**Core agent loop from scratch**:
A minimal implementation of an AI agent in a single file with no external frameworks. It demonstrates the ReAct pattern using:
 Prompt → LLM call → tool invocation → observation → repeat loop

This project focuses on understanding:
- How an agent is fundamentally just a control loop around an LLM
- How tools are defined and invoked
- How different model providers (OpenAI, Google, Hugging Face) behave via their APIs
- How responses are structured and interpreted

### [`02-travel-agent-gradio/`](./travel-agent-gradio/)
**Agent with real-world tools and interaction**:
Extends the basic loop into a practical travel planning assistant. Introduces:
- Multiple tools (web search, weather, itinerary planning, budgeting)
- Multi-turn conversation handling
- Prompt engineering for more reliable outputs
- Structured vs unstructured responses
- Handling context across turns
- Temperature and its effect on output variability (Temperature slider)
- Basic UI using Gradio for interaction

### [`03-travel-agent-fullstack/`](./travel-agent-full/)
**Separation of backend and frontend**:
The same agent functionality is implemented with a more production-like architecture:
- Backend: FastAPI handling agent execution
- Frontend: React + Vite UI
- Clear separation of concerns between:
	- agent logic
	- API layer
	- user interface

This project highlights:
  System design considerations for deploying agents
  How to structure agent services in real applications
  Decoupling UI from inference logic
  Building scalable interfaces around agent systems

**Demonstration**: [Watch video](demo.mov)

---

## What's coming

- Phase 4: LangChain + LangGraph integration
- Phase 5: Memory and Retrieval-Augmented Generation (RAG)
- Phase 6: Multi-agent orchestration (planner, researcher, reporter) with evaluation frameworks
- Phase 7: Multimodal agents, MCP, and edge inference

---

## Stack

- **LLM** — OpenAI (`gpt-4o-mini` for development, swap to `gpt-4o` for production)
- **Web search** — Tavily (free tier: 1000 searches/month)
- **Weather** — Open-Meteo (free, no API key)
- **UI** — Gradio (Projects 1–2), React + Vite (Project 3)
- **Backend** — FastAPI (Project 3)

## Setup

Each project has its own `README.md` with specific setup instructions. All projects require Python 3.11+.

```bash
git clone https://github.com/msdokania/agentic-ai-series
cd agentic-ai-series
```

Then follow the README inside whichever project folder you want to run.
