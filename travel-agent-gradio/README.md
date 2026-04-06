# 02 — Travel Agent (Gradio)

**The same agent loop, extended with real tools, external data, and domain-specific reasoning.**

---

## Purpose

A travel planning agent built on the same pattern as Project 1. What changed is the surface area: more tools, external APIs, and a domain where the model needs to make real decisions about which tool to call and when.

The UI is Gradio — deliberately minimal.
Goal:
- Demonstrate how the same agent loop scales to a practical use case
- Show how multiple tools interact in a single reasoning flow
- Explore prompt engineering for reliable tool usage
- Handle structured and unstructured outputs
- Work with external data sources (APIs + web search)

---

## What's new compared to Project 1

**Real external data.** The agent doesn't just call Python functions with hardcoded data — it makes live HTTP requests (weather), calls a search API (Tavily), and combines results from multiple tool calls in a single response.

**The system prompt does real work.** With two tools, prompt engineering is simple. With five tools, you have to explicitly tell the model *when* to use each one — otherwise it guesses, and it guesses wrong. Read `SYSTEM_PROMPT` in the code to see how this is handled.

**Multi-tool turns.** For a query like "Plan me 5 days in Tokyo with a budget estimate", the model calls `generate_itinerary`, then `estimate_budget`, then synthesises both into one response. The loop handles this naturally — each tool result is appended to the message history and the model decides whether it needs more information.

---

## Project structure

```
travel-agent-gradio/
└── agent.py      # Tools, schemas, agent loop, Gradio UI
```

---

## Tools

| Tool | Data source | What it does |
|------|-------------|-------------|
| `get_destinations(vibe)` | Hardcoded | Curated destination list filtered by trip type |
| `get_destination_info(destination)` | Tavily (web search) | Climate, highlights, visa info — searched live |
| `get_weather(city)` | Open-Meteo (free) | Current conditions + 5-day forecast |
| `generate_itinerary(destination, days, style)` | Model knowledge | Structured day-by-day plan |
| `estimate_budget(destination, days, style)` | Tavily (web search) | Current cost estimates searched live |

---

## Setup

```bash
pip install openai gradio tavily-python pytz
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...    # Free at app.tavily.com — 1000 searches/month
python agent.py
```

Open-Meteo (weather) needs no API key.

---

## Things worth reading in the code

- **`generate_itinerary()`** — this tool doesn't return an itinerary. It returns a *structured prompt* that tells the model how to format one. The model then writes the actual content from its own knowledge. This pattern (tool sets structure, model fills content) is useful whenever your content is too dynamic to hardcode.
- **`SYSTEM_PROMPT`** — compare this to Project 1's prompt. Notice how much more explicit it is about which tool to call in which situation. This is what prevents the model from hallucinating visa requirements instead of searching for them.
- **`search_web()` vs `get_destination_info()`** — two tools that both search, but for different purposes. The specific tool gives the model better framing for the result; the general tool gives it flexibility.
- **`MODE`** — Try between structured and unstructured responses and use Teaching mode to see incremental output.
- **`temperature_slider`** — play around with the slider input to see how model responses changes with different temperature values. Observe when model hallucinates and how it can be prevented.

---

## Try these prompts

- `I want a beach holiday for 7 days, mid-range budget — any suggestions?`
- `Plan me 5 days in Kyoto focused on food and temples`
- `What's the weather in Istanbul right now and is it a good time to visit?`
- `Do I need a visa for Japan with a UK passport?`
- `How much would 10 days in Barcelona cost for two people?`
