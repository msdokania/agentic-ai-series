# 01 — Utility Agent

**A minimal AI agent implemented from first principles. No frameworks.**

---

## Purpose

This project is designed to make the core mechanics of an AI agent fully visible and easy to reason about.

By keeping the implementation minimal and contained in a single file, you can:
	•	Read and understand the entire agent in one sitting
	•	Observe how tool calling works end-to-end
	•	See how state, memory, and reasoning are handled explicitly
	•	Build intuition before introducing frameworks or higher-level abstractions

The two tools are intentionally trivial — a timezone lookup and a calculator.

---

## How an agent works

At its core, an agent follows a simple control loop:

```
1. Receive user input
2. Construct messages (system prompt + conversation history)
3. Send messages to the LLM
4. Check the model’s response:
   - If it requests a tool → execute the tool
   - Append tool output to messages
   - Repeat from step 2
   - If it returns a final answer → return it to the user
```

Everything in modern agent frameworks is ultimately an abstraction over this loop.

---

## Project structure

```
utility-agent/
└── app_gemini.py          # The entire agent using Gemini model — tools, loop, and Gradio UI
└── app_openai.py          # The entire agent using OpenAI API — tools, loop, and Gradio UI
└── gemini-huggingface.py  # Demonstrating the working through a Propriety model vs via Inference Clients
   └── app.py         
   └── agent.py          
   └── llm.py       
   └── tools.py      
   └── prompts.py          
```

---

## Tools

| Tool | What it does |
|------|-------------|
| `get_current_time(timezone)` | Returns current date/time for any IANA timezone |
| `calculator(expression)` | Evaluates a safe arithmetic expression |

---

## Setup

```bash
pip install openai gradio pytz
export OPENAI_API_KEY=sk-...
python app_openai.py
```

Then open `http://localhost:7860` in your browser.

---

## Things worth reading in the code

- **`TOOLS_SCHEMA`** — this is how you describe a tool to the model. The description is what the model reads to decide *when* to call it. Wording matters.
- **`run_agent()`** — the loop. Read the `finish_reason` check: `"stop"` means text answer, `"tool_calls"` means the model wants to run something.
- **`messages.append({...})`** — "memory" is literally just this list. Every turn, you rebuild it from scratch and pass it to the API. There is no hidden state.

---

## Try these prompts

- `What time is it in Tokyo and New York right now?`
- `What is (1337 * 42) + 100?`
- `What time will it be in London in 3 hours?` ← requires the model to reason + calculate