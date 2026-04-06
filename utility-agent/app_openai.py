"""
my_agent.py — A complete AI agent in one file, using OpenAI.

Run it:
    pip install openai gradio pytz
    export OPENAI_API_KEY=sk-...
    python3 app_openai.py
"""

import os
import re
import json
import datetime
import pytz
import gradio as gr
from openai import OpenAI


# =============================================================================
# PART 1: TOOLS — the actual Python functions
# =============================================================================

def get_current_time(timezone: str) -> str:
    """Returns the current date and time in the given timezone."""
    try:
        tz = pytz.timezone(timezone)
        return datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return f"Error: {e}"


def calculator(expression: str) -> str:
    """Evaluates a basic math expression like '2 + 2' or '100 * 3.5'."""
    if not re.fullmatch(r"[\d\s\.\+\-\*/\(\)]+", expression):
        return "Error: only basic arithmetic allowed"
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"

TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculator":       calculator,
}


# =============================================================================
# PART 2: TOOL SCHEMAS
#
# Gemini:  you pass the actual function objects → it reads the docstrings
# OpenAI:  you write a JSON schema describing each tool explicitly
#
# The schema tells the model: what is the tool called, what does it do,
# and what arguments does it take. Same information as a docstring,
# just in a structured format the API can reliably parse.
# =============================================================================

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Returns the current date and time in a given timezone. "
                "Use this when the user asks what time it is somewhere."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "A valid IANA timezone, e.g. 'America/New_York' or 'Asia/Tokyo'.",
                    }
                },
                "required": ["timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluates a basic arithmetic expression. Use for any calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression like '100 * 3.5' or '(10 + 5) / 3'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


# =============================================================================
# PART 3: SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
Use a tool when you need real data. Don't guess answers you can look up.
Be concise in your final answers."""


# =============================================================================
# PART 4: THE AGENT LOOP
#
# The logic:
#   1. Call the model
#   2. Tool call? → run it, feed result back, loop
#   3. No tool call? → return the text
#
# =============================================================================

def run_agent(user_message: str, history: list):
    """
    Run one full conversation turn, yielding text as it becomes available.

    Args:
        user_message: What the user just typed.
        history:      Previous turns in OpenAI format:
                      [{"role": "user"|"assistant", "content": "..."}, ...]
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # OpenAI is stateless — unlike Gemini's start_chat which remembers history
    # here YOU pass the entire conversation every single call.
    #
    # Structure:
    #   system message  ← sets the personality / rules
    #   + history       ← all previous turns
    #   + new user msg  ← what was just typed
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_message},
    ]

    for step in range(10):
        print(f"Step: {step}")

        # ── Call the model ────────────────────────────────────────────────
        response = client.chat.completions.create(
            model="gpt-4o-mini",      
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",       # the model decides when to use tools
        )

        msg           = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        print(f"response msg: {msg}")
        print(f"finish_reason: {finish_reason}")

        # ── No tool call → final answer ───────────────────────────────────
        # "stop"       = model finished normally, produced text
        # "tool_calls" = model wants to call one or more tools
        if finish_reason == "stop":
            print("No tool call, returning text..")
            yield msg.content
            return

        # ── Tool calls then run them and loop ────────────────────────────────
        if finish_reason == "tool_calls":

            # If the model wrote any reasoning text before the tool call, yield it to chat
            if msg.content:
                print(f"Reasoning text: {msg.content}")
                yield msg.content

            # Step A: add the assistant's message (with tool_calls) to history.
            # OpenAI needs to see the tool_calls block before it can accept
            # the tool_result messages below.
            messages.append({
                "role":       "assistant",
                "content":    msg.content,
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,  # JSON string
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            # Step B: run each tool and append the result
            for tc in msg.tool_calls:
                tool_name = tc.function.name

                # arguments comes back as a JSON string. parse it to a dict
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                print(f"  → calling {tool_name}({tool_args})")

                if tool_name in TOOL_FUNCTIONS:
                    output = TOOL_FUNCTIONS[tool_name](**tool_args)
                else:
                    output = f"Error: unknown tool '{tool_name}'"

                print(f"  → tool output: '{output}'")

                # Tool results go back as role="tool" messages.
                # tool_call_id must match the id above so OpenAI can pair them.
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      output,
                })

            # Loop — model reads tool results and either calls more tools
            # or gives a final text answer

    yield "Sorry, I couldn't finish in time."


# =============================================================================
# PART 5: THE GRADIO UI
#
# OpenAI uses role="assistant" —
# =============================================================================

def _to_openai_history(gradio_history: list) -> list:
    """Convert Gradio history to OpenAI messages format."""
    messages = []

    for entry in gradio_history:
        # New Gradio format — dict
        if isinstance(entry, dict):
            role    = entry.get("role", "")
            content = entry.get("content", "")
        # Old Gradio format — [user_msg, assistant_msg]
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            if entry[0]:
                messages.append({"role": "user",      "content": str(entry[0])})
            if entry[1]:
                messages.append({"role": "assistant", "content": str(entry[1])})
            continue
        else:
            continue

        # Flatten list content (multimodal) to plain string
        if isinstance(content, list):
            text = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )
        else:
            text = str(content or "")

        if text.strip() and role in ("user", "assistant"):
            messages.append({"role": role, "content": text})

    return messages


def chat(user_message: str, gradio_history: list):
    history = _to_openai_history(gradio_history)
    print("user_message: ", user_message)
    print("history: ", history)
    yield from run_agent(user_message, history)


# =============================================================================
# PART 6: LAUNCH
# =============================================================================

if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=chat,
        title="My Agent (OpenAI)",
        description="A bare-bones agent. Try: 'What time is it in Tokyo?' or 'What is 1337 * 42?'",
        examples=[
            "What time is it in London?",
            "Calculate (100 + 50) * 3",
            "What time is it in Tokyo and New York right now?",
        ],
    )
    demo.launch(debug=True)