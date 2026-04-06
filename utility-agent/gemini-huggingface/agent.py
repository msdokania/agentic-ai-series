"""
agent.py — Agent loop using the Gemini API (google-generativeai) or HuggingFace.

Install:  pip install google-generativeai gradio pytz
Key:      export GEMINI_API_KEY=your_key_here

How Gemini tool calling works vs the others:
  - You declare tools as Python functions directly — no JSON schema needed
  - Gemini reads the function's docstring and type hints to understand it
  - When the model wants to call a tool, it returns a FunctionCall part
  - You execute the function and send back a FunctionResponse part
  - finish_reason "STOP" = done, "FUNCTION_CALL" doesn't exist as a string —
    instead you check whether any Part in the response is a function_call
"""

import os
import google.generativeai as genai

from llm import LLM
from tools import TOOL_FUNCTIONS, GEMINI_TOOLS, TOOLS_OPENAI_FORMAT
from prompts import SYSTEM_PROMPT

MAX_STEPS = 6


def run_agent(user_message: str, history: list[dict], backend: str = "huggingface"):
    """
    Run one full agent turn, yielding status + final text as strings.
 
    Args:
        user_message: New message from the user.
        history:      Previous turns. Format varies by backend:
                        Gemini: [{"role": "user"|"model", "parts": ["..."]}]
                        HF:     [{"role": "user"|"assistant", "content": "..."}]
        backend:      "gemini" or "huggingface"
 
    Yields:
        str — progress updates and the final answer.
    """
    llm = LLM(backend=backend)
 
    if backend == "gemini":
        yield from _run_gemini(llm, user_message, history)
    else:
        yield from _run_hf(llm, user_message, history)


# ── Gemini loop ────────────────────────────────────────────────────────────
 
def _run_gemini(llm: LLM, user_message: str, history: list):
    """
    Gemini-specific loop.
 
    Uses start_chat so Gemini manages history internally.
    Tool results go back as FunctionResponse parts.
    """
    chat = llm.start_gemini_chat(
        history=history,
        tools=GEMINI_TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )
 
    # First call: send the user message
    result = llm.chat(messages=user_message, tools=None, gemini_chat=chat)
 
    for step in range(MAX_STEPS):
        if result["type"] == "text":
            yield result["content"]
            return
 
        # Build FunctionResponse parts for each tool call
        tool_response_parts = []
        for call in result["calls"]:
            yield f"_> [{backend_label('gemini')}] Calling **{call['name']}** with `{call['args']}`…_\n\n"
 
            output = _dispatch(call["name"], call["args"])
 
            tool_response_parts.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=call["name"],
                        response={"result": output},
                    )
                )
            )
 
        # Send all tool results back in one message
        result = llm.chat(messages=tool_response_parts, tools=None, gemini_chat=chat)
 
    yield f"Reached the maximum of {MAX_STEPS} steps."


# ── HuggingFace loop ───────────────────────────────────────────────────────
 
def _run_hf(llm: LLM, user_message: str, history: list):
    """
    HuggingFace-specific loop.
 
    Stateless: we manually build and extend the messages list each step.
    Tool results go back as role="tool" messages.
    """
    print("Running hf agent...")
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + list(history)
        + [{"role": "user", "content": user_message}]
    )
 
    for _ in range(MAX_STEPS):
        result = llm.chat(messages=messages, tools=TOOLS_OPENAI_FORMAT, gemini_chat=None)
 
        if result["type"] == "text":
            print("Yielding text result..")
            yield result["content"]
            return
 
        print("HF final response: ", result)
        # Record the assistant's turn (must include the raw tool_calls block
        # so HF can correlate tool_use_id values)
        raw_msg = result["_raw_msg"]
        messages.append({
            "role":       "assistant",
            "content":    raw_msg.content,
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_msg.tool_calls
            ],
        })
 
        # Execute each tool and append results
        for call in result["calls"]:
            yield f"_> [{backend_label('huggingface')}] Calling **{call['name']}** with `{call['args']}`…_\n\n"
            output = _dispatch(call["name"], call["args"])
            messages.append({
                "role":         "tool",
                "tool_call_id": call["id"],
                "content":      output,
            })
        
    print("My messages: ", messages)
 
    yield f"Reached the maximum of {MAX_STEPS} steps."
 
 
# ── Shared helpers ─────────────────────────────────────────────────────────
 
def _dispatch(tool_name: str, args: dict) -> str:
    """Call the named tool function and return its string result."""
    if tool_name in TOOL_FUNCTIONS:
        return TOOL_FUNCTIONS[tool_name](**args)
    return f"Error: unknown tool '{tool_name}'"
 
 
def backend_label(backend: str) -> str:
    return "Gemini" if backend == "gemini" else "HF"