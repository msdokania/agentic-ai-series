"""
app.py — A complete AI agent in one file, explained line by line.

Run it:
    pip install google-generativeai gradio pytz
    export GEMINI_API_KEY=your_key_here
    python3 app_gemini.py

What this file teaches:
    1. What a "tool" actually is (just a Python function)
    2. What an "agent loop" actually is (just a while loop)
    3. What "memory" actually is (just a list)
    4. What "tool calling" actually is (just a Python function, model outputs a name+args, you run it)
"""

import os
import re
import datetime
import pytz
import gradio as gr
import google.generativeai as genai


# =============================================================================
# PART 1: TOOLS
#
# A tool is just a Python function. Nothing special about it.
# The only rule: it must return a string (or something you can turn into one).
#
# For Gemini specifically, the docstring is important — Gemini reads it to
# understand what the tool does and when to use it.
# =============================================================================

def get_current_time(timezone: str) -> str:
    """Returns the current date and time in the given timezone.

    Args:
        timezone: A valid timezone name like 'America/New_York' or 'Asia/Tokyo'.
    """
    try:
        tz = pytz.timezone(timezone)
        return datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return f"Error: {e}"


def calculator(expression: str) -> str:
    """Evaluates a basic math expression like '2 + 2' or '100 * 3.5'.

    Args:
        expression: A math expression using +, -, *, /. Parentheses are fine.
    """
    # Safety: only allow digits and math operators, never raw eval on user input
    if not re.fullmatch(r"[\d\s\.\+\-\*/\(\)]+", expression):
        return "Error: only basic arithmetic allowed"
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


# This dict maps tool names (strings) to the actual functions above.
# When Gemini says "call get_current_time", we look it up here and call it.
TOOLS = {
    "get_current_time": get_current_time,
    "calculator":       calculator,
}


# =============================================================================
# PART 2: THE SYSTEM PROMPT
#
# Tells the model what it is, what it can do, and how to behave.
# =============================================================================

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
Use a tool when you need real data. Don't guess answers you can look up.
Be concise in your final answers."""


# =============================================================================
# PART 3: THE AGENT LOOP
#
# This is the entire brain.
#
# The loop:
#   1. Send the user's message to the model
#   2. Did the model want to call a tool?
#      YES → run the tool, send the result back, go to step 1
#      NO  → the model gave a final answer, we're done
#
# =============================================================================

def run_agent(user_message: str, history: list):
    """
    Run one full conversation turn and return the final response.

    Args:
        user_message: What the user just typed.
        history:      Previous turns as Gemini expects them:
                      [{"role": "user"|"model", "parts": ["text"]}, ...]
    """
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
        tools=list(TOOLS.values()),   # pass the actual functions — Gemini reads the docstrings
    )

    # start_chat keeps the conversation history internally so we don't have
    # to manually append every message ourselves
    chat = model.start_chat(history=history)

    # Send the user's message — this is the first LLM call
    response = chat.send_message(user_message)

    # ── The loop ──────────────────────────────────────────────────────────
    for step in range(6):   # max 6 tool calls before giving up
        print("Step: ", step)
        print("Model response: ", response.candidates)

        # Look through the response for any tool call the model wants to make
        tool_calls = [
            part.function_call
            for candidate in response.candidates
            for part in candidate.content.parts
            if part.function_call.name   # empty name = not a tool call
        ]

        # No tool calls means the model gave us a plain text final answer
        if not tool_calls:
            print("No tool call, returning text..")
            yield response.text
            return

        print("Getting bare texts..")
        texts = [
            part.text
            for candidate in response.candidates
            for part in candidate.content.parts
            if part.text
        ]
        print("Texts: ", texts[0])
        yield "".join(texts[0])

        # There are tool calls then execute each one
        results = []
        for fc in tool_calls:
            tool_name = fc.name
            tool_args = dict(fc.args)

            print(f"  → calling {tool_name}({tool_args})") 

            # Look up and call the actual Python function
            if tool_name in TOOLS:
                output = TOOLS[tool_name](**tool_args)
                print(f"  → Tool output: '{output}") 
            else:
                output = f"Error: unknown tool '{tool_name}'"

            # Package the result in the format Gemini expects
            results.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=tool_name,
                        response={"result": output},
                    )
                )
            )
        print(f"  → Tools output: '{results}") 
        # Send all tool results back to the model and loop again
        # The model will now either call more tools OR give a final answer
        response = chat.send_message(results)

    return "Sorry, I couldn't finish in time."   # only hit if loop runs out


# =============================================================================
# PART 4: THE GRADIO UI
#
# Gradio handles the web interface. Our only job here is:
#   1. Convert Gradio's history format to Gemini's format
#   2. Call run_agent()
#   3. Return the response string
#
# Everything else (drawing the chatbox, storing messages) Gradio does for us.
# =============================================================================

def _to_gemini_history(gradio_history: list) -> list:
    """
    Gradio gives us history in its own format. Gemini wants a different format.
    This function converts between them.

    Gradio format (new):  [{"role": "user"|"assistant", "content": str|list}, ...]
    Gemini format (<v4):        [{"role": "user"|"model",     "parts": [str]}, ...]

    Two differences:
      - Gradio says "assistant", Gemini says "model"
      - Gradio content can be a list (multimodal), Gemini parts is always a list of str
    """
    gemini_history = []

    for entry in gradio_history:
        # Handle both old Gradio version format [[user,assistant],...] and new dict format
        if isinstance(entry, dict):
            role    = entry.get("role", "")
            content = entry.get("content", "")
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            # old format — entry[0] is user msg, entry[1] is assistant msg
            # add them as two separate turns
            if entry[0]:
                gemini_history.append({"role": "user",  "parts": [str(entry[0])]})
            if entry[1]:
                gemini_history.append({"role": "model", "parts": [str(entry[1])]})
            continue
        else:
            continue

        # Flatten content to a plain string (handles multimodal list content)
        if isinstance(content, list):
            text = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )
        else:
            text = str(content or "")

        if not text.strip():
            continue

        gemini_role = "model" if role == "assistant" else role
        gemini_history.append({"role": gemini_role, "parts": [text]})

    return gemini_history


def chat(user_message: str, gradio_history: list) -> str:
    history = _to_gemini_history(gradio_history)
    print("user_message: ", user_message)
    print("history: ", history)
    yield from run_agent(user_message, history)
    # return run_agent(user_message, history)


# =============================================================================
# PART 5: LAUNCH
# =============================================================================

if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=chat,
        title="My Agent",
        description="A bare-bones agent. Try: 'What time is it in Tokyo?' or 'What is 1337 * 42?'",
        examples=[
            "What time is it in London?",
            "Calculate (100 + 50) * 3",
            "What time is it in Tokyo and New York right now?",
        ],
    )
    demo.launch(debug=True)