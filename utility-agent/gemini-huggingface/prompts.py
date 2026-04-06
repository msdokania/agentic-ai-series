
# ---------------------------------------------------------------------------
# Main agent system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful assistant with access to tools.

Rules:
- Use a tool when you need real data (current time, calculations, search results).
- Do NOT guess or make up numbers. If you are uncertain, say so.
- When you have a final answer, state it directly and concisely.
- If a task requires multiple tool calls, do them one at a time.
"""


# ---------------------------------------------------------------------------
# Optional: a stricter "research" variant for tasks that need more rigour
# ---------------------------------------------------------------------------

RESEARCH_SYSTEM_PROMPT = """\
You are a rigorous research assistant with access to tools.

Rules:
- Always verify facts with at least one tool call before stating them.
- Cite which tool call produced each piece of information.
- If tools return conflicting information, say so explicitly.
- Structure your final answer with: Summary, Key Facts, Sources.
"""


# ---------------------------------------------------------------------------
# Helper: build the tool description block that gets injected into the prompt
#
# so we can see exactly what the LLM receives.
# ---------------------------------------------------------------------------

def build_tool_description(tools: list[dict]) -> str:
    """
    Given the TOOLS list from tools.py, return a human-readable description
    block suitable for injecting into a system prompt.

    Example output:
        Available tools:
        - get_current_time_in_timezone(timezone): Returns the current local time...
        - calculator(expression): Evaluates a basic arithmetic expression...

    Note: Many agent frameworks do this automatically under the hood. We have done manually here
    """
    lines = ["Available tools:"]
    for tool in tools:
        # Get the first parameter name for the signature hint
        params = list(tool["input_schema"]["properties"].keys())
        sig = ", ".join(params)
        lines.append(f"  - {tool['name']}({sig}): {tool['description']}")
    return "\n".join(lines)