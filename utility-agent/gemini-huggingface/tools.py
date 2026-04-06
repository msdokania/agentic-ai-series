"""
tools.py — Tool functions with schemas for both Gemini and HuggingFace.
"""

import re
import datetime
import pytz


# ---------------------------------------------------------------------------
# 1. Python functions
#    These are plain functions. They know nothing about the agent or the LLM.
#    They just take inputs and return a string result.
#    For Gemini, the docstring IS the tool description the model sees.
# ---------------------------------------------------------------------------
 
def get_current_time_in_timezone(timezone: str) -> str:
    """Returns the current local date and time in a specified timezone.
 
    Use this whenever the user asks what time it is in a specific city or
    country. Always use a valid IANA timezone string as the argument.
 
    Args:
        timezone: An IANA timezone string, e.g. 'America/New_York',
                  'Europe/London', 'Asia/Tokyo', 'Australia/Sydney'.
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"Current time in {timezone}: {local_time}"
    except pytz.exceptions.UnknownTimeZoneError:
        return f"Error: '{timezone}' is not a valid IANA timezone."
    except Exception as e:
        return f"Error: {e}"
 
 
def calculator(expression: str) -> str:
    """Evaluates a basic arithmetic expression and returns the numeric result.
 
    Use for addition, subtraction, multiplication, division. Do not use
    for anything other than pure number arithmetic.
 
    Args:
        expression: A mathematical expression, e.g. '(100 + 50) * 3 / 2'
    """
    if not re.fullmatch(r"[\d\s\.\+\-\*/\(\)]+", expression):
        return "Error: expression contains disallowed characters."
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: {e}"
 
 
# ---------------------------------------------------------------------------
# 2a. GEMINI_TOOLS — just the function objects
#     Gemini reads the name, docstring, and type hints automatically.
#     No schema to write.
# ---------------------------------------------------------------------------
 
GEMINI_TOOLS = [
    get_current_time_in_timezone,
    calculator,
]
 
 
# ---------------------------------------------------------------------------
# 2b. TOOLS_OPENAI_FORMAT — explicit JSON schema for HuggingFace
#     Same information as the docstrings above, but written out manually.
#     This is what Gemini is inferring automatically from 2a.
# ---------------------------------------------------------------------------
 
TOOLS_OPENAI_FORMAT = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time_in_timezone",
            "description": (
                "Returns the current local date and time in a specified timezone. "
                "Use when the user asks what time it is somewhere. "
                "Requires a valid IANA timezone string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone, e.g. 'America/New_York', 'Asia/Tokyo'.",
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
            "description": (
                "Evaluates a basic arithmetic expression and returns the result. "
                "Use for +, -, *, /. Parentheses supported."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '(100 + 50) * 3'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]
 
 
TOOL_FUNCTIONS = {
    "get_current_time_in_timezone": get_current_time_in_timezone,
    "calculator": calculator,
}