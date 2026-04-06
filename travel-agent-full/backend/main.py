"""
travel_agent backend — FastAPI + SSE streaming
Run: uvicorn main:app --reload --port 8000
"""

import os
import json
import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

from tools import TOOL_FUNCTIONS, TOOLS_SCHEMA

SYSTEM_PROMPT = """You are an expert travel agent with deep knowledge of destinations worldwide and access to real-time web search.
Your job is to help users plan amazing trips by giving specific, current, and genuinely useful travel advice.

Tool usage guide:
- User asks for destination ideas → call get_destinations (with a vibe if they mentioned one)
- User picks a destination → call get_destination_info + get_weather (run both) to get the facts, then share them naturally
- User wants an itinerary → call generate_itinerary, then write the full plan
- User asks about cost/budget → call estimate_budget
- User asks what to pack → call get_packing_list
- User asks about visas, events, current prices, advisories → search_web with a specific query
- Anything time-sensitive or that might have changed → search_web first

Rules:
- Always check weather when discussing a specific destination trip
- Be specific: name real restaurants, neighbourhoods, landmarks
- Be warm, enthusiastic, and specific. Don't just list places — paint a picture.
- If you don't have enough info (days? travel style? interests?), ask before calling tools
- After giving an itinerary, proactively offer: budget estimate, weather, packing tips
- Never make up visa requirements or prices — search for them"""


# =============================================================================
# FastAPI app
# =============================================================================

app = FastAPI(title="Travel Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: list  # list of {role, content} dicts


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def agent_stream(user_message: str, history: list) -> AsyncGenerator[str, None]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_message},
    ]

    for step in range(10):
        # synchronous OpenAI call in a thread so as not to block the event loop
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )

        msg           = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "stop":
            yield _sse("message", {"content": msg.content})
            yield _sse("done", {})
            return

        if finish_reason == "tool_calls":
            if msg.content:
                yield _sse("message", {"content": msg.content})

            # Collect tool calls to send for display
            tool_calls_payload = [
                {"name": tc.function.name, "args": tc.function.arguments}
                for tc in msg.tool_calls
            ]
            yield _sse("tool_calls", {"calls": tool_calls_payload})

            messages.append({
                "role":       "assistant",
                "content":    msg.content,
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                if tool_name in TOOL_FUNCTIONS:
                    output = await asyncio.to_thread(TOOL_FUNCTIONS[tool_name], **tool_args)
                else:
                    output = f"[Tool '{tool_name}' not yet wired up — returning stub]"

                yield _sse("tool_result", {"name": tool_name, "result": str(output)[:300]})

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      str(output),
                })

    yield _sse("message", {"content": "Sorry, I couldn't finish planning your trip. Please try again."})
    yield _sse("done", {})


@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        agent_stream(req.message, req.history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}