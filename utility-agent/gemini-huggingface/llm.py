"""
llm.py — A unified LLM interface that supports both Gemini and HuggingFace.

Usage in agent.py:
    from llm import LLM
    llm = LLM(backend="gemini")   # or backend="huggingface"
    response = llm.chat(messages, tools)

This file is the ONLY place that knows about backend differences.
agent.py just calls llm.chat() and gets back a standard response dict:
    {
        "type": "text",           # model gave a final answer
        "content": "..."
    }
    OR
    {
        "type": "tool_calls",     # model wants to call tools
        "calls": [
            {"id": "...", "name": "...", "args": {...}},
            ...
        ]
    }

The above is the entire contract between llm.py and agent.py.
Everything backend-specific is hidden inside this file.
"""

import os
import json


# ── Backend config ─────────────────────────────────────────────────────────

GEMINI_MODEL    = "gemini-2.5-flash"
# HF_MODEL        = "Qwen/Qwen2.5-Coder-32B-Instruct"
HF_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


# ── Public interface ───────────────────────────────────────────────────────

class LLM:
    """
    A thin, backend-agnostic LLM wrapper.

    Args:
        backend: "gemini" or "huggingface"

    The same LLM object is reused across all turns of the agent loop.
    For Gemini, start_chat() is called once per user turn (in run_agent),
    so the chat session lives in the agent, not here.
    For HuggingFace, each call is stateless — history is passed explicitly.
    """

    def __init__(self, backend: str = "gemini"):
        if backend not in ("gemini", "huggingface"):
            raise ValueError(f"backend must be 'gemini' or 'huggingface', got '{backend}'")
        self.backend = backend
        self._client = None  

    # ── Main method called by agent.py ────────────────────────────────────

    def chat(self, messages: list, tools: list, gemini_chat=None) -> dict:
        """
        Send messages to the LLM and return a normalised response.

        Args:
            messages:     Conversation so far (Gemini or OpenAI format
                          depending on backend — see agent.py).
            tools:        Tool definitions (format also varies by backend).
            gemini_chat:  For Gemini only — the active ChatSession object.
                          Pass None for HuggingFace (ignored).

        Returns:
            {"type": "text", "content": str}
            OR
            {"type": "tool_calls", "calls": [{"id", "name", "args"}, ...]}
        """
        if self.backend == "gemini":
            return self._chat_gemini(gemini_chat, messages)
        else:
            return self._chat_hf(messages, tools)

    def start_gemini_chat(self, history: list, tools: list, system_prompt: str):
        """
        Create a Gemini ChatSession. Called once per user turn by agent.py.
        Returns the chat object + the send function to kick off the loop.
        Only relevant for Gemini — HuggingFace doesn't use this.
        """
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system_prompt,
            tools=tools,
        )
        return model.start_chat(history=history)

    # ── Gemini internals ───────────────────────────────────────────────────

    def _chat_gemini(self, chat, messages) -> dict:
        """
        Send messages via an existing Gemini ChatSession and normalise response.
        `messages` here is either a string (first turn) or a list of Parts
        (tool results) — both are valid inputs to chat.send_message().
        """
        response = chat.send_message(messages)

        # Collect function_call parts. You can print the response and observe the different parts
        calls = []
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.function_call.name:   # empty name means not a tool call
                    calls.append({
                        "id":   part.function_call.name,  # Gemini has no call ID, use name
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args),
                    })

        if calls:
            return {"type": "tool_calls", "calls": calls}
        return {"type": "text", "content": response.text}

    # ── HuggingFace internals ──────────────────────────────────────────────

    def _get_hf_client(self):
        """Lazy-init the HF client so we don't import it when using Gemini."""
        if self._client is None:
            # from huggingface_hub import InferenceClient 
            # self._client = InferenceClient(
            #     provider="hf-inference",
            #     api_key=os.environ.get("HF_TOKEN"),
            # )
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=os.environ.get("HF_TOKEN"),
            )
        return self._client

    def _chat_hf(self, messages: list, tools: list) -> dict:
        """Send messages via HuggingFace client and normalise response."""
        client = self._get_hf_client()

        print("Chatting with HF Llama model now..")
        response = client.chat.completions.create(
            model=HF_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=1024,
            temperature=0.5,
        )

        # Print statements exist only for debugging and studying/understanding the responses
        print("HF model response: ", response.choices[0])
        msg           = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "tool_calls":
            calls = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                calls.append({
                    "id":   tc.id,
                    "name": tc.function.name,
                    "args": args,
                })
            return {"type": "tool_calls", "calls": calls, "_raw_msg": msg}

        return {"type": "text", "content": msg.content or ""}