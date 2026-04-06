"""
app.py — Gradio UI

Handles all Gradio history formats:
  - Old (< 4.x):   [[user_str, assistant_str], ...]
  - New (>= 4.x):  [{"role": "...", "content": str | list}, ...]
  - content as list: [{"type": "text", "text": "..."}, ...] (multimodal)
"""

import gradio as gr
from agent import run_agent


# ── Helpers ────────────────────────────────────────────────────────────────

def _content_to_str(content) -> str:
    """
    Safely convert any Gradio content value to a plain string.

    Gradio content can be:
      - str                            (normal text)
      - list of str                    (rare)
      - list of dicts like {"type": "text", "text": "..."}   (multimodal)
      - None
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or "")
        return "\n".join(p for p in parts if p)
    return str(content)


def _strip_tool_lines(content) -> str:
    """Convert content to str, then remove the _> status lines we inject."""
    text = _content_to_str(content)
    return "\n".join(
        line for line in text.split("\n")
        if not line.startswith("_>")
    ).strip()


def _gradio_to_hf(gradio_history: list) -> list:
    """Convert Gradio history (any format) → HF messages list."""
    messages = []

    for entry in gradio_history:
        # New Gradio format: dict with "role" and "content"
        if isinstance(entry, dict):
            role    = entry.get("role", "")
            content = entry.get("content")
            if role == "user":
                text = _content_to_str(content)
                if text:
                    messages.append({"role": "user", "content": text})
            elif role == "assistant":
                clean = _strip_tool_lines(content)
                if clean:
                    messages.append({"role": "assistant", "content": clean})

        # Old Gradio format: [user_str, assistant_str]
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            user_msg, assistant_msg = entry
            if user_msg:
                messages.append({"role": "user", "content": _content_to_str(user_msg)})
            if assistant_msg:
                clean = _strip_tool_lines(assistant_msg)
                if clean:
                    messages.append({"role": "assistant", "content": clean})

    return messages


def _hf_to_gemini(hf_messages: list) -> list:
    """Convert HF format → Gemini format ("assistant" role → "model")."""
    return [
        {
            "role":  "model" if m["role"] == "assistant" else m["role"],
            "parts": [m["content"]],
        }
        for m in hf_messages
    ]


# ── Chat handler ───────────────────────────────────────────────────────────

def chat(user_message: str, gradio_history: list, backend: str) -> str:
    print("user_message: ", user_message)
    print("gradio_history: ", gradio_history)
    hf_history = _gradio_to_hf(gradio_history)
    print("hf_history: ", hf_history)

    if backend == "gemini":
        history = _hf_to_gemini(hf_history)
        print("history: ", history)
    else:
        history = hf_history
    full_response = ""
    for chunk in run_agent(user_message, history, backend=backend):
        print("===========================")
        print("my chunk: ", chunk)
        print("===========================")
        full_response += chunk
    return full_response


# ── Gradio UI ──────────────────────────────────────────────────────────────

with gr.Blocks(title="Dual-backend Agent") as demo:
    gr.Markdown("## Agent — Gemini vs HuggingFace")
    gr.Markdown(
        "Switch backends with the toggle below. "
        "Both use the same tools and system prompt — only the LLM changes."
    )

    backend_selector = gr.Radio(
        choices=["gemini", "huggingface"],
        value="huggingface",
        label="Backend",
        info="Gemini = GEMINI_API_KEY  |  HuggingFace = HF_TOKEN",
    )

    chatbot = gr.ChatInterface(
        fn=chat,
        additional_inputs=[backend_selector],
        examples=[
            ["What time is it in Tokyo right now?", "gemini"],
            ["Calculate 1337 * 42",                 "huggingface"],
            ["What time is it in both London and New York?", "huggingface"],
        ],
    )

if __name__ == "__main__":
    demo.launch(debug=True)