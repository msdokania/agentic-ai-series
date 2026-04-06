# 03 — Travel Agent (Full-stack)

**Same agent. Proper architecture.**

---

## Purpose

This project focuses on system design around AI agents and deploys it with:
	•	Backend: FastAPI service exposing the agent over HTTP
	•	Frontend: React + Vite application with a chat interface

---

## Project structure

```
travel-agent-full/
├── backend/
│   ├── main.py              # FastAPI app — POST endpoint that runs the agent
│   ├── tools.py             # All tool functions
│   └── settings.py      
│
└── frontend/
    ├── src/
    │   ├── App.jsx       
    │   ├── main.jsx     
    │   ├── components/
    │   │   ├── Message.jsx
    │   └── hooks/
    │       └── useChat.js       # Single function that calls the backend
    ├── index.html
    └── vite.config.js
```

---

## What changed from Project 2

**Backend** — agent as a service

```python
@app.post("/chat")
async def chat(request: ChatRequest):
    response = ""
    for chunk in run_agent(request.message, request.history):
        response += chunk
    return {"response": response}
```

**Frontend** — UI and state management
React handles conversation state (the message list), renders the chat UI, and calls the backend on each submit. It knows nothing about tools, prompts, or the agent loop.

**Separation of concerns** — the frontend owns UI state, the backend owns agent logic.

---

## Setup

**Backend:**
```bash
cd backend
pip install fastapi uvicorn openai tavily-python pytz python-dotenv
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...
uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173`.

---

## Things worth reading in the code

- **`backend/main.py`**
- **`frontend/src/hooks/useChat.js`** — one function, one fetch call. The frontend treats the agent as a black box.
- **`frontend/src/components/Message.jsx`** — conversation history is a `useState` array in React. Same concept as the messages list in the backend — just on the other side of the HTTP boundary.

---

## Why not streaming?

Streaming (sending the response token by token as it generates) would be the next natural step here — it makes the UI feel significantly more responsive. It's not included in this project to keep the backend/frontend contract simple.

The next project in this series introduces streaming alongside LangChain.