import { useEffect, useRef, useState } from "react";
import Sidebar from "./components/Sidebar";
import ChatMessage from "./components/ChatMessage";
import ChatInput from "./components/ChatInput";
import { queryDocuments, listDocuments } from "./api";
import "./App.css";

const WELCOME = {
  role: "assistant",
  response: {
    answer:
      "Hello! Ingest one or more PDF papers using the sidebar, then ask me anything about them. I'll answer with citations back to the source documents.",
    citations: [],
    model: "",
    retrieval_strategy: "",
    chunks_retrieved: 0,
    total_latency_ms: 0,
    prompt_version: "",
  },
};

export default function App() {
  const [messages, setMessages] = useState([WELCOME]);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const bottomRef = useRef(null);

  async function fetchStats() {
    try {
      const data = await listDocuments();
      setStats(data);
    } catch {
      setStats(null);
    }
  }

  useEffect(() => {
    fetchStats();
    // Retry a few times in case the API is still starting up
    const t1 = setTimeout(fetchStats, 1500);
    const t2 = setTimeout(fetchStats, 4000);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleQuestion(question) {
    setMessages((prev) => [...prev, { role: "user", text: question }]);
    setLoading(true);
    try {
      const response = await queryDocuments({ question });
      setMessages((prev) => [...prev, { role: "assistant", response }]);
      fetchStats();
    } catch (e) {
      setMessages((prev) => [...prev, { role: "error", text: e.message }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <Sidebar stats={stats} onStatsChange={fetchStats} />

      <main className="chat-area">
        <header className="chat-header">
          <h1 className="chat-title">Research Assistant</h1>
          <span className="chat-subtitle">
            Deep RAG · {stats?.total_chunks ?? 0} chunks indexed
          </span>
        </header>

        <div className="messages-list">
          {messages.map((m, i) => (
            <ChatMessage key={i} message={m} />
          ))}
          {loading && (
            <div className="message assistant-message">
              <div className="bubble assistant-bubble thinking">
                <span className="spinner" /> Retrieving and generating…
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="input-area">
          <ChatInput onSubmit={handleQuestion} disabled={loading} />
          <p className="input-hint">Enter to send · Shift+Enter for newline</p>
        </div>
      </main>
    </div>
  );
}
