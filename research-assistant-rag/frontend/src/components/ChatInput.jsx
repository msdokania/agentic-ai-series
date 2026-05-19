import { useState } from "react";

export default function ChatInput({ onSubmit, disabled }) {
  const [value, setValue] = useState("");

  function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  function submit() {
    const q = value.trim();
    if (!q || disabled) return;
    onSubmit(q);
    setValue("");
  }

  return (
    <div className="chat-input-row">
      <textarea
        className="chat-textarea"
        placeholder="Ask a question about your documents…"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKey}
        rows={1}
        disabled={disabled}
      />
      <button
        className="send-btn"
        onClick={submit}
        disabled={disabled || !value.trim()}
        aria-label="Send"
      >
        {disabled ? <span className="spinner" /> : "↑"}
      </button>
    </div>
  );
}
