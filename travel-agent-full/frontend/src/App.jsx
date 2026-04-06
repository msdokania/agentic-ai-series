import React, { useRef, useEffect, useState } from 'react'
import { useChat } from './hooks/useChat'
import Message, { ToolPill } from './components/Message'
import styles from './App.module.css'

const SUGGESTIONS = [
  'Beach holiday for 7 days, mid-range budget?',
  '5-day Tokyo itinerary for food lovers',
  'Best time to visit Bali?',
  '10 days in Barcelona — what\'s the cost?',
  'What to pack for winter Japan?',
]

export default function App() {
  const { messages, isLoading, toolActivity, sendMessage, clearChat } = useChat()
  const [draft, setDraft] = useState('')
  const bottomRef  = useRef(null)
  const inputRef   = useRef(null)
  const isEmpty    = messages.length === 0

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, toolActivity])

  const submit = () => {
    const text = draft.trim()
    if (!text || isLoading) return
    setDraft('')
    sendMessage(text)
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className={styles.shell}>
      {/* ── Sidebar ─────────────────────────────────────── */}
      <aside className={styles.sidebar}>
        <div className={styles.logo}>
          <span className={styles.logoMark}>✈</span>
          <span className={styles.logoText}>voyage</span>
        </div>

        <nav className={styles.nav}>
          <button className={`${styles.navItem} ${styles.navActive}`}>
            <span>💬</span> Chat
          </button>
          <button className={styles.navItem} onClick={clearChat}>
            <span>＋</span> New trip
          </button>
        </nav>

        <div className={styles.sidebarBottom}>
          <p className={styles.sidebarHint}>Powered by GPT-4o&nbsp;mini</p>
        </div>
      </aside>

      {/* ── Main ────────────────────────────────────────── */}
      <main className={styles.main}>
        {/* Header */}
        <header className={styles.header}>
          <h1 className={styles.headerTitle}>Your travel planner</h1>
          {!isEmpty && (
            <button className={styles.clearBtn} onClick={clearChat}>Clear</button>
          )}
        </header>

        {/* Messages */}
        <div className={styles.feed}>
          {isEmpty ? (
            <div className={styles.hero}>
              <p className={styles.heroEyebrow}>Where to next?</p>
              <h2 className={styles.heroTitle}>Plan your perfect trip</h2>
              <p className={styles.heroSub}>
                Tell me where you'd like to go, how long you have, and your travel style —
                I'll handle the rest.
              </p>
              <div className={styles.chips}>
                {SUGGESTIONS.map(s => (
                  <button key={s} className={styles.chip} onClick={() => sendMessage(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className={styles.messages}>
              {messages.map((msg, i) => (
                <Message key={msg.id ?? i} msg={msg} />
              ))}

              {/* Tool activity strip */}
              {toolActivity.length > 0 && (
                <div className={styles.toolStrip}>
                  {toolActivity.map((t, i) => (
                    <ToolPill key={i} tool={t} />
                  ))}
                </div>
              )}

              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {/* Input */}
        <div className={styles.inputArea}>
          <div className={styles.inputWrap}>
            <textarea
              ref={inputRef}
              className={styles.textarea}
              placeholder="Ask anything about your trip…"
              value={draft}
              rows={1}
              onChange={e => {
                setDraft(e.target.value)
                // auto-grow
                e.target.style.height = 'auto'
                e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
              }}
              onKeyDown={handleKey}
              disabled={isLoading}
            />
            <button
              className={styles.sendBtn}
              onClick={submit}
              disabled={!draft.trim() || isLoading}
              aria-label="Send"
            >
              {isLoading
                ? <span className={styles.spinner} />
                : <SendIcon />}
            </button>
          </div>
          <p className={styles.inputHint}>
            Press <kbd>Enter</kbd> to send · <kbd>Shift+Enter</kbd> for new line
          </p>
        </div>
      </main>
    </div>
  )
}

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  )
}