import { useState, useCallback, useRef } from 'react'

export function useChat() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [toolActivity, setToolActivity] = useState([])   // [{name, args, result?}]
  const abortRef = useRef(null)

  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || isLoading) return

    // Add user message
    const userMsg = { role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)
    setToolActivity([])

    // Build history for the API (exclude the message we just added — backend appends it)
    const history = messages.map(m => ({ role: m.role, content: m.content }))

    // Placeholder for streaming assistant reply
    const assistantId = Date.now()
    setMessages(prev => [...prev, { id: assistantId, role: 'assistant', content: '', streaming: true }])

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history }),
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        let event = null
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            event = line.slice(7).trim()
          } else if (line.startsWith('data: ')) {
            const raw = line.slice(6).trim()
            if (!raw) continue
            try {
              const data = JSON.parse(raw)
              if (event === 'message') {
                setMessages(prev => prev.map(m =>
                  m.id === assistantId
                    ? { ...m, content: m.content + data.content }
                    : m
                ))
              } else if (event === 'tool_calls') {
                setToolActivity(prev => [
                  ...prev,
                  ...data.calls.map(c => ({ name: c.name, args: c.args })),
                ])
              } else if (event === 'tool_result') {
                setToolActivity(prev => prev.map((t, i) =>
                  i === prev.length - 1 && t.name === data.name
                    ? { ...t, result: data.result }
                    : t
                ))
              } else if (event === 'done') {
                setMessages(prev => prev.map(m =>
                  m.id === assistantId ? { ...m, streaming: false } : m
                ))
              }
            } catch (_) { /* ignore malformed JSON */ }
            event = null
          }
        }
      }
    } catch (err) {
      setMessages(prev => prev.map(m =>
        m.id === assistantId
          ? { ...m, content: 'Something went wrong. Is the backend running?', streaming: false, error: true }
          : m
      ))
    } finally {
      setIsLoading(false)
    }
  }, [messages, isLoading])

  const clearChat = useCallback(() => {
    setMessages([])
    setToolActivity([])
  }, [])

  return { messages, isLoading, toolActivity, sendMessage, clearChat }
}