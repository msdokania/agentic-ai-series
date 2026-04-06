import React from 'react'
import ReactMarkdown from 'react-markdown'
import styles from './Message.module.css'

const TOOL_LABELS = {
  get_destinations:       '🗺️  Finding destinations',
  search_web:             '🔍  Searching the web',
  get_destination_info:   '📍  Getting destination info',
  get_destination_info_static: '📍  Getting destination info',
  get_weather:            '🌤️  Checking weather',
  generate_itinerary:     '📅  Building itinerary',
  estimate_budget:        '💰  Estimating budget',
  estimate_budget_static: '💰  Estimating budget',
  get_packing_list:       '🎒  Generating packing list',
}

export default function Message({ msg }) {
  const isUser = msg.role === 'user'

  return (
    <div className={`${styles.row} ${isUser ? styles.userRow : styles.agentRow}`}>
      {!isUser && (
        <div className={styles.avatar}>✈</div>
      )}
      <div className={`${styles.bubble} ${isUser ? styles.userBubble : styles.agentBubble} ${msg.error ? styles.errorBubble : ''}`}>
        {isUser ? (
          <p>{msg.content}</p>
        ) : (
          <div className="prose">
            <ReactMarkdown>{msg.content || (msg.streaming ? '' : '…')}</ReactMarkdown>
            {msg.streaming && <span className={styles.cursor} />}
          </div>
        )}
      </div>
    </div>
  )
}

export function ToolPill({ tool }) {
  const label = TOOL_LABELS[tool.name] ?? `⚙️  ${tool.name}`
  const done   = !!tool.result
  return (
    <div className={`${styles.toolPill} ${done ? styles.toolDone : styles.toolRunning}`}>
      <span className={styles.toolDot} />
      <span className={styles.toolLabel}>{label}</span>
      {done && <span className={styles.toolCheck}>✓</span>}
    </div>
  )
}