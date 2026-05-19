import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function CitationBadge({ citation, index }) {
  const pages = citation.page_numbers?.length
    ? `p. ${citation.page_numbers.join(", ")}`
    : null;
  const score = citation.relevance_score
    ? `${Math.round(citation.relevance_score * 100)}%`
    : null;

  return (
    <div className="citation-badge">
      <span className="citation-index">[{index + 1}]</span>
      <div className="citation-details">
        <span className="citation-file">{citation.source_file}</span>
        {pages && <span className="citation-meta">{pages}</span>}
        {score && <span className="citation-score">{score} relevant</span>}
      </div>
    </div>
  );
}

function MetaRow({ label, value }) {
  return (
    <span className="meta-item">
      <span className="meta-label">{label}</span>
      <span className="meta-value">{value}</span>
    </span>
  );
}

export default function ChatMessage({ message }) {
  const { role, text, response } = message;

  if (role === "user") {
    return (
      <div className="message user-message">
        <div className="bubble user-bubble">{text}</div>
      </div>
    );
  }

  if (role === "error") {
    return (
      <div className="message assistant-message">
        <div className="bubble error-bubble">{text}</div>
      </div>
    );
  }

  return (
    <div className="message assistant-message">
      <div className="bubble assistant-bubble">
        <div className="answer-text">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{response.answer}</ReactMarkdown>
        </div>

        {response.citations?.length > 0 && (
          <div className="citations-section">
            <h4 className="citations-heading">Sources</h4>
            <div className="citations-list">
              {response.citations.map((c, i) => (
                <CitationBadge key={i} citation={c} index={i} />
              ))}
            </div>
          </div>
        )}

        <div className="response-meta">
          <MetaRow label="model" value={response.model} />
          <MetaRow label="strategy" value={response.retrieval_strategy} />
          <MetaRow label="chunks" value={response.chunks_retrieved} />
          <MetaRow
            label="latency"
            value={`${Math.round(response.total_latency_ms)}ms`}
          />
          <MetaRow label="prompt" value={response.prompt_version} />
        </div>
      </div>
    </div>
  );
}
