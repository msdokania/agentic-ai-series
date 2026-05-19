import { useRef, useState } from "react";
import { ingestFiles, resetStore } from "../api";

export default function Sidebar({ stats, onStatsChange }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [status, setStatus] = useState(null); // { type: "ok"|"err"|"loading", msg }

  async function handleFiles(files) {
    const pdfs = [...files].filter((f) => f.name.endsWith(".pdf"));
    if (!pdfs.length) {
      setStatus({ type: "err", msg: "Only PDF files are accepted." });
      return;
    }
    setStatus({ type: "loading", msg: `Ingesting ${pdfs.length} file(s)…` });
    try {
      const res = await ingestFiles(pdfs);
      setStatus({
        type: "ok",
        msg: `Done — ${res.chunks_created} chunks from ${res.documents_processed} doc(s)`,
      });
      onStatsChange();
    } catch (e) {
      setStatus({ type: "err", msg: e.message });
    }
  }

  async function handleReset() {
    if (!confirm("Clear all ingested documents?")) return;
    setStatus({ type: "loading", msg: "Resetting…" });
    try {
      await resetStore();
      setStatus({ type: "ok", msg: "Vector store cleared." });
      onStatsChange();
    } catch (e) {
      setStatus({ type: "err", msg: e.message });
    }
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <span className="logo-icon">⬡</span>
        <span className="logo-text">RAG Research</span>
      </div>

      <section className="sidebar-section">
        <h3 className="sidebar-label">Corpus</h3>
        <div className="stat-box">
          <span className="stat-num">{stats?.total_chunks ?? "—"}</span>
          <span className="stat-sub">chunks indexed</span>
        </div>
      </section>

      <section className="sidebar-section">
        <h3 className="sidebar-label">Ingest PDFs</h3>
        <div
          className={`drop-zone ${dragging ? "dragging" : ""}`}
          onClick={() => inputRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            handleFiles(e.dataTransfer.files);
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".pdf"
            multiple
            hidden
            onChange={(e) => handleFiles(e.target.files)}
          />
          <span className="drop-icon">↑</span>
          <span className="drop-text">Drop PDFs or click to browse</span>
        </div>

        {status && (
          <div className={`ingest-status ${status.type}`}>
            {status.type === "loading" && <span className="spinner" />}
            {status.msg}
          </div>
        )}
      </section>

      <section className="sidebar-section sidebar-footer">
        <button className="reset-btn" onClick={handleReset}>
          Reset corpus
        </button>
      </section>
    </aside>
  );
}
