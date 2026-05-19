const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function queryDocuments({ question, top_k }) {
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Query failed (${res.status})`);
  }
  return res.json();
}

export async function ingestFiles(files) {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  const res = await fetch(`${BASE}/ingest`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Ingest failed (${res.status})`);
  }
  return res.json();
}

export async function listDocuments() {
  const res = await fetch(`${BASE}/documents`);
  if (!res.ok) throw new Error("Could not fetch document list");
  return res.json();
}

export async function resetStore() {
  const res = await fetch(`${BASE}/reset`, { method: "POST" });
  if (!res.ok) throw new Error("Reset failed");
  return res.json();
}

export async function healthCheck() {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error("API unreachable");
  return res.json();
}
