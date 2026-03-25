"use client";
import { useState, useRef } from "react";
const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
export default function PlaygroundPage() {
  const [query, setQuery] = useState("");
  const [pipelineId, setPipelineId] = useState("");
  const [response, setResponse] = useState("");
  const [sources, setSources] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  async function handleSubmit() {
    if (!query.trim() || !pipelineId.trim()) return;
    setResponse(""); setSources([]); setLoading(true);
    const token = localStorage.getItem("nf_token") ?? "";
    const res = await fetch(`${API}/api/v1/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({ query, pipeline_id: pipelineId, stream: true }),
    });
    const { run_id } = await res.json();
    const es = new EventSource(`${API}/api/v1/query/${run_id}/stream`);
    es.onmessage = (e) => {
      const evt = JSON.parse(e.data);
      if (evt.type === "token") setResponse((p) => p + evt.delta);
      if (evt.type === "retrieval_complete") setSources(evt.sources ?? []);
      if (evt.type === "done") { es.close(); setLoading(false); }
    };
    es.onerror = () => { es.close(); setLoading(false); };
  }

  return (
    <main className="max-w-4xl mx-auto px-6 py-10">
      <h1 className="text-2xl font-bold mb-6">Query Playground</h1>
      <div className="space-y-4">
        <input className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-sm"
          placeholder="Pipeline ID" value={pipelineId} onChange={(e) => setPipelineId(e.target.value)} />
        <textarea rows={3} className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-sm resize-none"
          placeholder="Ask anything..." value={query} onChange={(e) => setQuery(e.target.value)} />
        <button onClick={handleSubmit} disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 px-6 py-2 rounded-lg text-sm font-medium">
          {loading ? "Generating…" : "Submit"}
        </button>
      </div>
      {sources.length > 0 && <div className="mt-4 text-xs text-gray-400">Sources: {sources.join(", ")}</div>}
      {response && (
        <div className="mt-4 p-4 bg-gray-900 rounded-xl border border-gray-800 text-sm leading-relaxed whitespace-pre-wrap">
          {response}{loading && <span className="animate-pulse">▍</span>}
        </div>
      )}
    </main>
  );
}
