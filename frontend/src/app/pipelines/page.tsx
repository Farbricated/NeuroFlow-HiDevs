"use client";
import { useEffect, useState } from "react";
const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
interface Pipeline { id: string; name: string; version: number; status: string; avg_score?: number; run_count?: number; }
export default function PipelinesPage() {
  const [pipelines, setPipelines] = useState<Pipeline[]>([]);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    const token = localStorage.getItem("nf_token") ?? "";
    fetch(`${API}/api/v1/pipelines`, { headers: { Authorization: `Bearer ${token}` } })
      .then((r) => r.json()).then(setPipelines).catch(console.error).finally(() => setLoading(false));
  }, []);
  const scoreColor = (s?: number) => !s ? "text-gray-500" : s > 0.8 ? "text-green-400" : s > 0.6 ? "text-yellow-400" : "text-red-400";
  return (
    <main className="max-w-5xl mx-auto px-6 py-10">
      <h1 className="text-2xl font-bold mb-6">Pipeline Manager</h1>
      {loading ? <p className="text-gray-400">Loading…</p> : (
        <div className="space-y-3">
          {pipelines.map((p) => (
            <div key={p.id} className="flex items-center justify-between p-4 bg-gray-900 rounded-xl border border-gray-800">
              <div>
                <div className="font-medium">{p.name} <span className="text-xs text-gray-500">v{p.version}</span></div>
                <div className="text-xs text-gray-500">{p.id}</div>
              </div>
              <div className="text-right">
                <div className={`text-lg font-bold ${scoreColor(p.avg_score)}`}>{p.avg_score != null ? p.avg_score.toFixed(2) : "—"}</div>
                <div className="text-xs text-gray-500">{p.run_count ?? 0} runs</div>
              </div>
            </div>
          ))}
          {pipelines.length === 0 && <p className="text-gray-500">No pipelines yet.</p>}
        </div>
      )}
    </main>
  );
}
