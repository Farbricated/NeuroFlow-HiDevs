"use client";
import { useEffect, useState } from "react";
const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
interface EvalItem { id: string; run_id: string; query: string; faithfulness: number; answer_relevance: number; context_precision: number; context_recall: number; overall_score: number; evaluated_at: string; }
function Bar({ value, label }: { value: number; label: string }) {
  const color = value > 0.8 ? "bg-green-500" : value > 0.6 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-24 text-gray-400 truncate">{label}</span>
      <div className="flex-1 bg-gray-800 rounded-full h-1.5"><div className={`${color} h-1.5 rounded-full`} style={{ width: `${value * 100}%` }} /></div>
      <span className="w-8 text-right">{value.toFixed(2)}</span>
    </div>
  );
}
export default function EvaluationsPage() {
  const [items, setItems] = useState<EvalItem[]>([]);
  useEffect(() => {
    const token = localStorage.getItem("nf_token") ?? "";
    fetch(`${API}/api/v1/evaluations?per_page=20`, { headers: { Authorization: `Bearer ${token}` } })
      .then((r) => r.json()).then((d) => setItems(d.items ?? [])).catch(console.error);
  }, []);
  return (
    <main className="max-w-4xl mx-auto px-6 py-10">
      <h1 className="text-2xl font-bold mb-6">Evaluation Feed</h1>
      <div className="space-y-3">
        {items.map((item) => (
          <div key={item.id} className="p-4 bg-gray-900 rounded-xl border border-gray-800">
            <div className="flex items-start justify-between mb-3">
              <p className="text-sm text-gray-300 truncate max-w-xs">{item.query}</p>
              <span className={`text-lg font-bold ml-4 ${item.overall_score > 0.8 ? "text-green-400" : item.overall_score > 0.6 ? "text-yellow-400" : "text-red-400"}`}>{item.overall_score.toFixed(2)}</span>
            </div>
            <div className="space-y-1">
              <Bar value={item.faithfulness} label="Faithfulness" />
              <Bar value={item.answer_relevance} label="Answer Rel." />
              <Bar value={item.context_precision} label="Ctx Precision" />
              <Bar value={item.context_recall} label="Ctx Recall" />
            </div>
            <div className="text-xs text-gray-600 mt-2">{new Date(item.evaluated_at).toLocaleString()}</div>
          </div>
        ))}
        {items.length === 0 && <p className="text-gray-500">No evaluations yet.</p>}
      </div>
    </main>
  );
}
