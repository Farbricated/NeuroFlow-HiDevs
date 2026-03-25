"use client";
import { useState } from "react";
const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
interface Doc { id: string; filename: string; status: string; }
const statusColor: Record<string, string> = { complete: "text-green-400", queued: "text-blue-400 animate-pulse", processing: "text-yellow-400 animate-pulse", failed: "text-red-400" };
export default function DocumentsPage() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);

  const uploadFile = async (file: File) => {
    setUploading(true);
    const token = localStorage.getItem("nf_token") ?? "";
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API}/api/v1/ingest`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: form });
      const data = await res.json();
      setDocs((prev) => [{ id: data.document_id, filename: file.name, status: data.status }, ...prev]);
    } catch (e) { console.error(e); } finally { setUploading(false); }
  };

  return (
    <main className="max-w-4xl mx-auto px-6 py-10">
      <h1 className="text-2xl font-bold mb-6">Documents</h1>
      <div onDrop={(e) => { e.preventDefault(); setDragging(false); Array.from(e.dataTransfer.files).forEach(uploadFile); }}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)}
        className={`border-2 border-dashed rounded-xl p-10 text-center mb-6 transition-colors ${dragging ? "border-blue-500 bg-blue-950" : "border-gray-700"}`}>
        {uploading ? <p className="text-blue-400">Uploading…</p> : <p className="text-gray-400">Drag & drop PDF, DOCX, CSV, or images here</p>}
      </div>
      <div className="space-y-2">
        {docs.map((d) => (
          <div key={d.id} className="flex items-center justify-between p-3 bg-gray-900 rounded-lg border border-gray-800 text-sm">
            <span className="font-medium">{d.filename}</span>
            <span className={statusColor[d.status] ?? "text-gray-400"}>{d.status}</span>
          </div>
        ))}
        {docs.length === 0 && <p className="text-gray-500 text-sm">No documents uploaded yet.</p>}
      </div>
    </main>
  );
}
