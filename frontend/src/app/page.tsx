import Link from "next/link";
const pages = [
  { href: "/playground",  label: "Query Playground",  desc: "Test pipelines with SSE streaming and citations" },
  { href: "/pipelines",   label: "Pipeline Manager",   desc: "Create, compare, and analyze RAG pipelines" },
  { href: "/evaluations", label: "Evaluation Feed",    desc: "Real-time quality scores as they complete" },
  { href: "/documents",   label: "Documents",          desc: "Upload and manage ingested documents" },
];
export default function Home() {
  return (
    <main className="max-w-4xl mx-auto px-6 py-16">
      <h1 className="text-4xl font-bold mb-2">NeuroFlow</h1>
      <p className="text-gray-400 mb-12">Multi-Modal LLM Orchestration Platform</p>
      <div className="grid grid-cols-2 gap-4">
        {pages.map((p) => (
          <Link key={p.href} href={p.href} className="block p-6 rounded-xl border border-gray-800 hover:border-blue-500 transition-colors">
            <div className="font-semibold text-lg mb-1">{p.label}</div>
            <div className="text-sm text-gray-400">{p.desc}</div>
          </Link>
        ))}
      </div>
    </main>
  );
}
