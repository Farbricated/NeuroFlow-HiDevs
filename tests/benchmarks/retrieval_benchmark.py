"""
Comprehensive retrieval benchmark.
Run: python -m tests.benchmarks.retrieval_benchmark
Requires: at least 50 document chunks ingested and TEST_SET populated.
"""
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


TEST_SET = [
    # Populate with {"query": "...", "relevant_chunk_ids": ["uuid1", ...]}
    # after ingesting test documents
]

# Sample test set structure for documentation
SAMPLE_TEST_SET = [
    {
        "query": "What is the attention mechanism?",
        "relevant_chunk_ids": [],
    },
    {
        "query": "How does multi-head attention work?",
        "relevant_chunk_ids": [],
    },
]


async def run_benchmark():
    from backend.db.pool import create_pool
    from backend.config import get_settings
    from pipelines.retrieval.query_processor import process_query
    from pipelines.retrieval.retriever import HybridRetriever
    from pipelines.retrieval.fusion import reciprocal_rank_fusion
    from pipelines.retrieval.reranker import rerank

    if not TEST_SET:
        print("⚠️  No test set defined. Using empty benchmark.")
        results = {
            "note": "Populate TEST_SET in tests/benchmarks/retrieval_benchmark.py with real chunk IDs",
            "strategies": {
                "dense_only": {"hit_rate_at_5": 0, "hit_rate_at_10": 0, "mrr_at_10": 0, "ndcg_at_10": 0},
                "sparse_only": {"hit_rate_at_5": 0, "hit_rate_at_10": 0, "mrr_at_10": 0, "ndcg_at_10": 0},
                "hybrid_rrf": {"hit_rate_at_5": 0, "hit_rate_at_10": 0, "mrr_at_10": 0, "ndcg_at_10": 0},
                "hybrid_reranked": {"hit_rate_at_5": 0, "hit_rate_at_10": 0, "mrr_at_10": 0, "ndcg_at_10": 0},
            },
            "improvement": {"hybrid_reranked_vs_dense_mrr": 0, "threshold_15pct_met": False},
        }
        with open("tests/benchmarks/retrieval_benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        _write_markdown(results)
        return results

    settings = get_settings()
    await create_pool(settings.asyncpg_dsn)
    retriever = HybridRetriever()

    def calc_metrics(result_lists_by_query: list[list[str]], relevant_sets: list[set], k: int = 10):
        hit_k5, hit_k10, rrs, ndcgs = [], [], [], []
        for retrieved, relevant in zip(result_lists_by_query, relevant_sets):
            top5 = set(retrieved[:5])
            top10 = set(retrieved[:k])
            hit_k5.append(1 if top5 & relevant else 0)
            hit_k10.append(1 if top10 & relevant else 0)
            rr = 0
            for i, r in enumerate(retrieved[:k]):
                if r in relevant:
                    rr = 1 / (i + 1)
                    break
            rrs.append(rr)
            dcg = sum(1 / (i + 1) for i, r in enumerate(retrieved[:k]) if r in relevant)
            idcg = sum(1 / (i + 1) for i in range(min(len(relevant), k)))
            ndcgs.append(dcg / idcg if idcg > 0 else 0)
        n = len(result_lists_by_query)
        return {
            "hit_rate_at_5": round(sum(hit_k5) / n, 4),
            "hit_rate_at_10": round(sum(hit_k10) / n, 4),
            "mrr_at_10": round(sum(rrs) / n, 4),
            "ndcg_at_10": round(sum(ndcgs) / n, 4),
        }

    dense_results = []
    sparse_results = []
    hybrid_results = []
    hybrid_reranked_results = []
    relevant_sets = [set(t["relevant_chunk_ids"]) for t in TEST_SET]

    for test in TEST_SET:
        processed = await process_query(test["query"])
        dense, sparse, metadata = await retriever.retrieve(
            query=processed.original, query_expansions=processed.expansions, k=20
        )
        fused = reciprocal_rank_fusion([dense, sparse, metadata])
        reranked = await rerank(test["query"], fused, top_k=10)

        dense_results.append([r.chunk_id for r in dense])
        sparse_results.append([r.chunk_id for r in sparse])
        hybrid_results.append([r.chunk_id for r in fused[:10]])
        hybrid_reranked_results.append([r.chunk_id for r in reranked])

    results = {
        "total_queries": len(TEST_SET),
        "strategies": {
            "dense_only": calc_metrics(dense_results, relevant_sets),
            "sparse_only": calc_metrics(sparse_results, relevant_sets),
            "hybrid_rrf": calc_metrics(hybrid_results, relevant_sets),
            "hybrid_reranked": calc_metrics(hybrid_reranked_results, relevant_sets),
        },
    }
    dense_mrr = results["strategies"]["dense_only"]["mrr_at_10"]
    reranked_mrr = results["strategies"]["hybrid_reranked"]["mrr_at_10"]
    improvement = (reranked_mrr - dense_mrr) / dense_mrr if dense_mrr > 0 else 0
    results["improvement"] = {
        "hybrid_reranked_vs_dense_mrr": round(improvement, 4),
        "threshold_15pct_met": improvement >= 0.15,
    }

    with open("tests/benchmarks/retrieval_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    _write_markdown(results)
    print(json.dumps(results, indent=2))
    return results


def _write_markdown(results: dict):
    lines = ["# Retrieval Benchmark Results\n"]
    lines.append("| Strategy | Hit@5 | Hit@10 | MRR@10 | NDCG@10 |")
    lines.append("|----------|-------|--------|--------|---------|")
    for name, m in results.get("strategies", {}).items():
        lines.append(f"| {name} | {m.get('hit_rate_at_5',0)} | {m.get('hit_rate_at_10',0)} | {m.get('mrr_at_10',0)} | {m.get('ndcg_at_10',0)} |")
    imp = results.get("improvement", {})
    lines.append(f"\n**Hybrid+Reranked vs Dense MRR improvement**: {imp.get('hybrid_reranked_vs_dense_mrr', 0):.1%}")
    lines.append(f"\n**15% threshold met**: {'✅' if imp.get('threshold_15pct_met') else '❌'}")
    with open("tests/benchmarks/retrieval_benchmark_results.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(run_benchmark())
