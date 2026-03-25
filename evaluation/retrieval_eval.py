"""
Retrieval evaluation script.
Usage: python -m evaluation.retrieval_eval
Requires at least 20 chunks ingested and a test set defined below.
"""
import asyncio
import json
from backend.db.pool import create_pool
from backend.config import get_settings
from pipelines.retrieval.query_processor import process_query
from pipelines.retrieval.retriever import HybridRetriever
from pipelines.retrieval.fusion import reciprocal_rank_fusion
from pipelines.retrieval.reranker import rerank

TEST_SET = [
    # Add your test queries with known relevant chunk IDs
    # {"query": "What is HNSW indexing?", "relevant_chunk_ids": ["<uuid>"]},
]


async def evaluate():
    settings = get_settings()
    await create_pool(settings.asyncpg_dsn)

    if not TEST_SET:
        print("No test set defined. Add entries to TEST_SET in retrieval_eval.py")
        return

    retriever = HybridRetriever()
    hit_count = 0
    reciprocal_ranks = []

    for test in TEST_SET:
        processed = await process_query(test["query"])
        dense, sparse, metadata = await retriever.retrieve(
            query=processed.original,
            query_expansions=processed.expansions,
            k=20,
        )
        fused = reciprocal_rank_fusion([dense, sparse, metadata])
        reranked = await rerank(test["query"], fused, top_k=10)

        result_ids = [r.chunk_id for r in reranked]
        relevant = set(test["relevant_chunk_ids"])

        hit = any(rid in relevant for rid in result_ids)
        if hit:
            hit_count += 1

        rank = next(
            (i + 1 for i, rid in enumerate(result_ids) if rid in relevant),
            None,
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    hit_rate = hit_count / len(TEST_SET)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    results = {
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "total_queries": len(TEST_SET),
        "hits": hit_count,
        "thresholds_met": {
            "hit_rate_gt_0.75": hit_rate > 0.75,
            "mrr_gt_0.55": mrr > 0.55,
        },
    }

    print(json.dumps(results, indent=2))

    with open("evaluation/retrieval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    asyncio.run(evaluate())
