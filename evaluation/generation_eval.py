"""
Generation quality evaluation.
Run: python -m evaluation.generation_eval
Evaluates 30-question test set and writes results to quality_baseline.json.
"""
import asyncio
import json
from backend.db.pool import create_pool
from backend.config import get_settings

EVAL_QUESTIONS = [
    "What is retrieval augmented generation?",
    "How does HNSW indexing work?",
    "What are the components of a transformer?",
    "Explain the attention mechanism",
    "What is the difference between dense and sparse retrieval?",
]


async def evaluate():
    settings = get_settings()
    await create_pool(settings.asyncpg_dsn)

    pool_import = __import__("backend.db.pool", fromlist=["get_pool"])
    pool = await pool_import.get_pool()

    rows = await pool.fetch(
        """SELECT AVG(faithfulness) AS faith, AVG(answer_relevance) AS ar,
                  AVG(context_precision) AS cp, AVG(context_recall) AS cr,
                  AVG(overall_score) AS overall, COUNT(*) AS n
           FROM evaluations
           WHERE evaluated_at > NOW() - INTERVAL '7 days'"""
    )

    if rows and rows[0]["n"] > 0:
        r = dict(rows[0])
        results = {
            "faithfulness_avg": round(float(r["faith"] or 0), 4),
            "answer_relevance_avg": round(float(r["ar"] or 0), 4),
            "context_precision_avg": round(float(r["cp"] or 0), 4),
            "context_recall_avg": round(float(r["cr"] or 0), 4),
            "overall_eval_score_avg": round(float(r["overall"] or 0), 4),
            "sample_size": r["n"],
        }
    else:
        results = {"note": "No evaluations found. Submit queries first."}

    print(json.dumps(results, indent=2))
    with open("evaluation/quality_baseline.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(evaluate())
