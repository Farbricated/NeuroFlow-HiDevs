import asyncio
import json
import uuid
from opentelemetry import trace

from evaluation.metrics.faithfulness import evaluate_faithfulness
from evaluation.metrics.answer_relevance import evaluate_answer_relevance
from evaluation.metrics.context_precision import evaluate_context_precision
from evaluation.metrics.context_recall import evaluate_context_recall
from backend.db.pool import get_pool

tracer = trace.get_tracer(__name__)

WEIGHTS = {
    "faithfulness": 0.35,
    "answer_relevance": 0.30,
    "context_precision": 0.20,
    "context_recall": 0.15,
}
TRAINING_THRESHOLD = 0.80


async def judge_run(run_id: str) -> dict:
    pool = await get_pool()

    run = await pool.fetchrow(
        """SELECT pr.query, pr.generation, pr.retrieved_chunk_ids,
                  array_agg(c.content ORDER BY c.chunk_index) AS chunk_contents
           FROM pipeline_runs pr
           LEFT JOIN chunks c ON c.id = ANY(pr.retrieved_chunk_ids)
           WHERE pr.id = $1
           GROUP BY pr.id, pr.query, pr.generation, pr.retrieved_chunk_ids""",
        uuid.UUID(run_id),
    )

    if not run or not run["generation"]:
        return {}

    query = run["query"]
    answer = run["generation"]
    chunks = list(run["chunk_contents"] or [])
    context = "\n\n".join(chunks)

    with tracer.start_as_current_span("evaluation.judge") as span:
        span.set_attribute("run_id", run_id)

        faithfulness, answer_relevance, context_precision, context_recall = await asyncio.gather(
            evaluate_faithfulness(query, answer, context),
            evaluate_answer_relevance(query, answer),
            evaluate_context_precision(query, chunks, answer),
            evaluate_context_recall(query, chunks, answer),
        )

        overall_score = (
            WEIGHTS["faithfulness"] * faithfulness
            + WEIGHTS["answer_relevance"] * answer_relevance
            + WEIGHTS["context_precision"] * context_precision
            + WEIGHTS["context_recall"] * context_recall
        )

        span.set_attribute("faithfulness", faithfulness)
        span.set_attribute("answer_relevance", answer_relevance)
        span.set_attribute("context_precision", context_precision)
        span.set_attribute("context_recall", context_recall)
        span.set_attribute("overall_score", overall_score)

    eval_id = uuid.uuid4()
    await pool.execute(
        """INSERT INTO evaluations
             (id, run_id, faithfulness, answer_relevance, context_precision,
              context_recall, overall_score, judge_model)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
        eval_id,
        uuid.UUID(run_id),
        faithfulness,
        answer_relevance,
        context_precision,
        context_recall,
        overall_score,
        "gpt-4o-mini",
    )

    if overall_score >= TRAINING_THRESHOLD:
        system_prompt = "You are a precise research assistant."
        await pool.execute(
            """INSERT INTO training_pairs
                 (run_id, system_prompt, user_message, assistant_message, quality_score)
               VALUES ($1, $2, $3, $4, $5)""",
            uuid.UUID(run_id),
            system_prompt,
            query,
            answer,
            overall_score,
        )

    return {
        "run_id": run_id,
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "overall_score": overall_score,
    }
