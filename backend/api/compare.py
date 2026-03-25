import asyncio
import uuid
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.db.pool import get_pool

router = APIRouter()


class CompareRequest(BaseModel):
    query: str
    pipeline_a_id: str
    pipeline_b_id: str


async def _run_single(pipeline_id: str, query: str, pool) -> dict:
    from pipelines.retrieval.query_processor import process_query
    from pipelines.retrieval.retriever import HybridRetriever
    from pipelines.retrieval.fusion import reciprocal_rank_fusion
    from pipelines.retrieval.reranker import rerank
    from pipelines.retrieval.context_assembler import assemble_context
    from pipelines.generation.generator import generate_and_stream

    pipeline = await pool.fetchrow(
        "SELECT id, config, version FROM pipelines WHERE id=$1 AND status='active'",
        uuid.UUID(pipeline_id),
    )
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

    config = dict(pipeline["config"])
    run_id = uuid.uuid4()

    await pool.execute(
        """INSERT INTO pipeline_runs (id, pipeline_id, pipeline_version, query, status)
           VALUES ($1, $2, $3, $4, 'running')""",
        run_id, uuid.UUID(pipeline_id), pipeline["version"], query,
    )

    start = time.monotonic()
    processed = await process_query(query)
    retriever = HybridRetriever()
    ret_config = config.get("retrieval", {})
    retrieval_start = time.monotonic()

    dense, sparse, metadata = await retriever.retrieve(
        query=processed.original,
        query_expansions=processed.expansions,
        metadata_filters=processed.metadata_filters,
        k=ret_config.get("dense_k", 20),
    )
    fused = reciprocal_rank_fusion([dense, sparse, metadata])
    reranked = await rerank(query, fused, top_k=ret_config.get("top_k_after_rerank", 8))
    retrieval_ms = int((time.monotonic() - retrieval_start) * 1000)

    context = assemble_context(
        reranked,
        token_budget=config.get("generation", {}).get("max_context_tokens", 4000),
    )

    generation_text = ""
    final_event = {}
    async for event in generate_and_stream(str(run_id), query, context, processed.query_type, config):
        if event["type"] == "done":
            final_event = event
        elif event["type"] == "token":
            generation_text += event.get("delta", "")

    total_ms = int((time.monotonic() - start) * 1000)

    return {
        "run_id": str(run_id),
        "generation": generation_text,
        "retrieval_latency_ms": retrieval_ms,
        "total_latency_ms": total_ms,
        "chunks_used": len(context.chunks_used),
        "citations": final_event.get("citations", []),
        "eval_score": None,
    }


@router.post("/pipelines/compare")
async def compare_pipelines(body: CompareRequest):
    pool = await get_pool()
    result_a, result_b = await asyncio.gather(
        _run_single(body.pipeline_a_id, body.query, pool),
        _run_single(body.pipeline_b_id, body.query, pool),
    )
    return {"query": body.query, "pipeline_a": result_a, "pipeline_b": result_b}
