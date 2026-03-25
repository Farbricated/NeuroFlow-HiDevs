import asyncio
import json
import uuid
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.db.pool import get_pool
from pipelines.retrieval.query_processor import process_query
from pipelines.retrieval.retriever import HybridRetriever
from pipelines.retrieval.fusion import reciprocal_rank_fusion
from pipelines.retrieval.reranker import rerank
from pipelines.retrieval.context_assembler import assemble_context
from pipelines.generation.generator import generate_and_stream

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    pipeline_id: str
    stream: bool = True


@router.post("/query")
async def create_query(body: QueryRequest):
    pool = await get_pool()

    # Load pipeline config
    pipeline = await pool.fetchrow(
        "SELECT id, config, version FROM pipelines WHERE id=$1 AND status='active'",
        uuid.UUID(body.pipeline_id),
    )
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    config = dict(pipeline["config"])

    # Create run record
    run_id = uuid.uuid4()
    await pool.execute(
        """INSERT INTO pipeline_runs (id, pipeline_id, pipeline_version, query, status)
           VALUES ($1, $2, $3, $4, 'running')""",
        run_id, uuid.UUID(body.pipeline_id), pipeline["version"], body.query,
    )

    if body.stream:
        return {"run_id": str(run_id)}

    # Non-streaming: run full pipeline and return
    result = await _run_pipeline(str(run_id), body.query, config)
    return result


async def _run_pipeline(run_id: str, query: str, config: dict) -> dict:
    pool = await get_pool()
    ret_config = config.get("retrieval", {})

    # Query processing
    processed = await process_query(query)

    retrieval_start = time.monotonic()

    # Retrieve
    retriever = HybridRetriever()
    dense, sparse, metadata = await retriever.retrieve(
        query=processed.original,
        query_expansions=processed.expansions,
        metadata_filters=processed.metadata_filters,
        k=ret_config.get("dense_k", 20),
    )

    fused = reciprocal_rank_fusion([dense, sparse, metadata])

    top_k = ret_config.get("top_k_after_rerank", 8)
    reranked = await rerank(query, fused, top_k=top_k)

    retrieval_ms = int((time.monotonic() - retrieval_start) * 1000)

    context = assemble_context(
        reranked,
        token_budget=config.get("generation", {}).get("max_context_tokens", 4000),
    )

    chunk_ids = [r.chunk_id for r in reranked]
    await pool.execute(
        "UPDATE pipeline_runs SET retrieved_chunk_ids=$1, retrieval_latency_ms=$2 WHERE id=$3",
        [uuid.UUID(cid) for cid in chunk_ids],
        retrieval_ms,
        uuid.UUID(run_id),
    )

    # Generate
    final_result = {}
    async for event in generate_and_stream(run_id, query, context, processed.query_type, config):
        if event["type"] == "done":
            final_result = event

    return final_result


@router.get("/query/{run_id}/stream")
async def stream_query(run_id: str):
    pool = await get_pool()
    run = await pool.fetchrow(
        "SELECT pipeline_id, query, status FROM pipeline_runs WHERE id=$1",
        uuid.UUID(run_id),
    )
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    pipeline = await pool.fetchrow(
        "SELECT config, version FROM pipelines WHERE id=$1",
        run["pipeline_id"],
    )
    config = dict(pipeline["config"]) if pipeline else {}

    async def event_generator():
        yield {"data": json.dumps({"type": "retrieval_start"})}

        # Run retrieval
        processed = await process_query(run["query"])
        retriever = HybridRetriever()
        ret_config = config.get("retrieval", {})

        dense, sparse, metadata = await retriever.retrieve(
            query=processed.original,
            query_expansions=processed.expansions,
            metadata_filters=processed.metadata_filters,
            k=ret_config.get("dense_k", 20),
        )
        fused = reciprocal_rank_fusion([dense, sparse, metadata])
        reranked = await rerank(run["query"], fused, top_k=ret_config.get("top_k_after_rerank", 8))
        context = assemble_context(reranked, token_budget=config.get("generation", {}).get("max_context_tokens", 4000))

        yield {
            "data": json.dumps({
                "type": "retrieval_complete",
                "chunk_count": len(context.chunks_used),
                "sources": list({c.document_name for c in context.chunks_used}),
            })
        }

        keepalive_task = asyncio.create_task(_keepalive())

        async for event in generate_and_stream(run_id, run["query"], context, processed.query_type, config):
            yield {"data": json.dumps(event)}

        keepalive_task.cancel()

    async def _keepalive():
        while True:
            await asyncio.sleep(15)

    return EventSourceResponse(event_generator())


@router.patch("/runs/{run_id}/rating")
async def rate_run(run_id: str, body: dict):
    rating = body.get("rating")
    if not isinstance(rating, int) or not 1 <= rating <= 5:
        raise HTTPException(status_code=422, detail="Rating must be integer 1-5")

    pool = await get_pool()
    await pool.execute(
        "UPDATE evaluations SET user_rating=$1 WHERE run_id=$2",
        rating, uuid.UUID(run_id),
    )
    return {"status": "ok"}
