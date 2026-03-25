import asyncio
import json
import uuid
import time
from typing import AsyncGenerator

from pipelines.generation.prompt_builder import build_prompt
from pipelines.generation.citations import parse_citations
from pipelines.retrieval.context_assembler import AssembledContext
from backend.db.pool import get_pool
from backend.providers.base import ChatMessage, RoutingCriteria
from backend.providers.router import ModelRouter
from backend.providers.client import get_client


async def generate_and_stream(
    run_id: str,
    query: str,
    context: AssembledContext,
    query_type: str,
    pipeline_config: dict,
) -> AsyncGenerator[dict, None]:
    pool = await get_pool()
    router = ModelRouter()
    gen_config = pipeline_config.get("generation", {})

    routing = RoutingCriteria(
        task_type="rag_generation",
        max_cost_per_call=gen_config.get("model_routing", {}).get("max_cost_per_call"),
    )
    provider = await router.select(routing)

    messages_raw = build_prompt(query, context.text, query_type)
    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages_raw]

    # Log prompt before calling LLM
    await pool.execute(
        "UPDATE pipeline_runs SET status='running', metadata=$1 WHERE id=$2",
        json.dumps({"prompt_logged": True, "context_tokens": context.total_tokens}),
        uuid.UUID(run_id),
    )

    full_response = []
    start = time.monotonic()

    yield {"type": "generation_start"}

    async for token in provider.stream(
        messages,
        temperature=gen_config.get("temperature", 0.3),
        max_tokens=gen_config.get("max_output_tokens", 1500),
    ):
        full_response.append(token)
        yield {"type": "token", "delta": token}

    generation_text = "".join(full_response)
    latency_ms = int((time.monotonic() - start) * 1000)

    # Resolve citations
    citations = parse_citations(generation_text, context.sources)
    citations_data = [
        {
            "reference": c.reference,
            "chunk_id": c.chunk_id,
            "document": c.document_name,
            "page": c.page_number,
            "invalid": c.invalid,
        }
        for c in citations
    ]

    # Update run record
    await pool.execute(
        """UPDATE pipeline_runs
           SET generation=$1, latency_ms=$2, model_used=$3, status='complete'
           WHERE id=$4""",
        generation_text,
        latency_ms,
        provider.model_name,
        uuid.UUID(run_id),
    )

    # Enqueue evaluation asynchronously (fire-and-forget)
    from backend.config import get_settings
    import redis.asyncio as aioredis
    settings = get_settings()
    try:
        r = aioredis.from_url(settings.redis_url)
        await r.lpush("queue:evaluate", json.dumps({"run_id": run_id}))
        await r.aclose()
    except Exception:
        pass

    yield {"type": "done", "run_id": run_id, "citations": citations_data}
