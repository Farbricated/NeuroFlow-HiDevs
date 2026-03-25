import asyncio
import json
from pipelines.retrieval.retriever import RetrievalResult
from backend.providers.base import ChatMessage, RoutingCriteria
from backend.providers.client import get_client

RERANK_TOP_N = 40


async def _score_pair(client, query: str, chunk: RetrievalResult) -> tuple[str, float]:
    prompt = (
        f"Rate the relevance of this passage to the query on a scale of 0-10.\n\n"
        f"Query: {query}\n\n"
        f"Passage: {chunk.content[:500]}\n\n"
        f"Return ONLY the numeric score (0-10), nothing else."
    )
    result = await client.chat(
        [ChatMessage(role="user", content=prompt)],
        routing_criteria=RoutingCriteria(task_type="classification"),
        max_tokens=5,
    )
    try:
        score = float(result.content.strip()) / 10.0
    except ValueError:
        score = 0.5
    return chunk.chunk_id, score


async def rerank(
    query: str,
    candidates: list[RetrievalResult],
    top_k: int = 10,
) -> list[RetrievalResult]:
    if not candidates:
        return []

    top_candidates = candidates[:RERANK_TOP_N]
    client = get_client()

    scores_raw = await asyncio.gather(
        *[_score_pair(client, query, c) for c in top_candidates]
    )
    score_map = dict(scores_raw)

    reranked = sorted(
        top_candidates,
        key=lambda c: score_map.get(c.chunk_id, 0),
        reverse=True,
    )

    for r in reranked:
        r.score = score_map.get(r.chunk_id, 0)

    return reranked[:top_k]
