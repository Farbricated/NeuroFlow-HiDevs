"""Reciprocal Rank Fusion (RRF) for merging multiple retrieval result lists."""
from __future__ import annotations

from collections import defaultdict

from pipelines.retrieval.retriever import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
) -> list[RetrievalResult]:
    """Merge result_lists using RRF, sorted by descending score."""
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, RetrievalResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            cid = result.chunk_id
            scores[cid] += 1.0 / (k + rank)
            if cid not in best:
                best[cid] = result

    fused: list[RetrievalResult] = []
    for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        r = best[cid]
        fused.append(
            RetrievalResult(
                chunk_id=r.chunk_id,
                content=r.content,
                document_name=r.document_name,
                page_number=r.page_number,
                score=score,
                retrieval_method="rrf_fused",
            )
        )

    return fused
