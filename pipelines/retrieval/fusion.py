from pipelines.retrieval.retriever import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
) -> list[RetrievalResult]:
    """
    Fuse multiple ranked result lists using RRF.
    score(d) = Σ 1 / (k + rank_i(d)) for each list where d appears.
    """
    scores: dict[str, float] = {}
    best: dict[str, RetrievalResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            cid = result.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in best or result.score > best[cid].score:
                best[cid] = result

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused = []
    for cid, rrf_score in ranked:
        r = best[cid]
        fused.append(RetrievalResult(
            chunk_id=r.chunk_id,
            content=r.content,
            document_name=r.document_name,
            page_number=r.page_number,
            score=rrf_score,
            retrieval_method="rrf_fused",
            metadata=r.metadata,
        ))

    return fused
