import asyncio
import json
import uuid
from dataclasses import dataclass, field
from backend.db.pool import get_pool
from backend.providers.client import get_client


@dataclass
class RetrievalResult:
    chunk_id: str
    content: str
    document_name: str
    page_number: int | None
    score: float
    retrieval_method: str
    metadata: dict = field(default_factory=dict)


class HybridRetriever:

    async def _dense_retrieval(
        self, query_texts: list[str], k: int = 20
    ) -> list[RetrievalResult]:
        client = get_client()
        pool = await get_pool()
        all_results: dict[str, RetrievalResult] = {}

        embeddings = await client.embed(query_texts)

        for embedding in embeddings:
            rows = await pool.fetch(
                """
                SELECT c.id, c.content, c.metadata, c.chunk_index,
                       d.filename,
                       1 - (c.embedding <=> $1::vector) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.status = 'complete'
                ORDER BY c.embedding <=> $1::vector
                LIMIT $2
                """,
                json.dumps(embedding),
                k,
            )
            for row in rows:
                cid = str(row["id"])
                if cid not in all_results or row["score"] > all_results[cid].score:
                    all_results[cid] = RetrievalResult(
                        chunk_id=cid,
                        content=row["content"],
                        document_name=row["filename"],
                        page_number=row["metadata"].get("page_number") if row["metadata"] else None,
                        score=row["score"],
                        retrieval_method="dense",
                        metadata=dict(row["metadata"] or {}),
                    )

        return sorted(all_results.values(), key=lambda r: r.score, reverse=True)[:k]

    async def _sparse_retrieval(self, query: str, k: int = 20) -> list[RetrievalResult]:
        pool = await get_pool()
        rows = await pool.fetch(
            """
            SELECT c.id, c.content, c.metadata,
                   d.filename,
                   ts_rank_cd(to_tsvector('english', c.content),
                               plainto_tsquery('english', $1)) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.status = 'complete'
              AND to_tsvector('english', c.content) @@ plainto_tsquery('english', $1)
            ORDER BY score DESC
            LIMIT $2
            """,
            query,
            k,
        )
        return [
            RetrievalResult(
                chunk_id=str(row["id"]),
                content=row["content"],
                document_name=row["filename"],
                page_number=row["metadata"].get("page_number") if row["metadata"] else None,
                score=float(row["score"]),
                retrieval_method="sparse",
                metadata=dict(row["metadata"] or {}),
            )
            for row in rows
        ]

    async def _metadata_retrieval(
        self, query: str, filters: dict, k: int = 20
    ) -> list[RetrievalResult]:
        if not filters:
            return []
        pool = await get_pool()
        client = get_client()
        embedding = (await client.embed([query]))[0]

        rows = await pool.fetch(
            """
            SELECT c.id, c.content, c.metadata, d.filename,
                   1 - (c.embedding <=> $1::vector) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.status = 'complete'
              AND c.metadata @> $2::jsonb
            ORDER BY c.embedding <=> $1::vector
            LIMIT $3
            """,
            json.dumps(embedding),
            json.dumps(filters),
            k,
        )
        return [
            RetrievalResult(
                chunk_id=str(row["id"]),
                content=row["content"],
                document_name=row["filename"],
                page_number=row["metadata"].get("page_number") if row["metadata"] else None,
                score=float(row["score"]),
                retrieval_method="metadata",
                metadata=dict(row["metadata"] or {}),
            )
            for row in rows
        ]

    async def retrieve(
        self,
        query: str,
        query_expansions: list[str] | None = None,
        metadata_filters: dict | None = None,
        k: int = 20,
    ) -> list[RetrievalResult]:
        all_queries = [query] + (query_expansions or [])

        dense_results, sparse_results, metadata_results = await asyncio.gather(
            self._dense_retrieval(all_queries, k),
            self._sparse_retrieval(query, k),
            self._metadata_retrieval(query, metadata_filters or {}, k),
        )

        return dense_results, sparse_results, metadata_results
