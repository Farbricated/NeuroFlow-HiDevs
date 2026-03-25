# ADR 001 — Vector Store Selection

## Context

NeuroFlow requires a vector store to persist chunk embeddings (1536 dimensions, text-embedding-3-small) and support hybrid search: approximate nearest neighbour (ANN) for dense retrieval and full-text search for sparse retrieval. Candidate options evaluated: Pinecone, Weaviate, Qdrant, and pgvector.

The project already requires Postgres for relational data (documents, pipeline_runs, evaluations, training_pairs). Adding a separate vector database means operating two data stores, two backup strategies, two connection pools, and two sets of consistency guarantees.

## Decision

Use **pgvector** (Postgres extension) with the HNSW index type.

Reasons:
1. **Operational simplicity** — one Postgres instance serves both relational and vector workloads. No separate service, no cross-service transactions, no dual backup.
2. **HNSW performance** — with `m=16, ef_construction=64`, HNSW achieves recall >95% at <10ms p99 for our embedding dimension and dataset sizes (<10M chunks). This matches or exceeds managed vector DBs for our scale.
3. **Native hybrid search** — Postgres full-text search (GIN index + `ts_rank_cd`) runs in the same query as the vector search, enabling true hybrid retrieval without a fan-out to two systems.
4. **Metadata filtering** — `chunks.metadata JSONB` with a GIN index allows arbitrary metadata predicates in the same query as the vector search. Pinecone metadata filtering has known edge cases with high-cardinality filters.
5. **Cost** — pgvector on a $20/month VPS outperforms Pinecone's Starter tier on every dimension for this project's scale.

Pinecone and Weaviate are better choices for multi-tenant SaaS with >100M vectors where operational burden justifies managed infrastructure. Qdrant is excellent for pure vector workloads but lacks native SQL and FTS.

## Consequences

- **Positive**: Simplified infrastructure, SQL joins between chunks and documents/evaluations, native transactions.
- **Negative**: Postgres must be tuned for vector workloads (`shared_buffers`, `work_mem`). At >50M chunks, HNSW index build time becomes significant. If the project scales beyond this, a migration to Qdrant or Weaviate may be warranted.
- **Monitoring**: Track `pg_stat_user_indexes` to watch HNSW index usage. Alert if ANN recall drops below 0.90 in retrieval evaluation benchmarks.
