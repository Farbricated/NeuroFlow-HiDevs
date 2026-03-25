# NeuroFlow-HiDevs

A production-grade Multi-Modal LLM Orchestration Platform with RAG, Fine-Tuning Pipelines & Real-Time Evaluation.

## Architecture

- **Ingestion**: PDF, DOCX, Image, CSV, URL → Chunking → Embeddings → pgvector
- **Retrieval**: Hybrid search (dense + sparse) → RRF fusion → Cross-encoder reranking
- **Generation**: Prompt assembly → Streaming SSE → Citation tracking
- **Evaluation**: LLM-as-judge with RAGAS metrics (faithfulness, relevance, precision, recall)
- **Fine-Tuning**: Training data extraction → MLflow tracking → Model registration
- **Resilience**: Circuit breakers, rate limiting, backpressure

## Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI (async) |
| Vector DB | Postgres + pgvector (HNSW) |
| Cache / Queue | Redis |
| ML Tracking | MLflow |
| Tracing | OpenTelemetry + Jaeger |
| LLM Providers | OpenAI, Anthropic |
| Frontend | Next.js |

## Quick Start

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY, ANTHROPIC_API_KEY, POSTGRES_PASSWORD, REDIS_PASSWORD

cd infra && docker compose up -d
cd ../backend && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Project Structure

```
NeuroFlow-HiDevs/
├── backend/          # FastAPI application
│   ├── api/          # Route handlers
│   ├── db/           # Database pool & migrations
│   ├── models/       # Pydantic schemas
│   ├── providers/    # LLM provider abstraction
│   └── resilience/   # Circuit breakers, rate limiting
├── pipelines/        # Core ML pipelines
│   ├── ingestion/    # Document ingestion
│   ├── retrieval/    # Hybrid retrieval
│   ├── generation/   # RAG generation
│   └── finetuning/   # Fine-tuning pipeline
├── evaluation/       # RAGAS-style evaluation
├── infra/            # Docker Compose + SQL schema
├── docs/             # Architecture docs & ADRs
└── frontend/         # Next.js dashboard
```

## Tasks Completed

- [x] Task 1 — System Architecture, API Contracts & ADRs
- [x] Task 2 — Infrastructure (Postgres/pgvector, Redis, MLflow, FastAPI)
- [x] Task 3 — LLM Provider Abstraction Layer
- [x] Task 4 — Multi-Modal Ingestion Pipeline
- [x] Task 5 — Retrieval Pipeline (Hybrid Search, RRF, Reranking)
- [x] Task 6 — RAG Generation Pipeline (SSE Streaming, Citations)
- [x] Task 7 — Automated Evaluation Framework (RAGAS)
- [x] Task 8 — Named Pipeline System & A/B Comparison
- [x] Task 9 — Fine-Tuning Pipeline (MLflow + Model Registration)
- [x] Task 10 — Production Async Resilience
