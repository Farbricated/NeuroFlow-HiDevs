# API Contracts

All endpoints require `Authorization: Bearer <token>` unless marked public.
Base URL: `http://localhost:8000`

---

## POST /ingest
Ingest a file or URL into NeuroFlow.

**Auth**: Required | **Rate limit**: 10 req/hour per IP

**Request** (multipart/form-data or JSON):
```json
{ "url": "https://example.com/paper.pdf" }
```
Or form-data with `file` (binary, max 100MB).

**Response 200**:
```json
{
  "document_id": "uuid",
  "status": "queued",
  "duplicate": false
}
```
**Errors**: `400` invalid file type | `413` file too large | `503` queue full

---

## POST /query
Execute a RAG query.

**Auth**: Required | **Rate limit**: 60 req/min per IP

**Request**:
```json
{
  "query": "What is HNSW indexing?",
  "pipeline_id": "uuid",
  "stream": true
}
```
**Response 200** (stream=false):
```json
{
  "run_id": "uuid",
  "generation": "Based on the provided sources...",
  "citations": [{"source": "Source 1", "chunk_id": "uuid", "document": "paper.pdf", "page": 3}],
  "latency_ms": 1240
}
```
**Response 200** (stream=true): `{ "run_id": "uuid" }` — then connect to SSE stream.

---

## GET /query/{run_id}/stream
SSE stream for a running query.

**Auth**: Required | **Rate limit**: none

**Events**:
```
data: {"type": "retrieval_start"}
data: {"type": "retrieval_complete", "chunk_count": 8, "sources": ["doc.pdf"]}
data: {"type": "token", "delta": "Based"}
data: {"type": "done", "run_id": "uuid", "citations": [...]}
data: {"type": "keepalive"}   // every 15s
```

---

## GET /evaluations
Paginated evaluation results.

**Query params**: `page=1&per_page=20&pipeline_id=uuid&min_score=0.7`

**Response 200**:
```json
{
  "items": [{
    "id": "uuid", "run_id": "uuid",
    "faithfulness": 0.91, "answer_relevance": 0.87,
    "context_precision": 0.78, "context_recall": 0.82,
    "overall_score": 0.86, "evaluated_at": "2024-01-15T10:00:00Z"
  }],
  "total": 142, "page": 1, "per_page": 20
}
```

---

## GET /evaluations/aggregate
Rolling quality metrics.

**Response 200**:
```json
{
  "period_days": 7,
  "avg_faithfulness": 0.88,
  "avg_answer_relevance": 0.84,
  "avg_context_precision": 0.76,
  "avg_context_recall": 0.80,
  "avg_overall_score": 0.83,
  "total_runs": 312
}
```

---

## POST /pipelines
Create a named pipeline configuration.

**Request**: Full PipelineConfig JSON (see data-models.md)

**Response 201**: `{ "pipeline_id": "uuid", "name": "legal-research-v2" }`

**Errors**: `422` schema validation failed | `409` name already exists

---

## GET /pipelines/{id}/runs
Pipeline execution history.

**Query params**: `page=1&per_page=50`

**Response 200**:
```json
{
  "items": [{
    "run_id": "uuid", "query": "...", "latency_ms": 1200,
    "input_tokens": 2100, "output_tokens": 340,
    "model_used": "gpt-4o-mini", "overall_score": 0.87
  }]
}
```

---

## POST /finetune/jobs
Submit a fine-tuning job.

**Request**: `{ "base_model": "gpt-4o-mini-2024-07-18", "min_quality_score": 0.82 }`

**Response 202**: `{ "job_id": "uuid", "training_pair_count": 47 }`

---

## GET /finetune/jobs/{id}
Fine-tuning job status and metrics.

**Response 200**:
```json
{
  "job_id": "uuid",
  "status": "running",
  "provider_job_id": "ftjob-abc123",
  "training_pair_count": 47,
  "mlflow_run_url": "http://localhost:5000/...",
  "metrics": { "training_loss": 0.42, "validation_loss": 0.51 }
}
```

---

## GET /health
System health check. **Auth**: None (public)

**Response 200**:
```json
{
  "status": "ok",
  "checks": {
    "postgres": {"status": "ok", "latency_ms": 3},
    "redis": {"status": "ok", "latency_ms": 1},
    "mlflow": {"status": "ok", "latency_ms": 45},
    "circuit_breakers": { "openai": {"state": "closed"} },
    "queue_depth": 5
  }
}
```
Status values: `ok` | `degraded` | `critical`

---

## GET /metrics
Prometheus-format metrics. **Auth**: None (public)

Returns text/plain Prometheus exposition format.
