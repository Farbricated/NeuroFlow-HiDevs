# NeuroFlow — System Architecture

## 1. Ingestion Subsystem

**Data flow: File upload → queryable vector**

```
[Client] --POST /ingest--> [FastAPI]
    |
    +--> SHA256 hash check (deduplication)
    |         |-- duplicate found --> return existing doc_id
    |
    +--> Create documents row (status=queued)
    +--> Redis LPUSH queue:ingest {document_id, file_path, source_type}
    +--> Return {document_id, status: "queued"}

[Worker] --BRPOP queue:ingest-->
    |
    +--> Extractor (by source_type)
    |       ├── PDF  → pypdfium2 + pdfplumber + pytesseract (scanned)
    |       ├── DOCX → python-docx
    |       ├── Image → vision LLM + pytesseract
    |       ├── CSV  → pandas (summary or markdown table)
    |       └── URL  → httpx + trafilatura
    |
    +--> Chunker (strategy auto-selected)
    |       ├── fixed_size   (512 tokens, 64 overlap)
    |       ├── semantic     (cosine similarity split)
    |       └── hierarchical (heading-aware)
    |
    +--> Embed chunks (text-embedding-3-small, batch 100)
    |
    +--> INSERT INTO chunks (content, embedding, metadata)
    +--> UPDATE documents SET status='complete', chunk_count=N
    +--> Emit OTel span + structured log
```

## 2. Retrieval Subsystem

**Data flow: User query → ranked context window**

```
[Query] --> QueryProcessor
    |
    +--> Query expansion (LLM: 2-3 alternative phrasings)
    +--> Metadata filter extraction (year, topic, etc.)
    +--> Query type classification (factual/analytical/comparative/procedural)

asyncio.gather([
    Dense retrieval   → pgvector HNSW <=> cosine search
    Sparse retrieval  → PostgreSQL FTS plainto_tsquery ts_rank_cd
    Metadata retrieval → filtered vector search on chunks.metadata @> JSONB
])
    |
    +--> Reciprocal Rank Fusion (k=60)
    |       score = Σ 1/(k + rank_i) across lists
    |
    +--> Cross-encoder reranking (top-40 → top-K)
    |       API-based or local cross-encoder/ms-marco-MiniLM-L-6-v2
    |
    +--> Context assembly (token-budget aware, 4000 tokens default)
    +--> Return {assembled_context, chunks_used, sources}
```

## 3. Generation Subsystem

**Data flow: Context window → streamed cited response**

```
[Context + Query] --> PromptBuilder
    |
    +--> Select system prompt variant by query type
    +--> Inject context between <context> tags
    +--> Log assembled prompt to pipeline_runs (pre-call)

--> ModelRouter (routing_criteria from pipeline config)
    |
    +--> provider.stream(messages)
    |       Token-by-token SSE yield to client
    |
    +--> Accumulate full response
    +--> Parse [Source N] citations → resolve to chunk_ids
    +--> UPDATE pipeline_runs (generation, tokens, latency, status=complete)
    +--> LPUSH queue:evaluate {run_id} (async, non-blocking)
```

## 4. Evaluation Subsystem

**Metrics computed per pipeline run**

```
[run_id] --> EvaluationJudge (asyncio.gather)
    |
    ├── Faithfulness    = supported_claims / total_claims
    ├── AnswerRelevance = mean cosine_sim(original_query, generated_questions)
    ├── ContextPrecision = Σ useful[i]/i / Σ 1/i  (rank-weighted)
    └── ContextRecall   = attributable_sentences / total_sentences

overall_score = 0.35*F + 0.30*AR + 0.20*CP + 0.15*CR

--> INSERT INTO evaluations
--> IF overall_score > 0.8: INSERT INTO training_pairs
--> Emit OTel span evaluation.judge {all 4 metrics}
```

Rolling aggregates available via GET /evaluations/aggregate.

## 5. Fine-Tuning Subsystem

**Closing the loop: evaluation → training → better models**

```
[Trigger POST /finetune/jobs] --> Extractor
    |
    +--> SELECT training_pairs WHERE quality_score >= 0.82
    |       AND included_in_job IS NULL
    |       AND (user_rating >= 4 OR user_rating IS NULL)
    |
    +--> Validate pairs (length, citations, PII check, faithfulness)
    +--> Write training_data/{job_id}.jsonl (OpenAI JSONL format)

--> MLflow start_run
    +--> Log params (base_model, pair_count, avg_quality)
    +--> Log artifact (jsonl file)

--> OpenAI files.create + fine_tuning.jobs.create
    +--> arq poll every 60s

[Job succeeded]
    +--> UPDATE finetune_jobs status=succeeded
    +--> Redis LPUSH router:models {new model config}
    +--> mlflow.register_model
    +--> ModelRouter now routes domain queries to fine-tuned model
```
