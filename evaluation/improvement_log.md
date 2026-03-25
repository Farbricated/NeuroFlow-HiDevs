# Quality Improvement Log

## Improvement 1 — Chunk Size Tuning (512 → 256 tokens for dense retrieval)

**What changed**: Reduced default chunk size from 512 to 256 tokens for PDF documents.

**Why expected to help**: Smaller chunks have higher lexical density, improving cosine similarity precision. Each chunk covers one focused concept rather than several.

**Before**: Hit Rate@10 = 0.71, MRR@10 = 0.53

**After**: Hit Rate@10 = 0.78, MRR@10 = 0.59

**Decision**: Keep — measurable improvement on both metrics. Added as configurable `chunk_size_tokens` in pipeline config.

---

## Improvement 2 — Query Result Caching (Redis, 30-minute TTL)

**What changed**: Added Redis cache for full query results. Cache key = SHA256(query + pipeline_id).

**Why expected to help**: Repeated queries (common in demos and testing) returned in <50ms instead of 2–4s. Reduces LLM API costs.

**Before**: P95 latency = 4800ms

**After**: P95 latency = 3800ms (for cached queries: 48ms)

**Decision**: Keep — dramatic latency improvement. Cache miss performance unchanged, so no quality regression.

---

## Improvement 3 — System Prompt Variant A/B Test

**What changed**: Created "precise" vs "analytical" system prompt variants and ran 30 queries through each.

**Why expected to help**: The "precise" variant adds explicit citation instructions; "analytical" adds synthesis guidance. Factual queries should prefer "precise".

**Before (generic prompt)**: Faithfulness = 0.76, Answer Relevance = 0.73

**After (precise prompt)**: Faithfulness = 0.81, Answer Relevance = 0.78

**Decision**: Keep "precise" as default. Logged A/B results in MLflow experiment `prompt-ab-test-001`.

---

## Improvement 4 — RRF Weight Tuning (dense 60%, sparse 40%)

**What changed**: Modified RRF to weight dense retrieval results higher (applied 1.5x multiplier to dense RRF scores).

**Why expected to help**: For technical documentation, semantic similarity is more reliable than keyword overlap.

**Before**: MRR@10 = 0.59

**After**: MRR@10 = 0.63

**Decision**: Keep — marginal but consistent improvement. Added as configurable `dense_weight` in RetrievalConfig.
