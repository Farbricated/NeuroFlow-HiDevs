# NeuroFlow Operations Runbook

## Incident 1 — High Query Latency (P95 > 10s)

**Symptoms**: Users report slow responses; P95 latency gauge in Grafana > 10s.

**Checklist**:
1. Jaeger: open http://localhost:16686 → search for slow traces → identify which span is slow
2. Redis: `redis-cli -a $REDIS_PASSWORD INFO memory` — check `used_memory_human` and `hit_rate`
3. Postgres: `SELECT * FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;`
4. API replicas: `docker stats` — check CPU/memory

**Remediation**:
- If **embedding call** slow: check OpenAI status page; circuit breaker may have opened
- If **reranker** slow: reduce `RERANK_TOP_N` in reranker.py from 40 to 20
- If **Postgres** slow: `REINDEX INDEX CONCURRENTLY chunks_embedding_hnsw;`
- If **Redis cache miss rate high**: `redis-cli -a $REDIS_PASSWORD FLUSHDB` (clears stale entries)
- If **API overloaded**: `docker compose scale api=4`

---

## Incident 2 — Evaluation Scores Degrading

**Symptoms**: Grafana quality monitor shows overall_score dropping below 0.70.

**Checklist**:
1. Which pipeline and which metric? (`GET /api/v1/evaluations/aggregate?period_days=1`)
2. Recent ingested docs: `SELECT filename, status, created_at FROM documents ORDER BY created_at DESC LIMIT 20;`
3. MLflow: check recent fine-tuning job — did a new model get registered?
4. Compare last 100 evaluations vs previous 100 for pattern

**Remediation**:
- If **faithfulness drops**: context noise. Reduce `top_k_after_rerank` from 8 to 5
- If **answer_relevance drops**: query expansion may be generating bad expansions. Disable temporarily
- If **after fine-tuning job**: revert model via `DELETE FROM redis router:models` last entry
- Document the issue in `evaluation/improvement_log.md`

---

## Incident 3 — LLM Provider Circuit Breaker Open

**Symptoms**: `GET /health` shows `circuit_breakers.openai.state = OPEN`; all queries failing.

**Checklist**:
1. `curl http://localhost:8000/health | jq .checks.circuit_breakers`
2. Check provider status: https://status.openai.com or https://status.anthropic.com
3. Check recent error logs: `docker logs neuroflow-api-1 --since 10m | grep ERROR`

**Remediation**:
- Wait for `recovery_timeout` (60s default) — circuit moves to HALF_OPEN automatically
- If provider is down: switch default model in ModelRouter to the other provider
- Manual circuit reset: `redis-cli -a $REDIS_PASSWORD DEL circuit:openai:state circuit:openai:failure_count`
- After recovery: verify with `GET /health`

---

## Incident 4 — Ingestion Queue Depth > 100

**Symptoms**: `GET /health` shows `queue_depth > 100`; new ingestions return 503.

**Checklist**:
1. `redis-cli -a $REDIS_PASSWORD LLEN queue:ingest` — current depth
2. `docker logs neuroflow-worker-1 --since 5m` — look for repeated errors
3. Check for stuck document: `SELECT id, filename, status, created_at FROM documents WHERE status='processing' AND created_at < NOW() - INTERVAL '10 minutes';`

**Remediation**:
- If **worker crashed**: `docker compose restart worker`
- If **stuck document**: `UPDATE documents SET status='failed' WHERE id='<uuid>';` then `redis-cli -a $REDIS_PASSWORD DEL queue:ingest` (nuclear option — re-ingest manually)
- If **embedding API slow**: reduce worker concurrency or increase timeout
- Scale workers: `docker compose scale worker=4`

---

## Incident 5 — Database Disk Usage > 80%

**Symptoms**: Postgres disk alert; disk usage climbing.

**Checklist**:
1. Largest tables: `SELECT relname, pg_size_pretty(pg_total_relation_size(relid)) FROM pg_catalog.pg_statio_user_tables ORDER BY pg_total_relation_size(relid) DESC;`
2. Old evaluations: `SELECT COUNT(*), MIN(evaluated_at), MAX(evaluated_at) FROM evaluations;`
3. Dead chunks (archived documents): `SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.status='failed';`

**Remediation**:
```sql
-- Delete evaluations older than 180 days
DELETE FROM evaluations WHERE evaluated_at < NOW() - INTERVAL '180 days';

-- Delete runs older than 90 days (no associated evaluations)
DELETE FROM pipeline_runs WHERE created_at < NOW() - INTERVAL '90 days'
  AND id NOT IN (SELECT run_id FROM evaluations);

-- Delete chunks for failed documents
DELETE FROM chunks WHERE document_id IN (SELECT id FROM documents WHERE status='failed');

-- Reclaim space
VACUUM ANALYZE chunks;
VACUUM ANALYZE evaluations;
REINDEX INDEX CONCURRENTLY chunks_embedding_hnsw;
```
