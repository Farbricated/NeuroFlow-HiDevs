import uuid
import json
from fastapi import APIRouter, HTTPException
from backend.db.pool import get_pool
from backend.models.pipeline import PipelineConfig

router = APIRouter()


@router.post("/pipelines", status_code=201)
async def create_pipeline(body: PipelineConfig):
    pool = await get_pool()
    existing = await pool.fetchrow("SELECT id FROM pipelines WHERE name=$1", body.name)
    if existing:
        raise HTTPException(status_code=409, detail="Pipeline name already exists")

    pipeline_id = uuid.uuid4()
    config_json = json.loads(body.model_dump_json())

    await pool.execute(
        "INSERT INTO pipelines (id, name, config, version) VALUES ($1, $2, $3, 1)",
        pipeline_id, body.name, json.dumps(config_json),
    )
    await pool.execute(
        "INSERT INTO pipeline_versions (pipeline_id, version, config) VALUES ($1, 1, $2)",
        pipeline_id, json.dumps(config_json),
    )
    return {"pipeline_id": str(pipeline_id), "name": body.name, "version": 1}


@router.get("/pipelines")
async def list_pipelines():
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT p.id, p.name, p.version, p.status, p.created_at,
                  COUNT(pr.id) AS run_count,
                  AVG(e.overall_score) AS avg_score
           FROM pipelines p
           LEFT JOIN pipeline_runs pr ON pr.pipeline_id = p.id
           LEFT JOIN evaluations e ON e.run_id = pr.id
           WHERE p.status != 'archived'
           GROUP BY p.id, p.name, p.version, p.status, p.created_at
           ORDER BY p.created_at DESC"""
    )
    return [dict(r) for r in rows]


@router.get("/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT * FROM pipelines WHERE id=$1 AND status!='archived'",
        uuid.UUID(pipeline_id),
    )
    if not row:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return dict(row)


@router.patch("/pipelines/{pipeline_id}")
async def update_pipeline(pipeline_id: str, body: PipelineConfig):
    pool = await get_pool()
    current = await pool.fetchrow(
        "SELECT version FROM pipelines WHERE id=$1", uuid.UUID(pipeline_id)
    )
    if not current:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    new_version = current["version"] + 1
    config_json = json.loads(body.model_dump_json())

    await pool.execute(
        "UPDATE pipelines SET config=$1, version=$2, name=$3 WHERE id=$4",
        json.dumps(config_json), new_version, body.name, uuid.UUID(pipeline_id),
    )
    await pool.execute(
        "INSERT INTO pipeline_versions (pipeline_id, version, config) VALUES ($1, $2, $3)",
        uuid.UUID(pipeline_id), new_version, json.dumps(config_json),
    )
    return {"pipeline_id": pipeline_id, "version": new_version}


@router.delete("/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    pool = await get_pool()
    await pool.execute(
        "UPDATE pipelines SET status='archived' WHERE id=$1", uuid.UUID(pipeline_id)
    )
    return {"status": "archived"}


@router.get("/pipelines/{pipeline_id}/runs")
async def get_pipeline_runs(pipeline_id: str, page: int = 1, per_page: int = 50):
    pool = await get_pool()
    offset = (page - 1) * per_page
    rows = await pool.fetch(
        """SELECT pr.id, pr.query, pr.latency_ms, pr.input_tokens,
                  pr.output_tokens, pr.model_used, pr.status, pr.created_at,
                  e.overall_score
           FROM pipeline_runs pr
           LEFT JOIN evaluations e ON e.run_id = pr.id
           WHERE pr.pipeline_id = $1
           ORDER BY pr.created_at DESC
           LIMIT $2 OFFSET $3""",
        uuid.UUID(pipeline_id), per_page, offset,
    )
    total = await pool.fetchval(
        "SELECT COUNT(*) FROM pipeline_runs WHERE pipeline_id=$1", uuid.UUID(pipeline_id)
    )
    return {"items": [dict(r) for r in rows], "total": total, "page": page, "per_page": per_page}


@router.get("/pipelines/{pipeline_id}/analytics")
async def get_pipeline_analytics(pipeline_id: str):
    pool = await get_pool()
    pid = uuid.UUID(pipeline_id)

    latency = await pool.fetchrow(
        """SELECT
             percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50,
             percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95,
             percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99,
             AVG(latency_ms) AS avg_latency,
             AVG(retrieval_latency_ms) AS avg_retrieval_latency
           FROM pipeline_runs
           WHERE pipeline_id=$1 AND status='complete'""",
        pid,
    )

    scores = await pool.fetchrow(
        """SELECT AVG(e.faithfulness) AS avg_faithfulness,
                  AVG(e.answer_relevance) AS avg_answer_relevance,
                  AVG(e.context_precision) AS avg_context_precision,
                  AVG(e.context_recall) AS avg_context_recall,
                  AVG(e.overall_score) AS avg_overall_score
           FROM evaluations e
           JOIN pipeline_runs pr ON pr.id = e.run_id
           WHERE pr.pipeline_id=$1""",
        pid,
    )

    daily = await pool.fetch(
        """SELECT DATE(created_at) AS day, COUNT(*) AS queries
           FROM pipeline_runs
           WHERE pipeline_id=$1 AND created_at > NOW() - INTERVAL '30 days'
           GROUP BY day ORDER BY day""",
        pid,
    )

    return {
        "latency": dict(latency) if latency else {},
        "evaluation_scores": dict(scores) if scores else {},
        "daily_queries": [dict(r) for r in daily],
    }


@router.get("/evaluations")
async def list_evaluations(page: int = 1, per_page: int = 20,
                            pipeline_id: str | None = None, min_score: float | None = None):
    pool = await get_pool()
    offset = (page - 1) * per_page
    conditions = []
    params: list = []
    idx = 1

    if pipeline_id:
        conditions.append(f"pr.pipeline_id = ${idx}")
        params.append(uuid.UUID(pipeline_id))
        idx += 1
    if min_score is not None:
        conditions.append(f"e.overall_score >= ${idx}")
        params.append(min_score)
        idx += 1

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    params += [per_page, offset]

    rows = await pool.fetch(
        f"""SELECT e.*, pr.query FROM evaluations e
            JOIN pipeline_runs pr ON pr.id = e.run_id
            {where}
            ORDER BY e.evaluated_at DESC
            LIMIT ${idx} OFFSET ${idx+1}""",
        *params,
    )
    total = await pool.fetchval(
        f"SELECT COUNT(*) FROM evaluations e JOIN pipeline_runs pr ON pr.id = e.run_id {where}",
        *params[:-2],
    )
    return {"items": [dict(r) for r in rows], "total": total, "page": page, "per_page": per_page}


@router.get("/evaluations/aggregate")
async def aggregate_evaluations(period_days: int = 7):
    pool = await get_pool()
    row = await pool.fetchrow(
        """SELECT
             AVG(faithfulness) AS avg_faithfulness,
             AVG(answer_relevance) AS avg_answer_relevance,
             AVG(context_precision) AS avg_context_precision,
             AVG(context_recall) AS avg_context_recall,
             AVG(overall_score) AS avg_overall_score,
             COUNT(*) AS total_runs
           FROM evaluations
           WHERE evaluated_at > NOW() - ($1 || ' days')::INTERVAL""",
        str(period_days),
    )
    return {"period_days": period_days, **dict(row)}
