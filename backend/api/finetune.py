import uuid
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.db.pool import get_pool
from pipelines.finetuning.extractor import extract_training_pairs, preview_training_pairs
from pipelines.finetuning.tracker import start_training_job
from pipelines.finetuning.job_manager import submit_finetune_job

router = APIRouter()


class FinetuneRequest(BaseModel):
    base_model: str = "gpt-4o-mini-2024-07-18"
    min_quality_score: float = 0.82


@router.post("/finetune/jobs", status_code=202)
async def create_finetune_job(body: FinetuneRequest):
    job_id = str(uuid.uuid4())
    pool = await get_pool()

    await pool.execute(
        """INSERT INTO finetune_jobs (id, base_model, status)
           VALUES ($1, $2, 'extracting')""",
        uuid.UUID(job_id), body.base_model,
    )

    pairs = await extract_training_pairs(job_id, min_quality=body.min_quality_score)
    if not pairs:
        await pool.execute(
            "UPDATE finetune_jobs SET status='failed' WHERE id=$1", uuid.UUID(job_id)
        )
        raise HTTPException(status_code=422, detail="No qualifying training pairs found")

    await pool.execute(
        "UPDATE finetune_jobs SET training_pair_count=$1, status='submitting' WHERE id=$2",
        len(pairs), uuid.UUID(job_id),
    )

    mlflow_run_id = start_training_job(job_id, body.base_model, pairs)
    await pool.execute(
        "UPDATE finetune_jobs SET mlflow_run_id=$1 WHERE id=$2",
        mlflow_run_id, uuid.UUID(job_id),
    )

    try:
        await submit_finetune_job(job_id, body.base_model)
    except Exception as e:
        await pool.execute(
            "UPDATE finetune_jobs SET status='failed' WHERE id=$1", uuid.UUID(job_id)
        )
        raise HTTPException(status_code=500, detail=f"Job submission failed: {e}")

    return {"job_id": job_id, "training_pair_count": len(pairs)}


@router.get("/finetune/jobs")
async def list_finetune_jobs():
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT * FROM finetune_jobs ORDER BY created_at DESC"
    )
    return [dict(r) for r in rows]


@router.get("/finetune/jobs/{job_id}")
async def get_finetune_job(job_id: str):
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT * FROM finetune_jobs WHERE id=$1", uuid.UUID(job_id)
    )
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    result = dict(row)
    if result.get("mlflow_run_id"):
        from backend.config import get_settings
        settings = get_settings()
        result["mlflow_run_url"] = f"{settings.mlflow_tracking_uri}/#/experiments/1/runs/{result['mlflow_run_id']}"

    return result


@router.get("/finetune/training-data/preview")
async def preview_training_data(min_quality: float = 0.82):
    pairs = await preview_training_pairs(min_quality=min_quality)
    return {"count": len(pairs), "samples": pairs}
