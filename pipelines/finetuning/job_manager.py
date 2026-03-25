import json
import uuid
import asyncio
from openai import AsyncOpenAI
from backend.db.pool import get_pool
from backend.config import get_settings
import redis.asyncio as aioredis


async def submit_finetune_job(job_id: str, base_model: str = "gpt-4o-mini-2024-07-18") -> str:
    jsonl_path = f"training_data/{job_id}.jsonl"
    client = AsyncOpenAI()

    with open(jsonl_path, "rb") as f:
        file_resp = await client.files.create(file=f, purpose="fine-tune")

    job = await client.fine_tuning.jobs.create(
        training_file=file_resp.id,
        model=base_model,
    )

    pool = await get_pool()
    await pool.execute(
        "UPDATE finetune_jobs SET provider_job_id=$1, status='training' WHERE id=$2",
        job.id, uuid.UUID(job_id),
    )
    return job.id


async def poll_job_status(job_id: str) -> dict:
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT provider_job_id, status FROM finetune_jobs WHERE id=$1",
        uuid.UUID(job_id),
    )
    if not row or not row["provider_job_id"]:
        return {"status": "unknown"}

    client = AsyncOpenAI()
    job = await client.fine_tuning.jobs.retrieve(row["provider_job_id"])

    if job.status == "succeeded":
        await _handle_success(job_id, job, pool)
    elif job.status in ("failed", "cancelled"):
        await pool.execute(
            "UPDATE finetune_jobs SET status=$1 WHERE id=$2",
            job.status, uuid.UUID(job_id),
        )

    return {"status": job.status, "fine_tuned_model": getattr(job, "fine_tuned_model", None)}


async def _handle_success(job_id: str, job, pool):
    settings = get_settings()
    metrics = {
        "training_loss": getattr(job, "trained_tokens", None),
    }
    await pool.execute(
        """UPDATE finetune_jobs
           SET status='succeeded', provider_job_id=$1, metrics=$2, completed_at=NOW()
           WHERE id=$3""",
        job.fine_tuned_model,
        json.dumps(metrics),
        uuid.UUID(job_id),
    )

    # Register in router:models Redis key
    new_model = {
        "provider": "openai",
        "model": job.fine_tuned_model,
        "task_type": "rag_generation",
        "status": "active",
        "job_id": job_id,
    }
    try:
        r = aioredis.from_url(settings.redis_url)
        existing_raw = await r.get("router:models")
        existing = json.loads(existing_raw) if existing_raw else []
        existing.append(new_model)
        await r.set("router:models", json.dumps(existing))
        await r.aclose()
    except Exception as e:
        print(f"Failed to update router:models: {e}")
