import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.config import get_settings
from backend.db.pool import create_pool
import redis.asyncio as aioredis


async def run_worker():
    settings = get_settings()
    await create_pool(settings.asyncpg_dsn)
    r = aioredis.from_url(settings.redis_url)
    print("Worker started. Listening on queue:ingest and queue:evaluate...")

    while True:
        try:
            # Ingest queue
            item = await r.brpop("queue:ingest", timeout=1)
            if item:
                _, data = item
                job = json.loads(data)
                print(f"Processing ingest job: {job.get('document_id')}")
                try:
                    from pipelines.ingestion.pipeline import process_document
                    await process_document(job)
                except Exception as e:
                    print(f"Ingest job failed: {e}")

            # Evaluate queue
            item = await r.brpop("queue:evaluate", timeout=1)
            if item:
                _, data = item
                job = json.loads(data)
                run_id = job.get("run_id")
                print(f"Processing evaluation for run: {run_id}")
                try:
                    from evaluation.judge import judge_run
                    await judge_run(run_id)
                except Exception as e:
                    print(f"Evaluation failed: {e}")

        except Exception as e:
            print(f"Worker error: {e}")
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(run_worker())
