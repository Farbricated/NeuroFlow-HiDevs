import asyncio
import time
import httpx
import redis.asyncio as aioredis
from backend.db.pool import get_pool
from backend.config import get_settings


async def check_postgres() -> dict:
    start = time.monotonic()
    try:
        pool = await get_pool()
        await pool.fetchval("SELECT 1")
        return {"status": "ok", "latency_ms": round((time.monotonic() - start) * 1000)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def check_redis() -> dict:
    start = time.monotonic()
    settings = get_settings()
    try:
        client = aioredis.from_url(settings.redis_url)
        await client.ping()
        await client.aclose()
        return {"status": "ok", "latency_ms": round((time.monotonic() - start) * 1000)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def check_mlflow() -> dict:
    start = time.monotonic()
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.mlflow_tracking_uri}/health")
            resp.raise_for_status()
        return {"status": "ok", "latency_ms": round((time.monotonic() - start) * 1000)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def run_all_checks() -> dict:
    pg, redis, mlflow = await asyncio.gather(
        check_postgres(), check_redis(), check_mlflow()
    )
    all_ok = all(c["status"] == "ok" for c in [pg, redis, mlflow])
    return {
        "postgres": pg,
        "redis": redis,
        "mlflow": mlflow,
        "all_ok": all_ok,
    }
