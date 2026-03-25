from fastapi import HTTPException
import redis.asyncio as aioredis


async def check_queue_depth(r: aioredis.Redis, settings) -> None:
    depth = await r.llen("queue:ingest")
    if depth >= settings.queue_block_depth:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "ingestion_queue_full",
                "queue_depth": depth,
                "retry_after": 30,
            },
        )
    if depth >= settings.queue_warn_depth:
        # Will return 202 with warning — set header via response object
        pass


async def get_queue_depth(r: aioredis.Redis) -> int:
    return await r.llen("queue:ingest")
