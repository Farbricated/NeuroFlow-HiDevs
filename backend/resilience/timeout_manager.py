import asyncio

TIMEOUTS = {
    "embedding": 10,
    "chat_completion": 60,
    "reranking": 15,
    "evaluation": 120,
    "file_extraction": 30,
    "url_fetch": 15,
}


async def with_timeout(coro, task_type: str):
    timeout = TIMEOUTS.get(task_type, 30)
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        import redis.asyncio as aioredis
        from backend.config import get_settings
        try:
            r = aioredis.from_url(get_settings().redis_url)
            await r.incr(f"timeouts:{task_type}")
            await r.aclose()
        except Exception:
            pass
        raise TimeoutError(f"Task '{task_type}' timed out after {timeout}s")
