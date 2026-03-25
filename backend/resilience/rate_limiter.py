import time
import asyncio
from fastapi import Request, HTTPException
import redis.asyncio as aioredis
from backend.config import get_settings

PROVIDER_LIMITS = {
    "openai": {"rpm": 3000, "refill_per_second": 50},
    "anthropic": {"rpm": 1000, "refill_per_second": 17},
}


async def _get_redis() -> aioredis.Redis:
    settings = get_settings()
    return aioredis.from_url(settings.redis_url)


async def acquire_provider_token(provider: str) -> bool:
    """Token bucket rate limiter for LLM provider calls."""
    config = PROVIDER_LIMITS.get(provider, {"rpm": 1000, "refill_per_second": 17})
    bucket_key = f"rpb:{provider}:tokens"
    last_refill_key = f"rpb:{provider}:last_refill"

    r = await _get_redis()
    try:
        now = time.time()
        last_refill = float(await r.get(last_refill_key) or now)
        elapsed = now - last_refill
        tokens = float(await r.get(bucket_key) or config["rpm"])

        # Refill tokens
        new_tokens = min(
            config["rpm"],
            tokens + elapsed * config["refill_per_second"],
        )

        if new_tokens < 1:
            wait = (1 - new_tokens) / config["refill_per_second"]
            await asyncio.sleep(wait)
            new_tokens = 1

        await r.set(bucket_key, new_tokens - 1)
        await r.set(last_refill_key, now)
        return True
    finally:
        await r.aclose()


async def check_endpoint_rate_limit(request: Request, limit: int, window_seconds: int):
    """Sliding window rate limiter for API endpoints."""
    client_ip = request.client.host if request.client else "unknown"
    path = request.url.path
    key = f"ratelimit:{path}:{client_ip}"

    r = await _get_redis()
    try:
        now = int(time.time())
        window_start = now - window_seconds

        pipe = r.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, window_seconds)
        results = await pipe.execute()

        current_count = results[1]
        if current_count >= limit:
            retry_after = window_seconds - (now - window_start)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )
    finally:
        await r.aclose()


async def check_pipeline_rate_limit(pipeline_id: str, limit_rpm: int):
    """Per-pipeline rate limiting."""
    bucket_key = f"rpb:pipeline:{pipeline_id}:tokens"
    r = await _get_redis()
    try:
        tokens = float(await r.get(bucket_key) or limit_rpm)
        if tokens < 1:
            raise HTTPException(status_code=429, detail="Pipeline rate limit exceeded")
        await r.set(bucket_key, tokens - 1)
        await r.expire(bucket_key, 60)
    finally:
        await r.aclose()
