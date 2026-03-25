import asyncio
import json
import time
from contextlib import asynccontextmanager
from enum import Enum
import redis.asyncio as aioredis
from backend.config import get_settings


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitOpenError(Exception):
    pass


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._settings = get_settings()

    def _keys(self):
        return (
            f"circuit:{self.name}:state",
            f"circuit:{self.name}:failure_count",
            f"circuit:{self.name}:opened_at",
            f"circuit:{self.name}:half_open_calls",
        )

    async def _get_redis(self) -> aioredis.Redis:
        return aioredis.from_url(self._settings.redis_url)

    async def get_state(self) -> CircuitState:
        r = await self._get_redis()
        state_key, _, opened_key, _ = self._keys()
        try:
            state = await r.get(state_key)
            if state is None:
                return CircuitState.CLOSED
            state = state.decode()
            if state == CircuitState.OPEN:
                opened_at = await r.get(opened_key)
                if opened_at and time.time() - float(opened_at) >= self.recovery_timeout:
                    await r.set(state_key, CircuitState.HALF_OPEN)
                    await r.set(f"circuit:{self.name}:half_open_calls", 0)
                    return CircuitState.HALF_OPEN
            return CircuitState(state)
        finally:
            await r.aclose()

    async def record_success(self):
        r = await self._get_redis()
        state_key, failure_key, _, half_key = self._keys()
        try:
            state = await self.get_state()
            if state == CircuitState.HALF_OPEN:
                calls = int((await r.incr(half_key)) or 0)
                if calls >= self.half_open_max_calls:
                    await r.set(state_key, CircuitState.CLOSED)
                    await r.set(failure_key, 0)
            else:
                await r.set(failure_key, 0)
        finally:
            await r.aclose()

    async def record_failure(self):
        r = await self._get_redis()
        state_key, failure_key, opened_key, _ = self._keys()
        try:
            state = await self.get_state()
            if state == CircuitState.HALF_OPEN:
                await r.set(state_key, CircuitState.OPEN)
                await r.set(opened_key, str(time.time()))
                return

            count = int((await r.incr(failure_key)) or 0)
            if count >= self.failure_threshold:
                await r.set(state_key, CircuitState.OPEN)
                await r.set(opened_key, str(time.time()))
        finally:
            await r.aclose()

    @asynccontextmanager
    async def __aenter__(self):
        state = await self.get_state()
        if state == CircuitState.OPEN:
            raise CircuitOpenError(f"Circuit '{self.name}' is OPEN")
        try:
            yield self
            await self.record_success()
        except CircuitOpenError:
            raise
        except Exception:
            await self.record_failure()
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @staticmethod
    async def get_all_states() -> dict:
        settings = get_settings()
        r = aioredis.from_url(settings.redis_url)
        try:
            keys = await r.keys("circuit:*:state")
            result = {}
            for key in keys:
                name = key.decode().split(":")[1]
                state = (await r.get(key) or b"CLOSED").decode()
                failure_count = int((await r.get(f"circuit:{name}:failure_count") or 0))
                opened_at = await r.get(f"circuit:{name}:opened_at")
                result[name] = {
                    "state": state,
                    "failure_count": failure_count,
                    "opened_at": opened_at.decode() if opened_at else None,
                }
            return result
        finally:
            await r.aclose()
