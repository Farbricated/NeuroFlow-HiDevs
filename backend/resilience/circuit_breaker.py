import enum
from typing import Dict, Any

try:
    import aioredis  # type: ignore
except ImportError:
    aioredis = None


class CircuitState(str, enum.Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitOpenError(Exception):
    pass


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: int = 60, half_open_max_calls: int = 3):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

    def _keys(self):
        return [f"cb:{self.name}:state", f"cb:{self.name}:fails", f"cb:{self.name}:half"]

    @classmethod
    async def get_all_states(cls) -> Dict[str, Any]:
        return {}
