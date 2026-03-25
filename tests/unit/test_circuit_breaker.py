"""Unit tests for circuit breaker state machine (mock Redis)."""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class InMemoryStore:
    """Simple in-memory store for testing circuit breaker without Redis."""
    def __init__(self):
        self._data = {}

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value):
        self._data[key] = str(value).encode() if not isinstance(value, bytes) else value

    def incr(self, key):
        val = int(self._data.get(key, 0))
        val += 1
        self._data[key] = str(val).encode()
        return val


store = InMemoryStore()


def test_circuit_breaker_import():
    from backend.resilience.circuit_breaker import CircuitBreaker, CircuitState, CircuitOpenError
    assert CircuitState.CLOSED == "CLOSED"
    assert CircuitState.OPEN == "OPEN"
    assert CircuitState.HALF_OPEN == "HALF_OPEN"


def test_circuit_open_error():
    from backend.resilience.circuit_breaker import CircuitOpenError
    err = CircuitOpenError("test circuit is OPEN")
    assert "test circuit" in str(err)


def test_circuit_breaker_instantiation():
    from backend.resilience.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(
        name="test",
        failure_threshold=5,
        recovery_timeout=60,
        half_open_max_calls=3,
    )
    assert cb.name == "test"
    assert cb.failure_threshold == 5
    assert cb.recovery_timeout == 60


def test_circuit_state_enum_values():
    from backend.resilience.circuit_breaker import CircuitState
    states = list(CircuitState)
    assert len(states) == 3
    assert CircuitState.CLOSED in states
    assert CircuitState.OPEN in states
    assert CircuitState.HALF_OPEN in states


def test_circuit_breaker_names_are_unique():
    from backend.resilience.circuit_breaker import CircuitBreaker
    cb1 = CircuitBreaker(name="openai")
    cb2 = CircuitBreaker(name="anthropic")
    assert cb1.name != cb2.name
    # Different keys in Redis
    keys1 = cb1._keys()
    keys2 = cb2._keys()
    assert keys1[0] != keys2[0]


@pytest.mark.asyncio
async def test_get_all_states_returns_dict():
    """Test that get_all_states returns a dict (mocked Redis)."""
    from backend.resilience.circuit_breaker import CircuitBreaker
    with patch("backend.resilience.circuit_breaker.aioredis") as mock_redis:
        mock_client = AsyncMock()
        mock_client.keys = AsyncMock(return_value=[])
        mock_redis.from_url.return_value = mock_client
        result = await CircuitBreaker.get_all_states()
    assert isinstance(result, dict)
