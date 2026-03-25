import json
import time
from opentelemetry import trace
import redis.asyncio as aioredis
from backend.providers.base import BaseLLMProvider, ChatMessage, GenerationResult, RoutingCriteria
from backend.providers.router import ModelRouter
from backend.providers.openai_provider import OpenAIProvider
from backend.config import get_settings

tracer = trace.get_tracer(__name__)
_client: "NeuroFlowClient | None" = None


class FallbackChain:
    """Try providers in order; fall back on non-retryable errors."""
    def __init__(self, providers: list[BaseLLMProvider]):
        self._providers = providers

    async def complete(self, messages: list[ChatMessage], **kwargs) -> GenerationResult:
        last_error = None
        for provider in self._providers:
            try:
                return await provider.complete(messages, **kwargs)
            except Exception as e:
                last_error = e
                continue
        raise last_error


class NeuroFlowClient:
    def __init__(self):
        self._settings = get_settings()
        self._router = ModelRouter()
        self._embed_provider = OpenAIProvider()  # Always use OpenAI for embeddings
        self._redis: aioredis.Redis | None = None

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(self._settings.redis_url)
        return self._redis

    async def _track_usage(self, model: str, result: GenerationResult):
        try:
            r = await self._get_redis()
            pipe = r.pipeline()
            pipe.incr(f"metrics:model:{model}:calls")
            pipe.incrbyfloat(f"metrics:model:{model}:cost_usd", result.cost_usd)
            await pipe.execute()
        except Exception:
            pass  # Metrics are best-effort

    async def chat(
        self,
        messages: list[ChatMessage],
        routing_criteria: RoutingCriteria | None = None,
        **kwargs,
    ) -> GenerationResult:
        criteria = routing_criteria or RoutingCriteria()
        provider = await self._router.select(criteria)

        with tracer.start_as_current_span("llm.complete") as span:
            span.set_attribute("model", provider.model_name)
            span.set_attribute("task_type", criteria.task_type)
            start = time.monotonic()
            result = await provider.complete(messages, **kwargs)
            span.set_attribute("input_tokens", result.input_tokens)
            span.set_attribute("output_tokens", result.output_tokens)
            span.set_attribute("cost_usd", result.cost_usd)
            span.set_attribute("latency_ms", result.latency_ms)

        await self._track_usage(provider.model_name, result)
        return result

    async def embed(self, texts: list[str]) -> list[list[float]]:
        with tracer.start_as_current_span("llm.embed") as span:
            span.set_attribute("text_count", len(texts))
            return await self._embed_provider.embed(texts)


def get_client() -> NeuroFlowClient:
    global _client
    if _client is None:
        _client = NeuroFlowClient()
    return _client
