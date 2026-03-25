import json
import redis.asyncio as aioredis
from backend.providers.base import BaseLLMProvider, RoutingCriteria
from backend.providers.openai_provider import OpenAIProvider
from backend.providers.anthropic_provider import AnthropicProvider
from backend.config import get_settings

# Static model registry (supplemented by Redis for fine-tuned models)
MODEL_REGISTRY = {
    "gpt-4o": {
        "provider": "openai", "model": "gpt-4o",
        "vision": True, "context_window": 128000,
        "cost_per_1k_input": 0.0025, "cost_per_1k_output": 0.010,
        "task_types": ["rag_generation", "evaluation", "classification"],
    },
    "gpt-4o-mini": {
        "provider": "openai", "model": "gpt-4o-mini",
        "vision": True, "context_window": 128000,
        "cost_per_1k_input": 0.00015, "cost_per_1k_output": 0.0006,
        "task_types": ["rag_generation", "embedding", "classification"],
    },
    "claude-3-haiku": {
        "provider": "anthropic", "model": "claude-3-haiku-20240307",
        "vision": True, "context_window": 200000,
        "cost_per_1k_input": 0.00025, "cost_per_1k_output": 0.00125,
        "task_types": ["rag_generation", "classification"],
    },
    "claude-3-5-sonnet": {
        "provider": "anthropic", "model": "claude-3-5-sonnet-20241022",
        "vision": True, "context_window": 200000,
        "cost_per_1k_input": 0.003, "cost_per_1k_output": 0.015,
        "task_types": ["rag_generation", "evaluation"],
    },
}

EVALUATION_MODELS = {"gpt-4o", "claude-3-5-sonnet"}


class ModelRouter:
    def __init__(self):
        self._settings = get_settings()

    async def _get_fine_tuned_models(self) -> list[dict]:
        try:
            client = aioredis.from_url(self._settings.redis_url)
            data = await client.get("router:models")
            await client.aclose()
            if data:
                return json.loads(data)
        except Exception:
            pass
        return []

    def _build_provider(self, config: dict) -> BaseLLMProvider:
        if config["provider"] == "openai":
            return OpenAIProvider(model=config["model"])
        elif config["provider"] == "anthropic":
            return AnthropicProvider(model=config["model"])
        raise ValueError(f"Unknown provider: {config['provider']}")

    async def select(self, criteria: RoutingCriteria) -> BaseLLMProvider:
        # Evaluation always uses a capable non-fine-tuned model
        if criteria.task_type == "evaluation":
            return OpenAIProvider(model="gpt-4o-mini")

        # Vision requirement
        if criteria.require_vision:
            return OpenAIProvider(model="gpt-4o-mini")

        # Long context requirement (>32k)
        if criteria.require_long_context:
            return AnthropicProvider(model="claude-3-5-sonnet-20241022")

        # Fine-tuned model preference
        if criteria.prefer_fine_tuned and criteria.domain:
            fine_tuned = await self._get_fine_tuned_models()
            for m in fine_tuned:
                if m.get("task_type") == criteria.domain and m.get("status") == "active":
                    return self._build_provider(m)

        # Cost filter
        if criteria.max_cost_per_call is not None:
            # For a 2k token call, estimate cost and filter
            cheap_models = []
            for name, cfg in MODEL_REGISTRY.items():
                est_cost = (2000 * cfg["cost_per_1k_input"] + 500 * cfg["cost_per_1k_output"]) / 1000
                if est_cost <= criteria.max_cost_per_call:
                    cheap_models.append((name, cfg, est_cost))
            if cheap_models:
                cheapest = min(cheap_models, key=lambda x: x[2])
                return self._build_provider(cheapest[1])

        # Default: cheapest model
        return OpenAIProvider(model="gpt-4o-mini")
