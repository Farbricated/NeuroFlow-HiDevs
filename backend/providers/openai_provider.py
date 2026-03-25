import asyncio
import time
from typing import AsyncGenerator
from openai import AsyncOpenAI, RateLimitError
from backend.providers.base import BaseLLMProvider, ChatMessage, GenerationResult

PRICE_TABLE = {
    "gpt-4o":          {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":     {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":     {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo":   {"input": 0.50,  "output": 1.50},
}
PER_MILLION = 1_000_000


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, model: str = "gpt-4o-mini", embed_model: str = "text-embedding-3-small"):
        self._model = model
        self._embed_model = embed_model
        self._client = AsyncOpenAI()
        self._prices = PRICE_TABLE.get(model, {"input": 0.15, "output": 0.60})

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def cost_per_input_token(self) -> float:
        return self._prices["input"] / PER_MILLION

    @property
    def cost_per_output_token(self) -> float:
        return self._prices["output"] / PER_MILLION

    @property
    def context_window(self) -> int:
        windows = {"gpt-4o": 128000, "gpt-4o-mini": 128000, "gpt-4-turbo": 128000}
        return windows.get(self._model, 16384)

    @property
    def supports_vision(self) -> bool:
        return self._model in ("gpt-4o", "gpt-4o-mini", "gpt-4-turbo")

    def _to_oai_messages(self, messages: list[ChatMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def complete(self, messages: list[ChatMessage], **kwargs) -> GenerationResult:
        start = time.monotonic()
        for attempt in range(3):
            try:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=self._to_oai_messages(messages),
                    **kwargs,
                )
                latency = (time.monotonic() - start) * 1000
                usage = resp.usage
                cost = self.estimate_cost(usage.prompt_tokens, usage.completion_tokens)
                return GenerationResult(
                    content=resp.choices[0].message.content,
                    model=self._model,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    latency_ms=latency,
                    cost_usd=cost,
                    finish_reason=resp.choices[0].finish_reason,
                )
            except RateLimitError as e:
                if attempt == 2:
                    raise
                wait = min(2 ** attempt * 2, 60)
                await asyncio.sleep(wait)

    async def stream(self, messages: list[ChatMessage], **kwargs) -> AsyncGenerator[str, None]:
        async with await self._client.chat.completions.create(
            model=self._model,
            messages=self._to_oai_messages(messages),
            stream=True,
            **kwargs,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    async def embed(self, texts: list[str]) -> list[list[float]]:
        BATCH = 100
        results = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            resp = await self._client.embeddings.create(
                model=self._embed_model,
                input=batch,
            )
            results.extend([item.embedding for item in resp.data])
        return results
