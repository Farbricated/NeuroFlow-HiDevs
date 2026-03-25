import time
from typing import AsyncGenerator
import anthropic
from backend.providers.base import BaseLLMProvider, ChatMessage, GenerationResult

PRICE_TABLE = {
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307":    {"input": 0.25, "output": 1.25},
    "claude-3-opus-20240229":     {"input": 15.00, "output": 75.00},
}
PER_MILLION = 1_000_000


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, model: str = "claude-3-haiku-20240307"):
        self._model = model
        self._client = anthropic.AsyncAnthropic()
        self._prices = PRICE_TABLE.get(model, {"input": 3.00, "output": 15.00})

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
        return 200000  # All Claude 3 models support 200k

    @property
    def supports_vision(self) -> bool:
        return True

    def _split_messages(self, messages: list[ChatMessage]):
        """Anthropic requires system message separate from conversation."""
        system = None
        conv = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                conv.append({"role": m.role, "content": m.content})
        return system, conv

    async def complete(self, messages: list[ChatMessage], **kwargs) -> GenerationResult:
        start = time.monotonic()
        system, conv = self._split_messages(messages)
        kwargs_clean = {k: v for k, v in kwargs.items() if k not in ("stream",)}
        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=kwargs_clean.pop("max_tokens", 2048),
            system=system or "",
            messages=conv,
            **kwargs_clean,
        )
        latency = (time.monotonic() - start) * 1000
        content = resp.content[0].text if resp.content else ""
        cost = self.estimate_cost(resp.usage.input_tokens, resp.usage.output_tokens)
        return GenerationResult(
            content=content,
            model=self._model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=latency,
            cost_usd=cost,
            finish_reason=resp.stop_reason or "stop",
        )

    async def stream(self, messages: list[ChatMessage], **kwargs) -> AsyncGenerator[str, None]:
        system, conv = self._split_messages(messages)
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=kwargs.get("max_tokens", 2048),
            system=system or "",
            messages=conv,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "Anthropic does not provide an embeddings API. Use OpenAI for embeddings."
        )
