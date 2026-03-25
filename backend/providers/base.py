from abc import ABC, abstractmethod
from typing import AsyncGenerator
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str | list  # str for text, list for multi-modal


@dataclass
class GenerationResult:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    finish_reason: str


@dataclass
class RoutingCriteria:
    task_type: str = "rag_generation"  # "rag_generation" | "evaluation" | "embedding" | "classification"
    max_cost_per_call: float | None = None
    require_vision: bool = False
    require_long_context: bool = False   # > 32k tokens
    latency_budget_ms: int | None = None
    prefer_fine_tuned: bool = False
    domain: str | None = None


class BaseLLMProvider(ABC):

    @abstractmethod
    async def complete(self, messages: list[ChatMessage], **kwargs) -> GenerationResult:
        """Single completion call."""
        ...

    @abstractmethod
    async def stream(self, messages: list[ChatMessage], **kwargs) -> AsyncGenerator[str, None]:
        """Streaming completion — yields token strings."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        ...

    @property
    @abstractmethod
    def cost_per_input_token(self) -> float:
        """Cost in USD per input token."""
        ...

    @property
    @abstractmethod
    def cost_per_output_token(self) -> float:
        """Cost in USD per output token."""
        ...

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Maximum context window in tokens."""
        ...

    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.cost_per_input_token +
            output_tokens * self.cost_per_output_token
        )
