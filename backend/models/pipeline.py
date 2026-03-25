"""Pydantic models for pipeline configuration."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator


class RetrievalConfig(BaseModel):
    model_config = {"extra": "forbid"}

    dense_k: int = 20
    sparse_k: int = 20
    top_k_after_rerank: int = 8
    reranker: Literal["cross-encoder", "none", "cohere"] = "cross-encoder"
    query_expansion: bool = True


class IngestionConfig(BaseModel):
    model_config = {"extra": "forbid"}

    chunking_strategy: Literal["fixed", "sentence", "semantic"] = "fixed"
    chunk_size: int = 512
    chunk_overlap: int = 64


class GenerationConfig(BaseModel):
    model_config = {"extra": "forbid"}

    max_context_tokens: int = 4000
    max_output_tokens: int = 1500
    temperature: float = 0.3
    system_prompt_variant: Literal["default", "analytical", "concise"] = "default"


class PipelineConfig(BaseModel):
    model_config = {"extra": "forbid"}

    name: str
    description: str = ""
    retrieval: RetrievalConfig = RetrievalConfig()
    ingestion: IngestionConfig = IngestionConfig()
    generation: GenerationConfig = GenerationConfig()

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("name must not be blank")
        return v
