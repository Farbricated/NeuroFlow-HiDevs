from pydantic import BaseModel, model_validator
from typing import Literal


class IngestionConfig(BaseModel):
    chunking_strategy: Literal["fixed_size", "semantic", "hierarchical", "auto"] = "auto"
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    extractors_enabled: list[str] = ["pdf", "docx", "image", "csv", "url"]

    class Config:
        extra = "forbid"


class RetrievalConfig(BaseModel):
    dense_k: int = 20
    sparse_k: int = 20
    reranker: Literal["cross-encoder", "none"] = "cross-encoder"
    top_k_after_rerank: int = 8
    query_expansion: bool = True
    metadata_filters_enabled: bool = True

    class Config:
        extra = "forbid"


class ModelRoutingConfig(BaseModel):
    task_type: str = "rag_generation"
    max_cost_per_call: float | None = None

    class Config:
        extra = "forbid"


class GenerationConfig(BaseModel):
    model_routing: ModelRoutingConfig = ModelRoutingConfig()
    max_context_tokens: int = 4000
    max_output_tokens: int = 1500
    temperature: float = 0.3
    system_prompt_variant: Literal["precise", "conversational", "analytical"] = "precise"

    class Config:
        extra = "forbid"


class EvaluationConfig(BaseModel):
    auto_evaluate: bool = True
    training_threshold: float = 0.82

    class Config:
        extra = "forbid"


class PipelineConfig(BaseModel):
    name: str
    description: str = ""
    ingestion: IngestionConfig = IngestionConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    class Config:
        extra = "forbid"

    @model_validator(mode="after")
    def validate_name(self) -> "PipelineConfig":
        if not self.name.strip():
            raise ValueError("Pipeline name cannot be empty")
        return self
