"""Unit tests for PipelineConfig Pydantic validation."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pydantic import ValidationError
from backend.models.pipeline import PipelineConfig, RetrievalConfig, GenerationConfig


def test_valid_pipeline_config():
    config = PipelineConfig(name="test-pipeline")
    assert config.name == "test-pipeline"
    assert config.retrieval.dense_k == 20
    assert config.generation.temperature == 0.3


def test_empty_name_rejected():
    with pytest.raises(ValidationError):
        PipelineConfig(name="")


def test_whitespace_name_rejected():
    with pytest.raises(ValidationError):
        PipelineConfig(name="   ")


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        PipelineConfig(name="ok", unknown_field="bad")


def test_retrieval_defaults():
    r = RetrievalConfig()
    assert r.dense_k == 20
    assert r.sparse_k == 20
    assert r.top_k_after_rerank == 8
    assert r.reranker == "cross-encoder"
    assert r.query_expansion is True


def test_retrieval_custom_values():
    r = RetrievalConfig(dense_k=30, top_k_after_rerank=5, reranker="none")
    assert r.dense_k == 30
    assert r.top_k_after_rerank == 5
    assert r.reranker == "none"


def test_invalid_reranker():
    with pytest.raises(ValidationError):
        RetrievalConfig(reranker="invalid-reranker")


def test_invalid_chunking_strategy():
    from backend.models.pipeline import IngestionConfig
    with pytest.raises(ValidationError):
        IngestionConfig(chunking_strategy="invalid")


def test_invalid_system_prompt_variant():
    with pytest.raises(ValidationError):
        GenerationConfig(system_prompt_variant="invalid")


def test_full_pipeline_serialization():
    config = PipelineConfig(
        name="prod-pipeline",
        description="Production RAG pipeline",
        retrieval=RetrievalConfig(dense_k=30, top_k_after_rerank=10),
        generation=GenerationConfig(temperature=0.1, max_output_tokens=2000),
    )
    d = config.model_dump()
    assert d["name"] == "prod-pipeline"
    assert d["retrieval"]["dense_k"] == 30
    assert d["generation"]["temperature"] == 0.1


def test_pipeline_with_all_fields():
    config = PipelineConfig(
        name="full-config",
        description="All fields set",
        retrieval=RetrievalConfig(dense_k=15, sparse_k=15, top_k_after_rerank=6),
        generation=GenerationConfig(
            max_context_tokens=3000,
            max_output_tokens=1000,
            temperature=0.5,
            system_prompt_variant="analytical",
        ),
    )
    assert config.generation.system_prompt_variant == "analytical"
    assert config.retrieval.dense_k == 15
