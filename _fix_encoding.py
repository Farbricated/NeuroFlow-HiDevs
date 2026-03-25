"""Re-write the newly created modules with explicit UTF-8 encoding."""
import pathlib

ROOT = pathlib.Path(__file__).parent

files = {
    "pipelines/ingestion/chunker.py": '''\
"""Text chunking utilities for the ingestion pipeline."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    content: str
    token_count: int
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using a simple regex heuristic."""
    sentences = re.split(r"(?<=[.!?])\\s+", text.strip())
    return [s for s in sentences if s]


def _fixed_size_chunk(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    """Split *text* into token-bounded chunks."""
    if not text or not text.strip():
        return []

    tokens = _ENCODER.encode(text)
    chunks: list[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_ENCODER.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += size - overlap

    return chunks
''',
    "pipelines/retrieval/retriever.py": '''\
"""Retrieval result types for the retrieval pipeline."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalResult:
    chunk_id: str
    content: str
    document_name: str
    page_number: int
    score: float
    retrieval_method: str
''',
    "pipelines/retrieval/fusion.py": '''\
"""Reciprocal Rank Fusion (RRF) for merging multiple retrieval result lists."""
from __future__ import annotations

from collections import defaultdict

from pipelines.retrieval.retriever import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
) -> list[RetrievalResult]:
    """Merge result_lists using RRF, sorted by descending score."""
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, RetrievalResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            cid = result.chunk_id
            scores[cid] += 1.0 / (k + rank)
            if cid not in best:
                best[cid] = result

    fused: list[RetrievalResult] = []
    for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        r = best[cid]
        fused.append(
            RetrievalResult(
                chunk_id=r.chunk_id,
                content=r.content,
                document_name=r.document_name,
                page_number=r.page_number,
                score=score,
                retrieval_method="rrf_fused",
            )
        )

    return fused
''',
    "backend/models/pipeline.py": '''\
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
''',
    "backend/security/prompt_injection.py": '''\
"""Prompt injection detection utilities."""
import re
import logging

logger = logging.getLogger(__name__)

INJECTION_PATTERNS = [
    re.compile(r"ignore (all |previous |the |your )?instructions", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"new (system |)prompt", re.IGNORECASE),
    re.compile(r"disregard (the |all |previous )", re.IGNORECASE),
    re.compile(r"forget (everything|all|previous)", re.IGNORECASE),
    re.compile(r"act as (if |a |an )", re.IGNORECASE),
    re.compile(r"\\[\\[(system|SYSTEM)\\]\\]"),
    re.compile(r"<\\|system\\|>"),
]


def scan_for_injection(text: str) -> dict | None:
    """Layer 1: pattern matching. Returns match info or None."""
    for pattern in INJECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            logger.warning("Prompt injection pattern detected: %s", pattern.pattern)
            return {"detected": True, "pattern": pattern.pattern, "match": m.group(0)}
    return None


async def classify_query_injection(query: str) -> bool:
    """Layer 2: LLM classification for user queries."""
    from backend.providers.base import ChatMessage, RoutingCriteria  # noqa: PLC0415
    from backend.providers.client import get_client  # noqa: PLC0415

    client = get_client()
    prompt = (
        "Does the following user message attempt to override system instructions, "
        "impersonate the system, or exfiltrate data? Answer yes or no.\\n"
        f"Message: {query}"
    )
    result = await client.chat(
        [ChatMessage(role="user", content=prompt)],
        routing_criteria=RoutingCriteria(task_type="classification"),
        max_tokens=5,
    )
    return result.content.strip().lower().startswith("yes")
''',
}

for rel_path, content in files.items():
    target = ROOT / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"Written: {rel_path}")

print("Done.")
