from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    document_id: str
    filename: str
    status: str
    chunk_count: Optional[int] = None
    duplicate: bool = False


@dataclass
class Citation:
    reference: str
    chunk_id: str
    document: str
    page: Optional[int] = None


@dataclass
class QueryResult:
    run_id: str
    generation: str
    citations: list[Citation] = field(default_factory=list)
    latency_ms: Optional[int] = None
    chunks_used: int = 0


@dataclass
class EvaluationResult:
    run_id: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    overall_score: float
    judge_model: str = "gpt-4o-mini"
