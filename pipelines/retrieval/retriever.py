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
