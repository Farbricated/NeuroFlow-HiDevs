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
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
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
