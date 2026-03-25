from dataclasses import dataclass, field
import tiktoken
from pipelines.retrieval.retriever import RetrievalResult

enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class AssembledContext:
    text: str
    chunks_used: list[RetrievalResult]
    total_tokens: int
    sources: list[dict]


def assemble_context(
    chunks: list[RetrievalResult],
    token_budget: int = 4000,
) -> AssembledContext:
    parts = []
    used = []
    total_tokens = 0

    for i, chunk in enumerate(chunks, start=1):
        header = f"[Source {i} — {chunk.document_name}"
        if chunk.page_number:
            header += f", page {chunk.page_number}"
        header += "]"
        block = f"{header}\n{chunk.content}"
        block_tokens = len(enc.encode(block))

        if total_tokens + block_tokens > token_budget:
            # Try to fit a truncated version snapped to sentence boundary
            remaining = token_budget - total_tokens
            if remaining < 50:
                break
            truncated_tokens = enc.encode(block)[:remaining]
            truncated = enc.decode(truncated_tokens)
            # Snap to last sentence end
            last_dot = truncated.rfind('. ')
            if last_dot > 100:
                truncated = truncated[:last_dot + 1]
            parts.append(truncated)
            used.append(chunk)
            total_tokens += len(enc.encode(truncated))
            break

        parts.append(block)
        used.append(chunk)
        total_tokens += block_tokens

    text = "\n\n".join(parts)
    sources = [
        {
            "index": i + 1,
            "chunk_id": c.chunk_id,
            "document": c.document_name,
            "page": c.page_number,
        }
        for i, c in enumerate(used)
    ]

    return AssembledContext(
        text=text,
        chunks_used=used,
        total_tokens=total_tokens,
        sources=sources,
    )
