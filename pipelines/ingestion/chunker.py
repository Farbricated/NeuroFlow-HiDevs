import re
from dataclasses import dataclass, field
import tiktoken
from pipelines.ingestion.extractors.pdf_extractor import ExtractedPage

enc = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SEMANTIC_THRESHOLD = 0.7


@dataclass
class Chunk:
    content: str
    token_count: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _count_tokens(text: str) -> int:
    return len(enc.encode(text))


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _fixed_size_chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        # Snap to sentence boundary within ±10%
        tolerance = int(size * 0.10)
        if end < len(tokens):
            boundary_end = min(end + tolerance, len(tokens))
            segment = enc.decode(tokens[end:boundary_end])
            sentence_end = segment.find('. ')
            if sentence_end != -1:
                extra = enc.encode(segment[:sentence_end + 1])
                chunk_text = enc.decode(tokens[start:end + len(extra)])
                end = end + len(extra)
        chunks.append(chunk_text)
        start = end - overlap
    return chunks


def _hierarchical_chunk(pages: list[ExtractedPage]) -> list[Chunk]:
    chunks = []
    idx = 0
    current_parent = None
    current_content = []

    for page in pages:
        lines = page.content.split('\n')
        for line in lines:
            if line.startswith('#'):
                # Flush current section
                if current_content:
                    text = '\n'.join(current_content).strip()
                    if text:
                        chunks.append(Chunk(
                            content=text,
                            token_count=_count_tokens(text),
                            chunk_index=idx,
                            metadata={"section": current_parent, "type": "hierarchical", **page.metadata},
                        ))
                        idx += 1
                current_parent = line.lstrip('#').strip()
                current_content = [line]
            else:
                current_content.append(line)

    if current_content:
        text = '\n'.join(current_content).strip()
        if text:
            chunks.append(Chunk(
                content=text,
                token_count=_count_tokens(text),
                chunk_index=idx,
                metadata={"section": current_parent, "type": "hierarchical"},
            ))

    return chunks


async def _semantic_chunk(text: str) -> list[str]:
    from backend.providers.client import get_client
    sentences = _split_sentences(text)
    if len(sentences) <= 2:
        return [text]

    client = get_client()
    embeddings = await client.embed(sentences)

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x ** 2 for x in a) ** 0.5
        nb = sum(x ** 2 for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0

    chunks = []
    buffer = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine(embeddings[i - 1], embeddings[i])
        if sim < SEMANTIC_THRESHOLD:
            chunks.append(' '.join(buffer))
            buffer = [sentences[i]]
        else:
            buffer.append(sentences[i])
    if buffer:
        chunks.append(' '.join(buffer))
    return chunks


def _select_strategy(pages: list[ExtractedPage]) -> str:
    for p in pages:
        if p.content_type == "table":
            return "fixed_size"
    has_headings = any('#' in p.content for p in pages)
    if has_headings:
        return "hierarchical"
    total_pages = len(pages)
    if total_pages > 50:
        return "semantic"
    return "fixed_size"


async def chunk_pages(pages: list[ExtractedPage], strategy: str | None = None) -> list[Chunk]:
    if not strategy:
        strategy = _select_strategy(pages)

    if strategy == "hierarchical":
        return _hierarchical_chunk(pages)

    full_text = "\n\n".join(p.content for p in pages)
    base_metadata = pages[0].metadata if pages else {}

    if strategy == "semantic":
        raw_chunks = await _semantic_chunk(full_text)
    else:
        raw_chunks = _fixed_size_chunk(full_text)

    return [
        Chunk(
            content=c,
            token_count=_count_tokens(c),
            chunk_index=i,
            metadata={**base_metadata, "strategy": strategy},
        )
        for i, c in enumerate(raw_chunks) if c.strip()
    ]
