import re
from dataclasses import dataclass
from pipelines.retrieval.retriever import RetrievalResult


@dataclass
class Citation:
    reference: str        # "Source 1"
    chunk_id: str
    document_name: str
    page_number: int | None
    content_preview: str  # first 100 chars
    invalid: bool = False


def parse_citations(
    response_text: str,
    sources: list[dict],
) -> list[Citation]:
    """
    Find all [Source N] patterns in response and resolve to chunk metadata.
    Flags citations where N exceeds available sources as hallucinated.
    """
    found = set(re.findall(r'\[Source (\d+)\]', response_text))
    citations = []
    source_map = {str(s["index"]): s for s in sources}

    for num in sorted(found, key=int):
        src = source_map.get(num)
        if src is None:
            citations.append(Citation(
                reference=f"Source {num}",
                chunk_id="",
                document_name="",
                page_number=None,
                content_preview="",
                invalid=True,
            ))
        else:
            citations.append(Citation(
                reference=f"Source {num}",
                chunk_id=src["chunk_id"],
                document_name=src["document"],
                page_number=src.get("page"),
                content_preview=src.get("preview", "")[:100],
                invalid=False,
            ))

    return citations
