# ADR 002 — Chunking Strategy

## Context

Chunking splits documents into segments that are embedded and stored independently. Chunk quality directly determines retrieval quality — chunks that are too large dilute relevance signals; chunks that are too small lose surrounding context that aids comprehension. Three strategies are commonly used:

1. **Fixed-size** — Split at token boundaries (e.g. 512 tokens, 64 overlap). Simple, predictable, fast. Works well for homogeneous content. Fails when important ideas span natural boundaries.
2. **Sentence-boundary** — Split at sentence ends. Preserves semantic units. Token counts vary widely, making context window management harder.
3. **Semantic** — Embed sentences, split where cosine similarity between adjacent sentences drops below a threshold. Produces topically coherent chunks. Slower (requires per-sentence embeddings during ingestion) and threshold is sensitive to domain.

## Decision

Use **all three strategies**, selected automatically based on content type:

| Content type | Strategy | Rationale |
|---|---|---|
| Table content | fixed_size | Tables have no prose sentences; semantic split fails. Fixed 512-token ensures rows aren't split mid-row. |
| DOCX with headings | hierarchical | Heading structure is authoritative. Parent chunk = section, children = sub-sections. Preserves document outline. |
| PDF > 50 pages | semantic | Long-form prose benefits from topic-coherent chunks. The embedding cost is amortised over large documents. |
| Default / short docs | fixed_size | Reliable, cheap, sufficient for most use cases. |

Fixed-size uses `tiktoken` (cl100k_base) for token counting. Sentence boundaries are respected within ±10% of the target size — the chunker finds the nearest sentence end within a 460–562 token window.

## Consequences

- **Positive**: Each content type gets a chunking strategy matched to its structure. No single strategy is optimal for all inputs.
- **Negative**: Three code paths to maintain. Hierarchical chunking requires reliable heading detection, which depends on extractor quality (DOCX headings are reliable; PDF headings detected via font size heuristics may mis-classify).
- **Switch condition**: If context precision evaluation scores are consistently below 0.70 for a particular pipeline, investigate whether the chunking strategy for its primary document type should be changed. The pipeline config makes this a configuration change, not a code change.
