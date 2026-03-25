"""Unit tests for chunker — no database required."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pipelines.ingestion.chunker import _fixed_size_chunk, _split_sentences, Chunk


SAMPLE_TEXT = """
The transformer architecture introduced the attention mechanism as the primary building block.
Unlike recurrent neural networks, transformers process all tokens in parallel.
The self-attention mechanism computes relationships between every pair of tokens.
Multi-head attention allows the model to attend to different representation subspaces.
Position encodings are added to token embeddings to preserve sequence order information.
Feed-forward networks follow each attention layer, applying the same transformation to each position.
Layer normalization is applied before each sub-layer in a pre-norm configuration.
Residual connections help gradients flow through the deep stack of transformer layers.
The encoder processes the input sequence and produces contextualized representations.
The decoder generates the output sequence one token at a time using the encoder output.
"""


def test_fixed_size_basic():
    chunks = _fixed_size_chunk(SAMPLE_TEXT, size=100, overlap=10)
    assert len(chunks) >= 1
    for c in chunks:
        assert isinstance(c, str)
        assert len(c) > 0


def test_fixed_size_overlap():
    chunks = _fixed_size_chunk(SAMPLE_TEXT, size=80, overlap=20)
    assert len(chunks) >= 2


def test_fixed_size_empty():
    chunks = _fixed_size_chunk("", size=512, overlap=64)
    assert chunks == [] or (len(chunks) == 1 and chunks[0].strip() == "")


def test_split_sentences_basic():
    text = "Hello world. This is a test. Another sentence here!"
    sentences = _split_sentences(text)
    assert len(sentences) >= 2
    for s in sentences:
        assert isinstance(s, str)
        assert s.strip()


def test_split_sentences_single():
    sentences = _split_sentences("Just one sentence")
    assert len(sentences) == 1
    assert sentences[0] == "Just one sentence"


def test_fixed_size_respects_budget():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    long_text = " ".join(["word"] * 2000)
    chunks = _fixed_size_chunk(long_text, size=512, overlap=64)
    for c in chunks:
        tokens = len(enc.encode(c))
        # Allow small overshoot from sentence snapping
        assert tokens <= 600, f"Chunk too large: {tokens} tokens"


def test_chunk_dataclass():
    c = Chunk(content="hello", token_count=1, chunk_index=0, metadata={"key": "val"})
    assert c.content == "hello"
    assert c.token_count == 1
    assert c.chunk_index == 0
    assert c.metadata["key"] == "val"


def test_fixed_size_non_overlapping():
    text = "A " * 300
    chunks_overlap = _fixed_size_chunk(text, size=100, overlap=50)
    chunks_no_overlap = _fixed_size_chunk(text, size=100, overlap=0)
    assert len(chunks_no_overlap) <= len(chunks_overlap)
