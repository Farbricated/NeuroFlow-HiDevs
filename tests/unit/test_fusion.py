"""Unit tests for RRF fusion."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pipelines.retrieval.fusion import reciprocal_rank_fusion
from pipelines.retrieval.retriever import RetrievalResult


def _make_result(chunk_id: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        content=f"Content of {chunk_id}",
        document_name="test.pdf",
        page_number=1,
        score=score,
        retrieval_method="test",
    )


def test_rrf_single_list():
    results = [_make_result(f"chunk-{i}") for i in range(5)]
    fused = reciprocal_rank_fusion([results])
    assert len(fused) == 5
    # Order should be preserved
    assert fused[0].chunk_id == "chunk-0"


def test_rrf_two_lists_same():
    list1 = [_make_result(f"c{i}") for i in range(3)]
    list2 = [_make_result(f"c{i}") for i in range(3)]
    fused = reciprocal_rank_fusion([list1, list2])
    # Items appearing in both lists should score higher
    assert len(fused) == 3


def test_rrf_two_lists_different():
    list1 = [_make_result("a"), _make_result("b")]
    list2 = [_make_result("c"), _make_result("d")]
    fused = reciprocal_rank_fusion([list1, list2])
    assert len(fused) == 4


def test_rrf_overlap_boosts_score():
    """Items appearing in multiple lists should rank higher than items in one list."""
    shared = "shared-chunk"
    unique = "unique-chunk"
    list1 = [_make_result(shared), _make_result(unique)]
    list2 = [_make_result(shared), _make_result("other")]
    fused = reciprocal_rank_fusion([list1, list2])
    chunk_ids = [r.chunk_id for r in fused]
    assert chunk_ids[0] == shared, "Shared chunk should rank first"


def test_rrf_empty_lists():
    fused = reciprocal_rank_fusion([[], []])
    assert fused == []


def test_rrf_known_scores():
    """Verify RRF score formula: 1/(k+rank). k=60 default."""
    k = 60
    list1 = [_make_result("a"), _make_result("b")]
    list2 = [_make_result("b"), _make_result("a")]
    fused = reciprocal_rank_fusion([list1, list2], k=k)
    score_map = {r.chunk_id: r.score for r in fused}
    # "a" is rank 1 in list1, rank 2 in list2
    expected_a = 1/(k+1) + 1/(k+2)
    # "b" is rank 2 in list1, rank 1 in list2
    expected_b = 1/(k+2) + 1/(k+1)
    assert abs(score_map["a"] - expected_a) < 1e-9
    assert abs(score_map["b"] - expected_b) < 1e-9
    # Both have same score since ranks are symmetric
    assert abs(score_map["a"] - score_map["b"]) < 1e-9


def test_rrf_returns_retrieval_results():
    results = [_make_result("x")]
    fused = reciprocal_rank_fusion([results])
    assert all(isinstance(r, RetrievalResult) for r in fused)
    assert fused[0].retrieval_method == "rrf_fused"
