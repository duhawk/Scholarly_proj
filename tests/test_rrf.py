"""Unit tests for Reciprocal Rank Fusion merge logic."""
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock all heavy deps (only present inside Docker / GPU env) before any import
for _mod in (
    "sqlalchemy",
    "sqlalchemy.ext",
    "sqlalchemy.ext.asyncio",
    "anthropic",
    "sentence_transformers",
    "torch",
    "numpy",
    "pydantic_settings",
    "pydantic",
):
    sys.modules.setdefault(_mod, MagicMock())

# Provide minimal settings mock
_settings_mock = MagicMock()
_settings_mock.top_k = 8
_settings_mock.rrf_k = 60
_settings_mock.reranker_top_n = 5
_settings_mock.anthropic_api_key = "test"
sys.modules["app.config"] = MagicMock(settings=_settings_mock)

sys.modules["app.retrieval.embedder"] = MagicMock(embed_text=MagicMock(return_value=[0.0] * 384))
sys.modules["app.retrieval.vector_search"] = MagicMock(vector_search=MagicMock())
sys.modules["app.retrieval.bm25"] = MagicMock(bm25_search=MagicMock())
sys.modules["app.retrieval.reranker"] = MagicMock(rerank=MagicMock())

from app.retrieval.hybrid import _reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(chunk_id: str, **extra) -> dict:
    return {"chunk_id": chunk_id, "document_title": "Doc", "content": "...", **extra}


# ---------------------------------------------------------------------------
# _reciprocal_rank_fusion
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    def test_empty_lists_return_empty(self):
        assert _reciprocal_rank_fusion([], []) == []

    def test_dense_only_list(self):
        dense = [_chunk("a"), _chunk("b"), _chunk("c")]
        result = _reciprocal_rank_fusion(dense, [])
        ids = [r["chunk_id"] for r in result]
        assert ids == ["a", "b", "c"]

    def test_sparse_only_list(self):
        sparse = [_chunk("x"), _chunk("y")]
        result = _reciprocal_rank_fusion([], sparse)
        ids = [r["chunk_id"] for r in result]
        assert ids == ["x", "y"]

    def test_chunk_in_both_lists_ranks_higher(self):
        dense = [_chunk("shared"), _chunk("dense_only")]
        sparse = [_chunk("shared"), _chunk("sparse_only")]
        result = _reciprocal_rank_fusion(dense, sparse)
        # "shared" appears in both → higher combined RRF score
        assert result[0]["chunk_id"] == "shared"

    def test_rrf_score_formula_rank1_k60(self):
        """Score for rank-1 item with k=60 is 1/(60+1) = 1/61."""
        dense = [_chunk("a")]
        result = _reciprocal_rank_fusion(dense, [], k=60)
        expected = 1.0 / (60 + 1)
        assert abs(result[0]["rrf_score"] - expected) < 1e-12

    def test_rrf_score_accumulates_across_lists(self):
        """A chunk at rank 1 in both lists should have score 2*(1/(k+1))."""
        dense = [_chunk("a")]
        sparse = [_chunk("a")]
        result = _reciprocal_rank_fusion(dense, sparse, k=60)
        expected = 2.0 / 61.0
        assert abs(result[0]["rrf_score"] - expected) < 1e-12

    def test_rrf_score_decreases_with_rank(self):
        """Higher rank number → lower score."""
        dense = [_chunk("first"), _chunk("second"), _chunk("third")]
        result = _reciprocal_rank_fusion(dense, [], k=60)
        scores = [r["rrf_score"] for r in result]
        assert scores[0] > scores[1] > scores[2]

    def test_results_sorted_descending(self):
        dense = [_chunk("a"), _chunk("b"), _chunk("c")]
        sparse = [_chunk("c"), _chunk("b"), _chunk("a")]
        result = _reciprocal_rank_fusion(dense, sparse)
        scores = [r["rrf_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_chunk_metadata_preserved_from_dense(self):
        dense = [_chunk("a", document_title="Dense Title", content="dense content")]
        sparse = [_chunk("a", document_title="Sparse Title", content="sparse content")]
        result = _reciprocal_rank_fusion(dense, sparse)
        # Dense list takes precedence for overlapping chunks
        assert result[0]["document_title"] == "Dense Title"

    def test_sparse_only_chunk_metadata_included(self):
        dense = [_chunk("d1")]
        sparse = [_chunk("s1", document_title="Sparse Only")]
        result = _reciprocal_rank_fusion(dense, sparse)
        ids = [r["chunk_id"] for r in result]
        assert "s1" in ids
        s1 = next(r for r in result if r["chunk_id"] == "s1")
        assert s1["document_title"] == "Sparse Only"

    def test_rrf_score_field_present_on_all_results(self):
        dense = [_chunk("a"), _chunk("b")]
        sparse = [_chunk("c")]
        result = _reciprocal_rank_fusion(dense, sparse)
        for r in result:
            assert "rrf_score" in r

    def test_custom_k_changes_scores(self):
        dense = [_chunk("a")]
        r_k60 = _reciprocal_rank_fusion(dense, [], k=60)
        r_k1 = _reciprocal_rank_fusion(dense, [], k=1)
        # k=1 gives 1/2=0.5; k=60 gives 1/61≈0.016
        assert r_k1[0]["rrf_score"] > r_k60[0]["rrf_score"]

    def test_no_duplicate_chunk_ids_in_output(self):
        # Same chunk at different positions in both lists
        dense = [_chunk("dup"), _chunk("a")]
        sparse = [_chunk("dup"), _chunk("b")]
        result = _reciprocal_rank_fusion(dense, sparse)
        ids = [r["chunk_id"] for r in result]
        assert len(ids) == len(set(ids))

    def test_all_unique_chunks_appear_in_result(self):
        dense = [_chunk("a"), _chunk("b")]
        sparse = [_chunk("c"), _chunk("d")]
        result = _reciprocal_rank_fusion(dense, sparse)
        ids = {r["chunk_id"] for r in result}
        assert ids == {"a", "b", "c", "d"}
