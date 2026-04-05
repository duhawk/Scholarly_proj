"""Unit tests for BM25Index scoring and tokenize_query."""
import math
import sys
import os
from unittest.mock import MagicMock

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock all deps only available inside Docker
_sa = MagicMock()
for _mod in (
    "sqlalchemy",
    "sqlalchemy.ext",
    "sqlalchemy.ext.asyncio",
    "pydantic_settings",
    "pydantic",
):
    sys.modules.setdefault(_mod, MagicMock())

# Provide a minimal settings mock so app.config imports cleanly
_settings_mock = MagicMock()
_settings_mock.bm25_k1 = 1.5
_settings_mock.bm25_b = 0.75
sys.modules["app.config"] = MagicMock(settings=_settings_mock)

from app.retrieval.bm25 import BM25Index, tokenize_query


# ---------------------------------------------------------------------------
# tokenize_query
# ---------------------------------------------------------------------------

class TestTokenizeQuery:
    def test_lowercases_input(self):
        tokens = tokenize_query("Neural Network")
        assert "neural" in tokens
        assert "network" in tokens

    def test_filters_stopwords(self):
        tokens = tokenize_query("the and or but")
        assert tokens == []

    def test_filters_short_tokens(self):
        # Words with <= 2 chars should be excluded
        tokens = tokenize_query("a ab abc abcd")
        assert "a" not in tokens
        assert "ab" not in tokens
        assert "abc" in tokens
        assert "abcd" in tokens

    def test_extracts_alphabetic_only(self):
        tokens = tokenize_query("word123 hello-world foo_bar")
        # re.findall(r"[a-z]+") splits on non-alpha characters
        assert "word" in tokens
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens
        assert "bar" in tokens

    def test_empty_string(self):
        assert tokenize_query("") == []

    def test_all_stopwords_returns_empty(self):
        assert tokenize_query("the is was are") == []


# ---------------------------------------------------------------------------
# BM25Index.build
# ---------------------------------------------------------------------------

class TestBM25IndexBuild:
    def _make_chunks(self, docs):
        """Helper: list of (chunk_id, tokens) -> chunk dicts."""
        return [
            {
                "chunk_id": str(i),
                "document_id": "doc-1",
                "document_title": "Test Doc",
                "content": " ".join(tokens),
                "chunk_index": i,
                "page_num": 1,
                "bm25_tokens": tokens,
            }
            for i, tokens in enumerate(docs)
        ]

    def test_build_sets_n_docs(self):
        idx = BM25Index()
        idx.build(self._make_chunks([["cat", "dog"], ["bird"]]))
        assert idx._n_docs == 2

    def test_build_computes_avgdl(self):
        idx = BM25Index()
        idx.build(self._make_chunks([["a", "b", "c"], ["d"]]))
        # lengths: 3, 1 → avg = 2.0
        assert idx._avgdl == 2.0

    def test_build_computes_df(self):
        idx = BM25Index()
        idx.build(self._make_chunks([["cat", "dog"], ["cat", "bird"]]))
        assert idx._df["cat"] == 2
        assert idx._df["dog"] == 1
        assert idx._df["bird"] == 1

    def test_build_empty_chunks(self):
        idx = BM25Index()
        idx.build([])
        assert idx._n_docs == 0
        assert idx._avgdl == 0.0


# ---------------------------------------------------------------------------
# BM25Index._idf
# ---------------------------------------------------------------------------

class TestBM25IndexIDF:
    def _build_simple(self):
        idx = BM25Index()
        idx._n_docs = 4
        idx._df = {"common": 4, "rare": 1, "medium": 2}
        idx._avgdl = 10.0
        idx._doc_freqs = {}
        idx._doc_lengths = {}
        idx._chunk_meta = {}
        return idx

    def test_idf_unknown_term_is_zero(self):
        idx = self._build_simple()
        assert idx._idf("unknown") == 0.0

    def test_idf_rare_term_higher_than_common(self):
        idx = self._build_simple()
        assert idx._idf("rare") > idx._idf("common")

    def test_idf_formula(self):
        idx = self._build_simple()
        # IDF(qi) = log((N - df + 0.5) / (df + 0.5) + 1)
        n, df = 4, 1
        expected = math.log((n - df + 0.5) / (df + 0.5) + 1)
        assert abs(idx._idf("rare") - expected) < 1e-9

    def test_idf_non_negative(self):
        idx = self._build_simple()
        for term in ["common", "rare", "medium"]:
            assert idx._idf(term) >= 0.0


# ---------------------------------------------------------------------------
# BM25Index.score
# ---------------------------------------------------------------------------

class TestBM25IndexScore:
    def _make_chunks(self, docs):
        return [
            {
                "chunk_id": str(i),
                "document_id": "doc-1",
                "document_title": "Test",
                "content": " ".join(tokens),
                "chunk_index": i,
                "page_num": 1,
                "bm25_tokens": tokens,
            }
            for i, tokens in enumerate(docs)
        ]

    def test_empty_query_returns_empty(self):
        idx = BM25Index()
        idx.build(self._make_chunks([["cat", "dog"]]))
        assert idx.score([]) == []

    def test_no_matching_term_returns_empty(self):
        idx = BM25Index()
        idx.build(self._make_chunks([["cat", "dog"]]))
        assert idx.score(["fish"]) == []

    def test_results_sorted_by_score_descending(self):
        idx = BM25Index()
        # doc 0 has "neural" once, doc 1 has "neural" three times
        idx.build(self._make_chunks([
            ["neural"],
            ["neural", "neural", "neural"],
        ]))
        results = idx.score(["neural"])
        assert len(results) == 2
        assert results[0]["bm25_score"] >= results[1]["bm25_score"]

    def test_more_term_occurrences_ranks_higher(self):
        idx = BM25Index()
        idx.build(self._make_chunks([
            ["deep"],                               # chunk 0: 1 occurrence
            ["deep", "deep", "deep", "deep"],       # chunk 1: 4 occurrences
        ]))
        results = idx.score(["deep"])
        # chunk 1 should rank first
        assert results[0]["chunk_id"] == "1"

    def test_length_normalization_penalizes_longer_docs(self):
        idx = BM25Index()
        # Both docs have "science" once, but doc 1 is much longer (more filler terms)
        filler = ["word"] * 50
        idx.build(self._make_chunks([
            ["science"],                     # chunk 0: short, 1 hit
            ["science"] + filler,            # chunk 1: long, 1 hit
        ]))
        results = idx.score(["science"])
        # shorter doc should score higher due to length normalization
        assert results[0]["chunk_id"] == "0"

    def test_result_contains_bm25_score(self):
        idx = BM25Index()
        idx.build(self._make_chunks([["retrieval", "augmented"]]))
        results = idx.score(["retrieval"])
        assert "bm25_score" in results[0]
        assert results[0]["bm25_score"] > 0.0

    def test_result_preserves_chunk_metadata(self):
        idx = BM25Index()
        idx.build(self._make_chunks([["information", "retrieval"]]))
        results = idx.score(["information"])
        assert results[0]["chunk_id"] == "0"
        assert results[0]["document_title"] == "Test"
        assert results[0]["chunk_index"] == 0

    def test_multi_term_query_accumulates_scores(self):
        idx = BM25Index()
        idx.build(self._make_chunks([
            ["attention", "mechanism"],         # chunk 0: both terms
            ["attention"],                      # chunk 1: one term only
        ]))
        results = idx.score(["attention", "mechanism"])
        # chunk 0 matches both terms → higher score
        assert results[0]["chunk_id"] == "0"

    def test_rebuild_clears_previous_state(self):
        idx = BM25Index()
        idx.build(self._make_chunks([["alpha"]]))
        idx.build(self._make_chunks([["beta"]]))
        # "alpha" should not score (old index cleared)
        assert idx.score(["alpha"]) == []
        assert len(idx.score(["beta"])) == 1
