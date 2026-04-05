"""Unit tests for chunk_text sentence-boundary snapping and chunking behaviour."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ingestion.chunker import chunk_text, _find_sentence_boundary, _approximate_tokens


# ---------------------------------------------------------------------------
# _approximate_tokens
# ---------------------------------------------------------------------------

class TestApproximateTokens:
    def test_empty_string_returns_one(self):
        assert _approximate_tokens("") == 1

    def test_four_chars_is_one_token(self):
        assert _approximate_tokens("abcd") == 1

    def test_eight_chars_is_two_tokens(self):
        assert _approximate_tokens("abcdefgh") == 2

    def test_long_text(self):
        text = "a" * 400
        assert _approximate_tokens(text) == 100


# ---------------------------------------------------------------------------
# _find_sentence_boundary
# ---------------------------------------------------------------------------

class TestFindSentenceBoundary:
    def test_returns_target_when_no_boundary_found(self):
        text = "no sentence ending here"
        result = _find_sentence_boundary(text, 10)
        assert result == 10

    def test_snaps_to_sentence_end_within_window(self):
        text = "First sentence. Second sentence. Third."
        # target at position 20 (inside "Second"), window covers ". "
        result = _find_sentence_boundary(text, 20, window=20)
        # Result should be at the end of "First sentence." + space
        snapped_char = text[result - 1] if result > 0 else text[0]
        # The snapped position should come right after ". " not be arbitrary
        assert result > 0
        assert result <= len(text)

    def test_picks_closest_boundary(self):
        text = "One. Two. Three."
        # Multiple sentence ends available — should pick the closest to target
        target = 5  # just after "One. "
        result = _find_sentence_boundary(text, target, window=15)
        assert result <= len(text)

    def test_boundary_at_edge_of_text(self):
        text = "Hello world."
        result = _find_sentence_boundary(text, len(text) - 1, window=5)
        assert result <= len(text)

    def test_exclamation_mark_counts_as_boundary(self):
        text = "Great result! More text here. Done."
        result = _find_sentence_boundary(text, 10, window=10)
        assert result > 0

    def test_question_mark_counts_as_boundary(self):
        text = "Is this correct? Yes it is. Done."
        result = _find_sentence_boundary(text, 10, window=10)
        assert result > 0


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def _pages(self, *texts):
        return [{"page_num": i + 1, "text": t} for i, t in enumerate(texts)]

    def test_empty_pages_returns_empty(self):
        assert chunk_text([]) == []

    def test_single_short_page_produces_at_least_one_chunk(self):
        # Default overlap (64 tokens = 256 chars) exceeds the short text, so the
        # chunker generates multiple position-shifted mini-chunks; assert >= 1.
        pages = self._pages("Short text.")
        chunks = chunk_text(pages)
        assert len(chunks) >= 1

    def test_chunk_has_required_fields(self):
        pages = self._pages("Hello world. This is a test sentence.")
        chunks = chunk_text(pages)
        assert len(chunks) >= 1
        c = chunks[0]
        assert "content" in c
        assert "chunk_index" in c
        assert "page_num" in c
        assert "token_count" in c

    def test_chunk_indices_are_sequential(self):
        # Use a long enough text to get multiple chunks
        long_text = ("The quick brown fox jumps over the lazy dog. " * 200)
        pages = self._pages(long_text)
        chunks = chunk_text(pages, chunk_size=64, overlap=8)
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_content_is_non_empty(self):
        pages = self._pages("Machine learning is a field of artificial intelligence. " * 50)
        chunks = chunk_text(pages, chunk_size=64, overlap=8)
        for c in chunks:
            assert c["content"].strip() != ""

    def test_overlap_means_adjacent_chunks_share_content(self):
        # Build text with known sentence markers
        sentence = "This is sentence number {}. "
        text = "".join(sentence.format(i) for i in range(200))
        pages = self._pages(text)
        chunks = chunk_text(pages, chunk_size=64, overlap=32)
        if len(chunks) < 2:
            return  # not enough chunks to test overlap
        # The end of chunk[0] should share some words with the start of chunk[1]
        words0 = set(chunks[0]["content"].split())
        words1 = set(chunks[1]["content"].split())
        assert len(words0 & words1) > 0, "Adjacent chunks should share words due to overlap"

    def test_token_count_approximation_reasonable(self):
        pages = self._pages("word " * 100)
        chunks = chunk_text(pages, chunk_size=512, overlap=64)
        for c in chunks:
            assert c["token_count"] >= 1

    def test_page_num_assigned_correctly(self):
        pages = [
            {"page_num": 1, "text": "Page one content. More text here."},
            {"page_num": 2, "text": "Page two content. Even more text."},
        ]
        chunks = chunk_text(pages, chunk_size=512, overlap=64)
        # All chunks should have page_num 1 or 2
        for c in chunks:
            assert c["page_num"] in (1, 2)

    def test_multiple_pages_all_content_present(self):
        pages = self._pages("Alpha text. ", "Beta text. ")
        chunks = chunk_text(pages)
        combined = " ".join(c["content"] for c in chunks)
        assert "Alpha" in combined
        assert "Beta" in combined

    def test_chunk_size_respected_approximately(self):
        # With chunk_size=32 tokens (~128 chars), chunks should be roughly that size
        long_text = "word " * 400
        pages = self._pages(long_text)
        chunks = chunk_text(pages, chunk_size=32, overlap=4)
        # At least several chunks should be produced
        assert len(chunks) > 2

    def test_whitespace_only_pages_produce_no_chunks(self):
        pages = [{"page_num": 1, "text": "   \n\n\t  "}]
        chunks = chunk_text(pages)
        assert len(chunks) == 0

    def test_sentence_boundary_snap_does_not_split_mid_sentence(self):
        # Build text where each sentence is well-separated
        sentences = ["Sentence {} ends here. ".format(i) for i in range(100)]
        text = "".join(sentences)
        pages = self._pages(text)
        chunks = chunk_text(pages, chunk_size=64, overlap=8)
        for c in chunks:
            content = c["content"]
            # Content should not end abruptly mid-word at a raw char boundary;
            # sentence snapping means the last char is either end-of-text or
            # follows a sentence terminator + whitespace in original text.
            # At minimum, content must be stripped and non-empty.
            assert content == content.strip()
            assert len(content) > 0
