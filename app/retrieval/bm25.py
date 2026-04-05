import math
import re
from collections import Counter

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "that", "this", "these",
    "those", "it", "its", "as", "if", "not", "no", "nor", "so", "yet",
    "both", "either", "neither", "each", "more", "most", "other", "some",
    "such", "than", "too", "very",
}


def tokenize_query(query: str) -> list[str]:
    tokens = re.findall(r"[a-z]+", query.lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


class BM25Index:
    """BM25 scorer built from chunk tokens loaded from the database.

    Formula:
        score(D, Q) = Σ IDF(qi) * f(qi, D) * (k1 + 1)
                                / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        # chunk_id -> {term: freq}
        self._doc_freqs: dict[str, dict[str, int]] = {}
        # chunk_id -> doc length
        self._doc_lengths: dict[str, int] = {}
        # term -> number of docs containing term
        self._df: dict[str, int] = {}
        self._n_docs: int = 0
        self._avgdl: float = 0.0
        # chunk metadata for returning results
        self._chunk_meta: dict[str, dict] = {}

    def build(self, chunks: list[dict]) -> None:
        """Build index from list of chunk dicts with bm25_tokens and metadata."""
        self._doc_freqs = {}
        self._doc_lengths = {}
        self._df = {}
        self._chunk_meta = {}

        for chunk in chunks:
            chunk_id = str(chunk["chunk_id"])
            tokens: list[str] = chunk.get("bm25_tokens") or []
            freq = Counter(tokens)
            self._doc_freqs[chunk_id] = dict(freq)
            self._doc_lengths[chunk_id] = len(tokens)
            self._chunk_meta[chunk_id] = {
                "chunk_id": chunk_id,
                "document_id": str(chunk.get("document_id", "")),
                "document_title": chunk.get("document_title", ""),
                "content": chunk.get("content", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "page_num": chunk.get("page_num"),
            }
            for term in freq:
                self._df[term] = self._df.get(term, 0) + 1

        self._n_docs = len(chunks)
        if self._n_docs > 0:
            self._avgdl = sum(self._doc_lengths.values()) / self._n_docs

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens: list[str]) -> list[dict]:
        """Score all documents for the query, return sorted list."""
        scores: dict[str, float] = {}

        for term in query_tokens:
            idf = self._idf(term)
            if idf == 0:
                continue
            for chunk_id, freq_map in self._doc_freqs.items():
                f = freq_map.get(term, 0)
                if f == 0:
                    continue
                dl = self._doc_lengths[chunk_id]
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1))
                term_score = idf * (f * (self.k1 + 1)) / denom
                scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score

        results = []
        for chunk_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            meta = self._chunk_meta[chunk_id].copy()
            meta["bm25_score"] = score
            results.append(meta)
        return results


async def bm25_search(
    query: str,
    session: AsyncSession,
    top_k: int = 10,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """Load all chunks from DB, build BM25 index, score query, return top_k results.

    Args:
        document_ids: optional list of document UUID strings to restrict the index to.
    """
    if document_ids:
        placeholders = ", ".join(f":doc_{i}::uuid" for i in range(len(document_ids)))
        sql = text(f"""
            SELECT
                c.id::text AS chunk_id,
                c.document_id::text,
                d.title AS document_title,
                c.content,
                c.chunk_index,
                c.page_num,
                c.bm25_tokens
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.document_id IN ({placeholders})
        """)
        params = {f"doc_{i}": doc_id for i, doc_id in enumerate(document_ids)}
        result = await session.execute(sql, params)
    else:
        sql = text("""
            SELECT
                c.id::text AS chunk_id,
                c.document_id::text,
                d.title AS document_title,
                c.content,
                c.chunk_index,
                c.page_num,
                c.bm25_tokens
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
        """)
        result = await session.execute(sql)
    rows = result.mappings().all()
    chunks = [dict(row) for row in rows]

    if not chunks:
        return []

    index = BM25Index(k1=settings.bm25_k1, b=settings.bm25_b)
    index.build(chunks)

    query_tokens = tokenize_query(query)
    if not query_tokens:
        return []

    return index.score(query_tokens)[:top_k]
