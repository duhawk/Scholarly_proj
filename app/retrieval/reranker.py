from sentence_transformers import CrossEncoder
from app.config import settings

_cross_encoder: CrossEncoder | None = None
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(_MODEL_NAME)
    return _cross_encoder


def rerank(query: str, chunks: list[dict], top_n: int | None = None) -> list[dict]:
    """Rerank chunks using cross-encoder relevance scores.

    Args:
        query: original user query
        chunks: list of chunk dicts with 'content' field
        top_n: number of top chunks to return (default: settings.reranker_top_n)

    Returns:
        chunks sorted by cross-encoder score descending, with 'rerank_score' added
    """
    if not chunks:
        return []

    n = top_n or settings.reranker_top_n
    model = get_cross_encoder()

    pairs = [(query, chunk["content"]) for chunk in chunks]
    scores = model.predict(pairs)

    scored = [
        {**chunk, "rerank_score": float(score)}
        for chunk, score in zip(chunks, scores)
    ]
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:n]


def warmup() -> None:
    """Pre-load cross-encoder model at startup."""
    get_cross_encoder()
