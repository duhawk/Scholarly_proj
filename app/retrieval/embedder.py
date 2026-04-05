import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings

_model: SentenceTransformer | None = None

# LRU-style cache for single-text embeddings (keyed on the raw string).
# Bounded at 512 entries — enough to absorb repeated queries without unbounded growth.
_embed_cache: dict[str, np.ndarray] = {}
_EMBED_CACHE_MAX = 512


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows to unit vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def embed_text(text: str) -> np.ndarray:
    """Embed a single string and return unit-normalized vector of shape (384,).

    Results are cached by text string (up to _EMBED_CACHE_MAX entries, LRU eviction).
    This avoids redundant model inference for repeated queries (e.g. agentic loop
    re-embeds the same hypothetical answer across iterations).
    """
    if text in _embed_cache:
        return _embed_cache[text]
    model = get_model()
    vector = model.encode([text], convert_to_numpy=True)
    result = _normalize(vector)[0]
    if len(_embed_cache) >= _EMBED_CACHE_MAX:
        _embed_cache.pop(next(iter(_embed_cache)))
    _embed_cache[text] = result
    return result


def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a list of strings and return unit-normalized matrix of shape (N, 384)."""
    model = get_model()
    vectors = model.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    return _normalize(vectors)


def warmup() -> None:
    """Pre-load model at startup."""
    get_model()
