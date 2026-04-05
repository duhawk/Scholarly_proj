"""Profile BM25Index build and query time across corpus sizes.

Run locally (no DB needed):
    python scripts/profile_bm25.py

Outputs a table of build_time and query_time at 1k / 5k / 10k / 50k / 100k chunks.
Use this to decide if an incremental index is needed (> ~10k chunks in the corpus).
"""
import random
import string
import time
import sys
import os
from unittest.mock import MagicMock

# Stub out DB/ORM deps so this runs without Docker
for _mod in ("sqlalchemy", "sqlalchemy.ext", "sqlalchemy.ext.asyncio",
             "pydantic_settings", "pydantic"):
    sys.modules.setdefault(_mod, MagicMock())
_settings_mock = MagicMock()
_settings_mock.bm25_k1 = 1.5
_settings_mock.bm25_b = 0.75
sys.modules["app.config"] = MagicMock(settings=_settings_mock)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.retrieval.bm25 import BM25Index  # noqa: E402


VOCAB = [w for w in string.ascii_lowercase * 3]
WORDS = [
    "neural", "network", "transformer", "attention", "embedding",
    "retrieval", "language", "model", "document", "query",
    "semantic", "dense", "sparse", "fusion", "ranking",
    "vector", "passage", "index", "corpus", "token",
]


def _random_tokens(n_tokens: int) -> list[str]:
    return [random.choice(WORDS) for _ in range(n_tokens)]


def _make_chunks(n: int, avg_tokens: int = 120) -> list[dict]:
    return [
        {
            "chunk_id": str(i),
            "document_id": f"doc-{i // 10}",
            "document_title": f"Document {i // 10}",
            "content": "",
            "chunk_index": i % 10,
            "page_num": 1,
            "bm25_tokens": _random_tokens(avg_tokens),
        }
        for i in range(n)
    ]


def profile(n_chunks: int, n_query_runs: int = 20) -> dict:
    chunks = _make_chunks(n_chunks)
    idx = BM25Index()

    t0 = time.perf_counter()
    idx.build(chunks)
    build_ms = (time.perf_counter() - t0) * 1000

    query_tokens = ["neural", "transformer", "retrieval", "embedding"]
    t0 = time.perf_counter()
    for _ in range(n_query_runs):
        idx.score(query_tokens)
    query_ms = (time.perf_counter() - t0) * 1000 / n_query_runs

    return {"n_chunks": n_chunks, "build_ms": build_ms, "query_ms": query_ms}


def main():
    sizes = [1_000, 5_000, 10_000, 50_000, 100_000]
    print(f"{'Chunks':>10}  {'Build (ms)':>12}  {'Query (ms)':>12}")
    print("-" * 40)
    for n in sizes:
        r = profile(n)
        flag = "  ← consider incremental index" if r["build_ms"] > 2000 else ""
        print(f"{r['n_chunks']:>10,}  {r['build_ms']:>12.1f}  {r['query_ms']:>12.2f}{flag}")

    print()
    print("Recommendation:")
    print("  ivfflat lists = max(1, sqrt(n_chunks)) for up to ~1M vectors.")
    print("  For BM25: if build_ms > 2000 at your corpus size, implement")
    print("  incremental index updates rather than full rebuilds per query.")


if __name__ == "__main__":
    main()
