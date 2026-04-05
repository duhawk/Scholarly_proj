import anthropic
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.retrieval.embedder import embed_text
from app.retrieval.vector_search import vector_search
from app.retrieval.bm25 import bm25_search
from app.retrieval.reranker import rerank


def _reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Merge dense and sparse ranked lists with Reciprocal Rank Fusion.

    RRF(d) = Σ 1 / (k + rank(d))
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        cid = chunk["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(sparse_results, start=1):
        cid = chunk["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in chunk_map:
            chunk_map[cid] = chunk

    merged = []
    for cid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        entry = chunk_map[cid].copy()
        entry["rrf_score"] = score
        merged.append(entry)

    return merged


def _generate_hypothetical_answer(query: str) -> str:
    """HyDE: generate a hypothetical answer to embed for retrieval."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Write a concise, factual answer to the following academic question. "
                    f"Your answer will be used to find relevant passages, not shown to users.\n\n"
                    f"Question: {query}"
                ),
            }
        ],
    )
    return response.content[0].text


async def hybrid_search(
    query: str,
    session: AsyncSession,
    top_k: int | None = None,
    use_hyde: bool = True,
    reranker_top_n: int | None = None,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """Full hybrid retrieval pipeline.

    1. HyDE: generate hypothetical answer, embed it
    2. Dense vector search with hypothetical embedding
    3. BM25 sparse search with original query
    4. RRF merge
    5. Cross-encoder rerank

    Args:
        document_ids: optional list of document UUID strings to restrict retrieval to.

    Returns reranked chunks list.
    """
    k = top_k or settings.top_k

    # 1. HyDE embedding
    if use_hyde:
        try:
            hypothetical = _generate_hypothetical_answer(query)
            search_embedding = embed_text(hypothetical)
        except Exception:
            # Fallback to direct query embedding on HyDE failure
            search_embedding = embed_text(query)
    else:
        search_embedding = embed_text(query)

    # 2. Dense vector search
    dense_results = await vector_search(search_embedding, session, top_k=k, document_ids=document_ids)

    # 3. BM25 sparse search (on original query)
    sparse_results = await bm25_search(query, session, top_k=k, document_ids=document_ids)

    # 4. RRF merge
    merged = _reciprocal_rank_fusion(dense_results, sparse_results, k=settings.rrf_k)

    # Limit candidates going into reranker (2x top_n to give reranker good options)
    n = reranker_top_n or settings.reranker_top_n
    candidates = merged[: n * 2]

    # 5. Cross-encoder rerank
    reranked = rerank(query, candidates, top_n=n)

    return reranked
