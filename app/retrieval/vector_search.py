import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def vector_search(
    query_embedding: np.ndarray,
    session: AsyncSession,
    top_k: int = 10,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """Search chunks by cosine similarity using pgvector <=> operator.

    Args:
        document_ids: optional list of document UUID strings to restrict search to.

    Returns list of {chunk_id, document_id, document_title, content, chunk_index,
                      page_num, similarity} sorted by similarity descending.
    """
    embedding_list = query_embedding.tolist()
    embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"

    if document_ids:
        # Build a safe parameterised IN-list via numbered placeholders
        placeholders = ", ".join(f":doc_{i}::uuid" for i in range(len(document_ids)))
        sql = text(f"""
            SELECT
                c.id::text AS chunk_id,
                c.document_id::text,
                d.title AS document_title,
                c.content,
                c.chunk_index,
                c.page_num,
                1 - (c.embedding <=> :embedding::vector) AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.document_id IN ({placeholders})
            ORDER BY c.embedding <=> :embedding::vector
            LIMIT :top_k
        """)
        params = {"embedding": embedding_str, "top_k": top_k}
        params.update({f"doc_{i}": doc_id for i, doc_id in enumerate(document_ids)})
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
                1 - (c.embedding <=> :embedding::vector) AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> :embedding::vector
            LIMIT :top_k
        """)
        result = await session.execute(
            sql,
            {"embedding": embedding_str, "top_k": top_k},
        )

    rows = result.mappings().all()
    return [dict(row) for row in rows]
