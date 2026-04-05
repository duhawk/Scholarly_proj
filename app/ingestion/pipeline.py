import re
import time
import uuid
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.parser import extract_text
from app.ingestion.chunker import chunk_text
from app.retrieval.embedder import embed_batch
from app.models import Document, Chunk


def _tokenize_for_bm25(text_content: str) -> list[str]:
    """Lowercase, strip punctuation, remove short tokens."""
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "that", "this", "these",
        "those", "it", "its", "as", "if", "not", "no", "nor", "so", "yet",
        "both", "either", "neither", "each", "more", "most", "other", "some",
        "such", "than", "too", "very",
    }
    tokens = re.findall(r"[a-z]+", text_content.lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


async def run_ingestion(
    document_id: uuid.UUID,
    pdf_bytes: bytes,
    title: str | None,
    authors: str | None,
    filename: str,
    session: AsyncSession,
    content_hash: str | None = None,
) -> dict:
    """Full ingestion pipeline: parse → chunk → embed → store."""
    start_ms = time.time()

    # 1. Parse PDF
    pages = extract_text(pdf_bytes)
    if not pages:
        raise ValueError("No text could be extracted from the PDF")

    # 2. Chunk
    raw_chunks = chunk_text(pages)
    if not raw_chunks:
        raise ValueError("No chunks produced from document")

    # 3. Embed in batch
    texts = [c["content"] for c in raw_chunks]
    embeddings = embed_batch(texts)  # shape: (N, 384), unit-normalized

    # 4. Store document
    doc = Document(
        id=document_id,
        filename=filename,
        title=title or filename,
        authors=authors,
        ingested_at=datetime.utcnow(),
        chunk_count=len(raw_chunks),
        content_hash=content_hash,
    )
    session.add(doc)
    await session.flush()

    # 5. Store chunks
    for i, (chunk_data, emb) in enumerate(zip(raw_chunks, embeddings)):
        bm25_tokens = _tokenize_for_bm25(chunk_data["content"])
        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=document_id,
            content=chunk_data["content"],
            embedding=emb.tolist(),
            chunk_index=chunk_data["chunk_index"],
            token_count=chunk_data["token_count"],
            bm25_tokens=bm25_tokens,
            page_num=chunk_data["page_num"],
        )
        session.add(chunk)

    await session.commit()

    elapsed_ms = int((time.time() - start_ms) * 1000)
    return {
        "document_id": str(document_id),
        "filename": filename,
        "chunk_count": len(raw_chunks),
        "ingestion_time_ms": elapsed_ms,
    }
