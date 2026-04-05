import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import Document

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
async def list_documents(session: AsyncSession = Depends(get_session)):
    """List all ingested documents."""
    result = await session.execute(select(Document).order_by(Document.ingested_at.desc()))
    docs = result.scalars().all()
    return [
        {
            "document_id": str(doc.id),
            "filename": doc.filename,
            "title": doc.title,
            "authors": doc.authors,
            "chunk_count": doc.chunk_count,
            "ingested_at": doc.ingested_at.isoformat() if doc.ingested_at else None,
        }
        for doc in docs
    ]


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """Delete a document and all its chunks (cascade). Returns 204 on success, 404 if not found."""
    result = await session.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    await session.delete(doc)
    await session.commit()
