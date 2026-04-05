import hashlib
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session, async_session
from app.models import Document, IngestionJobRecord
from app.ingestion.pipeline import run_ingestion
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])


async def _background_ingest(
    job_id: str,
    document_id: uuid.UUID,
    pdf_bytes: bytes,
    content_hash: str,
    title: str | None,
    authors: str | None,
    filename: str,
) -> None:
    """Background task: run ingestion pipeline and persist job status to DB."""

    # Mark as processing
    async with async_session() as session:
        job = await session.get(IngestionJobRecord, job_id)
        if job:
            job.status = "processing"
            job.updated_at = datetime.utcnow()
            await session.commit()

    logger.info(
        "ingestion_started",
        extra={"job_id": job_id, "filename": filename, "document_id": str(document_id)},
    )

    try:
        async with async_session() as session:
            result = await run_ingestion(
                document_id=document_id,
                pdf_bytes=pdf_bytes,
                content_hash=content_hash,
                title=title,
                authors=authors,
                filename=filename,
                session=session,
            )

        async with async_session() as session:
            job = await session.get(IngestionJobRecord, job_id)
            if job:
                job.status = "done"
                job.result = result
                job.updated_at = datetime.utcnow()
                await session.commit()

        logger.info(
            "ingestion_completed",
            extra={
                "job_id": job_id,
                "chunk_count": result.get("chunk_count"),
                "ingestion_time_ms": result.get("ingestion_time_ms"),
            },
        )

    except Exception as e:
        async with async_session() as session:
            job = await session.get(IngestionJobRecord, job_id)
            if job:
                job.status = "failed"
                job.error = str(e)
                job.updated_at = datetime.utcnow()
                await session.commit()

        logger.error(
            "ingestion_failed",
            extra={"job_id": job_id, "error": str(e)},
        )


@router.post("")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    authors: str | None = Form(default=None),
    session: AsyncSession = Depends(get_session),
):
    """Upload a PDF and start background ingestion. Returns job_id immediately."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # Enforce file size limit
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: max {settings.max_upload_size_mb} MB",
        )

    # Compute SHA-256 for duplicate detection
    content_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Reject duplicate uploads
    existing = await session.execute(
        select(Document).where(Document.content_hash == content_hash)
    )
    dup = existing.scalar_one_or_none()
    if dup:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Duplicate document",
                "document_id": str(dup.id),
                "filename": dup.filename,
                "title": dup.title,
            },
        )

    job_id = str(uuid.uuid4())
    document_id = uuid.uuid4()

    # Persist job to DB so status survives API restarts
    job_record = IngestionJobRecord(
        job_id=job_id,
        document_id=document_id,
        status="pending",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    session.add(job_record)
    await session.commit()

    background_tasks.add_task(
        _background_ingest,
        job_id=job_id,
        document_id=document_id,
        pdf_bytes=pdf_bytes,
        content_hash=content_hash,
        title=title,
        authors=authors,
        filename=file.filename,
    )

    logger.info(
        "ingest_job_created",
        extra={"job_id": job_id, "filename": file.filename, "size_bytes": len(pdf_bytes)},
    )

    return {
        "job_id": job_id,
        "document_id": str(document_id),
        "status": "pending",
    }


@router.get("/{job_id}")
async def get_ingest_status(
    job_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Poll the status of a background ingestion job."""
    job = await session.get(IngestionJobRecord, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response: dict = {"job_id": job.job_id, "status": job.status}
    if job.result:
        response["result"] = job.result
    if job.error:
        response["error"] = job.error
    return response
