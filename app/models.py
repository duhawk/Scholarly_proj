import uuid
from datetime import datetime
from typing import Literal

from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(512), nullable=False)
    title = Column(String(1024), nullable=True)
    authors = Column(String(1024), nullable=True)
    ingested_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    chunk_count = Column(Integer, default=0)
    content_hash = Column(String(64), nullable=True, index=True)  # SHA-256 hex for dedup

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384), nullable=True)
    chunk_index = Column(Integer, nullable=False)
    token_count = Column(Integer, default=0)
    bm25_tokens = Column(JSON, nullable=True)
    page_num = Column(Integer, nullable=True)

    document = relationship("Document", back_populates="chunks")


class Query(Base):
    __tablename__ = "queries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    citations = Column(JSON, nullable=True)
    retrieved_chunk_ids = Column(JSON, nullable=True)
    iterations = Column(Integer, default=1)
    faithfulness_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    eval_results = relationship("EvalResult", back_populates="query", cascade="all, delete-orphan")


class EvalResult(Base):
    __tablename__ = "eval_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id = Column(UUID(as_uuid=True), ForeignKey("queries.id", ondelete="CASCADE"), nullable=False)
    metric = Column(String(256), nullable=False)
    score = Column(Float, nullable=True)
    detail = Column(JSON, nullable=True)
    evaluated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    query = relationship("Query", back_populates="eval_results")


class IngestionJobRecord(Base):
    """Persisted ingestion job — survives API restarts."""
    __tablename__ = "ingestion_jobs"

    job_id = Column(String(64), primary_key=True)
    document_id = Column(UUID(as_uuid=True), nullable=False)
    status = Column(String(32), default="pending", nullable=False)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
