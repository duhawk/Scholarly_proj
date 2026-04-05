"""Initial schema with pgvector

Revision ID: 001
Revises:
Create Date: 2026-04-03
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("title", sa.String(1024), nullable=True),
        sa.Column("authors", sa.String(1024), nullable=True),
        sa.Column("ingested_at", sa.DateTime, nullable=False),
        sa.Column("chunk_count", sa.Integer, default=0),
    )

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", sa.Text, nullable=True),  # placeholder, altered below
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("token_count", sa.Integer, default=0),
        sa.Column("bm25_tokens", postgresql.JSON, nullable=True),
        sa.Column("page_num", sa.Integer, nullable=True),
    )
    # Replace text column with proper vector(384) type
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(384) USING NULL")

    op.create_table(
        "queries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("answer", sa.Text, nullable=True),
        sa.Column("citations", postgresql.JSON, nullable=True),
        sa.Column("retrieved_chunk_ids", postgresql.JSON, nullable=True),
        sa.Column("iterations", sa.Integer, default=1),
        sa.Column("faithfulness_score", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )

    op.create_table(
        "eval_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("query_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("queries.id", ondelete="CASCADE"), nullable=False),
        sa.Column("metric", sa.String(256), nullable=False),
        sa.Column("score", sa.Float, nullable=True),
        sa.Column("detail", postgresql.JSON, nullable=True),
        sa.Column("evaluated_at", sa.DateTime, nullable=False),
    )

    # Index for fast cosine similarity search
    op.execute("CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")


def downgrade() -> None:
    op.drop_table("eval_results")
    op.drop_table("queries")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.execute("DROP EXTENSION IF EXISTS vector")
