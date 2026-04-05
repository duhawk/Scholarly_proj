"""Tune ivfflat index lists parameter based on corpus size

Revision ID: 003
Revises: 002
Create Date: 2026-04-04
"""
import math
from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Query current chunk count to compute optimal lists value.
    # Formula: lists = max(1, floor(sqrt(n_chunks)))
    conn = op.get_bind()
    row = conn.execute(sa.text("SELECT COUNT(*) FROM chunks")).fetchone()
    n_chunks = row[0] if row else 0
    lists = max(1, math.floor(math.sqrt(n_chunks)))

    # Drop the existing auto-named ivfflat index and recreate with a stable name
    # and the tuned lists value.
    op.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
    # The auto-generated name from 001_initial may differ; drop by introspection as well.
    op.execute(
        "DO $$ "
        "DECLARE idx_name text; "
        "BEGIN "
        "  SELECT indexname INTO idx_name "
        "  FROM pg_indexes "
        "  WHERE tablename = 'chunks' AND indexdef ILIKE '%ivfflat%' "
        "  LIMIT 1; "
        "  IF idx_name IS NOT NULL THEN "
        "    EXECUTE 'DROP INDEX IF EXISTS ' || idx_name; "
        "  END IF; "
        "END $$"
    )
    op.execute(
        f"CREATE INDEX chunks_embedding_idx ON chunks "
        f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
    )


def downgrade() -> None:
    # Restore to default lists = 100
    op.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
    op.execute(
        "CREATE INDEX chunks_embedding_idx ON chunks "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )
