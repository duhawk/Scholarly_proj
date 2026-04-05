from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.evaluation.retrieval_quality import compute_retrieval_metrics
from app.models import EvalResult, Query as QueryModel

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.get("")
async def get_eval(
    query_id: str = Query(..., description="UUID of the query to retrieve evaluation for"),
    session: AsyncSession = Depends(get_session),
):
    """Retrieve stored evaluation results for a query."""
    result = await session.execute(
        select(EvalResult).where(EvalResult.query_id == query_id)
    )
    eval_results = result.scalars().all()

    if not eval_results:
        raise HTTPException(status_code=404, detail="No evaluation results found for this query")

    # Find faithfulness result
    faith = next((e for e in eval_results if e.metric == "faithfulness"), None)

    response = {
        "query_id": query_id,
        "faithfulness_score": faith.score if faith else None,
        "claims": faith.detail.get("claims", []) if faith and faith.detail else [],
        "metrics": [
            {
                "metric": e.metric,
                "score": e.score,
                "detail": e.detail,
                "evaluated_at": e.evaluated_at.isoformat() if e.evaluated_at else None,
            }
            for e in eval_results
        ],
    }
    return response


class RetrievalEvalRequest(BaseModel):
    query_id: str
    ground_truth_chunk_ids: List[str]


@router.post("/retrieval")
async def eval_retrieval(
    body: RetrievalEvalRequest,
    session: AsyncSession = Depends(get_session),
):
    """Compute retrieval precision/recall/F1 for a stored query against ground-truth chunk IDs."""
    result = await session.execute(
        select(QueryModel).where(QueryModel.id == body.query_id)
    )
    query_record = result.scalar_one_or_none()
    if query_record is None:
        raise HTTPException(status_code=404, detail="Query not found")

    retrieved_ids = query_record.retrieved_chunk_ids or []
    metrics = compute_retrieval_metrics(retrieved_ids, body.ground_truth_chunk_ids)

    return {
        "query_id": body.query_id,
        "retrieved_chunk_ids": retrieved_ids,
        "ground_truth_chunk_ids": body.ground_truth_chunk_ids,
        **metrics,
    }
