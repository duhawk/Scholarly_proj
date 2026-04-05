import uuid
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import EvalResult
from app.evaluation.faithfulness import evaluate_faithfulness


async def run_evaluation(
    query_id: uuid.UUID,
    answer: str,
    chunks: list[dict],
    session: AsyncSession,
) -> dict:
    """Run faithfulness evaluation and persist results.

    Returns faithfulness eval dict {score, claims}.
    """
    faith_result = evaluate_faithfulness(answer, chunks)

    eval_record = EvalResult(
        id=uuid.uuid4(),
        query_id=query_id,
        metric="faithfulness",
        score=faith_result["score"],
        detail={"claims": faith_result["claims"]},
        evaluated_at=datetime.utcnow(),
    )
    session.add(eval_record)
    await session.commit()

    return faith_result
