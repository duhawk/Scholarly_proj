import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session
from app.models import Query
from app.agentic.query_loop import run_query_loop
from app.evaluation.runner import run_evaluation
from app.generation.generator import generate_answer
from app.rate_limit import rate_limit
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str
    top_k: int = settings.top_k
    max_iterations: int = settings.max_iterations
    use_hyde: bool = True
    reranker_top_n: int | None = None


@router.post("", dependencies=[Depends(rate_limit)])
async def query(
    request: Request,
    body: QueryRequest,
    session: AsyncSession = Depends(get_session),
):
    """Run hybrid RAG retrieval and stream the answer as SSE."""
    request_id = getattr(request.state, "request_id", None)

    async def event_generator():
        # 1. Agentic retrieval loop
        loop_result = await run_query_loop(
            question=body.question,
            session=session,
            top_k=body.top_k,
            max_iterations=body.max_iterations,
            use_hyde=body.use_hyde,
            reranker_top_n=body.reranker_top_n,
        )
        chunks = loop_result["chunks"]
        iterations = loop_result["iterations"]

        logger.info(
            "retrieval_completed",
            extra={
                "request_id": request_id,
                "iterations": iterations,
                "chunks_retrieved": len(chunks),
                "chunk_ids": [c["chunk_id"] for c in chunks],
                "question_preview": body.question[:120],
            },
        )

        # 2. Generate answer (non-streaming pass for eval + faithfulness)
        query_id = uuid.uuid4()
        chunk_ids = [c["chunk_id"] for c in chunks]

        gen_result = await generate_answer(body.question, chunks)
        answer = gen_result["answer"]
        citations = gen_result["citations"]

        faith_result = await run_evaluation(query_id, answer, chunks, session)
        faithfulness_score = faith_result["score"]

        logger.info(
            "answer_generated",
            extra={
                "request_id": request_id,
                "query_id": str(query_id),
                "faithfulness_score": faithfulness_score,
                "citations_count": len(citations),
            },
        )

        # 3. Persist query record
        query_record = Query(
            id=query_id,
            question=body.question,
            answer=answer,
            citations=citations,
            retrieved_chunk_ids=chunk_ids,
            iterations=iterations,
            faithfulness_score=faithfulness_score,
            created_at=datetime.utcnow(),
        )
        session.add(query_record)
        await session.commit()

        # 4. Stream answer tokens + metadata as SSE
        import json

        words = answer.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"

        metadata = {
            "type": "metadata",
            "faithfulness_score": faithfulness_score,
            "iterations": iterations,
            "chunks_retrieved": len(chunks),
            "query_id": str(query_id),
        }
        yield f"data: {json.dumps(metadata)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{query_id}")
async def get_query(
    query_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """Retrieve a stored query and its answer by ID."""
    result = await session.execute(select(Query).where(Query.id == query_id))
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="Query not found")
    return {
        "query_id": str(record.id),
        "question": record.question,
        "answer": record.answer,
        "citations": record.citations,
        "faithfulness_score": record.faithfulness_score,
        "iterations": record.iterations,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }
