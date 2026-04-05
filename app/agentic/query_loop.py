import re
import anthropic
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.generation.prompt import SYSTEM_PROMPT, build_sufficiency_prompt
from app.retrieval.hybrid import hybrid_search


def _extract_refine_query(response_text: str) -> str | None:
    """Extract sub-query from REFINE: <sub-query> response."""
    match = re.match(r"REFINE:\s*(.+)", response_text.strip(), re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


async def run_query_loop(
    question: str,
    session: AsyncSession,
    top_k: int | None = None,
    max_iterations: int | None = None,
    document_ids: list[str] | None = None,
    use_hyde: bool = True,
    reranker_top_n: int | None = None,
) -> dict:
    """Iterative retrieval loop with agentic sufficiency checking.

    Args:
        document_ids: optional list of document UUID strings to restrict retrieval to.

    Returns {chunks, iterations} where chunks are deduplicated across all iterations.
    """
    k = top_k or settings.top_k
    max_iter = max_iterations or settings.max_iterations

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    all_chunks: dict[str, dict] = {}  # chunk_id -> chunk, dedup across iterations
    current_query = question
    iterations = 0

    while iterations < max_iter:
        iterations += 1

        # Retrieve chunks for current query
        new_chunks = await hybrid_search(
            current_query,
            session,
            top_k=k,
            document_ids=document_ids,
            use_hyde=use_hyde,
            reranker_top_n=reranker_top_n,
        )
        for chunk in new_chunks:
            cid = chunk["chunk_id"]
            if cid not in all_chunks:
                all_chunks[cid] = chunk

        combined_chunks = list(all_chunks.values())

        # Ask Claude if context is sufficient
        prompt = build_sufficiency_prompt(question, combined_chunks)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = response.content[0].text.strip()

        # Check if Claude wants to refine the search
        sub_query = _extract_refine_query(response_text)
        if sub_query is None or iterations >= max_iter:
            # Sufficient context or hit max iterations — stop
            break

        # Loop with the refined sub-query
        current_query = sub_query

    return {
        "chunks": list(all_chunks.values()),
        "iterations": iterations,
    }
