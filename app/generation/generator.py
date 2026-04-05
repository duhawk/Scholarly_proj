import json
import re
from collections.abc import AsyncGenerator

import anthropic

from app.config import settings
from app.generation.prompt import SYSTEM_PROMPT, build_user_message


def _parse_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """Extract inline citations from answer text and map to chunk metadata."""
    # Match patterns like [Title, chunk N] or [Title, chunk N]
    pattern = re.compile(r'\[([^\]]+),\s*chunk\s+(\d+)\]')
    seen = set()
    citations = []

    chunk_map: dict[tuple[str, int], dict] = {}
    for c in chunks:
        key = (c.get("document_title", ""), c.get("chunk_index", 0))
        chunk_map[key] = c

    for match in pattern.finditer(answer):
        title = match.group(1).strip()
        chunk_idx = int(match.group(2))
        key = (title, chunk_idx)
        if key in seen:
            continue
        seen.add(key)

        chunk = chunk_map.get(key)
        citation = {
            "document_title": title,
            "chunk_index": chunk_idx,
            "excerpt": chunk["content"][:200] + "..." if chunk else "",
        }
        citations.append(citation)

    return citations


async def generate_answer(
    question: str,
    chunks: list[dict],
) -> dict:
    """Non-streaming generation. Returns {answer, citations}."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(question, chunks)}],
    )
    answer = message.content[0].text
    citations = _parse_citations(answer, chunks)
    return {"answer": answer, "citations": citations}


async def stream_answer(
    question: str,
    chunks: list[dict],
    faithfulness_score: float | None,
    iterations: int,
) -> AsyncGenerator[str, None]:
    """Stream answer as SSE events.

    Yields SSE-formatted strings:
        data: {"type": "token", "content": "..."}
        data: {"type": "citations", "citations": [...]}
        data: {"type": "metadata", ...}
        data: [DONE]
    """
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    full_answer = ""

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(question, chunks)}],
    ) as stream:
        for text_delta in stream.text_stream:
            full_answer += text_delta
            payload = json.dumps({"type": "token", "content": text_delta})
            yield f"data: {payload}\n\n"

    citations = _parse_citations(full_answer, chunks)
    yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"

    metadata = {
        "type": "metadata",
        "faithfulness_score": faithfulness_score,
        "iterations": iterations,
        "chunks_retrieved": len(chunks),
    }
    yield f"data: {json.dumps(metadata)}\n\n"
    yield "data: [DONE]\n\n"
