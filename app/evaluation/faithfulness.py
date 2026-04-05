import re
import anthropic
from app.config import settings


def _split_into_claims(answer: str) -> list[str]:
    """Split answer text into individual sentence-level claims."""
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _build_context_str(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        title = chunk.get("document_title", "Unknown")
        idx = chunk.get("chunk_index", 0)
        content = chunk.get("content", "")
        parts.append(f"[{title}, chunk {idx}]: {content}")
    return "\n\n".join(parts)


def _check_claim(claim: str, context: str, client: anthropic.Anthropic) -> dict:
    """Check a single claim against the context. Returns {claim, supported, justification}."""
    prompt = (
        f"Retrieved passages:\n\n{context}\n\n"
        f"---\n\n"
        f"Claim to verify: \"{claim}\"\n\n"
        f"Is this claim directly supported by the retrieved passages above?\n"
        f"Answer with YES or NO on the first line, followed by a brief justification."
    )
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    lines = text.split("\n", 1)
    first_line = lines[0].strip().upper()
    supported = first_line.startswith("YES")
    justification = lines[1].strip() if len(lines) > 1 else text
    return {"claim": claim, "supported": supported, "justification": justification}


def evaluate_faithfulness(answer: str, chunks: list[dict]) -> dict:
    """Evaluate faithfulness of answer against retrieved chunks.

    Makes one Claude API call per claim.

    Returns:
        {score: float, claims: [{claim, supported, justification}]}
    """
    claims = _split_into_claims(answer)
    if not claims:
        return {"score": 1.0, "claims": []}

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    context = _build_context_str(chunks)

    results = []
    for claim in claims:
        result = _check_claim(claim, context, client)
        results.append(result)

    supported_count = sum(1 for r in results if r["supported"])
    score = supported_count / len(results) if results else 1.0

    return {"score": score, "claims": results}
