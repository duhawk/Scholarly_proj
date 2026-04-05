SYSTEM_PROMPT = """You are a scholarly research assistant. Answer questions based ONLY on the provided context passages.

Rules:
- Cite sources inline using the format: [Title, chunk N]
- Do not use any knowledge outside the provided passages
- If the context is insufficient to answer, say so clearly
- Be precise and academic in tone
- For every factual claim, include a citation
"""


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into numbered context block."""
    lines = []
    for chunk in chunks:
        title = chunk.get("document_title", "Unknown")
        idx = chunk.get("chunk_index", 0)
        content = chunk.get("content", "").strip()
        lines.append(f"[{title}, chunk {idx}]\n{content}")
    return "\n\n---\n\n".join(lines)


def build_user_message(question: str, chunks: list[dict]) -> str:
    context = build_context_block(chunks)
    return (
        f"Context passages:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}\n\n"
        f"Answer using only the context above. Cite every claim with [Title, chunk N]."
    )


def build_sufficiency_prompt(question: str, chunks: list[dict]) -> str:
    """Prompt for agentic loop sufficiency check."""
    context = build_context_block(chunks)
    return (
        f"Context passages:\n\n{context}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Can you fully answer this question from the context above?\n"
        f"If YES, just answer the question.\n"
        f"If NO, respond with exactly: REFINE: <specific sub-question to search next>"
    )
