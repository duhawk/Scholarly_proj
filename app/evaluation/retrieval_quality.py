def compute_retrieval_metrics(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
) -> dict:
    """Compute precision and recall for retrieval quality.

    Args:
        retrieved_chunk_ids: list of chunk IDs returned by retrieval
        relevant_chunk_ids: ground-truth relevant chunk IDs (if available)

    Returns:
        {precision, recall, f1}
    """
    if not retrieved_chunk_ids:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not relevant_chunk_ids:
        return {"precision": None, "recall": None, "f1": None}

    retrieved_set = set(retrieved_chunk_ids)
    relevant_set = set(relevant_chunk_ids)
    tp = len(retrieved_set & relevant_set)

    precision = tp / len(retrieved_set) if retrieved_set else 0.0
    recall = tp / len(relevant_set) if relevant_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
