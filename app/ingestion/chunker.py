import re


def _approximate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token."""
    return max(1, len(text) // 4)


def _find_sentence_boundary(text: str, target_pos: int, window: int = 100) -> int:
    """Find the nearest sentence boundary (. ! ?) around target_pos."""
    start = max(0, target_pos - window)
    end = min(len(text), target_pos + window)
    region = text[start:end]

    # Find all sentence-ending positions within the window
    sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+', region)]

    if not sentence_ends:
        return target_pos

    # Pick the closest boundary to the target position relative to start
    relative_target = target_pos - start
    best = min(sentence_ends, key=lambda pos: abs(pos - relative_target))
    return start + best


def chunk_text(pages: list[dict], chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """Chunk pages into overlapping fixed-size chunks with sentence-aware boundaries.

    Args:
        pages: list of {page_num, text} from parser
        chunk_size: target chunk size in tokens
        overlap: overlap between chunks in tokens

    Returns:
        list of {content, chunk_index, page_num, token_count}
    """
    # Concatenate all pages while tracking page boundaries
    full_text = ""
    page_map: list[tuple[int, int]] = []  # (char_start, page_num)

    for page in pages:
        start = len(full_text)
        full_text += page["text"] + "\n\n"
        page_map.append((start, page["page_num"]))

    def get_page_num(char_pos: int) -> int:
        page_num = page_map[0][1]
        for char_start, pn in page_map:
            if char_start <= char_pos:
                page_num = pn
            else:
                break
        return page_num

    chunks = []
    chunk_index = 0
    chunk_size_chars = chunk_size * 4  # ~4 chars per token
    overlap_chars = overlap * 4

    pos = 0
    while pos < len(full_text):
        end = min(pos + chunk_size_chars, len(full_text))

        # Snap end to sentence boundary if not at end of text
        if end < len(full_text):
            end = _find_sentence_boundary(full_text, end)

        content = full_text[pos:end].strip()
        if content:
            chunks.append({
                "content": content,
                "chunk_index": chunk_index,
                "page_num": get_page_num(pos),
                "token_count": _approximate_tokens(content),
            })
            chunk_index += 1

        # Advance with overlap
        next_pos = end - overlap_chars
        if next_pos <= pos:
            next_pos = pos + 1
        pos = next_pos

    return chunks
