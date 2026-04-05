import fitz  # PyMuPDF


def extract_text(pdf_bytes: bytes) -> list[dict]:
    """Extract text per page from PDF bytes.

    Returns list of {page_num, text} dicts.
    """
    pages = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text and text.strip():
                pages.append({"page_num": page_num + 1, "text": text.strip()})
        doc.close()
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {e}") from e
    return pages
