
def chunk_documents(pages, chunk_size=500, overlap=100):
    """
    Split extracted PDF pages into smaller text chunks.

    Each chunk retains its original page number.
    """

    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page_number": page_number
                })

            start += chunk_size - overlap

    return chunks
