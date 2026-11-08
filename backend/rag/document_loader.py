
import fitz
from pathlib import Path


def load_pdf(file_path: str):
    """
    Extract text from a PDF document.
    Returns text along with page numbers.
    """

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {file_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    documents = []

    with fitz.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            if text:
                documents.append({
                    "page_number": page_number,
                    "text": text
                })

    if not documents:
        raise ValueError(
            "No extractable text found in the PDF."
        )

    return documents
