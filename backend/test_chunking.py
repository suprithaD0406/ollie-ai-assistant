
from rag.document_loader import load_pdf
from rag.chunking import chunk_documents

pdf_path = r"C:\Users\SUPRITHA D\Downloads\average_resume.pdf"

pages = load_pdf(pdf_path)

chunks = chunk_documents(pages)

print("Chunking successful!")
print("Total pages:", len(pages))
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print("Page:", chunk["page_number"])
    print(chunk["text"])
