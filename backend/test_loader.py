
from rag.document_loader import load_pdf

pdf_path = r"C:\Users\SUPRITHA D\Downloads\average_resume.pdf"

pages = load_pdf(pdf_path)

print("PDF extraction successful!")
print("Total extracted pages:", len(pages))

for page in pages:
    print(f"\nPage {page['page_number']}:")
    print(page["text"][:300])
