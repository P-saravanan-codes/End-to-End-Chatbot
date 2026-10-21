import fitz  # PyMuPDF
from typing import List, Dict, Any
from pydantic import BaseModel


class ExtractedPage(BaseModel):
    page_number: int
    text: str
    tables: List[str]
    metadata: Dict[str, Any]


class PDFParser:
    """Extracts clean text, structured tables, and page metadata from PDFs for RAG pipelines."""

    def __init__(self, extract_tables: bool = True):
        self.extract_tables = extract_tables

    def parse(self, pdf_path: str) -> List[ExtractedPage]:
        doc = fitz.open("data/Medical_book.pdf")
        extracted_pages: List[ExtractedPage] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # 1. Extract raw page text
            page_text = page.get_text("text").strip()

            # 2. Extract tables into Markdown format if enabled
            formatted_tables = []
            if self.extract_tables:
                tables = page.find_tables()
                for table in tables:
                    # Convert table grid into clean Markdown
                    md_table = table.to_markdown()
                    if md_table:
                        formatted_tables.append(md_table)

            # 3. Assemble page data with metadata needed for chunking & retrieval
            page_data = ExtractedPage(
                page_number=page_num + 1,
                text=page_text,
                tables=formatted_tables,
                metadata={
                    "source": pdf_path,
                    "total_pages": len(doc),
                    "page": page_num + 1,
                }
            )
            extracted_pages.append(page_data)

        doc.close()
        return extracted_pages


# Quick Test / Verification Execution
if __name__ == "__main__":
    import sys

    # Quick sanity check
    parser = PDFParser(extract_tables=True)
    print("PDFParser initialized successfully.")