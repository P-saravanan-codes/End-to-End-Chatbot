from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.parsers.cleaner import CleanedPage
from config.setting import settings


class Chunk(BaseModel):
    """Represents an individual text chunk ready for vector embedding."""
    chunk_id: str
    text: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextChunker:
    """Splits CleanedPage objects into semantic text chunks while preserving metadata."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Uses recursive splitting to preserve paragraphs and sentence boundaries
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_page(self, page: CleanedPage) -> List[Chunk]:
        """Splits a single CleanedPage into chunks."""
        chunks: List[Chunk] = []

        # 1. Chunk main page text
        if page.text.strip():
            raw_splits = self.splitter.split_text(page.text)
            for idx, split in enumerate(raw_splits):
                chunk_meta = page.metadata.copy()
                chunk_meta.update({
                    "page_number": page.page_number,
                    "chunk_type": "text",
                    "chunk_size": len(split)
                })
                
                chunk = Chunk(
                    chunk_id=f"page_{page.page_number}_text_chunk_{idx}",
                    text=split,
                    chunk_index=idx,
                    metadata=chunk_meta
                )
                chunks.append(chunk)

        # 2. Treat Markdown tables as distinct, unsplit chunks to prevent layout destruction
        for table_idx, table_str in enumerate(page.tables):
            table_meta = page.metadata.copy()
            table_meta.update({
                "page_number": page.page_number,
                "chunk_type": "table",
                "table_index": table_idx
            })
            
            table_chunk = Chunk(
                chunk_id=f"page_{page.page_number}_table_chunk_{table_idx}",
                text=table_str,
                chunk_index=len(chunks),
                metadata=table_meta
            )
            chunks.append(table_chunk)

        return chunks

    def chunk_document(self, pages: List[CleanedPage]) -> List[Chunk]:
        """Processes a full document list of CleanedPages into a flat list of Chunks."""
        all_chunks: List[Chunk] = []
        for page in pages:
            all_chunks.extend(self.chunk_page(page))
        return all_chunks


if __name__ == "__main__":
    # Quick Verification Script
    from src.parsers.cleaner import CleanedPage

    sample_page = CleanedPage(
        page_number=1,
        text="Amlodipine is a calcium channel blocker used to treat high blood pressure. " * 20,
        tables=["| Drug | Dose |\n|---|---|\n| Amlodipine | 5mg |"],
        metadata={"source": "cardiology_guide.pdf"}
    )

    chunker = TextChunker(chunk_size=200, chunk_overlap=30)
    result_chunks = chunker.chunk_page(sample_page)

    print(f"Generated {len(result_chunks)} chunk(s):")
    for c in result_chunks:
        print(f"[{c.chunk_id}] ({c.metadata['chunk_type']}): {c.text[:60]}...")