from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.parsers.cleaner import CleanedPage


class HierarchicalChunk(BaseModel):
    """Represents a child chunk linked to its larger parent context."""
    child_id: str
    parent_id: str
    child_text: str
    parent_text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HierarchicalChunker:
    """Creates parent-child chunk pairs for high-precision retrieval."""

    def __init__(
        self,
        parent_chunk_size: int = 1200,
        parent_overlap: int = 100,
        child_chunk_size: int = 300,
        child_overlap: int = 30
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap
        )

    def chunk_page(self, page: CleanedPage) -> List[HierarchicalChunk]:
        hierarchical_chunks: List[HierarchicalChunk] = []

        if not page.text.strip():
            return hierarchical_chunks

        # 1. Split text into parent chunks
        parent_texts = self.parent_splitter.split_text(page.text)

        for p_idx, p_text in enumerate(parent_texts):
            parent_id = f"p_{page.page_number}_{p_idx}"

            # 2. Split each parent chunk into smaller child chunks
            child_texts = self.child_splitter.split_text(p_text)

            for c_idx, c_text in enumerate(child_texts):
                child_id = f"c_{page.page_number}_{p_idx}_{c_idx}"
                
                meta = page.metadata.copy()
                meta.update({
                    "page_number": page.page_number,
                    "parent_id": parent_id,
                    "child_id": child_id
                })

                hierarchical_chunks.append(
                    HierarchicalChunk(
                        child_id=child_id,
                        parent_id=parent_id,
                        child_text=c_text,
                        parent_text=p_text,
                        metadata=meta
                    )
                )

        return hierarchical_chunks


if __name__ == "__main__":
    print("HierarchicalChunker initialized successfully.")