import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class CleanedPage(BaseModel):
    """Represents a sanitized document page ready for chunking."""
    page_number: int
    text: str
    tables: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextCleaner:
    """Sanitizes raw extracted PDF text for downstream chunking and embedding."""

    def __init__(self, min_line_length: int = 3):
        self.min_line_length = min_line_length

    def _fix_hyphenation(self, text: str) -> str:
        """Fixes words split across line breaks (e.g., 'med-\nical' -> 'medical')."""
        return re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    def _remove_page_numbers_and_footers(self, text: str) -> str:
        """Strips standalone page numbers and common header/footer patterns."""
        # Removes lines that are just numbers (e.g., "Page 12", "12", "Page 12 of 45")
        text = re.sub(r'(?i)^\s*(page\s+)?\d+(\s+of\s+\d+)?\s*$', '', text, flags=re.MULTILINE)
        return text

    def _clean_whitespace(self, text: str) -> str:
        """Normalizes irregular spaces while retaining paragraph structure."""
        # Replace multiple horizontal spaces/tabs with a single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Reduce 3 or more consecutive newlines to double newlines (paragraph boundaries)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _remove_junk_characters(self, text: str) -> str:
        """Removes non-printable ASCII noise and replacement characters."""
        # Strips null bytes and unicode replacement characters (\ufffd)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd]', '', text)
        return text

    def clean_text(self, raw_text: str) -> str:
        """Applies full cleaning pipeline to a raw text string."""
        if not raw_text:
            return ""

        text = self._remove_junk_characters(raw_text)
        text = self._fix_hyphenation(text)
        text = self._remove_page_numbers_and_footers(text)
        text = self._clean_whitespace(text)
        return text

    def clean_page(
        self,
        page_number: int,
        raw_text: str,
        tables: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CleanedPage:
        """Cleans page text and returns a validated CleanedPage Pydantic model."""
        cleaned_str = self.clean_text(raw_text)
        
        # Clean extracted tables (ensure no leading/trailing blank lines)
        cleaned_tables = [t.strip() for t in (tables or []) if t.strip()]

        # Enrich metadata
        meta = metadata.copy() if metadata else {}
        meta["cleaned"] = True
        meta["char_count"] = len(cleaned_str)

        return CleanedPage(
            page_number=page_number,
            text=cleaned_str,
            tables=cleaned_tables,
            metadata=meta
        )


# Verification & Direct Execution Block
if __name__ == "__main__":
    raw_sample = """
    Page 14
    
    Clinical Guideline: Hyper-\ntension Treatment
    
    1. Introduction \ufffd
    This document outlines standard care   protocols for adult patients.
    
    
    Page 14 of 50
    """

    cleaner = TextCleaner()
    cleaned_output = cleaner.clean_page(
        page_number=14,
        raw_text=raw_sample,
        tables=["| Condition | Drug |\n|---|---|\n| Stage 1 | Amlodipine |"],
        metadata={"source": "Medical_book.pdf"}
    )

    print("--- Pydantic Cleaned Page Model ---")
    print(cleaned_output.model_dump_json(indent=2))