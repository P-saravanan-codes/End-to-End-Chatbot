import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in Python path for standalone script execution
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.setting import settings
from src.connector.file_loader import FileLoader
from src.parsers.pdf_parser import PDFParser
from src.parsers.cleaner import TextCleaner, CleanedPage
from src.chunkers.semantic import TextChunker, Chunk
from src.embedder.mini_embedder import MiniLMEmbedder, EmbeddedChunk
from src.vector_store.qdrant_client import QdrantVectorStore


class RAGIngestionPipeline:
    """Master orchestrator for the RAG data ingestion pipeline."""

    def __init__(self, in_memory_db: bool = True):
        print("=" * 60)
        print("  Initializing RAG Ingestion Pipeline Components")
        print("=" * 60)
        
        # 1. Initialize Connectors & Processors
        self.loader = FileLoader()
        self.parser = PDFParser(extract_tables=True)
        self.cleaner = TextCleaner()
        self.chunker = TextChunker()
        
        # 2. Initialize Embedder (Using local 384-dim MiniLM on CPU)
        print("Loading local MiniLM embedding model...")
        self.embedder = MiniLMEmbedder()
        
        # 3. Initialize Vector Store (Matching 384 dimensions)
        self.vector_store = QdrantVectorStore(
            collection_name=settings.VECTOR_DB_COLLECTION,
            in_memory=in_memory_db,
            vector_size=384
        )
        print("Pipeline components loaded successfully.\n")

    def run(self) -> Dict[str, Any]:
        """Executes the full ingestion pipeline on all files in data/raw/."""
        start_time = time.time()
        
        # Step 1: Scan local files
        print("[Step 1/5] Scanning data/raw/ directory...")
        files = self.loader.list_files()
        if not files:
            print("  ⚠️ No supported files (.pdf, .txt, .md) found in data/raw/.")
            return {
                "status": "warning",
                "processed_files": 0,
                "processed_pages": 0,
                "generated_chunks": 0,
                "indexed_vectors": 0,
                "execution_time_sec": round(time.time() - start_time, 2)
            }

        print(f"  Found {len(files)} file(s) for processing.")

        # Step 2: Parse and Clean PDF Documents
        print("\n[Step 2/5] Parsing & Cleaning documents...")
        cleaned_pages: List[CleanedPage] = []
        
        for file_info in files:
            print(f"  Processing: {file_info.file_name}")
            if file_info.file_extension == ".pdf":
                raw_pages = self.parser.parse(str(file_info.file_path))
                for page in raw_pages:
                    cleaned_page = self.cleaner.clean_page(
                        page_number=page.page_number,
                        raw_text=page.text,
                        tables=page.tables,
                        metadata=page.metadata
                    )
                    cleaned_pages.append(cleaned_page)

        print(f"  Successfully extracted {len(cleaned_pages)} page(s).")

        # Step 3: Chunk Processed Pages
        print("\n[Step 3/5] Chunking cleaned pages...")
        all_chunks: List[Chunk] = []
        for page in cleaned_pages:
            chunks = self.chunker.chunk_page(page)
            all_chunks.extend(chunks)

        print(f"  Generated {len(all_chunks)} text and table chunk(s).")

        # Step 4: Generate Dense Vector Embeddings
        print("\n[Step 4/5] Generating vector embeddings (384-dim MiniLM)...")
        embedded_chunks: List[EmbeddedChunk] = self.embedder.embed_chunks(all_chunks)
        print(f"  Generated {len(embedded_chunks)} vector embeddings.")

        # Step 5: Index Vectors into Qdrant
        print("\n[Step 5/5] Indexing vectors into Qdrant Vector Store...")
        indexed_count = self.vector_store.upsert_chunks(embedded_chunks)

        elapsed_time = round(time.time() - start_time, 2)

        summary = {
            "status": "success",
            "processed_files": len(files),
            "processed_pages": len(cleaned_pages),
            "generated_chunks": len(all_chunks),
            "indexed_vectors": indexed_count,
            "execution_time_sec": elapsed_time
        }

        print("\n" + "=" * 60)
        print("  Pipeline Execution Summary")
        print("=" * 60)
        print(f"  Status:          {summary['status'].upper()}")
        print(f"  Files Processed: {summary['processed_files']}")
        print(f"  Pages Parsed:    {summary['processed_pages']}")
        print(f"  Chunks Created:  {summary['generated_chunks']}")
        print(f"  Vectors Stored:  {summary['indexed_vectors']}")
        print(f"  Total Time:      {summary['execution_time_sec']} seconds")
        print("=" * 60 + "\n")

        return summary


if __name__ == "__main__":
    # Test pipeline execution directly
    pipeline = RAGIngestionPipeline(in_memory_db=True)
    pipeline.run()