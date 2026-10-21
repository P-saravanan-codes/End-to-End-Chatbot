import sys
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field

# Ensure root directory is in Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sentence_transformers import SentenceTransformer
from config.setting import settings
from src.chunkers.semantic import Chunk


class EmbeddedChunk(BaseModel):
    """Represents a text chunk paired with its 384-dim vector embedding."""
    chunk_id: str
    text: str
    vector: List[float] = Field(..., description="384-dim dense vector")
    metadata: dict


class MiniLMEmbedder:
    """Generates dense vector embeddings locally on CPU using sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Force CPU execution to prevent CUDA driver mismatch errors
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_text(self, text: str) -> List[float]:
        """Generates a 384-dim vector for a single text string."""
        if not text.strip():
            return []
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_chunks(self, chunks: List[Chunk]) -> List[EmbeddedChunk]:
        """Batch embeds a list of Chunk objects."""
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        embedded_chunks = []
        for chunk, vec in zip(chunks, vectors):
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    vector=vec.tolist(),
                    metadata=chunk.metadata
                )
            )

        return embedded_chunks


if __name__ == "__main__":
    embedder = MiniLMEmbedder()
    vec = embedder.embed_text("Amlodipine treats high blood pressure.")
    print("MiniLM Embedder initialized successfully!")
    print(f"Generated Vector Dimension: {len(vec)}")  # Expected: 384