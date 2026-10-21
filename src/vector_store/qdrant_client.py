import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in Python path for direct script execution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from config.setting import settings
from src.embedder.mini_embedder import EmbeddedChunk, MiniLMEmbedder


class QdrantVectorStore:
    """Handles collection creation, vector upserts, and vector searches in Qdrant."""

    def __init__(
        self,
        collection_name: str = settings.VECTOR_DB_COLLECTION,
        host: str = settings.QDRANT_HOST,
        port: int = settings.QDRANT_PORT,
        in_memory: bool = True,
        vector_size: int = 384  # Matches 384-dim vectors from MiniLM / BGE-small
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size

        # In-memory for fast prototyping, or connect via host/port
        if in_memory:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(host=host, port=port)

        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Creates the Qdrant collection if it does not already exist."""
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Created Qdrant collection '{self.collection_name}' (vector size: {self.vector_size})")

    def upsert_chunks(self, embedded_chunks: List[EmbeddedChunk]) -> int:
        """Upserts a list of EmbeddedChunk objects into the vector database."""
        if not embedded_chunks:
            return 0

        points: List[PointStruct] = []

        for chunk in embedded_chunks:
            # Generate deterministic UUIDv5 from chunk_id to prevent duplicate points
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

            # Store both metadata and chunk text in the vector payload
            payload = chunk.metadata.copy()
            payload["text"] = chunk.text
            payload["chunk_id"] = chunk.chunk_id

            points.append(
                PointStruct(
                    id=point_id,
                    vector=chunk.vector,
                    payload=payload
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Indexed {len(points)} vector point(s) into collection '{self.collection_name}'.")
        return len(points)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 3,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Searches vector space for matches nearest to query_vector."""
        if not query_vector:
            return []

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold
        ).points

        hits = []
        for point in search_result:
            hits.append({
                "score": point.score,
                "text": point.payload.get("text", ""),
                "chunk_id": point.payload.get("chunk_id", ""),
                "metadata": point.payload
            })

        return hits


if __name__ == "__main__":
    # Sanity Check & Verification Script
    print("--- Testing Qdrant Vector Store with Local Embedder ---")

    # 1. Initialize Vector Store & Local Embedder
    vector_store = QdrantVectorStore(in_memory=True)
    embedder = MiniLMEmbedder()

    # 2. Create Sample Medical Documents & Embed Them
    sample_text_1 = "Amlodipine is a calcium channel blocker used to treat hypertension."
    sample_text_2 = "Metformin is a first-line medication for type 2 diabetes management."

    chunk_1 = EmbeddedChunk(
        chunk_id="doc1_p1_c0",
        text=sample_text_1,
        vector=embedder.embed_text(sample_text_1),
        metadata={"source": "medical_book.pdf", "page": 1}
    )
    chunk_2 = EmbeddedChunk(
        chunk_id="doc2_p1_c0",
        text=sample_text_2,
        vector=embedder.embed_text(sample_text_2),
        metadata={"source": "medical_book.pdf", "page": 1}
    )

    # 3. Upsert Chunks into Qdrant
    vector_store.upsert_chunks([chunk_1, chunk_2])

    # 4. Search Query
    query = "What drug manages high blood pressure?"
    query_vector = embedder.embed_text(query)
    results = vector_store.search(query_vector=query_vector, top_k=1)

    print("\n--- Search Query Results ---")
    print(f"Query:        '{query}'")
    print(f"Matched Text:  {results[0]['text']}")
    print(f"Similarity:    {results[0]['score']:.4f}")
    print(f"Source File:   {results[0]['metadata'].get('source')}")