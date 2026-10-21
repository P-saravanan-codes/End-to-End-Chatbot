import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root directory is in Python path
sys.path.append(str(Path(__file__).resolve().parent))

from google import genai
from config.setting import settings
from src.pipeline import RAGIngestionPipeline


class RAGQueryEngine:
    """Retrieves relevant context chunks from Qdrant and generates an answer using Gemini."""

    def __init__(self, pipeline: RAGIngestionPipeline):
        self.pipeline = pipeline
        self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.llm_model = "gemini-3.8-flash"

    def ask(self, question: str, top_k: int = 3) -> str:
        """Retrieves context from Qdrant and invokes Gemini to generate an answer."""
        if not question.strip():
            return "Please enter a valid question."

        # 1. Embed query using MiniLM local embedder
        query_vector = self.pipeline.embedder.embed_text(question)

        # 2. Search Qdrant vector store
        search_results: List[Dict[str, Any]] = self.pipeline.vector_store.search(
            query_vector=query_vector,
            top_k=top_k
        )

        if not search_results:
            return "No relevant information found in the ingested documents."

        # 3. Format retrieved chunks into a context block
        context_blocks = []
        for idx, res in enumerate(search_results, 1):
            source = res["metadata"].get("source", "Unknown Document")
            page = res["metadata"].get("page_number", "N/A")
            context_blocks.append(
                f"[Source {idx}: {Path(source).name}, Page {page}]\n{res['text']}"
            )

        context_str = "\n\n".join(context_blocks)

        # 4. Construct prompt for Gemini
        prompt = f"""You are a helpful medical assistant. Answer the user's question using ONLY the provided context below.
If the context does not contain enough information to answer, state clearly that you don't know based on the provided documents.

CONTEXT:
{context_str}

USER QUESTION:
{question}

ANSWER:"""

        # 5. Generate response via Gemini API
        response = self.gemini_client.models.generate_content(
            model=self.llm_model,
            contents=prompt
        )

        return response.text


def main():
    print("=" * 65)
    print("      Medical Chatbot - RAG Ingestion & Q/A System")
    print("=" * 65)

    # 1. Initialize and run ingestion pipeline
    pipeline = RAGIngestionPipeline(in_memory_db=True)
    summary = pipeline.run()

    if summary["indexed_vectors"] == 0:
        print("\n⚠️ No document vectors were indexed.")
        print("Please place your PDF files in 'data/raw/' and run main.py again.")
        return

    # 2. Start Interactive RAG Q&A
    query_engine = RAGQueryEngine(pipeline=pipeline)

    print("\n" + "=" * 65)
    print("  RAG System Ready! Type your question below (or 'exit' to quit).")
    print("=" * 65 + "\n")

    while True:
        try:
            user_query = input("Ask a Question > ").strip()
            if not user_query:
                continue

            if user_query.lower() in ["exit", "quit", "q"]:
                print("\nExiting RAG system. Goodbye!")
                break

            print("\nSearching vector database and generating answer...\n")
            answer = query_engine.ask(user_query)

            print("--- RESPONSE ---")
            print(answer)
            print("-" * 65 + "\n")

        except KeyboardInterrupt:
            print("\nExiting system.")
            break
        except Exception as e:
            print(f"\nError processing query: {e}\n")


if __name__ == "__main__":
    main()