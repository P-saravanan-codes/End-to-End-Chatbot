import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Define base directory (root of the medical_chatbot project)
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application Settings managed via Environment Variables / .env file."""

    # Project Information
    PROJECT_NAME: str = "Medical Chatbot RAG Pipeline"
    VERSION: str = "0.1.0"
    DEBUG: bool = False

    # API Keys
    GEMINI_API_KEY: str = Field(..., description="Google Gemini API Key is required")
    OPENAI_API_KEY: Optional[str] = Field(None, description="Optional OpenAI API Key")

    # Data Paths
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"

    # Embedding & Vector Database Settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    VECTOR_DB_COLLECTION: str = "medical_docs"

    # Vector DB Connection Defaults
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    # Load from .env configuration
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore unmentioned environment variables
        case_sensitive=True
    )


# Instantiate settings instance for single-point import
settings = Settings()


if __name__ == "__main__":
    # Sanity Check Script
    print(f"Loaded Project Name: {settings.PROJECT_NAME}")
    print(f"Raw Data Directory:  {settings.DATA_RAW_DIR}")
    print(f"Embedding Model:     {settings.EMBEDDING_MODEL}")
    
    if settings.GEMINI_API_KEY:
        masked_key = settings.GEMINI_API_KEY[:4] + "..." + settings.GEMINI_API_KEY[-4:]
        print(f"Gemini API Key:      {masked_key} (Successfully Loaded)")
    else:
        print("Gemini API Key:      NOT FOUND")