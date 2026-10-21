import os
import sys
from pathlib import Path
from typing import List, Generator, Optional
from pydantic import BaseModel, Field

# Ensure project root directory is in Python path for standalone script execution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.setting import settings


class LoadedFile(BaseModel):
    """Metadata model representing a discovered local file."""
    file_path: Path
    file_name: str
    file_extension: str
    file_size_bytes: int


class FileLoader:
    """Scans and loads local documents from the configured raw data directory."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        supported_extensions: Optional[List[str]] = None
    ):
        # Default to DATA_RAW_DIR from settings if no custom path is provided
        self.base_dir = Path(base_dir) if base_dir else Path(settings.DATA_RAW_DIR)
        self.supported_extensions = supported_extensions or [".pdf", ".txt", ".md"]

    def list_files(self) -> List[LoadedFile]:
        """Scans directory and returns metadata for all supported raw files."""
        if not self.base_dir.exists():
            # Automatically create the directory if it doesn't exist yet
            self.base_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created missing raw data directory: {self.base_dir}")
            return []

        loaded_files: List[LoadedFile] = []

        # Recursively search through base directory
        for file_path in self.base_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                file_info = LoadedFile(
                    file_path=file_path,
                    file_name=file_path.name,
                    file_extension=file_path.suffix.lower(),
                    file_size_bytes=file_path.stat().st_size
                )
                loaded_files.append(file_info)

        return loaded_files

    def load_stream(self) -> Generator[LoadedFile, None, None]:
        """Generator that yields loaded files one by one (useful for memory efficiency)."""
        for file in self.list_files():
            yield file


if __name__ == "__main__":
    # Quick Test / Sanity Check Execution
    print("--- Testing File Loader ---")
    loader = FileLoader()
    print(f"Scanning target directory: {loader.base_dir}")
    
    files = loader.list_files()
    print(f"Found {len(files)} supported document(s):")
    
    for f in files:
        print(f" - {f.file_name} ({f.file_size_bytes} bytes)")