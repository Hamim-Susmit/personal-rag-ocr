from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Settings:
    docs_dir: Path = Path(os.getenv("DOCS_DIR", "data/raw_docs"))
    index_dir: Path = Path(os.getenv("INDEX_DIR", "data/index"))
    sqlite_path: Path = Path(os.getenv("SQLITE_PATH", "data/metadata.db"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    ocr_min_chars_threshold: int = int(os.getenv("OCR_MIN_CHARS_THRESHOLD", "80"))
    ocr_lang: str = os.getenv("OCR_LANG", "eng")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")
    ollama_timeout_seconds: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))

    def ensure_dirs(self) -> None:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
