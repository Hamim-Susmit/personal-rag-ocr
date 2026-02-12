from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np

from rag.core.config import Settings
from rag.core.logging import get_logger
from rag.core.utils import sha256_file
from rag.embeddings.embedder import Embedder
from rag.extractors.ocr import OCRProcessor
from rag.extractors.router import SUPPORTED_SUFFIXES, extract_file
from rag.processing.chunking import chunk_pages
from rag.processing.dedupe import dedupe_chunks
from rag.store.metadata_store import MetadataStore
from rag.store.vector_store import VectorStore

logger = get_logger(__name__)


class Indexer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ocr = OCRProcessor(lang=settings.ocr_lang)
        self.embedder = Embedder(settings.embedding_model)
        self.metadata_store = MetadataStore(settings.sqlite_path)
        self.vector_store = VectorStore(settings.index_dir, dim=384)

    def index_path(self, root_path: Path, progress_cb=None) -> dict[str, int]:
        files = [p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
        skipped = 0
        reindexed = 0
        for i, file_path in enumerate(files, start=1):
            file_hash = sha256_file(file_path)
            existing = self.metadata_store.get_doc_by_path(str(file_path))
            if existing and existing["file_hash"] == file_hash:
                skipped += 1
                if progress_cb:
                    progress_cb(i, len(files), f"Skipped unchanged: {file_path.name}")
                continue

            doc_id = str(uuid4()) if not existing else existing["doc_id"]
            self.metadata_store.upsert_document(doc_id, str(file_path), file_path.name, file_hash)
            self.metadata_store.delete_doc_chunks(doc_id)

            pages = extract_file(file_path, doc_id, self.ocr, self.settings.ocr_min_chars_threshold)
            chunks = dedupe_chunks(chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap))
            self.metadata_store.insert_chunks(chunks)
            reindexed += 1
            logger.info("Indexed %s with %d chunks", file_path, len(chunks))
            if progress_cb:
                progress_cb(i, len(files), f"Indexed: {file_path.name} ({len(chunks)} chunks)")

        self._rebuild_vector_index()
        return {"total": len(files), "skipped": skipped, "reindexed": reindexed}

    def _rebuild_vector_index(self) -> None:
        rows = self.metadata_store.all_chunks()
        chunk_ids = [row["chunk_id"] for row in rows]
        texts = [row["text"] for row in rows]
        embeddings = self.embedder.embed(texts) if texts else np.zeros((0, 384), dtype="float32")
        self.vector_store.rebuild(embeddings, chunk_ids)

    def close(self) -> None:
        self.metadata_store.close()
