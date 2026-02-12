from __future__ import annotations

import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.core.config import Settings
from rag.rag_pipeline.indexer import Indexer
from rag.rag_pipeline.retriever import Retriever
from rag.embeddings.embedder import Embedder
from rag.store.metadata_store import MetadataStore
from rag.store.vector_store import VectorStore


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        docs = root / "docs"
        idx_dir = root / "index"
        db = root / "metadata.db"
        docs.mkdir()
        (docs / "note.txt").write_text("My passport renewal appointment is on 2025-02-10.")

        settings = Settings(docs_dir=docs, index_dir=idx_dir, sqlite_path=db)
        indexer = Indexer(settings)
        summary = indexer.index_path(docs)
        indexer.close()

        embedder = Embedder(settings.embedding_model)
        ms = MetadataStore(db)
        vs = VectorStore(idx_dir, dim=384)
        retriever = Retriever(embedder, vs, ms)
        results = retriever.retrieve("When is passport renewal appointment?", 3)
        ms.close()

        assert summary["reindexed"] == 1
        assert results, "No retrieval results"
        assert "2025-02-10" in results[0].text
        print("smoke_test passed")


if __name__ == "__main__":
    main()
