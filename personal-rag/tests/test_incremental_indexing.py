from pathlib import Path

from rag.core.config import Settings
from rag.rag_pipeline.indexer import Indexer
from rag.store.metadata_store import MetadataStore


def test_deleted_file_is_removed_from_metadata(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    idx_dir = tmp_path / "index"
    db = tmp_path / "metadata.db"

    first_file = docs / "a.txt"
    first_file.write_text("alpha text")

    settings = Settings(docs_dir=docs, index_dir=idx_dir, sqlite_path=db)
    indexer = Indexer(settings)
    first_summary = indexer.index_path(docs)
    assert first_summary["reindexed"] == 1

    first_file.unlink()
    second_summary = indexer.index_path(docs)
    indexer.close()

    assert second_summary["deleted"] == 1

    ms = MetadataStore(db)
    assert ms.all_chunks() == []
    ms.close()
