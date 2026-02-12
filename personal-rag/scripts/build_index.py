from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.core.config import get_settings
from rag.rag_pipeline.indexer import Indexer


if __name__ == "__main__":
    settings = get_settings()
    indexer = Indexer(settings)
    summary = indexer.index_path(Path(settings.docs_dir))
    indexer.close()
    print(summary)
