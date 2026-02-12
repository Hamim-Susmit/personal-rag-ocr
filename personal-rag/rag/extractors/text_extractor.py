from __future__ import annotations

from pathlib import Path

from rag.core.models import Page


def extract_text_like(path: Path, doc_id: str) -> list[Page]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Page(doc_id, str(path), path.name, None, "document", text, {"type": path.suffix.lower().lstrip('.')})]
