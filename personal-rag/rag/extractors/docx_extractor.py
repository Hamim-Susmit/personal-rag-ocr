from __future__ import annotations

from pathlib import Path
from docx import Document

from rag.core.models import Page


def extract_docx(path: Path, doc_id: str) -> list[Page]:
    doc = Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [Page(doc_id, str(path), path.name, None, "document", text, {"type": "docx"})]
