from __future__ import annotations

from pathlib import Path
from bs4 import BeautifulSoup

from rag.core.models import Page


def extract_html(path: Path, doc_id: str) -> list[Page]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text("\n", strip=True)
    return [Page(doc_id, str(path), path.name, None, "document", text, {"type": "html"})]
