from __future__ import annotations

from pathlib import Path
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT

from rag.core.models import Page


def extract_epub(path: Path, doc_id: str) -> list[Page]:
    book = epub.read_epub(str(path))
    pages: list[Page] = []
    counter = 1
    for item in book.get_items():
        if item.get_type() != ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text("\n", strip=True)
        pages.append(Page(doc_id, str(path), path.name, counter, f"section {counter}", text, {"type": "epub"}))
        counter += 1
    return pages
