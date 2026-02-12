from __future__ import annotations

from pathlib import Path

from rag.core.logging import get_logger
from rag.core.models import Page
from rag.extractors.docx_extractor import extract_docx
from rag.extractors.epub_extractor import extract_epub
from rag.extractors.html_extractor import extract_html
from rag.extractors.image_extractor import extract_image
from rag.extractors.ocr import OCRProcessor
from rag.extractors.pdf_extractor import extract_pdf
from rag.extractors.pptx_extractor import extract_pptx
from rag.extractors.text_extractor import extract_text_like
from rag.extractors.xlsx_extractor import extract_xlsx

logger = get_logger(__name__)


SUPPORTED_SUFFIXES = {
    ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".html", ".htm", ".epub"
}


def extract_file(path: Path, doc_id: str, ocr: OCRProcessor, ocr_min_chars_threshold: int) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path, doc_id, ocr, ocr_min_chars_threshold)
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return extract_image(path, doc_id, ocr)
    if suffix == ".docx":
        return extract_docx(path, doc_id)
    if suffix == ".pptx":
        return extract_pptx(path, doc_id)
    if suffix == ".xlsx":
        return extract_xlsx(path, doc_id)
    if suffix in {".txt", ".md"}:
        return extract_text_like(path, doc_id)
    if suffix in {".html", ".htm"}:
        return extract_html(path, doc_id)
    if suffix == ".epub":
        return extract_epub(path, doc_id)

    logger.warning("Unsupported file type skipped: %s", path)
    return []
