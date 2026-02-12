from __future__ import annotations

from pathlib import Path
import fitz
from PIL import Image

from rag.core.models import Page
from rag.extractors.ocr import OCRProcessor


def extract_pdf(path: Path, doc_id: str, ocr: OCRProcessor, ocr_min_chars_threshold: int) -> list[Page]:
    pages: list[Page] = []
    with fitz.open(path) as pdf:
        for idx, page in enumerate(pdf, start=1):
            native_text = page.get_text("text").strip()
            text = native_text
            used_ocr = False
            if len(native_text) < ocr_min_chars_threshold:
                pix = page.get_pixmap(dpi=240)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = ocr.image_to_text(image)
                if ocr_text:
                    text = ocr_text
                    used_ocr = True
            pages.append(
                Page(
                    doc_id=doc_id,
                    file_path=str(path),
                    file_name=path.name,
                    page_number=idx,
                    locator=f"p.{idx}",
                    text=text,
                    extra_metadata={"type": "pdf", "ocr_used": used_ocr},
                )
            )
    return pages
