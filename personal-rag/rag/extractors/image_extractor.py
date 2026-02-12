from __future__ import annotations

from pathlib import Path
from PIL import Image

from rag.core.models import Page
from rag.extractors.ocr import OCRProcessor


def extract_image(path: Path, doc_id: str, ocr: OCRProcessor) -> list[Page]:
    with Image.open(path) as img:
        text = ocr.image_to_text(img)
    return [
        Page(
            doc_id=doc_id,
            file_path=str(path),
            file_name=path.name,
            page_number=None,
            locator=path.name,
            text=text,
            extra_metadata={"type": "image"},
        )
    ]
