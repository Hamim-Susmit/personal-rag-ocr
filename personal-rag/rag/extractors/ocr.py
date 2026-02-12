from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from PIL import Image, ImageOps
import pytesseract

from rag.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OCRProcessor:
    lang: str = "eng"
    threshold: int | None = 170
    max_dim: int = 2000

    def preprocess(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("L")
        width, height = image.size
        largest = max(width, height)
        if largest > self.max_dim:
            ratio = self.max_dim / largest
            image = image.resize((int(width * ratio), int(height * ratio)))
        if self.threshold is not None:
            image = image.point(lambda p: 255 if p > self.threshold else 0)
        return image

    def image_to_text(self, image: Image.Image) -> str:
        try:
            processed = self.preprocess(image)
            return pytesseract.image_to_string(processed, lang=self.lang).strip()
        except Exception as exc:
            logger.exception("OCR failed: %s", exc)
            return ""

    def bytes_to_text(self, content: bytes) -> str:
        try:
            image = Image.open(BytesIO(content))
            return self.image_to_text(image)
        except Exception as exc:
            logger.exception("Unable to load image bytes for OCR: %s", exc)
            return ""
