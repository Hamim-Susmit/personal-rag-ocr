from __future__ import annotations

from uuid import uuid4

from rag.core.models import Chunk, Page


def chunk_pages(pages: list[Page], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = max(1, chunk_size - chunk_overlap)
    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        idx = 0
        chunk_index = 0
        while idx < len(text):
            piece = text[idx: idx + chunk_size].strip()
            if piece:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid4()),
                        doc_id=page.doc_id,
                        file_name=page.file_name,
                        text=piece,
                        chunk_index=chunk_index,
                        locator=page.locator,
                        metadata={"page_number": page.page_number, **page.extra_metadata},
                    )
                )
                chunk_index += 1
            idx += step
    return chunks
