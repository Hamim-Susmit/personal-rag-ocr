from __future__ import annotations

from rag.core.models import Chunk


def dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    seen: set[tuple[str, str, str]] = set()
    result: list[Chunk] = []
    for chunk in chunks:
        key = (chunk.doc_id, chunk.locator, chunk.text)
        if key in seen:
            continue
        seen.add(key)
        result.append(chunk)
    return result
