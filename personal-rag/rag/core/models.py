from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Page:
    doc_id: str
    file_path: str
    file_name: str
    page_number: int | None
    locator: str
    text: str
    extra_metadata: dict[str, Any]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    file_name: str
    text: str
    chunk_index: int
    locator: str
    metadata: dict[str, Any]


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    file_name: str
    locator: str
