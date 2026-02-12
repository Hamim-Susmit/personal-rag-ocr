from __future__ import annotations

from pathlib import Path
import json
import faiss
import numpy as np


class VectorStore:
    def __init__(self, index_dir: Path, dim: int) -> None:
        self.index_dir = index_dir
        self.index_path = index_dir / "faiss.index"
        self.map_path = index_dir / "row_map.json"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.row_to_chunk_id: list[str] = []
        self.load()

    def load(self) -> None:
        if self.index_path.exists() and self.map_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.row_to_chunk_id = json.loads(self.map_path.read_text())

    def persist(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.map_path.write_text(json.dumps(self.row_to_chunk_id))

    def rebuild(self, embeddings: np.ndarray, chunk_ids: list[str]) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        self.row_to_chunk_id = list(chunk_ids)
        if len(chunk_ids):
            self.index.add(embeddings)
        self.persist()

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        scores, indices = self.index.search(query_embedding, top_k)
        results: list[tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.row_to_chunk_id):
                continue
            results.append((self.row_to_chunk_id[idx], float(score)))
        return results
