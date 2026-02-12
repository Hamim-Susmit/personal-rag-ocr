from __future__ import annotations

from rag.core.models import RetrievedChunk
from rag.embeddings.embedder import Embedder
from rag.store.metadata_store import MetadataStore
from rag.store.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, metadata_store: MetadataStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.metadata_store = metadata_store

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed([query])
        matches = self.vector_store.search(query_embedding, top_k)
        ids = [chunk_id for chunk_id, _ in matches]
        chunks = self.metadata_store.get_chunks_by_ids(ids)
        score_map = {chunk_id: score for chunk_id, score in matches}
        for chunk in chunks:
            chunk.score = score_map.get(chunk.chunk_id, 0.0)
        return chunks
