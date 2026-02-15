from __future__ import annotations

import hashlib
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.core.logging import get_logger

logger = get_logger(__name__)


class Embedder:
    def __init__(self, model_name: str, dim: int = 384) -> None:
        self.dim = dim
        self.model = None
        self._mode = "hash"

        allow_download = os.getenv("EMBEDDING_ALLOW_DOWNLOAD", "0") == "1"

        try:
            self.model = SentenceTransformer(model_name, local_files_only=not allow_download)
            model_dim = self.model.get_sentence_embedding_dimension()
            if model_dim != self.dim:
                raise ValueError(
                    f"Embedding dimension mismatch: model={model_dim}, configured={self.dim}. "
                    "Set EMBEDDING_DIM to match the model output."
                )
            self._mode = "model"
        except Exception as exc:
            logger.warning(
                "Falling back to deterministic hash embeddings because model '%s' could not be loaded: %s",
                model_name,
                exc,
            )

    def _hash_embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:8], "little") % self.dim
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")

        if self._mode == "model":
            vectors = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            vectors = vectors.astype("float32")
        else:
            vectors = np.vstack([self._hash_embed(t) for t in texts]).astype("float32")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms
