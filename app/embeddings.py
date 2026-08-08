"""Embedding backends.

Uses sentence-transformers when available; falls back to a deterministic,
dependency-free hashing embedder so the service runs and tests offline.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import List

DIM = 384


class HashingEmbedder:
    """Deterministic local embedder for dev/test/offline use."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def encode(self, texts: List[str]) -> List[List[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]{2,}", text.lower()):
            digest = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = digest % self.dim
            sign = 1.0 if (digest >> 16) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


class SentenceTransformerEmbedder:
    """Real embedding model backend (all-MiniLM-L6-v2 by default)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts).tolist()


def make_embedder():
    """Build an embedder based on EMBEDDING_BACKEND (auto|hash|st)."""
    backend = os.getenv("EMBEDDING_BACKEND", "auto").lower()
    if backend == "hash":
        return HashingEmbedder()
    if backend == "st":
        return SentenceTransformerEmbedder(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    # auto: prefer sentence-transformers, fall back to hashing
    try:
        return SentenceTransformerEmbedder(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    except Exception:
        return HashingEmbedder()
